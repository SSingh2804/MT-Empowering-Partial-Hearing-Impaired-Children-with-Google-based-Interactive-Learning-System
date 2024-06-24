import streamlit as st
import os
import requests
import random
import tempfile

from io import BytesIO
from google.cloud import texttospeech
from google.cloud import aiplatform
from PIL import Image

from model_utils import (preprocess_image, process_yolo_response, get_all_detected_labels)
from helper_util import (display_random_image, parse_and_convert_to_uppercase, search_asl_images,
                         read_serialized_response)
from cloud_utils import (list_images_in_bucket, generate_signed_url, fetch_image,
                         fetch_all_images_from_gcs, fetch_all_asl_images, google_tts)



# Set Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "acs-ssingh-yolo-421023-e659f224998a.json"
PROJECT = "acs-ssingh-yolo-421023"
REGION = "europe-west3"
BUCKET_NAME = 'object-prediction-ml-model'
FOLDER_SAMPLE_OBJECTS = 'Object_images/'
FOLDER_ASL = "Hand_signs"
ENDPOINT_ID = "7495436737246658560"

# Define the path for the temporary directory
temp_dir = os.path.join(os.getcwd(), 'temp')
# Create the directory if it does not exist
os.makedirs(temp_dir, exist_ok=True)

def predict_with_endpoint(REGION, PROJECT, ENDPOINT_ID, image_tensor):
    """
    Connects to a deployed model endpoint in Google Vertex AI and makes a prediction.

    Args:
        project_id (str): GCP project ID.
        region (str): GCP region where the model is deployed.
        endpoint_id (str): ID of the deployed endpoint.
        instances (list): List of instances to predict on. Each instance should be a dictionary.
        parameters (dict, optional): Optional parameters for the prediction. Default is None.
        api_endpoint (str, optional): API endpoint for Vertex AI. Default is None, which will use the region to infer the endpoint.

    Returns:
        dict: The prediction response from the model.
    """
    #if api_endpoint is None:
    api_endpoint = f'{REGION}-aiplatform.googleapis.com'
    #print("API_endpoint", api_endpoint)
    # Create a client
    client_options = {'api_endpoint': api_endpoint}
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)

    # Get the fully qualified endpoint name
    endpoint = client.endpoint_path(project=PROJECT, location=REGION, endpoint=ENDPOINT_ID)
    #print("Endpoint", endpoint)
    # Prepare the request payload
    instances = image_tensor.tolist()
     
    response = client.predict(endpoint=endpoint, instances=instances, parameters=None)
    with open('temp/cloud_response.txt', 'w') as file:
            file.write(str(response))
    
    return response


# Initialize detected_objects with the detected label
detected_objects = []
individual_letters = []
# Set Streamlit page configuration
st.set_page_config(layout="wide", page_title="Object Detection for Partial Hearing")

# Application Title
st.title("Object Identifier")

# Sidebar Configuration
st.sidebar.header('Categories')
selected_category = st.sidebar.radio(
    "Choose one",
    ("Home :house_buildings:", "Kitchen :spoon::bowl_with_spoon:", "Food :broccoli:",
     "Animals :cow:", "Transport :car:", "Sports :field_hockey_stick_and_ball:",
     "Randomized :question:"),
    index=6  # Setting the default to "Random"
).split()[0]

# Debugging: Display selected category
st.sidebar.markdown(f"**Selected Category:** {selected_category}")

# Initialize session state for various flags
if 'view_active' not in st.session_state:
    st.session_state.view_active = False
if 'upload_active' not in st.session_state:
    st.session_state.upload_active = False

# Sidebar options for other functionalities
st.sidebar.subheader("Additional Options")
if st.sidebar.button(':eye:', key='view'):
    st.session_state.view_active = not st.session_state.view_active
if st.sidebar.button(':open_file_folder:', key='upload'):
    st.session_state.upload_active = not st.session_state.upload_active

all_gcs_images_list = list_images_in_bucket(BUCKET_NAME, FOLDER_SAMPLE_OBJECTS)
all_hand_signs = list_images_in_bucket(BUCKET_NAME, FOLDER_ASL)


# Randomly choose an image from the image list in GCS bucket.
def display_random_image(all_gcs_images_list, category):
    if category == "Randomized":
        return random.choice(all_gcs_images_list)
    else:
        # Filter image paths by category prefix
        filtered_paths = [path for path in all_gcs_images_list
                          if path.startswith(f'Object_images/{category}')]

        if not filtered_paths:
            raise ValueError(f"No images found for the category '{category}'.")
        return random.choice(filtered_paths)


@st.cache_data
def fetch_all_images_from_gcs_cache():
    return fetch_all_images_from_gcs(BUCKET_NAME, FOLDER_SAMPLE_OBJECTS)


# Fetch all images from the GCS bucket
all_gcs_images = fetch_all_images_from_gcs_cache()

# Shuffle the images
shuffled_images = list(all_gcs_images.items())
random.shuffle(shuffled_images)

# Function to fetch all ASL images from the storage bucket and store in cache
@st.cache_data
def fetch_all_asl_images_cache():
    return fetch_all_asl_images(BUCKET_NAME, FOLDER_ASL)

# Function to search relevant letter images from the pool of all ASL images
def search_asl_images(letters, asl_images):
    # Retrieve images from the cached dictionary
    images = []
    for letter in letters:
        if letter in asl_images:
            images.append(asl_images[letter])
    return images


# Request the image
client = texttospeech.TextToSpeechClient()

# Get the ASL images from the cached dictionary
asl_images_cache = fetch_all_asl_images_cache()

# Initialize session state for random_image_path
if 'random_image_path' not in st.session_state:
    st.session_state.random_image_path = None
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None


# Image Upload Functionality
if st.session_state.get('upload_active', False):
    st.subheader("Upload an Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg"])

    # If an image is uploaded, display it
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_filename = temp_file.name
            temp_file.write(uploaded_file.getvalue())
            st.session_state.uploaded_image = temp_filename
            st.session_state.random_image_path = None  # Reset random image path

    # If no image is uploaded, do not display any image
    else:
        st.session_state.uploaded_image = None
        st.session_state.random_image_path = None  # Reset random image path


# YOLO parameters
net_h, net_w = 128, 128
obj_thresh, nms_thresh = 0.5, 0.45
anchors = [[116, 90, 156, 198, 373, 326], [30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]]
labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
          "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
          "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
          "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
          "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
          "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
          "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
          "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
          "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
          "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

# Main container for layout
with st.container():

    # Define the main columns
    image_col, result_col = st.columns([1, 1])
    
    with image_col:
        st.markdown("### Image Selection and Run Prediction")

        # Divide image_col into two sub-columns
        col_detect_button, col_next_button = st.columns(2)
        
        # Control buttons
        if col_next_button.button('Next', disabled=st.session_state.upload_active):
            try:
                random_gcs_image = display_random_image(all_gcs_images_list, selected_category)
                signed_url = generate_signed_url(BUCKET_NAME, random_gcs_image)
                 # Fetch the image from the signed URL
                response = requests.get(signed_url)
                image = Image.open(BytesIO(response.content))
                
                # Save the image to the local temporary directory
                temp_filename = os.path.join(temp_dir, 'random_image.jpg')
                image.save(temp_filename)
                st.session_state.random_image_path = temp_filename
                # random_gcs_image = fetch_image(signed_url)
                # with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as gcs_temp_file:
                #     gcs_temp_filename = gcs_temp_file.name
                #     random_gcs_image.save(gcs_temp_filename)
                #st.session_state.random_image_path = gcs_temp_filename
                
                st.session_state.uploaded_image = None
            except Exception as e:
                st.error(f"Error selecting random image: {e}")
            st.rerun()  # Rerun the script to update the displayed image

        
        if col_detect_button.button('Detect'):
            if st.session_state.random_image_path:
                image, new_image, image_h, image_w = preprocess_image(st.session_state.random_image_path, net_h, net_w)
            elif st.session_state.uploaded_image:
                image, new_image, image_h, image_w = preprocess_image(st.session_state.uploaded_image, net_h, net_w)

            #Function call to request deployed model for prediction
            predict_with_endpoint(REGION, PROJECT, ENDPOINT_ID, new_image)

            #Convert response from google into array
            response_yolos = read_serialized_response()

            #Post processing of received response to detect labels
            boxes = process_yolo_response(response_yolos, anchors, obj_thresh, nms_thresh, net_h, net_w, image_h, image_w)
            detected_objects = get_all_detected_labels(boxes, labels, obj_thresh)
            if detected_objects:
                #Parse detected labels into individual characters
                individual_letters = [parse_and_convert_to_uppercase(obj) for obj in detected_objects]

        # Display image based on the current state
        if st.session_state.upload_active and st.session_state.uploaded_image:
            # Display uploaded image if upload button is active and an image is uploaded
            image_path = st.session_state.uploaded_image
        elif not st.session_state.upload_active and st.session_state.random_image_path:
            # Display random image based on the chosen category if upload button is not active
            image_path = st.session_state.random_image_path
        else:
            # If no condition is met, do not display any image
            image_path = None

        # Display image in image_col
        if image_path:
            image = image_path
            st.image(image, width=350)

    with result_col:
        st.markdown("### Detected Object and ASL Images")
        if detected_objects:
            for detected_object in detected_objects:
                parsed_object = parse_and_convert_to_uppercase(detected_object)
                st.markdown(f"#### Detected Object: {parsed_object}")
                
                #Audio Using Google TTS API
                result = google_tts(detected_object,client)
                audio_file = open("temp/output.mp3", "rb")
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format="audio/mp3", autoplay = True)

                # Display sign images
                st.markdown("### ASL Symbols")
                
                # Get the ASL images from the cached dictionary
                asl_images = search_asl_images(parsed_object, asl_images_cache)

                # Display the ASL images in Streamlit
                sign_image_cols = st.columns(len(asl_images))
                for col, img in zip(sign_image_cols, asl_images):
                    col.image(img, width=60)


# Image Gallery
if st.session_state.get('view_active', False):
    st.subheader("Image Gallery")
    if shuffled_images:
        cols = st.columns(5)  # Number of columns in the grid
        for i, (image_name, image_url) in enumerate(shuffled_images):
            with cols[i % 5]:  # Iterate over columns cyclically
                try:
                    image = fetch_image(image_url)
                    st.image(image, use_column_width=True)
                except Exception as e:
                    st.error(f"Error fetching image {image_name}: {e}")
    else:
        st.write("No JPG images found in the directory.")