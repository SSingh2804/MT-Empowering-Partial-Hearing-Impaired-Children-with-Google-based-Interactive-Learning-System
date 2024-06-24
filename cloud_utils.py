from google.cloud import storage
from google.cloud import texttospeech
import requests
from PIL import Image
from io import BytesIO
import os

# Function to list images in a GCS bucket
def list_images_in_bucket(bucket_name, folder):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=folder)

    return [blob.name for blob in blobs if blob.name.endswith(('.jpg', '.png'))]

# Function to generate a signed URL for a GCS object
def generate_signed_url(bucket_name, blob_name):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    url = blob.generate_signed_url(version="v4", expiration=3600)  # URL valid for 1 hour
    return url

# Function to fetch an image from a URL
def fetch_image(url):
    response = requests.get(url)
    response.raise_for_status()  # Ensure we notice bad responses
    img = Image.open(BytesIO(response.content))
    return img

# Function to fetch all images from the GCS bucket and store them in a dictionary
def fetch_all_images_from_gcs(bucket_name, folder):
    image_paths = list_images_in_bucket(bucket_name, folder)
    image_urls = {}
    for path in image_paths:
        try:
            url = generate_signed_url(bucket_name, path)
            image_urls[os.path.basename(path)] = url
        except Exception as e:
            print(f"Error generating signed URL for {path}: {e}")
    return image_urls

# Function to fetch all ASL images from the storage bucket and store in cache
def fetch_all_asl_images(bucket_name, folder):
    # Get the list of image paths in the bucket
    image_paths = list_images_in_bucket(bucket_name, folder)
    
    # Create a dictionary to store the images
    asl_images = {}

    # Fetch each image and store it in the dictionary
    for image_path in image_paths:
        signed_url = generate_signed_url(bucket_name, image_path)
        image = fetch_image(signed_url)
        image_name = os.path.basename(image_path).split('.')[0]  # Extract the image name without extension
        asl_images[image_name] = image

    return asl_images

def google_tts(text_block, client):

    synthesis_input = texttospeech.SynthesisInput(text=text_block)

    voice = texttospeech.VoiceSelectionParams(
    language_code = "en-US",
    name = 'en-US-Standard-G'    
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding = texttospeech.AudioEncoding.MP3,
        speaking_rate = 1,
        pitch = 1
    )

    response = client.synthesize_speech(
        input = synthesis_input,
        voice = voice, 
        audio_config = audio_config
    )
    with open("temp/output.mp3","wb") as output:
        output.write(response.audio_content)
        #print('Audio content written to file output.mp3 ')



