import random
import numpy as np
import re

#Randomly chose a image from the image list in GCS bucket.
def display_random_image(all_gcs_images_list, category):
    if category == "Randomized":
        return random.choice(all_gcs_images_list)
    else:
        # Filter image paths by category prefix
        filtered_paths = [path for path in all_gcs_images_list
     if path.startswith(f'Object_images/{category}')]
        print(filtered_paths)
        if not filtered_paths:
            raise ValueError(f"No images found for the category '{category}'.")
        return random.choice(filtered_paths)
    
def parse_and_convert_to_uppercase(input_string):
    # Convert the input string to uppercase letters
    return [char.upper() for char in input_string]


#Function to search relevant letter images from the pool of all ASL images
def search_asl_images(letters, asl_images):
    # Retrieve images from the cached dictionary
    images = []
    for letter in letters:
        if letter in asl_images:
            images.append(asl_images[letter])
    return images


# Function to extract numbers for a given key
def extract_data_from_response(cloud_response, key):
    if not isinstance(cloud_response, str):
        raise TypeError("Content must be a string.")
    numbers = []
    # Regular expression to find the key and capture everything until the next 'fields' or the end of the content
    pattern = re.compile(rf'key:\s*"{key}".*?(\{{.*?)(?=\bfields\b|\Z)', re.DOTALL)
    match = pattern.search(cloud_response)
    
    if match:
        # Extract the block of text related to the key
        block = match.group(1)
        # Find all number_value entries within the block
        number_pattern = re.compile(r'number_value:\s*(-?\d+\.?\d*)')
        numbers.extend([float(num) for num in number_pattern.findall(block)])
    
    return numbers

def reshape_array(input_array, dimension):

    # Check if the array can be reshaped to the desired dimensions
    if len(input_array) == dimension * dimension * 255:
        # Reshape the array and add an additional dimension
        reshaped_array = np.reshape(input_array, (1, dimension, dimension, 255))
        return reshaped_array
    else:
        print(f"The array for dimension {dimension} cannot be reshaped to the desired dimensions. Please check the array size.")
        return None
    
#Function to parse Google serialized response into numpy array
def read_serialized_response():
    # Read the file contents
    with open('temp/cloud_response.txt', 'r') as file:
        cloud_reponse = file.read()

    # Initialize arrays to hold the numbers for each key
    conv_81 = extract_data_from_response(cloud_reponse, 'conv_81')
    conv_93 = extract_data_from_response(cloud_reponse, 'conv_93')
    conv_105 = extract_data_from_response(cloud_reponse, 'conv_105')

    # Reshape the arrays and add an additional dimension
    reshaped_conv_81 = reshape_array(conv_81, 4)
    reshaped_conv_93 = reshape_array(conv_93, 8)
    reshaped_conv_105 = reshape_array(conv_105, 16)

    # Combine the reshaped arrays into a bigger array
    response_yolos = [reshaped_conv_81, reshaped_conv_93, reshaped_conv_105]

    return response_yolos