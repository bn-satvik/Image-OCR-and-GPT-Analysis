import os
import json
import base64
import requests
from dotenv import load_dotenv
import time

# Start timer to measure script execution time
start_time = time.time()

# Load environment variables from .env file
load_dotenv()

# Get Sage API token from environment
API_TOKEN = os.getenv("API_TOKEN")

# Function to convert an image to a base64-encoded string
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Function to send the image and prompt to Sage GPT and extract text with bounding boxes
def extract_text_and_boxes_with_sage(image_path, token):
    # Convert image to base64
    base64_image = encode_image_to_base64(image_path)

    # Prompt to instruct Sage GPT to extract text and bounding boxes
    prompt = (
        "Please extract all visible text in this image and return it as a list of objects "
        "with this format: {text: '...', bounding_box: [x1, y1, x2, y2]}, where coordinates are in image pixel space."
    )

    # Sage API endpoint
    url = "https://api.sage.cudasvc.com/openai/chat/completions"
    
    # HTTP headers including authorization
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    # Compose request data: prompt + image (in base64)
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": prompt},  # Send prompt
            {
                "role": "user",
                "content": [  # Attach image as base64
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
    }

    # Send POST request to Sage API
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()  # Raise error if request fails

    # Extract GPT response text
    result = response.json()
    return result['choices'][0]['message']['content']

# Main function to run the full process
def main():
    image_path = "assets/example.jpg"         # Input image path
    output_path = "output/result.json"        # Output file path

    # Get result from Sage GPT
    sage_response = extract_text_and_boxes_with_sage(image_path, API_TOKEN)

    # Save Sage result to JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "sage_gpt_result": sage_response
        }, f, indent=4)

    print(f"Processed {image_path}, results saved to {output_path}")

# Run the script
if __name__ == "__main__":
    main()
    # Print total execution time
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
