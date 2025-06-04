import os
import json
import requests
from PIL import Image
import pytesseract
from pytesseract import Output
from dotenv import load_dotenv
import time  # For measuring execution time

# Start timer
start_time = time.time()

# Load environment variables from .env file
load_dotenv()

# Load API keys and paths from .env
API_TOKEN = os.getenv("API_TOKEN")
TESSERACT_PATH = os.getenv("TESSERACT_PATH")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Not used currently

# Set Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# Extract text and bounding boxes from image using OCR
def extract_text_with_boxes(image_path):
    image = Image.open(image_path)
    data = pytesseract.image_to_data(image, output_type=Output.DICT)
    
    results = []
    for i in range(len(data['text'])):
        # Keep only confident, non-empty text
        if data['text'][i].strip() and int(data['conf'][i]) > 0:
            results.append({
                'text': data['text'][i],
                'bbox': [
                    data['left'][i],
                    data['top'][i],
                    data['left'][i] + data['width'][i],
                    data['top'][i] + data['height'][i]
                ]
            })
    return results

# Send extracted text to Sage GPT for analysis
def analyze_with_sage_gpt(text, token):
    url = "https://api.sage.cudasvc.com/openai/chat/completions"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": text}
        ]
    }
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    result = response.json()
    return result['choices'][0]['message']['content']

# Main function to run everything
def main():
    image_path = "assets/example.jpg"         # Input image path
    output_path = "output/result.json"        # Output JSON path

    # Run OCR on the image
    ocr_results = extract_text_with_boxes(image_path)

    # Combine all OCR text into one string
    combined_text = " ".join([item['text'] for item in ocr_results])

    # Analyze text using Sage GPT
    sage_result = analyze_with_sage_gpt(combined_text, API_TOKEN)

    # Save OCR and GPT results to JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "ocr_results": ocr_results,
            "sage_gpt_analysis": sage_result
        }, f, indent=4)

    print(f"Processed {image_path}, results saved to {output_path}")

# Run the script
if __name__ == "__main__":
    main()
    # Show total time taken
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
