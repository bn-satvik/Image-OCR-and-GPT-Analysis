import os
import json
import requests
from PIL import Image
import pytesseract
from pytesseract import Output
from dotenv import load_dotenv
import time  # import time module

# Start timer
start_time = time.time()

# Load environment variables from .env
load_dotenv()

API_TOKEN = os.getenv("API_TOKEN")
TESSERACT_PATH = os.getenv("TESSERACT_PATH")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

def extract_text_with_boxes(image_path):
    image = Image.open(image_path)
    data = pytesseract.image_to_data(image, output_type=Output.DICT)
    
    results = []
    for i in range(len(data['text'])):
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

def main():
    image_path = "assets/example.jpg"
    output_path = "output/result.json"
    
    ocr_results = extract_text_with_boxes(image_path)
    combined_text = " ".join([item['text'] for item in ocr_results])
    
    sage_result = analyze_with_sage_gpt(combined_text, API_TOKEN)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "ocr_results": ocr_results,
            "sage_gpt_analysis": sage_result
        }, f, indent=4)
    
    print(f"Processed {image_path}, results saved to {output_path}")

if __name__ == "__main__":
    main()
    # Stop timer and print elapsed time
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
