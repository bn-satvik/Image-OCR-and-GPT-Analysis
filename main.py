import os
import json
import base64
import requests
import fitz  # PyMuPDF
from dotenv import load_dotenv
import time

# Start timer
start_time = time.time()

# Load environment variables
load_dotenv()
API_TOKEN = os.getenv("API_TOKEN")

# Utility: encode image bytes to base64
def encode_image_bytes_to_base64(image_bytes):
    return base64.b64encode(image_bytes).decode("utf-8")

# Utility: encode file (image path) to base64
def encode_image_file_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Extract all images from PDF as base64 list
def extract_images_from_pdf(pdf_path):
    images_base64 = []
    doc = fitz.open(pdf_path)

    for page_index in range(len(doc)):
        for img_index, img in enumerate(doc.get_page_images(page_index)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_b64 = encode_image_bytes_to_base64(image_bytes)
            images_base64.append({
                "page": page_index + 1,
                "image_index": img_index + 1,
                "base64": image_b64
            })

    return images_base64

# Send base64 image to Sage GPT
def extract_text_and_boxes_with_sage(base64_image, token):
    prompt = (
        "Please extract all visible text in this image and return it as a list of objects "
        "with this format: {text: '...', bounding_box: [x1, y1, x2, y2]}, where coordinates are in image pixel space."
    )

    url = "https://api.sage.cudasvc.com/openai/chat/completions"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": prompt},
            {
                "role": "user",
                "content": [
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

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    result = response.json()
    return result['choices'][0]['message']['content']

# Main logic
def main():
    input_path = "assets/example.pdf"  # Or "assets/image.jpg"
    output_dir = "output/results"
    os.makedirs(output_dir, exist_ok=True)

    ext = os.path.splitext(input_path)[1].lower()
    all_results = []

    if ext == ".pdf":
        print(f"Detected PDF: {input_path}")
        images_info = extract_images_from_pdf(input_path)

        if not images_info:
            print("No images found in the PDF.")
            return

        for i, img_info in enumerate(images_info):
            try:
                print(f"Processing Page {img_info['page']}, Image {img_info['image_index']}")
                result_text = extract_text_and_boxes_with_sage(img_info["base64"], API_TOKEN)

                result_path = os.path.join(
                    output_dir, f"result_page{img_info['page']}_img{img_info['image_index']}.json"
                )
                with open(result_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "page": img_info['page'],
                        "image_index": img_info['image_index'],
                        "sage_gpt_result": result_text
                    }, f, indent=4)

                all_results.append({
                    "page": img_info['page'],
                    "image_index": img_info['image_index'],
                    "sage_gpt_result": result_text
                })
            except Exception as e:
                print(f"Failed on Page {img_info['page']} Image {img_info['image_index']}: {e}")

    elif ext in [".jpg", ".jpeg", ".png"]:
        print(f"Detected image: {input_path}")
        try:
            base64_image = encode_image_file_to_base64(input_path)
            result_text = extract_text_and_boxes_with_sage(base64_image, API_TOKEN)

            result_path = os.path.join(output_dir, "result_image.json")
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "file": input_path,
                    "sage_gpt_result": result_text
                }, f, indent=4)

            all_results.append({
                "file": input_path,
                "sage_gpt_result": result_text
            })

        except Exception as e:
            print(f"Error processing image: {e}")

    else:
        print("Unsupported file type. Please use a .pdf, .jpg, .jpeg, or .png file.")
        return

    # Save combined result
    combined_path = os.path.join(output_dir, "all_combined_results.json")
    with open(combined_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4)

    print(f"All results saved to {output_dir}")

if __name__ == "__main__":
    main()
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
