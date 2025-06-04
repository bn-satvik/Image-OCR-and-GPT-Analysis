# Image OCR and GPT Analysis

This project extracts text with bounding boxes from an image using Tesseract OCR, then sends the extracted text to Sage GPT (`gpt-4o-mini`) for analysis. The results are saved to a JSON file.

## Requirements

- Python 3.x
- Tesseract OCR installed ([Download here](https://github.com/tesseract-ocr/tesseract))
- Required Python packages: `pillow`, `pytesseract`, `requests`, `python-dotenv`

## Setup

1. Install dependencies:
   ```bash
   pip install pillow pytesseract requests python-dotenv
