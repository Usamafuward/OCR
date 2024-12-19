import fitz  # PyMuPDF
from PIL import Image
from openai import OpenAI
from typing import List, Dict
from dotenv import load_dotenv
import json
import os
import io
import base64
import requests
import instructor
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def save_image(image: Image.Image, page_num: int, img_num: int) -> str:
    """Save the image and return the file path."""
    image_path = f"output_image_page_{page_num}_img_{img_num}.png"
    image.save(image_path, format="PNG")
    return image_path

def render_pdf_as_images(pdf_path: str) -> List[Dict[str, str]]:
    """Render PDF pages as images and save them with URLs."""
    all_images = []
    try:
        pdf_document = fitz.open(pdf_path)
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            image_list = page.get_images(full=True)

            for img_num, img in enumerate(image_list):
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                # Create an image from bytes
                image = Image.open(io.BytesIO(image_bytes))

                if image_ext != "png":
                    image = image.convert("RGB")
                image_path = save_image(image, page_num, img_num)
                all_images.append({"image_path": image_path, "image": image})
                
    except Exception as e:
        print(f"An error occurred while processing the PDF: {e}")
    finally:
        pdf_document.close()

    return all_images

def extract_data_with_gpt4_from_image(image: Image.Image) -> str:
    """Extract data directly from an image using OpenAI's GPT-4 model."""
    try:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        buffered.seek(0)

        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')  # Decode to string
        
        client = instructor.patch(OpenAI(api_key=openai.api_key))

        instruction_prompt = (
            "You have received an image of a form with various fields, including checkboxes, text inputs, and numbers. Your task is to analyze the input image and extract the values of the following fields:"
            "1. Extract values from any text input fields."
            "2. Identify which checkboxes are checked and which are not. Values should be either 'checked' or 'unchecked'."
            "3. Extract alphabetic values for names (e.g., 'name: John Doe')."
            "4. Extract numerical values for account and contact numbers (e.g., 'account no: 12345678', 'contact no: 9876543210')."
            "5. Look for standard email formats and extract them (e.g., 'email: example@example.com')."
            "6. Identify any dates and extract them (e.g., 'date of birth: 01/01/1990')."
            "7. Extract values containing both alphabetic and numerical data for addresses (e.g., 'address: 123 Main St')."
            "8. Extract values where the first character is an alphabet followed by numerals (e.g., 'passport no: A1234567')."
            "9. Extract any unique identification codes or reference numbers."
            "10. Identify any instances of alphanumeric combinations (e.g., 'reference code: A123B')."

            "After analyzing the form, provide the extracted data as key: value."

            "Start the analysis now."
        )

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": instruction_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_base64}"
                            }
                        }
                    ]
                }
            ],
            temperature=0,
            seed=42
        )
        
        extracted_data = response.choices[0].message.content.strip()
        print(f"GPT-4 response: {extracted_data}")

        return extracted_data
    except Exception as e:
        print(f"Error extracting data from image with GPT-4 model: {e}")
        return ""

def extract_data_from_pdf(pdf_path: str):
    print("Rendering PDF pages to images...")
    images_info = render_pdf_as_images(pdf_path)
    print(f"Rendered {len(images_info)} pages as images.")
    
    all_data = ""
    for index, image_info in enumerate(images_info):
        print(f"Processing page {index + 1}...")

        extracted_data = extract_data_with_gpt4_from_image(image_info['image'])
        
        if extracted_data:
            all_data += extracted_data + "\n"

    print("Extracted Data:")
    print(all_data)

    with open("extracted_data.txt", "w", encoding="utf-8") as f:
        f.write(all_data)
    print("Results saved to 'extracted_data.txt'")
    return all_data

if __name__ == "__main__":
    pdf_path = "test.pdf"
    extract_data_from_pdf(pdf_path)