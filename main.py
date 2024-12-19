import fitz  # PyMuPDF
from PIL import Image
from dotenv import load_dotenv
import os
import io
import google.generativeai as genai  # Import the necessary Google Generative AI library
import instructor

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))  # Use the environment variable for API key

import fitz  # PyMuPDF
from io import BytesIO
from dotenv import load_dotenv
import google.generativeai as genai  # Import the necessary Google Generative AI library

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))  # Use the environment variable for API key

def extract_data_from_image(image_bytes: bytes) -> str:
    """Extract data from an image using the Gemini model."""
    try:
        instruction_prompt = (
            "Analyze the following image of a form and extract the data from it. "
            "Extract values from any text input fields, identify checked checkboxes, "
            "Extract value from Y/N checkboxes Y=true and if not checked N=false, "
            "extract names, account numbers, contact numbers, email addresses, dates, "
            "addresses, and any reference codes."
            "give equal key value pairs for each field."
        )
        
        # instruction_prompt = (
        #     "Carefully analyze this document image. Extract data with the following guidelines:\n"
        #     "1. Identify the document type (e.g., application form, invoice, receipt)\n"
        #     "2. Extract all visible text fields with their corresponding values\n"
        #     "3. Capture key information such as:\n"
        #     "   - Full names\n"
        #     "   - Contact information (phone, email)\n"
        #     "   - Addresses\n"
        #     "   - Reference numbers\n"
        #     "4. For checkbox fields, only include checked boxes\n"
        #     "5. Preserve the context and meaning of each field\n"
        #     "6. If unsure about a field, leave it empty rather than guessing\n"
        #     "7. Provide the output in a clean, structured JSON format\n"
        #     "\nBe precise, thorough, and focus on information clarity and accuracy."
        # )

        # Generate content using the Gemini model
        model = genai.GenerativeModel("gemini-1.5-flash") 
        response = model.generate_content(
        [
            {
                "mime_type": "image/jpeg",
                "data": image_bytes,
            },
            instruction_prompt,
        ]
    )

        extracted_data = response.text.strip()
        print(f"Gemini response: {extracted_data}")

        return extracted_data
    except Exception as e:
        print(f"Error extracting data from image with Gemini model: {e}")
        return ""

def render_pdf_page_as_image(page, dpi=300):
    """Render a PDF page as a high-quality image."""
    try:
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        return buffer.getvalue()  # Return image bytes
    except Exception as e:
        print(f"Error rendering page as image: {e}")
        return None

def process_pdf(pdf_path: str, dpi=300) -> str:
    """Render PDF pages as high-quality images and extract data using Gemini."""
    all_data = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page_num, page in enumerate(doc):
                print(f"Processing page {page_num + 1}...")
                image_bytes = render_pdf_page_as_image(page, dpi=dpi)
                if image_bytes:
                    extracted_data = extract_data_from_image(image_bytes)
                    if extracted_data:
                        all_data += extracted_data + "\n"
    except Exception as e:
        print(f"An error occurred while processing the PDF: {e}")

    return all_data

if __name__ == "__main__":
    pdf_path = "test.pdf"
    extracted_data = process_pdf(pdf_path)

    print("Extracted Data:")
    print(extracted_data)

    with open("extracted_data.txt", "w", encoding="utf-8") as f:
        f.write(extracted_data)
    print("Results saved to 'extracted_data.txt'")
