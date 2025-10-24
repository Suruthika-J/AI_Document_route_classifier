import ollama
import json
from PyPDF2 import PdfReader
import io
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image

# --- For Windows users, you might need to set the tesseract path explicitly ---
# If you did NOT add Tesseract to your system PATH, uncomment the line below.
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text_with_ocr(file_bytes):
    """
    Extracts text from PDF pages using OCR if they are images.
    Includes performance and accuracy improvements.
    """
    text = ""
    page_limit = 5  # IMPROVEMENT: Limit to first 5 pages for performance.
    
    # Convert PDF bytes to a list of PIL images
    images = convert_from_bytes(file_bytes)
    
    # Only process pages up to the specified limit
    for i, image in enumerate(images[:page_limit]):
        try:
            # IMPROVEMENT: Specify language for better accuracy.
            text += pytesseract.image_to_string(image, lang='eng') + "\n"
        except Exception as e:
            print(f"Error during OCR on page {i+1}: {e}")
            continue
    return text

def extract_text_from_file(file):
    """
    Extracts text from an uploaded file (supports .txt and .pdf).
    Now includes an OCR fallback for image-based PDFs.
    """
    file_bytes = file.getvalue()

    if file.name.endswith('.pdf'):
        # --- Method 1: Try standard text extraction first (fastest) ---
        try:
            pdf_reader = PdfReader(io.BytesIO(file_bytes))
            standard_text = ""
            for page in pdf_reader.pages:
                standard_text += page.extract_text() or ""
            
            # If standard extraction gives good results, use it
            if len(standard_text.strip()) > 100:
                print("Successfully extracted text using standard method.")
                return standard_text
        except Exception as e:
            print(f"Standard PDF reading failed: {e}. Falling back to OCR.")

        # --- Method 2: Fallback to OCR if standard method fails or yields little text ---
        print("Standard method yielded little/no text. Attempting OCR...")
        try:
            ocr_text = extract_text_with_ocr(file_bytes)
            return ocr_text
        except Exception as e:
            return f"Error: OCR processing failed: {e}"

    elif file.name.endswith('.txt'):
        try:
            return file_bytes.decode('utf-8')
        except Exception as e:
            return f"Error: Could not read TXT file: {e}"
    else:
        return None

def classify_and_route_document(text_content):
    """
    Uses Ollama to classify the document and suggest routing.
    """
    if not text_content or len(text_content) < 20:
        return {"error": "Content is too short to classify or text could not be extracted."}

    truncated_text = text_content[:4000]

    # IMPROVEMENT: More "bulletproof" prompt for the LLM.
    prompt = f"""
    You are an expert AI agent for a back office. Your task is to analyze the following document content, classify it, and recommend a department for routing.

    The possible document classifications are:
    - Invoice
    - Purchase Order
    - Contract
    - Other

    The corresponding departments for routing are:
    - Invoice -> Finance Department
    - Purchase Order -> Procurement Department
    - Contract -> Legal Department
    - Other -> General Administration

    Analyze the text provided below. Based on your analysis, provide a confidence score from 0 to 100 on how certain you are of the classification.

    **Document Text:**
    ---
    {truncated_text}
    ---

    **Instructions:**
    Your response must be a single, valid JSON object and nothing else. Do not include any introductory text, explanations, apologies, or markdown formatting like ```json. Your entire response must start with `{{` and end with `}}`.

    The JSON object must have the following structure:
    {{
      "classification": "...",
      "confidence_score": ...,
      "routing_department": "...",
      "reasoning": "A brief explanation of why you chose this classification."
    }}
    """

    try:
        response = ollama.chat(
            model='mistral:latest',
            messages=[{'role': 'user', 'content': prompt}],
            format='json'
        )
        json_response = json.loads(response['message']['content'])
        return json_response
    except json.JSONDecodeError:
        return {"error": "Failed to parse LLM response. The model may have produced invalid JSON."}
    except Exception as e:
        return {"error": f"An error occurred while communicating with Ollama: {e}"}