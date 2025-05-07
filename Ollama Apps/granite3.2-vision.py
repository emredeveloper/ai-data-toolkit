import requests
import base64
import os
import argparse
from PIL import Image
import io
import tkinter as tk
from tkinter import filedialog
import tempfile
import fitz  # PyMuPDF for PDF handling
import shutil

OLLAMA_API_HOST = "http://localhost:11434"
PROJECT_ROOT = os.getcwd()

def resolve_path(file_path):
    """Resolve a path that might be relative to the project root."""
    if os.path.isabs(file_path) or not file_path:
        return file_path
    
    # Try project root first, then script directory
    for base_dir in [PROJECT_ROOT, os.path.dirname(os.path.abspath(__file__))]:
        resolved = os.path.join(base_dir, file_path)
        if os.path.exists(resolved):
            return resolved
    return file_path

def process_image(image_path, prompt, model="granite3.2-vision:latest"):
    """Process an image with the model."""
    # Resize image if needed
    with Image.open(image_path) as img:
        max_size = 1024
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            img = img.resize((int(img.size[0] * ratio), int(img.size[1] * ratio)), Image.LANCZOS)
            buffer = io.BytesIO()
            img.save(buffer, format=img.format if img.format else 'JPEG')
            buffer.seek(0)
            base64_image = base64.b64encode(buffer.read()).decode('utf-8')
        else:
            with open(image_path, "rb") as img_file:
                base64_image = base64.b64encode(img_file.read()).decode('utf-8')
    
    # Query model
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3},
        "images": [base64_image]
    }
    
    response = requests.post(f"{OLLAMA_API_HOST}/api/generate", json=payload)
    if response.status_code == 200:
        result = response.json()
        print("\nResponse from model:")
        print("-" * 50)
        print(result.get("response", "No response text received"))
        print("-" * 50)
        print(f"Total processing time: {result.get('total_duration', 0) // 1000000}ms")
        return result
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

def process_pdf(pdf_path, prompt, model):
    """Process all images in a PDF."""
    temp_dir = tempfile.mkdtemp()
    try:
        image_paths = []
        pdf_document = fitz.open(pdf_path)
        
        for page_num in range(len(pdf_document)):
            for img_index, img_info in enumerate(pdf_document[page_num].get_images(full=True)):
                base_image = pdf_document.extract_image(img_info[0])
                img_path = os.path.join(temp_dir, f"page{page_num+1}_img{img_index+1}.{base_image['ext']}")
                
                with open(img_path, "wb") as img_file:
                    img_file.write(base_image["image"])
                image_paths.append(img_path)
        
        print(f"Extracted {len(image_paths)} images from the PDF")
        
        for i, img_path in enumerate(image_paths, 1):
            print(f"\nProcessing image {i}/{len(image_paths)} from PDF...")
            image_prompt = f"[Image {i}/{len(image_paths)} from PDF] {prompt}"
            process_image(img_path, image_prompt, model)
            
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

def main():
    parser = argparse.ArgumentParser(description="Query vision models with images or PDFs")
    parser.add_argument("--file", "-f", help="Path to image or PDF file")
    parser.add_argument("--prompt", "-p", help="Text prompt to send to the model")
    parser.add_argument("--model", "-m", default="granite3.2-vision:latest", help="Model name")
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama API host")
    args = parser.parse_args()
    
    global OLLAMA_API_HOST
    OLLAMA_API_HOST = args.host
    
    # Get file path
    file_path = resolve_path(args.file) if args.file else None
    if not file_path or not os.path.exists(file_path):
        print("Please select a file from the dialog...")
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="Select Image or PDF File",
            filetypes=[
                ("All supported files", "*.png;*.jpg;*.jpeg;*.gif;*.bmp;*.pdf"),
                ("Image files", "*.png;*.jpg;*.jpeg;*.gif;*.bmp"),
                ("PDF files", "*.pdf"),
                ("All files", "*.*")
            ]
        )
        root.destroy()
        if not file_path:
            print("No file selected. Exiting.")
            return
    
    # Get prompt
    prompt = args.prompt or input("Enter your prompt for the model: ")
    
    try:
        if file_path.lower().endswith('.pdf'):
            process_pdf(file_path, prompt, args.model)
        else:
            process_image(file_path, prompt, args.model)
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nMake sure Ollama is running and the model is installed.")

if __name__ == "__main__":
    main()
