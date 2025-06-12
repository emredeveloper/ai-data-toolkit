import fitz  # PyMuPDF
import easyocr
import os

pdf_path = "ornek.pdf"
output_dir = "pdf_sayfa_gorselleri"
os.makedirs(output_dir, exist_ok=True)

reader = easyocr.Reader(['tr'])

doc = fitz.open(pdf_path)

for page_num in range(len(doc)):
    page = doc.load_page(page_num)
    pix = page.get_pixmap(dpi=300)  # Yüksek çözünürlük için dpi artırıldı
    image_filename = f"{output_dir}/sayfa{page_num+1}.png"
    pix.save(image_filename)
    print(f"{image_filename} kaydedildi.")

    # OCR işlemi
    ocr_result = reader.readtext(image_filename, detail=0, paragraph=True)
    print(f"OCR Sonucu ({image_filename}):\n", "\n".join(ocr_result))
    print("-" * 40)

doc.close()
