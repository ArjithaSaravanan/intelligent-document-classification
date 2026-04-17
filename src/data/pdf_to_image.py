from pdf2image import convert_from_path
from pathlib import Path
from tqdm import tqdm

def convert_pdf_folder_to_images(input_dir: Path, output_dir: Path):
    """
    Convert all PDFs in a folder into images *first page only*
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_files = list(input_dir.glob('*.pdf'))
    
    for pdf_file in tqdm(pdf_files, desc=f"Processing {input_dir.name}"):
        try:
            # Convert the first page of the PDF to an image
            images = convert_from_path(str(pdf_file), first_page=1, last_page=1)
            if images:
                image = images[0]
                output_path = output_dir / f"{pdf_file.stem}.jpg"
                image.save(output_path, "JPEG")
        except Exception as e:
            print(f"Error converting {pdf_file}: {e}")

def process_all_classes(base_input: Path, base_output: Path):
    """
    Process all class folders in the base input directory and convert PDFs to images.
    """
    classes = [
        "financial_invoice",
        "retail_receipt",
        "hospitality_receipt",
        "fuel_receipt",
    ]

    for cls in classes:
        input_dir = base_input / cls
        output_dir = base_output / cls
        convert_pdf_folder_to_images(input_dir, output_dir)

if __name__ == "__main__":
    BASE_INPUT_DIR = Path("data/raw/pdfs")
    BASE_OUTPUT_DIR = Path("data/processed/images")
    process_all_classes(BASE_INPUT_DIR, BASE_OUTPUT_DIR)