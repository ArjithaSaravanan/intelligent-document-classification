from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
import shutil
import uuid

from src.inference.predict import predict_image

app = FastAPI(
    title = "Intelligent Document Classifier API",
    version="1.0.0",
    description="API for classifying document images and PDFs into document types."
)

TEMP_DIR = Path("temp_uploads")
TEMP_DIR.mkdir(exist_ok=True)

@app.get("/")
def health_check():
    return {
        "status": "ok",
        "message": "Document classification API is running."
    }

@app.post("/v1/document/classify")
async def classify_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")
    
    suffix = Path(file.filename).suffix.lower()
    allowed_extensions = ['.jpg', '.jpeg', '.png', '.pdf']

    if suffix not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}. Allowed types are: {', '.join(allowed_extensions)}")

    temp_file_path = TEMP_DIR / f"{uuid.uuid4()}{suffix}"

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        predicted_class, confidence = predict_image(str(temp_file_path))
        return {
            "filename": file.filename,
            "document_type": predicted_class,
            "confidence": round(confidence, 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    finally:
        if temp_file_path.exists():
            temp_file_path.unlink()
