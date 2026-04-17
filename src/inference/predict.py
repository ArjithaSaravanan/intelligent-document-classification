from PIL import Image
import torch
from torchvision import transforms
import sys
from pdf2image import convert_from_path
from pathlib import Path

from src.models.classifier import get_model

CLASS_NAMES = ['financial_invoice', 'fuel_receipt', 'hospitality_receipt', 'retail_receipt']

def load_image(image_path: str):
    "Handle both image files and PDFs (convert first page to image)"
    path = Path(image_path)
    if path.suffix.lower() == '.pdf':
        images = convert_from_path(str(path), first_page=1, last_page=1)
        if images:
            return images[0].convert('RGB')
        else:
            raise ValueError(f"Could not convert PDF {path} to image.")
    else:
        return Image.open(path).convert('RGB')


def predict_image(image_path: str):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = load_image(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension

    model = get_model(num_classes=4)
    model.load_state_dict(
        torch.load("outputs/models/doc_classifier.pth", map_location="cpu")
    )
    model.eval()

    with torch.no_grad():
        outputs = model(image)
        #Convert raw scores to probabilities and get the predicted class index and confidence
        probabilities = torch.softmax(outputs, dim=1)
        predicted_index = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_index].item()

    predicted_class = CLASS_NAMES[predicted_index]
    return predicted_class, confidence

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.inference.predict <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    predicted_class, confidence = predict_image(file_path)
    print(f"Predicted class for {file_path}: {predicted_class} with confidence {confidence:.4f}")