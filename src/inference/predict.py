from PIL import Image
import torch
from torchvision import transforms
import sys

from src.models.classifier import get_model

CLASS_NAMES = ['financial_invoice', 'fuel_receipt', 'hospitality_receipt', 'retail_receipt']

def predict_image(image_path: str):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension

    model = get_model(num_classes=4)
    model.load_state_dict(torch.load("outputs/models/doc_classifier.pth", map_location="cpu"))
    model.eval()

    with torch.no_grad():
        outputs = model(image)
        predicted_index = torch.argmax(outputs, dim=1).item()

    predicted_class = CLASS_NAMES[predicted_index]
    return predicted_class

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.inference.predict <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    prediction = predict_image(image_path)
    print(f"Predicted class for {image_path}: {prediction}")