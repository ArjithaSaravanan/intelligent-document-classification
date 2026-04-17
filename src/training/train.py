import torch
import torch.nn as nn
import torch.optim as optim
import os

from src.models.classifier import get_model
from src.data.dataset import get_dataloader

def train():
    dataloader = get_dataloader("data/processed/images", batch_size=4, shuffle=True)
    model = get_model(num_classes=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10

    for epoch in range(num_epochs):
        total_loss = 0
        for images, labels in dataloader:
            #Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            #Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}")
    
    os.makedirs("outputs/models", exist_ok=True)
    torch.save(model.state_dict(), "outputs/models/doc_classifier.pth")
    print("Model saved to outputs/models/doc_classifier.pth")

if __name__ == "__main__":
    train()