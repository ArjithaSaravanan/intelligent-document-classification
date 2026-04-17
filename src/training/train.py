import torch
import torch.nn as nn
import torch.optim as optim
import os

from src.models.classifier import get_model
from src.data.dataset import get_dataloader

def train():
    train_dataloader, val_dataloader = get_dataloader("data/processed/images", batch_size=4)
    model = get_model(num_classes=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for images, labels in train_dataloader:
            #Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            #Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_dataloader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_dataloader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}]"
               f" Train Loss: {avg_train_loss:.4f}"
               f" Val Loss: {avg_val_loss:.4f}")
    
    os.makedirs("outputs/models", exist_ok=True)
    torch.save(model.state_dict(), "outputs/models/doc_classifier.pth")
    print("Model saved to outputs/models/doc_classifier.pth")

if __name__ == "__main__":
    train()