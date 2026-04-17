import torch
import torch.nn as nn
import torch.optim as optim

from src.models.classifier import get_model
from src.data.dataset import get_dataloader

#Load a batch of data
dataloader = get_dataloader("data/processed/images", batch_size=4, shuffle=True)
batch_images, batch_labels = next(iter(dataloader))
#Build the model
model = get_model(num_classes=4)
#Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
#Forward pass
outputs = model(batch_images)
#Compute loss
loss = criterion(outputs, batch_labels)
print(f"Loss before backward pass: {loss.item():.4f}")

#Backward pass
optimizer.zero_grad()  # Clear previous gradients
loss.backward()        # Compute gradients
optimizer.step()       # Update weights

print("Completed one training step.")
