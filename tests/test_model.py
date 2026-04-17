from src.models.classifier import get_model
from src.data.dataset import get_dataloader

dataloader = get_dataloader("data/processed/images", batch_size=4, shuffle=True)
batch_images, batch_labels = next(iter(dataloader))

model = get_model(num_classes=4)

outputs = model(batch_images)
print(f"Input batch shape: {batch_images.shape}")
print(f"Output batch shape: {outputs.shape}")
print(f"Batch labels: {batch_labels}")