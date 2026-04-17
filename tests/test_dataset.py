from src.data.dataset import get_dataset, get_dataloader

dataset = get_dataset("data/processed/images")

print(f"Number of samples in dataset: {len(dataset)}")
print(f"Classes: {dataset.classes}")
print(f"Class to index: {dataset.class_to_idx}")

image, label = dataset[0]
print(f"Single Image shape: {image.shape}")
print(f"Single Label: {label}")

dataloader = get_dataloader("data/processed/images", batch_size=4, shuffle=True)
batch_images, batch_labels = next(iter(dataloader))
print(f"Batch Images shape: {batch_images.shape}")
print(f"Batch Labels: {batch_labels}")