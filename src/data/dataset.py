from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_dataset(data_dir: str):
    """
    Loads dataset from the specified directory using ImageFolder.
    """
    transform = transforms.Compose([
        transforms.Resize((224,224)),# Resize images to 224x224
        transforms.RandomRotation(5), # Randomly rotate images by up to 5 degrees
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # Randomly adjust brightness and contrast
        transforms.RandomHorizontalFlip(p=0.1), # Randomly flip images horizontally with a probability of 0.1
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(
        root=data_dir,
        transform=transform
    )

    return dataset

def get_dataloader(data_dir: str, batch_size: int = 4):
    """
    Returns train and validation dataloaders
    """
    dataset = get_dataset(data_dir)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    # Split the dataset into training and validation sets
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_dataloader, val_dataloader