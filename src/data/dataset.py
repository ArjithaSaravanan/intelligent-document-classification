from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataset(data_dir: str):
    """
    Loads dataset from the specified directory using ImageFolder.
    """
    transform = transforms.Compose([
        transforms.Resize((224,224)),# Resize images to 224x224
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(
        root=data_dir,
        transform=transform
    )

    return dataset

def get_dataloader(data_dir: str, batch_size: int = 4, shuffle: bool = True):
    """
    Loads dataset and wraps it in a DataLoader
    """
    dataset = get_dataset(data_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
    return dataloader