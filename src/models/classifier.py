import torch.nn as nn
from torchvision import models

def get_model(num_classes: int):
    """
    Loads a pre-trained ResNet18 model and modifies the
    final layer for the specified number of classes.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False
        
    in_features = model.fc.in_features
    # Replace the final fully connected layer
    model.fc = nn.Linear(in_features, num_classes)
    return model