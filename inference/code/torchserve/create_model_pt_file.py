from torchvision.models import resnet50, ResNet50_Weights
import torch
import os

model = resnet50(weights=ResNet50_Weights.DEFAULT)
path = os.path.join(os.path.dirname(__file__), "default-resnet-50-model.pt")
torch.save(model.state_dict(), path)
