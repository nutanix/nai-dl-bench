from torchvision.models import get_model, get_weight
import torch
import os
import argparse

def create(model_name, weight_type):
    weights = get_weight(weight_type)
    model = get_model(model_name, weights=weights)

    filename = model_name + "-default.pt"
    path = os.path.join(os.path.dirname(__file__), filename)
    torch.save(model.state_dict(), path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inference run script')
    parser.add_argument('--model_name', type=str, default="", required= True,
                        metavar='n', help='Name of the model')
    
    parser.add_argument('--weight', type=str, default="", required= True,
                        metavar='w', help='The type of weight to be set for the model')

    args = parser.parse_args()
    create(args.model_name, args.weight)