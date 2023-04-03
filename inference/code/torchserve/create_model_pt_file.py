from torchvision.models import get_model, get_weight
import torch
import os
import argparse

def create(model_name, weight_type, output):
    weights = get_weight(weight_type)
    model = get_model(model_name, weights=weights)

    filename = model_name + "-default.pt"
    path = os.path.join(output, filename)
    torch.save(model.state_dict(), path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate model file')
    parser.add_argument('--model_name', type=str, default="", required= True,
                        metavar='n', help='Name of the model')
    
    parser.add_argument('--weights', type=str, default="", required= True,
                        metavar='w', help='The type of weight to be set for the model')

    parser.add_argument('--output', type=str, default="", required= True,
                        metavar='o', help='The absolute path for saving the model')

    args = parser.parse_args()
    create(args.model_name, args.weights, args.output)