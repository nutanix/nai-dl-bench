import os,sys
import argparse
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torchvision as tv
import torchvision.transforms as transforms
from torch.utils.data import DataLoader as dl
import torch.nn as nn
import random
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import time
from datetime import datetime
import multiprocessing
from multiprocessing import Array
from multiprocessing.managers import SyncManager
from torchvision import models

global_rank = int(os.environ["OMPI_COMM_WORLD_RANK"]) if "OMPI_COMM_WORLD_RANK" in os.environ  else int(os.environ["RANK"])

def set_random_seeds(random_seed=0):
    # pytorch random number generator is made deterministic
    torch.manual_seed(random_seed)
    # convolution operation is made deterministic
    torch.backends.cudnn.deterministic = True
    # algorithm chosen by cuDNN library is made deterministic
    torch.backends.cudnn.benchmark = False
    # numpy random number generator is made deterministic
    np.random.seed(random_seed)
    # python random number generator is made deterministic
    random.seed(random_seed)


def init_backend_processes(backend):
    world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"]) if "OMPI_COMM_WORLD_SIZE" in os.environ  else int(os.environ["WORLD_SIZE"])
    if torch.cuda.is_available():
        dist.init_process_group(rank = global_rank, world_size=world_size ,backend="nccl", timeout=timedelta(seconds=15))
    else:
        dist.init_process_group(rank = global_rank, world_size=world_size ,backend="gloo", timeout=timedelta(seconds=15))


def train(argv):
    global_rank = int(os.environ["OMPI_COMM_WORLD_RANK"]) if "OMPI_COMM_WORLD_RANK" in os.environ  else int(os.environ["RANK"])
    local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK']) if 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ else (int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0)
    num_epochs = argv.num_epochs
    batch_size = argv.batch_size
    learning_rate = argv.learning_rate
    random_seed = argv.random_seed
    backend = argv.backend
    model_type = argv.model
    workers = argv.workers
    pf = argv.pf
    dataset_folder = argv.data_folder
    output_folder = argv.output_folder
    model_filename = argv.output_model_file
    model_filepath = os.path.join(output_folder, model_filename)

    set_random_seeds(random_seed=random_seed)
    print("TRYING TO INITIALIZE BACKEND")
    init_backend_processes(backend)
    print("BACKEND INITIALIZED")

    device = torch.device(
        f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    print(device, 'DEVICE')

    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = tv.datasets.ImageFolder(dataset_folder, transform=preprocess)

    if model_type == "resnet34":
        model = models.resnet34().to(device) 
    elif model_type == "resnet50":
        model = models.resnet50().to(device) 
    else:
        raise AssertionError("Wrong resnet type")

    if torch.cuda.is_available():
        ddp_model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank)
    else:
        ddp_model = nn.parallel.DistributedDataParallel(
            model, device_ids=None, output_device=None)

    train_sampler = DistributedSampler(dataset=train_dataset)
    train_dl = dl(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=workers, prefetch_factor=pf, pin_memory=True)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, weight_decay=0.001, momentum=0.9)

    # Train the model
    start_time = datetime.fromtimestamp(datetime.now().timestamp())

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_dl):
            # Move tensors to the configured device
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            # Forward pass
            outputs = ddp_model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad(set_to_none=True)
            # Backward and optimize
            loss.backward()
            optimizer.step()
    end_time = datetime.fromtimestamp(datetime.now().timestamp())
    print(f"Total training time for node - {global_rank}",end_time - start_time)
    if global_rank == 0:
        torch.save(ddp_model.module.state_dict(), model_filepath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training script')
    parser.add_argument('--num-epochs', type=int, default=1,
                        metavar='e', help='number of epochs to train (default: 1)')
    parser.add_argument('--batch-size', type=int, default=16, metavar='bs',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--learning-rate', type=float, default=0.01,
                        metavar='LR', help='learning rate (default: 0.01)')
    parser.add_argument('--random-seed', type=int, default=1,
                        metavar='S', help='random seed (default: 1)')
    
    parser.add_argument("--model", type=str, help = "model type", default="resnet50" )

    parser.add_argument("--workers", type=int, help = "workers", default=1 )

    parser.add_argument("--pf", type=int, help = "prefetch factor", default=2 )

    parser.add_argument("--data-folder", type=str, help="train dataset", default='', required= True )
    parser.add_argument("--output-folder", type=str, help="output folder", default="training_results", required=True)
    parser.add_argument("--output-model-file", type=str, help="output Model filename.", default="resnet.pth")

    if dist.is_available():
        parser.add_argument('--backend', type=str, help='Distributed backend',
                            choices=[dist.Backend.GLOO,
                                     dist.Backend.NCCL, dist.Backend.MPI],
                            default=dist.Backend.NCCL)
    argv = parser.parse_args()
    print("ARGUMENTS",argv)
    Path(argv.output_folder).mkdir(parents=True, exist_ok=True)
    dataset_dir = os.listdir(argv.data_folder)
    # Checking if dataset is empty or not
    if len(dataset_dir) == 0:
        print("Empty dataset directory")
        sys.exit(1)
    train(argv)
    print(f"\nSUCCESSFULLY COMPLETED TRAINING ON NODE - {global_rank}")
    sys.exit()