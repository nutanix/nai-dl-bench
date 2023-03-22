import os,sys
import argparse
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torchvision as tv
import torchvision.transforms as transforms
from torch.utils.data import DataLoader as dl
import torch.nn as nn
import matplotlib.pyplot as plt
import random
import numpy as np
import gc
from datetime import datetime, timedelta
from pathlib import Path
from torch.cuda import nvtx
import psutil
import GPUtil
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
import multiprocessing
from multiprocessing import Array
from multiprocessing.managers import SyncManager
from models.resnet34 import ResNet34, ResidualBlock
from models.resnet50 import ResNet50, Bottleneck
from utils.graph import save_time_series_plt
from utils.monitor import monitor

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
    global_rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
    world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
    dist.init_process_group(rank = global_rank, world_size=world_size ,backend="nccl", timeout=timedelta(seconds=15))

def train(argv):
    print("TRAINING STARTED")
    local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
    num_epochs = argv.num_epochs
    batch_size = argv.batch_size
    learning_rate = argv.learning_rate
    random_seed = argv.random_seed
    model_filename = argv.model_filename
    backend = argv.backend
    output_folder = argv.output_folder
    model_filepath = os.path.join(output_folder, model_filename)
    nvtx_profile  = argv.profile
    resnet_type = argv.resnet
    workers = argv.workers
    pf = argv.pf
    datasetFolder = argv.dataset 

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

    # 37 gb dataset -> '/nfs/datasets/ILSVRC2014_DET_train', 200gb -> '/nfs/datasets/ILSVRC'
    train_dataset = tv.datasets.ImageFolder(datasetFolder, transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ]))


    cls_to_label_map = {}
    with open('labels.txt') as f:
        while True:
            line = f.readline()
            if not line:
                # print(line=='')
                break
            # print(line)
            cls = line.split(':')[0]
            value = line.split(':')[1].split(',')[0]
            cls_to_label_map[cls] = value

    idx_to_label_map = {}
    for cls,idx in train_dataset.class_to_idx.items():
        idx_to_label_map[idx] = cls_to_label_map[cls]  if cls in cls_to_label_map else "unknown"
    
    num_of_classes = len(idx_to_label_map.keys())
    print("NUMBER OF CLASSES", num_of_classes)

    if resnet_type ==34:
        model = ResNet34(ResidualBlock, [3, 4, 6, 3], num_of_classes).to(device)
    elif resnet_type == 50:
        model = ResNet50(Bottleneck, [3, 4, 6, 3], num_of_classes).to(device)
    else:
        raise AssertionError("Wrong resnet type")

    ddp_model = nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank)
    train_sampler = DistributedSampler(dataset=train_dataset)
    train_dl = dl(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=workers, multiprocessing_context="spawn", prefetch_factor=pf) #

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, weight_decay=0.001, momentum=0.9)

    # Train the model
    start_time = datetime.fromtimestamp(datetime.now().timestamp())
    if nvtx_profile:
        torch.cuda.cudart().cudaProfilerStart()
        for epoch in range(num_epochs):
            print("EPOCH", epoch)
            # print("TIME",datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))

            nvtx.range_push("DATA LOADING")
            for i, (images, labels) in enumerate(train_dl):
                nvtx.range_pop()
                nvtx.range_push(f"batch-{i}")
                # Move tensors to the configured device
                nvtx.range_push("COPY TO DEVICE")
                images = images.to(device)
                labels = labels.to(device)
                nvtx.range_pop()

                nvtx.range_push("FORWARD PASS")
                # Forward pass
                outputs = ddp_model(images)
                # print(outputs.min(), outputs.max(), outputs.shape)
                # print(labels, labels.shape)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                nvtx.range_pop()

                nvtx.range_push("BACKWARD PASS")
                # Backward and optimize
                loss.backward()
                optimizer.step()
                nvtx.range_pop()
                # del images, labels, outputs
                # torch.cuda.empty_cache()
                # gc.collect()
                nvtx.range_pop()
                nvtx.range_push("DATA LOADING")
               
            nvtx.range_pop()
            torch.cuda.cudart().cudaProfilerStop()
            print('Epoch [{}/{}], Loss: {:.4f}'
                .format(epoch+1, num_epochs, loss.item()))
    else:
        for epoch in range(num_epochs):
            print("EPOCH", epoch)
            # print("TIME",datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))

            for i, (images, labels) in enumerate(train_dl):
                # Move tensors to the configured device
                images = images.to(device)
                labels = labels.to(device)
                # Forward pass
                outputs = ddp_model(images)
                # print(outputs.min(), outputs.max(), outputs.shape)
                # print(labels, labels.shape)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                # Backward and optimize
                loss.backward()
                optimizer.step()
                # del images, labels, outputs
                # torch.cuda.empty_cache()
                # gc.collect()

            print('Epoch [{}/{}], Loss: {:.4f}'
                .format(epoch+1, num_epochs, loss.item()))
    end_time = datetime.fromtimestamp(datetime.now().timestamp())
    print("Total training time",end_time - start_time)
    torch.save(ddp_model.state_dict(), model_filepath)


if __name__ == '__main__':
    model_filename_default = "resnet_distributed.pth"

    parser = argparse.ArgumentParser(description='training script')
    parser.add_argument('--num_epochs', type=int, default=1,
                        metavar='e', help='number of epochs to train (default: 10)')
    parser.add_argument('--batch_size', type=int, default=128, metavar='bs',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        metavar='LR', help='learning rate (default: 0.01)')
    parser.add_argument('--random_seed', type=int, default=1,
                        metavar='S', help='random seed (default: 1)')

    parser.add_argument("--model_filename", type=str,
                        help="Model filename.", default=model_filename_default)
    
    parser.add_argument("--profile", help="generate nvtx profiler result", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--output_folder", type=str, help="output folder", default="training_results", required=True)

    parser.add_argument("--resnet", type=int, help = "resnet type", default=50 )

    parser.add_argument("--workers", type=int, help = "workers", default=2 )

    parser.add_argument("--pf", type=int, help = "prefetch factor", default=2 )

    parser.add_argument("--dataset", type=str, help="train dataset", default='/nfs/datasets/ILSVRC/Data/CLS-LOC/train' )

    if dist.is_available():
        parser.add_argument('--backend', type=str, help='Distributed backend',
                            choices=[dist.Backend.GLOO,
                                     dist.Backend.NCCL, dist.Backend.MPI],
                            default=dist.Backend.NCCL)
    argv = parser.parse_args()
    Path(argv.output_folder).mkdir(parents=True, exist_ok=True)
    multiprocessing.set_start_method("spawn")
    manager = SyncManager()
    manager.start()
    dates = manager.list([])
    gpus = GPUtil.getGPUs()
    gpu_usages = manager.list([ manager.list([]) for gpu in gpus])
    cpu_usages = manager.list([])
    mem_usages = manager.list([])
    m1 = multiprocessing.Process(target=monitor, args=(dates,gpu_usages, cpu_usages, mem_usages, argv.output_folder))
    t1 = multiprocessing.Process(target=train,args=(argv,))
    m1.start()
    t1.start()
    t1.join()
    print("TRAINING DONE")
    m1.terminate()
    # print("CONTENT", dates, gpu_usages[0])
    save_time_series_plt(dates,gpu_usages,f"gpu.jpg", argv.output_folder)
    save_time_series_plt(dates,cpu_usages,f"cpu.jpg", argv.output_folder)  
    save_time_series_plt(dates,mem_usages,f"mem.jpg", argv.output_folder)  
    manager.shutdown()
    sys.exit()
