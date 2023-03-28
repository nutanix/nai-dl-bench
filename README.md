# nai-dl-bench

**SETUP**

MPIRun setup:

Install MPIRun in each node: 
```
sudo apt install openmpi-bin
```

Enable Passwordless SSH Login from master node to all other nodes. Choose any one of the participating nodes to be the master node.  
ssh-keygen on master node
Run the following command on the master node for each of the worker node, assuming ubuntu is the username for all vms: 
```
ssh-copy-id -i /home/ubuntu/.ssh/id_rsa <username>@<worker node ip>
```

- Dataset has to be accessible from all nodes eg: NFS
- Absolute path of the dataset and training code need to be the same across all nodes

**TRAINING**

To start multi node training, use the following command

```
bash training/code/run.sh -n <NUM_PROCS> -h <HOSTS_IP_LIST> -m <MASTER_ADDRESS> -c <COMMAND_TO_RUN_PYTHON_SCRIPT>

-n Number of training processes
-h Comma separated list of Host IPs
-m IP Address of master node
-c Python command with space separated list of --<option> <argument> inside double quotes for the training script. data and output options are mandatory for the training.py script
```

In the following examples a node with ip 10.112.26.105 is chosen as the master node.

single node training command using 1 gpus:-
```
bash training/code/run.sh -n 1 -h 10.112.26.105 -m 10.112.26.105  -c "python3 training.py --data-folder /home/ubuntu/data --output-folder /home/ubuntu/output --model resnet50 --output-model-file resnet.pth --batch-size 12 --workers 1 --pf 2 --num-epochs 1"
```

2 nodes (10.112.26.94 and 10.112.26.105) training command using 1 gpu each, run the command on master node:-
```
bash training/code/run.sh -n 2 -h 10.112.26.94:1,10.112.26.105:1 -m 10.112.26.105   -c "python3 training.py --data-folder /home/ubuntu/data --output-folder /home/ubuntu/output --model resnet50 --output-model-file resnet.pth --batch-size 12 --workers 1 --pf 2 --num-epochs 1"
```

## INFERENCE

### Setup TorchServe

Install openjdk, pip3  

```
sudo apt-get install openjdk-17-jdk python3-pip
```

Nvidia driver installation:
Reference: https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html#runfile


Clone this repo and select torchserve folder 
Install TS libraries 

```
cd inference/code/torchserve
pip install -r requirements.txt
```

### Create .mar file for resnet50

Generate new resnet-50.pt using the eager mode 

```
python create_model_pt_file.py --model_name resnet50 --weight ResNet50_Weights.DEFAULT
```

Generate resnet50.mar file

```
torch-model-archiver --model-name resnet50 --version 1.0 --model-file models/resnet50/resnet50_arch.py --serialized-file resnet50-default.pt --handler image_classifier --extra-files index_to_name.json
```

Create a folder and move the .mar file inside it

```
mkdir model_store
mv resnet50.mar model_store/resnet50.mar
```

### Start Torchserve Server 

##### Torchserve Start command

```
torchserve --start --ncs --model-store model_store --ts-config config.properties --log-config log4j2.xml
```

##### Health Check

```
curl http://localhost:8080/ping
```

##### Register model

curl -X POST  "http://{inference_endpoint}:{management_port}/models?url={model_location}&initial_workers={number}&synchronous=true&batch_size={number}&max_batch_delay={delay_in_ms}"

```
curl -X POST  "http://localhost:8081/models?url=resnet50.mar&initial_workers=1&synchronous=true&batch_size=1&max_batch_delay=20"
```

##### Describe registered model

GET /models/{model_name}

```
curl http://localhost:8081/models/resnet50
```

##### Edit config for a registered model

```
curl -v -X PUT "http://localhost:8081/models/resnet50?min_worker=3&max_worker=6"
```

##### Inference Check

curl http://{inference_endpoint}/predictions/{model_name} -T {input_file}

Test input file can be found in data folder

```
curl http://localhost:8080/predictions/resnet50 -T input.jpg
```

##### Unregister a model

DELETE /models/{model_name}/{version}

```
curl -X DELETE http://localhost:8081/models/resnet50/1.0
```

##### Torchserve Stop command

```
torchserve --stop
```

For more detailed explanations on using the management endpoint. Check out - https://pytorch.org/serve/management_api.html

By default, TorchServe uses all available GPUs for inference. Use number_of_gpu in the config.properties file to limit the usage of GPUs

Properties in config.properties file can be updated as required
Reference: https://pytorch.org/serve/configuration.html

Log level can be set as required by modifying the log4j2.xml file


### Automated Setup and Inference run
You can test your trained model end to end with running the inference run script with the requirement arguments 
run.sh is inside inference/code/torchserve folder

command to run inference
```
Usage: bash run.sh -n <MODEL_NAME> -d <INPUT_DATA_ABSOLUTE_PATH> -m <MODEL_ABSOLUTE_PATH> -f <MODEL_ARCH_FILE_ABSOLUTE_PATH> -c <CLASSES_MAPPING_ABSOLUTE_PATH> -h <HANDLER_FILE_ABSOLUTE_PATH> -e <EXTRA_FILES> -g <NUM_OF_GPUS> [OPTIONAL -k]

-n Name of the Model
-d Absolute path to the inputs folder that contains data to be predicted.
-m Absolute path to the saved model file
-f Absolute path to the model arch file
-c Absolute path classes mapping file
-h Absolute path handler file
-e Comma separated absolute paths of all the additional paths required by the model
-g Number of gpus to be used to execute. Default will be 0, cpu used
-k Keep the torchserve server alive after run completion. Default, stops the server if not set
```

- Run Inference on the existing standard resnet50 model provided in this repo.
Set the name and data folder parameter as required. "-k" keeps the torchserve server alive and needs to be stopped explicitly
```
bash inference/code/torchserve/run.sh -n resnet50 -d /home/ubuntu/data -k
```

- Run Inference on the trained resnet50 model that was generated using the training code provided in this repo.
Set the name and data folder parameter as required and for GPU to be used set "-g <num>"
```
bash inference/code/torchserve/run.sh -n resnet50 -d /home/ubuntu/data -m resnet50.pt -g 1
```

- Run Inference on the custom model of your choice.
Make sure to set all the parameters as shown in the example
```
bash inference/code/torchserve/run.sh -n resnet50 -d /home/ubuntu/data -m /home/ubuntu/model/resnet50.pt -f /home/ubuntu/model/model.py -c /home/ubuntu/index_to_name.json -h image_classifier -e /home/gavrishdemo/test/resnet50.py -g 2
```

Should print "Inference Run Successful" as a message at the end

##### Add custom models as default

- create a folder inside "models" folder with the name of the model and add all the required files

```
models/
    -custom100
        - model.pt
        - arch.py
        - handler.py
        - class_map.json
```

- make an entry for this custom model in "models/models.json"

```
{
    .
    .
    .
    "custom100": {
        "model_arch_file": "arch.py",
        "handler": "handler.py",
        "class_map": "class_map.json"
    }
}
```

- run command 

```
bash inference/code/torchserve/run.sh -n custom100 -d /home/ubuntu/data -m models/custom100/model.pt
```


