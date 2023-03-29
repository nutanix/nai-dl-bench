import os
import platform
import sys
import time
import requests
import marsgen as mg

torchserve_command = {
    "Windows": "torchserve.exe",
    "Darwin": "torchserve",
    "Linux": "torchserve"
}

torch_model_archiver_command = {
        "Windows": "torch-model-archiver.exe",
        "Darwin": "torch-model-archiver",
        "Linux": "torch-model-archiver"
    }

def start_torchserve(gen_folder,
        ncs=False, model_store="model_store", workflow_store="",
        models="", config_file="", log_file="", log_config_file="", wait_for=10, gen_mar=True, gpus=0):
    if gen_mar:
        mg.gen_mar(gen_folder, model_store)
    print("## Starting TorchServe \n")
    cmd = f"TS_NUMBER_OF_GPU={gpus} {torchserve_command[platform.system()]} --start --ncs --model-store={model_store}"
    if models:
        cmd += f" --models={models}"
    if ncs:
        cmd += " --ncs"
    if config_file:
        cmd += f" --ts-config={config_file}"
    if log_config_file:
        cmd += f" --log-config {log_config_file}"
    if log_file:
        print(f"## Console logs redirected to file: {log_file} \n")
        dirpath = os.path.dirname(log_file)
        cmd += f" >> {os.path.join(dirpath,log_file)}"
    print(f"## In directory: {os.getcwd()} | Executing command: {cmd} \n")
    status = os.system(cmd)
    if status == 0:
        print("## Successfully started TorchServe \n")
        time.sleep(wait_for)
        return True
    else:
        print("## TorchServe failed to start ! Make sure it's not running already\n")
        return False


def stop_torchserve(wait_for=10):
    print("## Stopping TorchServe \n")
    cmd = f"{torchserve_command[platform.system()]} --stop"
    status = os.system(cmd)
    if status == 0:
        print("## Successfully stopped TorchServe \n")
        time.sleep(wait_for)
        return True
    else:
        print("## TorchServe failed to stop ! \n")
        return False


# Takes model name and mar name from model zoo as input
def register_model(model_name, protocol="http", host="localhost", port="8081"):
    print(f"\n## Registering {model_name} model \n")
    marfile = f"{model_name}.mar"
    params = (
        ("model_name", model_name),
        ("url", marfile),
        ("initial_workers", "1"),
        ("synchronous", "true"),
    )
    url = f"{protocol}://{host}:{port}/models"
    response = requests.post(url, params=params, verify=False)
    return response


def run_inference(model_name, file_name, protocol="http", host="localhost", port="8080", timeout=120):
    print(f"## Running inference on {model_name} model \n")
    url = f"{protocol}://{host}:{port}/predictions/{model_name}"
    files = {"data": (file_name, open(file_name, "rb"))}
    response = requests.post(url, files=files, timeout=timeout)
    print(response)
    return response


def unregister_model(model_name, protocol="http", host="localhost", port="8081"):
    print(f"## Unregistering {model_name} model \n")
    url = f"{protocol}://{host}:{port}/models/{model_name}"
    response = requests.delete(url, verify=False)
    return response