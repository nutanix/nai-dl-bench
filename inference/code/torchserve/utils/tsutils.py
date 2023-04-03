import os
import platform
import sys
import time
import requests
import marsgen as mg
import json

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

def generate_ts_start_cmd(ncs, model_store, models, config_file, log_file, log_config_file, gpus, debug):
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
    debug and print(f"## In directory: {os.getcwd()} | Executing command: {cmd} \n")
    return cmd


def start_torchserve(gen_folder, ncs=False, model_store="model_store", models="", 
        config_file="", log_file="", log_config_file="", wait_for=10, gen_mar=True, gpus=0, debug=False):
    if gen_mar:
        new_mar_file = mg.gen_mar(gen_folder, model_store, debug)
    print("## Starting TorchServe \n")
    cmd = generate_ts_start_cmd(ncs, model_store, models, config_file, log_file, log_config_file, gpus, debug)
    status = os.system(cmd)
    if status == 0:
        print("## Successfully started TorchServe \n")
        time.sleep(wait_for)
        return (True, new_mar_file if gen_mar else None)
    else:
        print("## TorchServe failed to start ! Make sure it's not running already\n")
        return (False, None)


def stop_torchserve(wait_for=10):
    try:
        requests.get('http://localhost:8080/ping')
    except Exception as e:
        return

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


def get_params_for_registration(model_name):
    dirpath = os.path.dirname(__file__)
    initial_workers = batch_size = max_batch_delay = response_timeout = None
    with open(os.path.join(dirpath, '../models/models.json'), 'r') as f:
        model_config = json.loads(f.read())
        if model_name in model_config:
            if "initial_workers" in model_config[model_name]:
                initial_workers = model_config[model_name]['initial_workers']

            if "batch_size" in model_config[model_name]:
                batch_size = model_config[model_name]['batch_size']

            if "max_batch_delay" in model_config[model_name]:
                max_batch_delay = model_config[model_name]['max_batch_delay']

            if "response_timeout" in model_config[model_name]:
                response_timeout = model_config[model_name]['response_timeout']

    return initial_workers, batch_size, max_batch_delay, response_timeout


def register_model(model_name, marfile, protocol="http", host="localhost", port="8081"):
    print(f"\n## Registering {marfile} model \n")
    initial_workers, batch_size, max_batch_delay, response_timeout = get_params_for_registration(model_name)

    params = (
        ("url", marfile),
        ("initial_workers", initial_workers or 1),
        ("batch_size", batch_size or 1),
        ("max_batch_delay", max_batch_delay or 200),
        ("response_timeout", response_timeout or 2000),
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