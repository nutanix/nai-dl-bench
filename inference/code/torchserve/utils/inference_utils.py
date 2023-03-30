import os
import sys
import nvgpu

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)))
MAR_CONFIG_FILE_PATH = os.path.join(REPO_ROOT, "mar_config.json")
sys.path.append(REPO_ROOT)

import tsutils as ts
import system_utils
import time
import json
import subprocess
import requests

def error_msg_print():
    print(f"\n**************************************")
    print(f"*\n*\n*  Error found - Unsuccessful ")
    print(f"*\n*\n**************************************")
    stopped = ts.stop_torchserve()

def create_init_json(model_name, model_arch_path, model_path, handler_path, index_file_path, extra_files):
    init_json = {}
        
    init_json['model_name'] = model_name
    init_json['version'] = "1.0"
    init_json['model_file'] = model_arch_path
    init_json['serialized_file_local'] = model_path
    init_json['handler'] = handler_path
    init_json['extra_files'] = index_file_path

    if extra_files:
            init_json['extra_files'] += ", "+ extra_files

    return [init_json]


def get_model_name(input_dict, modelUrl):
    for entry in input_dict['models']:
        if(modelUrl == entry['modelUrl']):
            return entry['modelName']
    
    print('\n model not found among registered models')
    error_msg_print()
    sys.exit(1)

def get_inference(model_name, model_path, handler_path, index_file_path, input_path, 
    model_arch_path, extra_files, gpus, gen_folder, gen_mar, mar_file, debug=False):

    try:
        dirpath = os.path.dirname(__file__)

        if(not mar_file):
            init_json = create_init_json(model_name, model_arch_path, model_path, handler_path, index_file_path, extra_files)
            debug and print("\n",init_json, "\n")
            with open(os.path.join(dirpath, gen_folder, 'mar_config.json'), 'w') as f:
                json.dump(init_json, f)
            gen_mar=True

        else:
            gen_mar = False

        inputs = [os.path.join(input_path, item) for item in os.listdir(input_path)] if input_path else []

        inference_model = {
            "name": model_name,
            "inputs": inputs,
            "handler": handler_path,
        }

        models_to_validate = [
            inference_model
        ]

        ts_log_file = os.path.join(dirpath, gen_folder, "logs/ts_console.log")
        ts_log_config = os.path.join(dirpath, "../log4j2.xml")
        ts_config_file = os.path.join(dirpath, '../config.properties')
        ts_model_store = os.path.join(dirpath, gen_folder, "model_store")

        gen_path =os.path.join(dirpath, gen_folder, "logs")
        os.environ["LOG_LOCATION"] = gen_path
        os.environ["METRICS_LOCATION"] = gen_path

        os.makedirs(os.path.join(dirpath, gen_folder, "model_store"), exist_ok=True)
        os.makedirs(os.path.join(dirpath, gen_folder, "logs"), exist_ok=True)

        mar_dest = os.path.join(ts_model_store, mar_file.split('/')[-1])
        mar_file and subprocess.check_output(f'cp {mar_file} {mar_dest}', shell=True)

        if gpus > 0 and system_utils.is_gpu_instance():
            import torch

            if not torch.cuda.is_available():
                sys.exit("## Ohh its NOT running on GPU ! \n")
            print(f'\n## Running on {gpus} GPU(s) \n')
        
        else:
            print('\n## Running on CPU \n')
            gpus=0
        
        started, new_mar_file = ts.start_torchserve(gen_folder, model_store=ts_model_store, log_file=ts_log_file, 
            log_config_file=ts_log_config, config_file=ts_config_file, gpus=gpus, gen_mar=gen_mar, debug=debug)
        if not started:
            error_msg_print()
            sys.exit(1)

        cmd = "curl localhost:8080/ping"
        os.system(cmd)  

        for model in models_to_validate:
            model_name = model["name"]
            model_inputs = model["inputs"]
            model_handler = model["handler"]

            input_mar = mar_file if(mar_file) else new_mar_file

            # Run REST inference
            response = ts.register_model(model_name, input_mar)    
            if response and response.status_code == 200:
                print(f"## Successfully registered {input_mar} model with torchserve \n")
            else:
                print("## Failed to register model with torchserve \n")
                error_msg_print()
                sys.exit(1)

            if (mar_file):
                result = requests.get('http://localhost:8081/models/').json()  
                model_name = get_model_name(result, mar_file)

            # For each input execute inference n=4 times
            for input in model_inputs:
                print(input)
                response = ts.run_inference(model_name, input)  
                if response and response.status_code == 200:
                    print(f"## Successfully ran inference on {model_name} model. \n\n Output - {response.text}\n\n")
                else:
                    print(f"## Failed to run inference on {model_name} model \n")
                    error_msg_print()
                    sys.exit(1)

            if model != inference_model:
                response = ts.unregister_model(model_name)
                if response and response.status_code == 200:
                    print(f"## Successfully unregistered {model_name} \n")
                else:
                    print(f"## Failed to unregister {model_name} \n")
                    error_msg_print()
                    sys.exit(1)

            debug and os.system(f"curl http://localhost:8081/models/{model_name}")
            print(f"## {model_handler} Handler is stable. \n")
    except Exception as e:
        error_msg_print()
        print(e)
        sys.exit(1)