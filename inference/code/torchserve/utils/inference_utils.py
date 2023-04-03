import os
import sys
import nvgpu

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)

import tsutils as ts
import system_utils
import time
import json
import subprocess
import requests
import inference_data_model
import traceback

def error_msg_print():
    print(f"\n**************************************")
    print(f"*\n*\n*  Error found - Unsuccessful ")
    print(f"*\n*\n**************************************")
    stopped = ts.stop_torchserve()

def create_init_json(data_model, debug):
    init_json = {}
        
    init_json['model_name'] = data_model.model_name
    init_json['version'] = "1.0"
    init_json['model_file'] = data_model.model_arch_path
    init_json['serialized_file_local'] = data_model.model_path
    init_json['handler'] = data_model.handler_path
    init_json['extra_files'] = data_model.index_file_path

    if data_model.extra_files:
            init_json['extra_files'] += ", "+ data_model.extra_files

    debug and print("\n",init_json, "\n")
    with open(os.path.join(data_model.dir_path, data_model.gen_folder, 'mar_config.json'), 'w') as f:
        json.dump([init_json], f)


def get_model_name(input_dict, modelUrl):
    for entry in input_dict['models']:
        if(modelUrl == entry['modelUrl']):
            return entry['modelName']
    
    print('\n model not found among registered models')
    error_msg_print()
    sys.exit(1)


def get_inputs_from_folder(input_path):
    return [os.path.join(input_path, item) for item in os.listdir(input_path)] if input_path else []


def set_compute_setting(gpus):
    if gpus > 0 and system_utils.is_gpu_instance():
        import torch

        if not torch.cuda.is_available():
            sys.exit("## Ohh its NOT running on GPU ! \n")
        print(f'\n## Running on {gpus} GPU(s) \n')
    
    else:
        print('\n## Running on CPU \n')
        gpus=0


def prepare_settings(data_model):
    data_model.dir_path = os.path.dirname(__file__)
    data_model.ts_log_file = os.path.join(data_model.dir_path, data_model.gen_folder, "logs/ts_console.log")
    data_model.ts_log_config = os.path.join(data_model.dir_path, "../log4j2.xml")
    data_model.ts_config_file = os.path.join(data_model.dir_path, '../config.properties')
    data_model.ts_model_store = os.path.join(data_model.dir_path, data_model.gen_folder, "model_store")

    gen_path = os.path.join(data_model.dir_path, data_model.gen_folder, "logs")
    os.environ["LOG_LOCATION"] = gen_path
    os.environ["METRICS_LOCATION"] = gen_path

    os.makedirs(os.path.join(data_model.dir_path, data_model.gen_folder, "model_store"), exist_ok=True)
    os.makedirs(os.path.join(data_model.dir_path, data_model.gen_folder, "logs"), exist_ok=True)


def ts_health_check():
    os.system("curl localhost:8080/ping")  


def start_ts_server(gen_folder, ts_model_store, ts_log_file, ts_log_config, ts_config_file, gpus, gen_mar, debug):
    started, generated_mar_file = ts.start_torchserve(gen_folder, model_store=ts_model_store, log_file=ts_log_file, 
        log_config_file=ts_log_config, config_file=ts_config_file, gpus=gpus, gen_mar=gen_mar, debug=debug)
    if not started:
        error_msg_print()
        sys.exit(1)

    return generated_mar_file


def execute_inference_on_inputs(model_inputs, model_name):
    for input in model_inputs:
        print(input)
        response = ts.run_inference(model_name, input)  
        if response and response.status_code == 200:
            print(f"## Successfully ran inference on {model_name} model. \n\n Output - {response.text}\n\n")
        else:
            print(f"## Failed to run inference on {model_name} model \n")
            error_msg_print()
            sys.exit(1)


def register_model(model_name, input_mar):
    response = ts.register_model(model_name, input_mar)    
    if response and response.status_code == 200:
        print(f"## Successfully registered {input_mar} model with torchserve \n")
    else:
        print("## Failed to register model with torchserve \n")
        error_msg_print()
        sys.exit(1)


def unregister_model(model_name):
    response = ts.unregister_model(model_name)
    if response and response.status_code == 200:
        print(f"## Successfully unregistered {model_name} \n")
    else:
        print(f"## Failed to unregister {model_name} \n")
        error_msg_print()
        sys.exit(1)


def validate_inference_model(models_to_validate, input_mar, model_name, is_mar_generated, debug):
    for model in models_to_validate:
        model_name = model["name"]
        model_inputs = model["inputs"]
        model_handler = model["handler"]

        register_model(model_name, input_mar)

        if (not is_mar_generated):
            # For pre-existing mar file the model name might be set different while creation
            # Fetch the correct model name for inference requests
            result = requests.get('http://localhost:8081/models/').json()  
            model_name = get_model_name(result, input_mar)

        execute_inference_on_inputs(model_inputs, model_name)

        unregister_model(model_name)

        debug and os.system(f"curl http://localhost:8081/models/{model_name}")
        print(f"## {model_handler} Handler is stable. \n")


def get_inference_internal(data_model, generate_mar, debug):
    dm = data_model
    inputs = get_inputs_from_folder(dm.input_path)

    inference_model = {
        "name": dm.model_name,
        "inputs": inputs,
        "handler": dm.handler_path,
    }

    models_to_validate = [
        inference_model
    ]

    set_compute_setting(dm.gpus)
    
    # mar file is generated only if generate_mar is True else it will be None
    generated_mar_file = start_ts_server(dm.gen_folder, dm.ts_model_store, dm.ts_log_file, dm.ts_log_config, dm.ts_config_file, dm.gpus, generate_mar, debug)

    ts_health_check()

    # Use input provided mar if it exists
    input_mar = generated_mar_file if(generate_mar) else dm.mar_filepath
    validate_inference_model(models_to_validate, input_mar, dm.model_name, generate_mar, debug)


def get_inference_with_mar(data_model, debug=False):
    try:
        prepare_settings(data_model)

        # copy mar file to model_store
        mar_dest = os.path.join(data_model.ts_model_store, data_model.mar_filepath.split('/')[-1])
        if (data_model.mar_filepath != mar_dest):
            subprocess.check_output(f'cp {data_model.mar_filepath} {mar_dest}', shell=True)

        get_inference_internal(data_model, generate_mar=False, debug=debug)

    except Exception as e:
        error_msg_print()
        print(e)
        debug and traceback.print_exc()
        sys.exit(1)


def get_inference(data_model, debug=False):
    try:
        prepare_settings(data_model)

        # create json config file for mar file generation
        init_json = create_init_json(data_model, debug)

        get_inference_internal(data_model, generate_mar=True, debug=debug)
    except Exception as e:
        error_msg_print()
        print(e)
        debug and traceback.print_exc()
        sys.exit(1)