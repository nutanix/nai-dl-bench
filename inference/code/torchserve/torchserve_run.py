from utils.inference_utils import get_inference, get_inference_with_mar, error_msg_print
from utils.shell_utils import rm_dir, rm_file
from utils import tsutils as ts
from utils.system_utils import check_if_path_exists, create_folder_if_not_exits
import os
import argparse
import json
import inference_data_model as dm
import subprocess
import sys

def error_handling(error_reason, model_name):
    print(f'\n{error_reason} not found in models json under {model_name}\n')
    error_msg_print()
    sys.exit(1)


def set_default_values(data_model, model_name, model_path, model_arch_path, classes, handler_path, gen_folder):
    dir_path = os.path.dirname(__file__)
    gen_folder_path = os.path.join(dir_path, 'utils', gen_folder)
    with open(os.path.join(dir_path, 'models/models.json'), 'r') as f:
        model_config = json.loads(f.read())
        if(model_name not in model_config):
            print(f"{model_name} is not present in the models json")
            error_msg_print()
            sys.exit(1)

        data_model.model_name = model_name
        models_folder_path = os.path.join(dir_path, 'models', model_name)
        if(not model_path):
            if("weights" not in model_config[model_name]):
                error_handling("weights", model_name)

            weights = model_config[model_name]["weights"]
            create_pt_file = 'create_model_pt_file.py'
            subprocess.check_output(f'python3 {os.path.join(dir_path, create_pt_file)} --model_name {model_name} --weights {weights} --output {gen_folder_path}', shell=True)
            
            file_name = f'{model_name}-default.pt'
            dest_path = os.path.join(gen_folder_path, file_name)
            check_if_path_exists(dest_path)
            data_model.model_path = dest_path


        if(not model_arch_path):
            if("model_arch_file" not in model_config[model_name]):
                error_handling("model_arch_file", model_name)

            dest_path = os.path.join(models_folder_path, model_config[model_name]["model_arch_file"])
            check_if_path_exists(dest_path)
            data_model.model_arch_path = dest_path


        if(not classes):
            if("class_map" not in model_config[model_name]):
                error_handling("class_map", model_name)

            dest_path = os.path.join(models_folder_path, model_config[model_name]["class_map"])
            check_if_path_exists(dest_path)
            data_model.index_file_path = dest_path
        

        if(not handler_path):
            if("handler" not in model_config[model_name]):
                error_handling("handler", model_name)
            
            handler = model_config[model_name]["handler"]
            if(handler not in ['image_classifier', 'image_segmenter', 'object_detector', 'text_classifier']):
                handler = os.path.join(models_folder_path, handler)
                check_if_path_exists(handler)
        
            data_model.handler_path = handler
        

def run_inference_with_mar(args):
    check_if_path_exists(args.mar)
    data_model = dm.set_data_model(args.data, args.gpus, args.gen_folder_name, model_name=args.model_name, mar_filepath=args.mar)
    
    get_inference_with_mar(data_model, args.debug_mode)


def run_inference_with_custom_params(args):
    data_model = dm.set_data_model(args.data, args.gpus, args.gen_folder_name, args.model_name, args.model_path, args.handler_path, args.classes,  
        args.model_arch_path, args.extra_files)
    
    if(not args.model_path or not args.model_arch_path or not args.classes or not args.handler_path):
        set_default_values(data_model, args.model_name, args.model_path, args.model_arch_path, args.classes, args.handler_path, args.gen_folder_name)

    get_inference(data_model, args.debug_mode)


def run_inference(args):
    # validate gen folder
    create_folder_if_not_exits(os.path.join(os.path.dirname(__file__), 'utils', args.gen_folder_name))

    if(args.mar):
        run_inference_with_mar(args)

    elif(not args.model_name):
        print("Some of the required parameters are empty -n <MODEL_NAME>")
        error_msg_print()
        sys.exit(1)
    
    else:
        run_inference_with_custom_params(args)


def torchserve_run(args):
    try:
        # Stop the server if anything is running
        cleanup(args.gen_folder_name, True, False)

        # data folder exists check
        if(args.data):
            check_if_path_exists(args.data)

        run_inference(args)

        print(f"\n**************************************")
        print(f"*\n*\n*  Inference Run Successful  ")
        print(f"*\n*\n**************************************")

    finally:
        cleanup(args.gen_folder_name, args.stop_server, args.ts_cleanup)


def cleanup(gen_folder, ts_stop = True, ts_cleanup = True):
    if ts_stop:
        ts.stop_torchserve()
        dirpath = os.path.dirname(__file__)
        # clean up the logs folder to reset logs before the next run
        # TODO - To reduce logs from taking a lot of storage it is being cleared everytime it is stopped
        # Understand on how this can be handled better by rolling file approach
        rm_dir(os.path.join(dirpath, 'utils', gen_folder, 'logs'))

        if ts_cleanup:
            # clean up the entire generate folder
            rm_dir(os.path.join(dirpath, 'utils', gen_folder))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inference run script')
    parser.add_argument('--model_path', type=str, default="", 
                        metavar='m', help='absolute path to the saved model file')
    
    parser.add_argument('--model_arch_path', type=str, default="", 
                        metavar='a', help='absolute path to the model architecture file')
    
    parser.add_argument('--handler_path', type=str, default="", 
                        metavar='h', help='absolute path to the handler file')

    parser.add_argument('--data', type=str, default="",
                        metavar='d', help='absolute path to the inputs folder that contains data to be predicted.')
    
    parser.add_argument('--classes', type=str, default="", 
                        metavar='c', help='absolute path classes mapping file')
    
    parser.add_argument('--model_name', type=str, default="",
                        metavar='n', help='name of the model file')

    parser.add_argument('--extra_files', type=str, default="", 
                        metavar='e', help='any additional files required by your model to execute')
    
    parser.add_argument('--gpus', type=int, default=0,
                        metavar='g', help='number of gpus to use for execution')

    parser.add_argument('--gen_folder_name', type=str, default="gen", 
                        metavar='f', help='Name for generate folder used to create temp files')

    parser.add_argument('--gen_mar', type=int, default=1, 
                        metavar='gm', help='generate mar file before starting')

    parser.add_argument('--stop_server', type=int, default=1, 
                        metavar='stop', help='Stop torchserve after run completion')

    parser.add_argument('--ts_cleanup', type=int, default=1, 
                        metavar='cleanup', help='clean up torchserve temp files after run completion')

    parser.add_argument('--debug_mode', type=int, default=0, 
                        metavar='debug', help='run debug mode')

    parser.add_argument('--mar', type=str, default="", 
                        metavar='mar', help='absolute path to the model archive file')

    args = parser.parse_args()
    torchserve_run(args)
