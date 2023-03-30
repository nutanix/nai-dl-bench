from utils.inference_utils import get_inference
from utils.shell_utils import rm_dir, rm_file
from utils import tsutils as ts
import os
import argparse


def torchserve_run(args):
    try:
        if(not args.mar and (not args.model_path or not args.model_arch_path or not args.classes or not args.handler_path)):
            print("the following arguments are required: --model_path, --model_arch_path, --handler_path, --classes")
            sys.exit(1)

        # Run Inference
        get_inference(args.model_name, args.model_path, args.handler_path, args.classes, args.data, args.model_arch_path,
             args.extra_files, args.gpus, args.gen_folder_name, args.gen_mar, args.mar, args.debug_mode)

        print(f"\n**************************************")
        print(f"*\n*\n*  Inference Run Successful  ")
        print(f"*\n*\n**************************************")

    finally:
        cleanup(args.gen_folder_name, args.stop_server, args.ts_cleanup)


def cleanup(gen_folder, ts_stop = True, ts_cleanup = True):
    if ts_stop:
        ts.stop_torchserve()
    
        if ts_cleanup:
            dirpath = os.path.dirname(__file__)
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
    
    parser.add_argument('--gpus', type=int, default=0, required= True,
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