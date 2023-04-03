#!/usr/bin/env bash
SCRIPT=$(realpath "$0")
wdir=$(dirname "$SCRIPT")

helpFunction()
{
   echo ""
   echo "Usage: $0 -n <MODEL_NAME> -d <INPUT_DATA_ABSOLUTE_PATH> -m <MODEL_ABSOLUTE_PATH> -f <MODEL_ARCH_FILE_ABSOLUTE_PATH> -c <CLASSES_MAPPING_ABSOLUTE_PATH> -h <HANDLER_FILE_ABSOLUTE_PATH> -e <EXTRA_FILES> -g <NUM_OF_GPUS> -a <ABOSULTE_PATH_MODEL_ARCHIVE_FILE>"
   echo -e "\t-n Name of the Model"
   echo -e "\t-d Absolute path to the inputs folder that contains data to be predicted."
   echo -e "\t-m Absolute path to the saved model file"
   echo -e "\t-f Absolute path to the model architecture file"
   echo -e "\t-c Absolute path classes mapping file"
   echo -e "\t-h Absolute path handler file"
   echo -e "\t-e Comma separated absolute paths of all the additional paths required by the model"
   echo -e "\t-g Number of gpus to be used to execute. Default will be 0, cpu used"
   echo -e "\t-a Absolute path to the model archive file (.mar)"
   echo -e "\t-k Keep the torchserve server alive after run completion. Default stops the server if not set"
   exit 1 # Exit script after printing help
}

while getopts ":kn:d:m:f:c:h:e:g:a:" opt
do
   case "$opt" in
        n ) model_name="$OPTARG" ;;
        d ) data="$OPTARG" ;;
        m ) model_file_path="$OPTARG" ;;
        f ) model_arch_path="$OPTARG" ;;
        c ) classes="$OPTARG" ;;
        h ) handler_path="$OPTARG" ;;
        e ) extra_files="$OPTARG" ;;
        g ) gpus="$OPTARG" ;;
        a ) mar_file_path="$OPTARG" ;;
        k ) stop_server=0 ;;
        ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

gen_folder="gen"
mkdir -p $wdir/utils/$gen_folder
cmd="python3 $wdir/torchserve_run.py --gen_folder_name $gen_folder"

if [ ! -z $model_name ] ; then
    cmd+=" --model_name $model_name"
fi

if [ ! -z $mar_file_path ] ; then
    cmd+=" --mar $mar_file_path"
fi

if [ ! -z $model_file_path ] ; then
    cmd+=" --model_path $model_file_path"
fi

if [ ! -z $model_arch_path ] ; then
    cmd+=" --model_arch_path $model_arch_path"
fi

if [ ! -z $classes ] ; then
    cmd+=" --classes $classes"
fi

if [ ! -z $handler_path ] ; then
    cmd+=" --handler_path $handler_path"
fi

if [ -z "$gpus" ] ; then
    gpus=0
else
    sys_gpus=$(nvidia-smi --list-gpus | wc -l)
    if [ "$gpus" -gt "$sys_gpus" ]; then
        echo "Machine has fewer GPUs ($sys_gpus) then input provided - $gpus";
        helpFunction  
    fi
fi
cmd+=" --gpus $gpus"

if [ ! -z $stop_server ] ; then
    cmd+=" --stop_server $stop_server"
fi

if [ ! -z "$data" ] ; then
    cmd+=" --data $data"
fi

echo "Running the Inference script";
echo "$cmd"
$cmd


