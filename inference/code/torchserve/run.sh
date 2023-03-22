#!/usr/bin/env bash
SCRIPT=$(realpath "$0")
wdir=$(dirname "$SCRIPT")

helpFunction()
{
   echo ""
   echo "Usage: $0 -n <MODEL_NAME> -d <INPUT_DATA_ABSOLUTE_PATH> -m <MODEL_ABSOLUTE_PATH> -f <MODEL_ARCH_FILE_ABSOLUTE_PATH> -c <CLASSES_MAPPING_ABSOLUTE_PATH> -h <HANDLER_FILE_ABSOLUTE_PATH> -e <EXTRA_FILES> -g <NUM_OF_GPUS>"
   echo -e "\t-n Name of the Model"
   echo -e "\t-d Absolute path to the inputs folder that contains data to be predicted."
   echo -e "\t-m Absolute path to the saved model file"
   echo -e "\t-f Absolute path to the model arch file"
   echo -e "\t-c Absolute path classes mapping file"
   echo -e "\t-h Absolute path handler file"
   echo -e "\t-e Comma separated absolute paths of all the additional paths required by the model"
   echo -e "\t-g Number of gpus to be used to execute. Default will be 0, cpu used"
   exit 1 # Exit script after printing help
}

while getopts "n:d:m:f:c:h:e:g:" opt
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
        ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Print helpFunction in case parameters are empty
if [ -z "$model_name" ] || [ -z "$data" ]
then
   echo "Some of the required parameters are empty -n <MODEL_NAME> -d <INPUT_DATA_ABSOLUTE_PATH>";
   helpFunction
fi

cmd="python3 $wdir/torchserve_run.py --model_name $model_name --data $data"

function set_default_params(){
    model_arch_path=$wdir'/model_arch.py'
    classes=$wdir'/index_to_name.json'
    handler_path='image_classifier'
    if [ -z "$gpus" ] ; then
        gpus=0
    fi
}

function validate_params() {
    validate_model_file_path

    if [ -z "$model_arch_path" ] ; then
        echo "Model Arch file path as not been provided"
        helpFunction
    elif [ -z "$classes" ] ; then
        echo "Classes mapping file path as not been provided"
        helpFunction
    elif [ -z "$handler_path" ] ; then
        echo "Handler file path as not been provided"
        helpFunction
    fi
}

function validate_model_file_path() {
    FILE=$model_file_path
    if [ ! -f "$FILE" ]
    then
        echo "$FILE does not exist. Please set the correct aboslute path to the saved model file";
        helpFunction
    fi
}

gen_folder="gen"
mkdir $wdir/utils/$gen_folder
if [ ! -z "$model_file_path" ]; then
    if [ ! -z "$model_arch_path" ] || [ ! -z "$classes" ] || [ ! -z "$handler_path" ] ; then
        echo "Model with given params"
        validate_params
    else
        echo "Model file with defualt params"
        validate_model_file_path
        set_default_params
    fi   

else
    echo "Default params"
    python3 $wdir/create_model_pt_file.py
    mv $wdir/default-resnet-50-model.pt $wdir/utils/gen/default-resnet-50-model.pt

    model_file_path=$wdir'/utils/gen/default-resnet-50-model.pt'
    set_default_params
fi

cmd+=" --model_path $model_file_path --model_arch_path $model_arch_path  --classes $classes --handler_path $handler_path  --gpus $gpus --gen_folder_name $gen_folder"

if [ ! -z $extra_files ] ; then
    cmd+=" --extra_files $extra_files"
fi

echo "Running the Inference script";
echo "$cmd"
$cmd


