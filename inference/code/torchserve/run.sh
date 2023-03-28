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
   echo -e "\t-k Keep the torchserve server alive after run completion. Default stops the server if not set"
   exit 1 # Exit script after printing help
}

while getopts ":kn:d:m:f:c:h:e:g:" opt
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
        k ) stop_server=0 ;;
        ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Print helpFunction in case parameters are empty
if [ -z "$model_name" ] || [ -z "$data" ]
then
   echo "Some of the required parameters are empty -n <MODEL_NAME> -d <INPUT_DATA_ABSOLUTE_PATH>";
   helpFunction
fi

function validate_file_path() {
    file=$1
    if [ ! -f "$file" ]
    then
        echo "$file does not exist. Please set the correct aboslute path to the file";
        helpFunction
    fi
}

function validate_all_params() {
    validate_file_path $model_file_path
    validate_file_path $model_arch_path
    validate_file_path $classes
    if [ ! "$handler_path" = "image_classifier" ]; then
        validate_file_path $handler_path
    fi
}

function get_value() {
    item=$1
    out=$(python3 -c "import json,sys;f=open('$wdir/models/models.json');obj=json.load(f);print(obj['$model_name'].get('$item', '#'));")
    if [ "$out" = "#" ]; then
        echo "$item - value is not present in the under the provided model in the models json";
        helpFunction        
    fi

    echo "$out"
}

function set_default_params() {
    if [ -z "$model_file_path" ] ; then
        weights=$(get_value "weights")
        python3 $wdir/create_model_pt_file.py --model_name $model_name --weight $weights
        filename=$model_name'-default.pt'
        mv $wdir/$filename $wdir/utils/gen/$filename

        model_file_path=$wdir'/utils/gen/'$filename
    fi

    if [ -z "$model_arch_path" ] ; then
        res=$(get_value "model_arch_file")
        model_arch_path=$wdir'/models/'$model_name'/'$res
    fi

    if [ -z "$classes" ] ; then
        res=$(get_value "class_map")
        classes=$wdir'/models/'$model_name'/'$res
    fi

    if [ -z "$handler_path" ] ; then
        res=$(get_value "handler")
        if [ ! "$res" = "image_classifier" ]; then
            handler_path=$wdir'/models/'$model_name'/'$res
        else
            handler_path='image_classifier'
        fi
    fi

    if [ -z "$gpus" ] ; then
        gpus=0
    fi
}

gen_folder="gen"
mkdir $wdir/utils/$gen_folder
if [ -z "$model_file_path" ] || [ -z "$model_arch_path" ] || [ -z "$classes" ] || [ -z "$handler_path" ]
then
    echo "Some params are not provided, try using default params"
    out=$(python3 -c "import json,sys;f=open('$wdir/models/models.json');obj=json.load(f);print(obj.get('$model_name', '#'));")
    if [ "$out" = "#" ]; then
        echo "$model_name is not present in the default models json";
        helpFunction
    fi
    set_default_params
fi

validate_all_params

cmd="python3 $wdir/torchserve_run.py --model_name $model_name --data $data"
cmd+=" --model_path $model_file_path --model_arch_path $model_arch_path  --classes $classes --handler_path $handler_path  --gpus $gpus --gen_folder_name $gen_folder"

if [ ! -z $extra_files ] ; then
    cmd+=" --extra_files $extra_files"
fi

if [ ! -z $stop_server ] ; then
    cmd+=" --stop_server $stop_server"
fi

echo "Running the Inference script";
echo "$cmd"
$cmd


