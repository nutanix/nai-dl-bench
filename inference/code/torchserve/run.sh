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

# TorchServe Default Handler for image classification
declare -A ts_default_handler
ts_default_handler["image_classifier"]=1
ts_default_handler["image_segmenter"]=1
ts_default_handler["object_detector"]=1
ts_default_handler["text_classifier"]=1

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
    if [ ! -v ts_default_handler[$handler_path] ]; then
        validate_file_path $handler_path
    fi
}

function get_value() {
    name=$1
    item=$2
    echo "$(python3 $wdir/utils/shell_utils.py --function load_item_from_json --params $name,$item)"
}

function check_value() {
    param=$1
    item=$2
    if [ -z "$item" ] || [ "$item" = '#' ]; then
        echo "$param - value is not present in the under the provided model in the models json";
        helpFunction  
    fi
}

function set_default_params() {
    if [ -z "$model_file_path" ] ; then
        weights=$(get_value $model_name "weights")
        check_value weights $weights
        python3 $wdir/create_model_pt_file.py --model_name $model_name --weight $weights
        filename=$model_name'-default.pt'
        mv $wdir/$filename $wdir/utils/gen/$filename

        model_file_path=$wdir'/utils/gen/'$filename
    fi

    if [ -z "$model_arch_path" ] ; then
        res=$(get_value $model_name "model_arch_file")
        check_value model_arch_file $res
        model_arch_path=$wdir'/models/'$model_name'/'$res
    fi

    if [ -z "$classes" ] ; then
        res=$(get_value $model_name "class_map")
        check_value class_map $res
        classes=$wdir'/models/'$model_name'/'$res
    fi

    if [ -z "$handler_path" ] ; then
        res=$(get_value $model_name "handler")
        check_value handler $res
        handler_path=$res
        if [ ! -v ts_default_handler[$res] ]; then
            handler_path=$wdir'/models/'$model_name'/'$res
        fi
    fi
}

function create_cmd() {
    validate_all_params 
    cmd+=" --model_name $model_name --model_path $model_file_path --model_arch_path $model_arch_path  --classes $classes --handler_path $handler_path"

    if [ ! -z $extra_files ] ; then
        cmd+=" --extra_files $extra_files"
    fi
}

gen_folder="gen"
mkdir -p $wdir/utils/$gen_folder
cmd="python3 $wdir/torchserve_run.py --gen_folder_name $gen_folder"
if [ ! -z "$mar_file_path" ]
then
    validate_file_path $mar_file_path
    cmd+=" --mar $mar_file_path"

    if [ ! -z "$model_name" ]; then 
        cmd+=" --model_name $model_name"
    fi

elif [ -z "$model_name" ]
then
   echo "Some of the required parameters are empty -n <MODEL_NAME>";
   helpFunction

elif [ -z "$model_file_path" ] || [ -z "$model_arch_path" ] || [ -z "$classes" ] || [ -z "$handler_path" ]
then
    echo "Some params are not provided, try using default params"
    check_value "$model_name - key," $(get_value $model_name)
    set_default_params
    create_cmd
else
    create_cmd
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


