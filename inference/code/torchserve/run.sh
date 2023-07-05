#!/usr/bin/env bash
SCRIPT=$(realpath "$0")
wdir=$(dirname "$SCRIPT")

helpFunction()
{
   echo ""
   echo "Usage: $0 -n <MODEL_NAME> -d <INPUT_DATA_ABSOLUTE_PATH> -m <MODEL_ABSOLUTE_PATH> -f <MODEL_ARCH_FILE_ABSOLUTE_PATH> -c <CLASSES_MAPPING_ABSOLUTE_PATH> -h <HANDLER_FILE_ABSOLUTE_PATH> -e <EXTRA_FILES> -g <NUM_OF_GPUS> -a <ABOSULTE_PATH_MODEL_ARCHIVE_FILE>"
   echo -e "\t-o Choice of compute infra to be run on"
   echo -e "\t-i Container Image"
   echo -e "\t-r resorces to be provided to the container in quotes '--cpu <cpu to be used> --mem <memory to be provided to the container>'"
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

while getopts ":kn:d:m:f:c:h:e:g:a:o:i:r:" opt
do
   case "$opt" in
        o ) compute_choice="$OPTARG" ;;
        i ) image="$OPTARG" ;;
        r ) resources="$OPTARG" ;;
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

function validate_and_get_resources(){
    # Split the string into an array of arguments
    IFS=" " read -r -a res_array <<< "$resources"

    for ((i=0; i<${#res_array[@]}; i++)); 
    do
    if [[ ${res_array[i]} == "--cpu" ]]; then
        # Get the value from the next index
        cpu=${res_array[i+1]}
    fi
    if [[ ${res_array[i]} == "--mem" ]]; then
        # Get the value from the next index
        mem=${res_array[i+1]}
    fi
    done

    if [ -z "$image"  ]
    then
        echo "Some or all of the parameters are empty";
        helpFunction
    fi

    if [ -z "$mem"  ] ||  [ -z "$cpu"  ]
    then
        echo "one of the resources is empty";
        helpFunction
    fi
}

function create_execution_cmd()
{
    gen_folder="gen"
    if [ $compute_choice = "vm" ] ;
    then
        cmd="python3 $wdir/torchserve_run.py"
    else
        cmd="python code/torchserve/torchserve_run.py"
    fi

    cmd+=" --gen_folder_name $gen_folder"

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
}

function inference_exec_vm(){
    echo "Running the Inference script";
    echo "$cmd"
    $cmd
}

function inference_exec_kubernetes()
{   
    if [ -z "$KUBECONFIG" ]; then
        echo "Kube config environment variable is not set - KUBECONFIG"
        exit 1 
    fi

    if [ -z "$gpus"  ] || [ "$gpus" -eq 0 ] 
    then
        gpus="0"
    fi

    validate_and_get_resources

    mem+='Gi'
    echo "Running the Inference script";
    python $wdir/kubernetes_run.py --image $image --command "$cmd" --gpu $gpus --cpu $cpu --mem $mem
}

function inference_exec_container()
{
    validate_and_get_resources
    mem+='g'

    echo "Running the Inference script";
    docker_exec_cmd="docker run --cpus $cpu --memory $mem --gpus $gpus -p 8080:8080 -p 8081:8081 -p 8082:8082 $image $cmd"
    echo "$docker_exec_cmd"
    $docker_exec_cmd
}

# Entry Point
if [ -z "$compute_choice"  ] 
then
    compute_choice="vm"
fi

create_execution_cmd
case $compute_choice in
    "docker")
        echo "Compute choice is docker."
        inference_exec_container
        ;;

    "k8s")
        echo "Compute choice is Kubernetes."
        inference_exec_kubernetes
        ;;

    "vm")
        echo "Compute choice is VM."
        inference_exec_vm
        ;;
    *)
        echo "Invalid choice. Exiting."
        echo "Please select a valid option:"
        echo "1. k8s for Kubernetes env"
        echo "2. vm for virtual machine env"
        echo "3. docker for docker env"
        exit 1
        ;;
esac

