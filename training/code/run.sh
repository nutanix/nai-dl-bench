#!/bin/bash

function containerHelpFunction()
{
   echo "inside container help function"
   echo "Usage: $0 -i <IMAGE> -c <COMMAND_TO_RUN_PYTHON_SCRIPT>"
   echo "\t-i container image name used for training"
   echo "\t-c Python command with space separated list of --<option> <argument> inside double quotes for the training script. data and output options are mandatory for the training.py script"
   echo "\t-r resorces to be provided to the container in quotes '--gpu <number of gpus to be used> --cpu <cpu resources to be used> --mem <ram to be provided to the container> --shm <shared memory to be provided to the container>' "
   exit 1 # Exit script after printing help
}

function kubernetesvmHelpFunction()
{
   echo "inside kubernetes help function"
   echo "Usage: $0 -n <NUM_PROCS> -i <IMAGE> -c <COMMAND_TO_RUN_PYTHON_SCRIPT>"
   echo "\t-n Number of training processes"
   echo "\t-i container image name used for training"
   echo "\t-c Python command with space separated list of --<option> <argument> inside double quotes for the training script. data and output options are mandatory for the training.py script"
   echo "\t-r resorces to be provided to the container in quotes '--gpu <number of gpus to be used> --cpu <cpu resources to be used> --mem <ram to be provided to the container> --shm <shared memory to be provided to the container>' "
   exit 1 # Exit script after printing help
}

function vmHelpFunction()
{
   echo "inside vm help function"
   echo "Usage: $0 -n <NUM_PROCS> -h <HOSTS_IP_LIST> -m <MASTER_ADDRESS> -c <COMMAND_TO_RUN_PYTHON_SCRIPT>"
   echo  "\t-n Number of training processes"
   echo  "\t-h Comma separated list of Host IPs"
   echo  "\t-m IP Address of master node"
   echo  "\t-c Python command with space separated list of --<option> <argument> inside double quotes for the training script. data and output options are mandatory for the training.py script"
   exit 1 # Exit script after printing help
}

function training(){
    echo "hosts - $hosts"
    echo "Running the training script";
    echo "mpirun -np $processes -H $hosts -x MASTER_ADDR=$masterNode -x MASTER_PORT=29500 -x PATH -bind-to none -map-by slot -mca pml ob1 -mca btl ^openib -wdir $wdir $command"
    mpirun -np $processes -H $hosts -x MASTER_ADDR=$masterNode -x MASTER_PORT=29500 -x PATH -bind-to none -map-by slot -mca pml ob1 -mca btl ^openib -wdir $wdir $command
    if [ $? -gt 0 ] ;then
        exit 1
    else
        echo "completed running the training script"
    fi
    
}

function vmTraining(){
    echo "inside vm training function"
    wdir=$(dirname $BASH_SOURCE)

    # Print vmHelpFunction in case parameters are empty
    if [ -z "$processes" ] || [ -z "$ips" ] || [ -z "$masterNode" ]
    then
        echo "Some or all of the parameters are empty";
        vmHelpFunction
    fi

    if [ -z "$command" ]
    then
        echo "Command to execute training script is empty"
        exit 1
    fi

    # Begin script in case all parameters are correct
    echo "Processes - $processes"
    echo "IPs - $ips"
    echo "Master Node - $masterNode"

    ipList=(${ips//,/ })
    count=${#ipList[@]}

    # Set variable for type input variant in IPs
    if [[ "${ips}" == *:* ]]; then  
        proc_set=1
        ipList_with_demitter=(${ips//:/ })
    else
        proc_set=0
    fi

    # Master Node validation
    if [[ ! "${ipList[*]}" =~ "${masterNode}" ]]
    then
        echo "Master Node must be among the IPs";
        vmHelpFunction
    fi

    # Working Directory validation in current node
    FILE=${wdir}/training.py
    if [ ! -f "$FILE" ]
    then
        echo "$FILE does not exist. Please set the correct working directory parameter to the root of the nai-dl-bench folder. Absolute path must be the same across all hosts";
        vmHelpFunction
    fi

    function check_ip_valid() {
        ips_to_check=("$@")
        rx='([1-9]?[0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])'
        for ip in "${ips_to_check[@]}"
        do
            if [ "${proc_set}" -eq 1 ]; then
                split=(${ip//:/ })
                ip=${split[0]}
            fi

            if [[ ! $ip =~ ^$rx\.$rx\.$rx\.$rx$ ]]; then
                echo "$ip is not in the valid format"
                vmHelpFunction
            fi
        done
    }

    check_ip_valid ${ipList[@]} # Validate IP format

    # Assign Hosts
    if [ "${proc_set}" -eq 1 ] # Process count has been set with IP
    then
        # Validate if Process count is set for all IPs
        max=$(( count+1 ))
        if [[ "${max}" -ne "${#ipList_with_demitter[@]}" ]]
        then
            echo "Number of processes has not been assigned to all the provided IPs";
            vmHelpFunction
        fi

        proc=0
        for item in "${ipList[@]}"
        do
            split=(${item//:/ })
            ((proc=split[1]+proc))
        done

        # Validate if total process count matches initial set number
        if [[ "${processes}" -ne "${proc}" ]]
        then
            echo "Sum of processes assigned to all hosts need to match the number of processes set at -n";
            vmHelpFunction
        fi

        hosts=${ips}
    else 

        # Validate if minimum number IPs has been provided
        if [ "${count}" -gt "${processes}" ] 
        then
            echo "Number of processes should be >= number of IPs";
            vmHelpFunction
        fi

        if [ $cpuOnly != true ]
        then
            # Check nvidia driver files are present
            FILE=/proc/driver/nvidia/version
            if [ ! -f "$FILE" ]
            then
                echo "$FILE does not exist. Please make sure Nvidia Driver has been installed";
                vmHelpFunction
            fi

            gpu=$(nvidia-smi --list-gpus | wc -l)
            echo "GPUs per host, Assuming same number GPUs in every host - $gpu"
            max=$(( count*gpu ))
            if [ "${max}" -lt "${processes}" ]
            then
                echo "Assuming same number of GPUs setup, number of processes should be < number of host IPs * number of GPUs per host";
                echo "For alternate scenarios, assign processes per IP. Format -  <IP_n>:<NUM_PROCS_for_ip_n>";
                echo "Example: 0.0.0.0:1,0.0.0.1:2";
                vmHelpFunction
            fi
        fi
        declare -a processes_per_IP
        index=0
        n=${processes}
        max=${#ipList[@]}
        ((max--))
        while [ ${n} -gt 0 ];
        do
            if [ -z ${processes_per_IP[$index]} ]
            then
                processes_per_IP[index]=0
            fi

            ((processes_per_IP[$index]=processes_per_IP[$index]+1))
            ((n--))
            ((index++))
            if [ ${index} -gt ${max} ]
            then
                index=0
            fi
        done

        n=${#processes_per_IP[@]} 
        hosts=""
        while [ ${n} -gt 0 ];
        do
            if [ ${n} -ne ${#processes_per_IP[@]}  ]
            then
                hosts+=","
            fi
            hosts+=${ipList[${n}-1]}":"${processes_per_IP[${n}-1]}
            ((n--))
        done
    fi

    training

}

function fetchResources(){

    # Split the string into an array of arguments
    IFS=" " read -r -a res_array <<< "$resources"

    for ((i=0; i<${#res_array[@]}; i++)); 
    do
    # Check if the argument is '--data-folder'
    if [[ ${res_array[i]} == "--gpu" ]]; then
        # Get the value from the next index
        gpu=${res_array[i+1]}
    fi
    if [[ ${res_array[i]} == "--cpu" ]]; then
        # Get the value from the next index
        cpu=${res_array[i+1]}
    fi
    if [[ ${res_array[i]} == "--mem" ]]; then
        # Get the value from the next index
        mem=${res_array[i+1]}
    fi
    if [[ ${res_array[i]} == "--shm" ]]; then
        # Get the value from the next index
        shm=${res_array[i+1]}
    fi
    done
}

function kubernetesTraining()
{   echo "inside kubernetes training"
    variable_name="KUBECONFIG"
    is_exported=$(printenv | grep -q "^$variable_name=" && echo "true" || echo "false")

    if [ "$is_exported" = "true" ]; then
        echo "Kube config variable is set"
    else
        echo "Kube config variable is not set"
        kubernetesvmHelpFunction
    fi

    if [ -z "$processes" ] || [ -z "$image"  ]
    then
        echo "Some or all of the parameters are empty";
        kubernetesvmHelpFunction
    fi

    echo "$command"
    if [ -z "$command" ]
    then
        echo "Command to execute training script is empty"
        exit 1
    fi

    fetchResources

    if [ -z "$mem"  ] || [ -z "$gpu"  ] || [ -z "$shm"  ] || [ -z "$cpu"  ]
    then
        echo "one of the resources is empty";
        kubernetesHelpFunction
    fi

    wdir=$(dirname $BASH_SOURCE)

    echo "python "$wdir/utils/k8s.py" --processes $processes -i $image -c "$command" --gpu $gpu --cpu $cpu --mem $mem --shm $shm"
    python "$wdir/utils/k8s.py" --processes $processes -i $image -c "$command" --gpu $gpu --cpu $cpu --mem $mem --shm $shm

    if [ $? -gt 0 ] ;then
        exit 1
    else
        echo "completed running the training script"
    fi
}

function containerTraining()
{

    data_folder=""
    output_folder=""
    cpu=""
    gpu=""
    mem=""
    shm=""

    if [ -z "$command" ]
    then
        echo "python command empty";
        containerHelpFunction
    fi

    if [ -z "$image" ]
    then
        echo "docker image empty";
        containerHelpFunction
    fi

    # Split the string into an array of arguments
    IFS=" " read -r -a arg_array <<< "$command"

    for ((i=0; i<${#arg_array[@]}; i++)); do
    # Check if the argument is '--data-folder'
    if [[ ${arg_array[i]} == "--data-folder" ]]; then
        # Get the value from the next index
        data_folder=${arg_array[i+1]}
    fi
    if [[ ${arg_array[i]} == "--output-folder" ]]; then
        # Get the value from the next index
        output_folder=${arg_array[i+1]}
    fi
    done
    
    if [ -z "$data_folder" ] || [ -z "$output_folder"  ]
    then
        echo "data folder and output folder empty";
        containerHelpFunction
    fi

    fetchResources

    if [ -z "$mem"  ] || [ -z "$gpu"  ] || [ -z "$shm"  ] || [ -z "$cpu"  ]
    then
        echo "one of the resources is empty";
        containerHelpFunction
    fi

    indexes=""
    n=$gpu
    for ((i=0; i<n; i++)); do
    if [[ $i != $((n-1)) ]] ; then
        indexes+="$i,"
    else
        indexes+="$i"
    fi
    done

    string=$command
    first_word_removed_command="${string#* }"
    docker network create testnet

    if [[ $gpu != 0 ]]; then
        echo "docker run --cpus $cpu --memory $mem --net testnet --name master -t -v $data_folder:$data_folder -v $output_folder:$output_folder --gpus device=$indexes --shm-size $shm  $image torchrun --nnodes=1 --nproc_per_node=$gpu --rdzv_endpoint=master:29500 --rdzv_backend=c10d $first_word_removed_command"
        docker run --cpus $cpu --memory $mem --net testnet --name master -t -v $data_folder:$data_folder -v $output_folder:$output_folder --gpus device=$indexes --shm-size $shm  $image torchrun --nnodes=1 --nproc_per_node=$gpu --rdzv_endpoint=master:29500 --rdzv_backend=c10d $first_word_removed_command
    else
        echo "docker run --cpus $cpu --memory $mem  --net testnet --name master -t -v $data_folder:$data_folder -v $output_folder:$output_folder  --shm-size $shm  $image torchrun --nnodes=1 --nproc_per_node=1 --rdzv_endpoint=master:29500 --rdzv_backend=c10d $first_word_removed_command"
        docker run --cpus $cpu --memory $mem  --net testnet --name master -t -v $data_folder:$data_folder -v $output_folder:$output_folder  --shm-size $shm  $image torchrun --nnodes=1 --nproc_per_node=1 --rdzv_endpoint=master:29500 --rdzv_backend=c10d $first_word_removed_command
    fi

    if [ $? -gt 0 ] ;then
        exit 1
    else
        echo "completed running the training script"
    fi
}

choice="vm"

while getopts ":e:" opt; do
    case $opt in
        e)
            choice=$OPTARG
            ;;
    esac
done

shift $((OPTIND-3))
OPTIND=2

case $choice in
    "docker")
        echo "You chose docker."
        while getopts ":i:c:r:" opt; do
            case $opt in
                i)
                    image=$OPTARG
                    ;;
                c)
                    command=$OPTARG
                    ;;
                r)
                    resources=$OPTARG
                    ;;
                ?)
                    containerHelpFunction
                    ;;
            esac
        done
        containerTraining
        ;;
    "k8s")
        echo "You chose Kubernetes."
        while getopts ":n:i:c:r:" opt; do
            case $opt in
                n)
                    processes=$OPTARG
                    ;;
                i)
                    image=$OPTARG
                    ;;
                c)
                    command=$OPTARG
                    ;;
                r)
                    resources=$OPTARG
                    ;;
                ?)
                    kubernetesvmHelpFunction
                    ;;
            esac
        done
        kubernetesTraining
        ;;
    "vm")
        echo "You chose VM."
        cpuOnly=false
        while [ $# -gt 0 ]; do
                case $1 in
                -n) 
                    shift
                    processes=$1
                    ;;
                -h)
                    shift
                    ips=$1
                    ;;
                -m)
                    shift
                    masterNode=$1
                    ;;
                -c)
                    shift
                    command=$1
                    ;;
                -cpuOnly)
                    cpuOnly=true
                    ;;
                esac
            shift
        done
        vmTraining
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