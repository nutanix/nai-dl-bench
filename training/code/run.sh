#!/usr/bin/env bash
wdir=$(dirname $BASH_SOURCE)

function training(){
    echo "hosts - $hosts"
    echo "Running the training script";
    echo "mpirun -np $processes -H $hosts -x MASTER_ADDR=$masterNode -x MASTER_PORT=29500 -x PATH -bind-to none -map-by slot -mca pml ob1 -mca btl ^openib -wdir $wdir $command"
    mpirun -np $processes -H $hosts -x MASTER_ADDR=$masterNode -x MASTER_PORT=29500 -x PATH -bind-to none -map-by slot -mca pml ob1 -mca btl ^openib -wdir $wdir $command
    echo "completed running the training script";
}

helpFunction()
{
   echo ""
   echo "Usage: $0 -n <NUM_PROCS> -h <HOSTS_IP_LIST> -m <MASTER_ADDRESS> -c <COMMAND_TO_RUN_PYTHON_SCRIPT>"
   echo -e "\t-n Number of training processes"
   echo -e "\t-h Comma separated list of Host IPs"
   echo -e "\t-m IP Address of master node"
   echo -e "-c Python command with space separated list of --<option> <argument> inside double quotes for the training script. data and output options are mandatory for the training.py script"
   exit 1 # Exit script after printing help
}

while getopts "n:h:m:c:" opt
do
   case "$opt" in
        n ) processes="$OPTARG" ;;
        h ) ips="$OPTARG" ;;
        m ) masterNode="$OPTARG" ;;
        c ) command="$OPTARG";;
        ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Print helpFunction in case parameters are empty
if [ -z "$processes" ] || [ -z "$ips" ] || [ -z "$masterNode" ]
then
   echo "Some or all of the parameters are empty";
   helpFunction
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
    helpFunction
fi

# Working Directory validation in current node
FILE=${wdir}/training.py
if [ ! -f "$FILE" ]
then
    echo "$FILE does not exist. Please set the correct working directory parameter to the root of the nai-dl-bench folder. Absolute path must be the same across all hosts";
    helpFunction
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
            helpFunction
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
        helpFunction
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
        helpFunction
    fi

    hosts=${ips}
else 

    # Validate if minimum number IPs has been provided
    if [ "${count}" -gt "${processes}" ] 
    then
        echo "Number of processes should be >= number of IPs";
        helpFunction
    fi

    # Check nvidia driver files are present
    FILE=/proc/driver/nvidia/version
    if [ ! -f "$FILE" ]
    then
        echo "$FILE does not exist. Please make sure Nvidia Driver has been installed";
        helpFunction
    fi

    gpu=$(nvidia-smi --list-gpus | wc -l)
    echo "GPUs per host, Assuming same number GPUs in every host - $gpu"
    max=$(( count*gpu ))
    if [ "${max}" -lt "${processes}" ]
    then
        echo "Assuming same number of GPUs setup, number of processes should be < number of host IPs * number of GPUs per host";
        echo "For alternate scenarios, assign processes per IP. Format -  <IP_n>:<NUM_PROCS_for_ip_n>";
        echo "Example: 0.0.0.0:1,0.0.0.1:2";
        helpFunction
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