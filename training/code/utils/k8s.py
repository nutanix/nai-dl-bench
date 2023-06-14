import argparse
import sys
import os.path as path
import os
from kubernetes.client import V1PodTemplateSpec
from kubernetes.client import V1ObjectMeta
from kubernetes.client import V1PodSpec
from kubernetes.client import V1Container


from kubeflow.training import TrainingClient
from kubeflow.training.models import KubeflowOrgV1PyTorchJob, KubeflowOrgV1PyTorchJobSpec, V1ReplicaSpec
from kubeflow.training import V1RunPolicy, V1SchedulingPolicy
from kubeflow.training.constants import constants

# Create the argument parser
print("YAML generation script started")
parser = argparse.ArgumentParser(description='Script to generate the yaml.')

# Add arguments
parser.add_argument('--processes', type=int, help='number of processes')
parser.add_argument('-i', '--image', type=str, help='training image name')
parser.add_argument('-c', '--command', type=str, help='training command')
parser.add_argument('--gpu', type=int, help='number of gpus')
parser.add_argument('--cpu', type=str, help='number of cpus in m units')
parser.add_argument('--mem', type=str, help='memory required by the container')
parser.add_argument('--shm', type=str,
                    help='shared memory required by the container')

kubMemUnits = ['Ei', 'Pi', 'Ti', 'Gi', 'Mi', 'Ki']

# Parse the command-line arguments
args = parser.parse_args()

if not any(unit in args.mem for unit in kubMemUnits):
    print("container memory unit has to be one of", kubMemUnits)
    sys.exit(1)

if not any(unit in args.shm for unit in kubMemUnits):
    print("shared memory unit has to be one of", kubMemUnits)
    sys.exit(1)

gpus = args.gpu
cpus = args.cpu
memory = args.mem
shared_memory = args.shm
# Access the parsed arguments
processes = args.processes
fc = args.command.split(' ')
ffc = [e for e in fc if e != '']
parsed_command = []
for i, e in enumerate(ffc):
    if 'data-folder' in e:
        il = ffc[i+1]
        input_pvc = il.split(':')[0]
        input_folder = il.split(':')[1]
        if input_folder[0] != '/':
            input_folder = '/' + input_folder
        input_volume_mount_path = "/workspace" + input_folder
    elif 'output-folder' in e:
        ol = ffc[i+1]
        output_pvc = ol.split(':')[0]
        output_folder = ol.split(':')[1]
        if output_folder[0] != '/':
            output_folder = '/' + output_folder
        output_volume_mount_path = "/workspace" + output_folder
    if ':' not in e:
        parsed_command.append(e)
    else:
        if 'data-folder' in ffc[i-1]:
            parsed_command.append(input_volume_mount_path)
        elif 'output-folder' in ffc[i-1]:
            parsed_command.append(output_volume_mount_path)

volumes_list = [{"name": input_pvc, "persistentVolumeClaim": {"claimName": input_pvc}},
                {"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": shared_memory}}]

volumes_mount_list = [{"name": input_pvc, "mountPath": input_volume_mount_path},
                      {"name": "dshm", "mountPath": "/dev/dshm"}]
if output_pvc != input_pvc:
    volumes_list = volumes_list + \
        [{"name": output_pvc, "persistentVolumeClaim": {"claimName": output_pvc}}]
    volumes_mount_list = volumes_mount_list + \
        [{"name": output_pvc, "mountPath": output_volume_mount_path}]

container = V1Container(
    name="pytorch",
    image=args.image,
    image_pull_policy='Always',
    args=parsed_command,
    resources={"limits": {"cpu": cpus, "memory": memory} if gpus ==
               0 else {"nvidia.com/gpu": gpus, "cpu": cpus, "memory": memory}},
    volume_mounts=volumes_mount_list
)
master = V1ReplicaSpec(
    replicas=1,
    restart_policy="OnFailure",
    template=V1PodTemplateSpec(
        metadata={"labels": {"kind": "pytorchjob"}, "annotations": {
            "test_version": "1", "sidecar.istio.io/inject": "false"}},
        spec=V1PodSpec(
            containers=[container],
            volumes=volumes_list
        )
    )
)

worker = V1ReplicaSpec(
    replicas=processes-1,
    restart_policy="OnFailure",
    template=V1PodTemplateSpec(
        metadata={"labels": {"kind": "pytorchjob"}, "annotations": {
            "test_version": "1", "sidecar.istio.io/inject": "false"}},
        spec=V1PodSpec(
            containers=[container],
            volumes=volumes_list
        )
    )
)

pytorchjob = KubeflowOrgV1PyTorchJob(
    api_version="kubeflow.org/v1",
    kind="PyTorchJob",
    metadata=V1ObjectMeta(name="pytorch-training-job", namespace='default'),
    spec=KubeflowOrgV1PyTorchJobSpec(
        run_policy=V1RunPolicy(
                clean_pod_policy="None",
                scheduling_policy=V1SchedulingPolicy(),
        ),
        pytorch_replica_specs={"Master": master,
                               "Worker": worker}
    )
)


def underscore_to_camelcase(string):
    if string == 'test_version':
        return string
    words = string.split('_')
    return words[0] + ''.join(word.capitalize() for word in words[1:])


def remove_none_values(data):
    if isinstance(data, dict):
        return {
            underscore_to_camelcase(key): remove_none_values(value)
            for key, value in data.items()
            if value is not None
        }
    elif isinstance(data, list):
        return [remove_none_values(item) for item in data if item is not None]
    else:
        return data

def describe_pod(job):
    os.system('kubectl describe pod pytorch-training-job-master-0')
    os.system('kubectl logs pytorch-training-job-master-0 -c pytorch')


jobSpec = pytorchjob.to_dict()
cleanedJobSpec = remove_none_values(jobSpec)


pytorchjob_client = TrainingClient()
pytorchjob_client.create_pytorchjob(pytorchjob, namespace='default')
pytorchjob_client.wait_for_job_conditions(name='pytorch-training-job', namespace='default',
                                          job_kind='PyTorchJob', expected_conditions={'Succeeded'}, polling_interval=30, timeout=1800, callback=describe_pod)
status = pytorchjob_client.is_job_succeeded(
    name='pytorch-training-job', namespace='default', job_kind='PyTorchJob')

pytorchjob_client.get_job_logs(name='pytorch-training-job', namespace='default', container=constants.PYTORCHJOB_CONTAINER, follow=True)

if status != 'Succeeded':
    print("job status is -", status)
    sys.exit(1)
else:
    print("job status is -", status)
