import argparse
import sys
import time
from kubernetes import client, config

# Create the argument parser
parser = argparse.ArgumentParser(description='Script to generate the yaml.')

# Add arguments
parser.add_argument('--image', type=str, help='image name')
parser.add_argument('--command', type=str, help='execution command')
parser.add_argument('--gpu', type=int, help='number of gpus')
parser.add_argument('--cpu', type=int, help='number of cpus')
parser.add_argument('--mem', type=str, help='memory required by the container')

kubMemUnits = ['Ei', 'Pi', 'Ti', 'Gi', 'Mi', 'Ki']

# Parse the command-line arguments
args = parser.parse_args()

if not any(unit in args.mem for unit in kubMemUnits):
    print("container memory unit has to be one of", kubMemUnits)
    sys.exit(1)

gpus = args.gpu
cpus = args.cpu
memory = args.mem
image = args.image
command = args.command.split(' ')
# Access the parsed arguments

config.load_kube_config()
api = client.AppsV1Api()

deployment = client.V1Deployment(
    api_version="apps/v1",
    kind="Deployment",
    metadata=client.V1ObjectMeta(
        name="torchserve"
    ),
    spec=client.V1DeploymentSpec(
        replicas=1,
        selector=client.V1LabelSelector(
            match_labels={
                "app": "torchserve"
            }
        ),
        template=client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(
                labels={
                    "app": "torchserve"
                }
            ),
            spec=client.V1PodSpec(
                containers=[
                    client.V1Container(
                        name="torchserve",
                        image=image,
                        args=command,
                        image_pull_policy="IfNotPresent",
                        ports=[
                            client.V1ContainerPort(
                                name="ts",
                                container_port=8080
                            ),
                            client.V1ContainerPort(
                                name="ts-management",
                                container_port=8081
                            ),
                            client.V1ContainerPort(
                                name="ts-metric",
                                container_port=8082
                            )
                        ],
                        resources=client.V1ResourceRequirements(
                            limits={
                                "cpu": cpus,
                                "memory": memory,
                                "nvidia.com/gpu": gpus
                            }
                        )
                    )
                ]
            )
        )
    )
)

api.create_namespaced_deployment(namespace='default', body=deployment)

time.sleep(10)

deployment_object=api.read_namespaced_deployment('torchserve', 'default')

if deployment_object.status.available_replicas is None:
    print("Kube Deployment Failed")
    sys.exit()
else:
    print("Kube Deployment Active")
