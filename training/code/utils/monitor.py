
import os,sys
import psutil
import GPUtil
import numpy as np
import time
from datetime import datetime

def monitor(dates, gpu_usages, cpu_usages, mem_usages, output_folder):
    print("MONITORING STARTED")
    sys.stdout.flush()
    while 1:
        cpu_usage = psutil.cpu_percent()
        cpu_usages.append(cpu_usage)
        mem_usage = psutil.virtual_memory().percent
        mem_usages.append(mem_usage)
        #os.path.expanduser('~') + f"/experiments/{output_folder}/{name}"
        with open(os.path.curdir + f"/{output_folder}/monitor_out.txt", 'a') as f:
            f.write('\n')
            timestamp = datetime.now()
            dates.append(timestamp)
            f.write(timestamp.strftime("%Y-%m-%d-%H:%M:%S"))
            f.write('\n')
            f.write(f"CPU {cpu_usage}% ",)
            f.write(f"MEMORY {mem_usage}% ")
            f.write("GPU")
            f.write('\n')
            gpus = GPUtil.getGPUs()
            f.write(' ID  GPU  MEM')
            f.write('\n')
            f.write('--------------')
            f.write('\n')
            for gpu in gpus: 
                gpu_usages[int(gpu.id)].append(gpu.load*100)   
                f.write(' {0:2d} {1:3.0f}% {2:3.0f}%'.format(gpu.id, gpu.load*100, gpu.memoryUtil*100))
                f.write('\n')
            # GPUtil.showUtilization()
        time.sleep(15)
