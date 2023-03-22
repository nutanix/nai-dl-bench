import requests
import os
import shutil
import glob
from pathlib import Path

def rm_file(path, regex=False):
    if regex:
        file_list = glob.glob(path, recursive=True)
    else:
        file_list = [path]
    for file in file_list:
        path = Path(file)
        if os.path.exists(path):
            print(f"Removing file : {path}")
            os.remove(path)


def rm_dir(path):
    path = Path(path)
    if os.path.exists(path):
        print(f"Deleting directory : {path}")
        shutil.rmtree(path)
