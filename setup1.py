import os

os.system("sudo apt update")
os.system("sudo apt install docker")

os.system("docker pull tensorflow/tensorflow:1.6.0-gpu-py3")
os.system("docker run -it tensorflow/tensorflow:1.6.0-gpu-py3 bash")
