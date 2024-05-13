import os

os.system("apt update")

os.system("sudo python3.5 -m pip install -r requirements3.5.txt --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org")
os.system("sudo python3.9 -m pip install -r requirements3.9.txt")
os.system("apt install python3-tk -y")
