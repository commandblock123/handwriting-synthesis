import os

os.system("apt update")

os.system("python -m pip install -r requirements.txt --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org")
os.system("apt install python3-tk -y")
