import os

os.system("sudo apt update")
os.system("sudo apt install git")
os.system("git clone https://github.com/commandblock123/handwriting-synthesis.git")

os.system("cd handwriting-synthesis")
os.system("python -m pip install -r requirements.txt")
