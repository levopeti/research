import sys
import subprocess
import os
import glob
import pip

if sys.version_info[0] != 3:
    print("Python 3 required. Run with python3 install.py")
    exit(-1)

venv = os.path.expanduser("~/.venv/research")

print(venv)

"""
devnull = open(os.devnull, 'w')
completed_process = subprocess.run(["which", "nvcc"], stdout=devnull, stderr=devnull)
has_gpu = (completed_process.returncode == 0)

print("Installing %s version to %s." % ("GPU" if has_gpu else "CPU", venv))

# create virtual environment if doesn't exist
if not os.path.exists(venv):
    subprocess.run("virtualenv -p python3 %s" % venv, shell=True, check=True)

pip_path = glob.glob(venv + "/**/pip3")[0]
python_path = glob.glob(venv + "/**/python")[0]
activate_this = glob.glob(venv + "/**/activate_this.py")[0]

# activate the environment
# exec (compile(open(activate_this, "rb").read(), activate_this, 'exec'), dict(__file__=activate_this))

# update env variables to use the virtual environment
os.environ.update({'VIRTUAL_ENV': venv, 'PATH': "%s/bin:%s" % (venv, os.environ["PATH"]), '__PYVENV_LAUNCHER__': python_path})

# install packages from pip_freeze.txt. if GPU present, install tensorflow-gpu instead of tensorflow
packages = [x.strip() for x in open("pip_freeze.txt", "r").read().strip().split("\n")]
to_install = []
for package in packages:
    [name, version] = package.split("==")
    if name in ["tensorflow", "tensorflow-gpu"]:
        to_install.append("%s==%s" % ("tensorflow-gpu" if has_gpu else "tensorflow", version))
    else:
        to_install.append(package)

subprocess.run("pip install %s" % " ".join(to_install), shell=True, check=True)
"""
