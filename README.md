# Word prediction in historical linguistics
This is a Jupyter notebook and Python library to demonstrate the use of word prediction using deep learning as an aid in historical linguistics.

## Installation
### Linux/Mac

* Chaining search is a Jupyter notebook, which depends on Python 3, pip (PyPi) and venv. Please first install Python 3, pip and development hearders for libxml2, libz and libopenblas via your package management system. E.g. for Ubuntu:
 ```
 sudo apt install python3-pip libxml2-dev libopenblas-dev libz-dev
 ```

* Now, run our install script in a terminal, as a normal user (without `sudo`):
   ```
   ./install.sh
   ```
   If permission is denied, issue the following command once:
   ```
   sudo chmod +x install.sh
   ```
   and then run the install script.

* Every time you want to run the notebook, run the `run.sh` script as a normal user (without `sudo`):
   ```
   ./run.sh
   ```
   A browser window will open. Now, click `Word prediction in historical linguistics.ipynb`. The first time you use it, pick the kernel `ph-env` from menu `Kernel > Change kernel > ph-env`.


### Windows

Chaining search can be easily installed using our install script. This will install all prerequisites for Chaining search.

* Open a command prompt (Windows key + R, then issue "cmd").
* Change to the Chaining search directory (the directory where this README is located):
 ```
 cd CHAINING\SEARCH\DIRECTORY
 ```
* If you don't have Python yet, install it now:
 ```
 python_install.bat
 ```
* Close the command prompt after this (required!)

Now we're ready to install our notebook:
* Open a command prompt (again: Windows key + R, then type "cmd").
* Change to the Chaining search directory (the directory where this README is located): 
* Invoke the install script:
 ```
 install.bat
 ```

Every time you would like to run chaining search, invoke our run script:

* Open a command prompt (Windows key + R, then issue `cmd`).
* Change to the Chaining search directory (the directory where this README is located):
 ```
 cd CHAINING\SEARCH\DIRECTORY
 ```
* Invoke the run script:
 ```
 run.bat
 ```
* A browser window will open. Now, click `Word prediction in historical linguistics.ipynb`. The first time you use it, pick the kernel `ph-env` from menu `Kernel > Change kernel > ph-env`.

