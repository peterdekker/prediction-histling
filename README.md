# Word Prediction in Computational Historical Linguistics
This is a Jupyter notebook and Python library to demonstrate the use of word prediction using deep learning as an aid in historical linguistics. This notebook accompanies the following paper: [Dekker, P., & Zuidema, W. (2021). Word Prediction in Computational Historical Linguistics. Journal of Language Modelling, 8(2), 295–336.](https://doi.org/10.15398/jlm.v8i2.268) The results yielded by this demonstrational notebook may differ somewhat from the results in the article.

Any questions or problems?
 * [File a bug report](https://github.com/peterdekker/prediction-histling/issues)
 * [Contact us](https://peterdekker.eu/#contact)

## Installation
### Linux/Mac

* Please first install Python 3, pip, Python venv and development hearders for libxml2, libz and libopenblas via your package management system. For GPU support, also install pygpu and headers for libgpuarray. E.g. for Ubuntu:
 ```
 sudo apt install python3-pip python3-venv libxml2-dev libopenblas-dev libz-dev python3-pygpu libgpuarray-dev
 ```
* Open a terminal and move to the directory where this README is located.
* Now, run the install script, as a normal user (without `sudo`):
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
   A browser window will open. Now, click the notebook: `Word prediction in computational historical linguistics.ipynb`. The first time you use it, pick the kernel `ph-env` from menu `Kernel > Change kernel > env`.


### Windows (experimental)
Running this notebook on Windows is not fully tested. If you run into any problems, [file an issue](https://github.com/peterdekker/prediction-histling/issues).

 * Open a command prompt (Windows key + R, then issue "cmd").
 * Change the directory where this README is located:
 ```
 cd PREDICTION-HISTLING\DIRECTORY
 ```
 * If you don't have Python yet, install it now:
 ```
 python_install.bat
 ```
 * Close the command prompt after this (required!)

Now we're ready to install our notebook:
 * Open a command prompt (again: Windows key + R, then type "cmd").
 * Change to the directory where this README is located: 
 * Invoke the install script:
 ```
 install.bat
 ```

Every time you would like to run the notebook, invoke our run script:
 * Open a command prompt (Windows key + R, then issue `cmd`).
 * Change to the directory
 * Invoke the run script:
 ```
 run.bat
 ```
 * A browser window will open. Now, click the notebook: `Word prediction in computational historical linguistics.ipynb`. The first time you use it, pick the kernel `ph-env` from menu `Kernel > Change kernel > env`.

 Thanks to Mathieu Fannee for describing the steps of running Python virtual environments on Windows.

