# Word prediction in historical linguistics
This is a Jupyter notebook and Python library to demonstrate the use of word prediction using deep learning as an aid in historical linguistics. This notebook is based on [master thesis work by Peter Dekker](http://peterdekker.eu/projects/#mscthesis). The results yielded by this demonstrational notebook may differ somewhat from the results in the thesis.

Any questions or problems?
 * [Contact me](https://peterdekker.eu/#contact)
 * [File a bug report](https://github.com/peterdekker/prediction-histling/issues)

## Installation
### Linux/Mac

* This Jupyter notebook depends on Python 3, pip (PyPi) and venv. Please first install Python 3, pip and development hearders for libxml2, libz and libopenblas via your package management system. E.g. for Ubuntu:
 ```
 sudo apt install python3-pip libxml2-dev libopenblas-dev libz-dev
 ```

* Install the Python dependencies systemwide, using the `pip3` package manager as root. First install a number of packages separately:
```
sudo pip3 install wheel cython numpy
sudo pip3 install -r requirements.txt
```

* Every time you want to run the notebook, change to the prediction-histling directory (the directory where this README is located), and run the Jupyter notebook as a normal user:
   ```
   cd PATH/TO/PREDICTION-HISTLING
   jupyter notebook
   ```
   A browser window will open. Now, click `Word prediction in historical linguistics.ipynb`.

### Windows (experimental)
Running this notebook on Windows is not fully tested. If you run into any problems, [file an issue](https://github.com/peterdekker/prediction-histling/issues).

* Open a command prompt (Windows key + R, then issue "cmd").
* Change to the prediction-histling directory (the directory where this README is located):
 ```
 cd PATH\TO\PREDICTION-HISTLING
 ```
* If you don't have Python yet, install it now:
 ```
 python_install.bat
 ```
* Close the command prompt after this (required!)


* Install the Python dependencies using the `pip` package manager. First install a number of packages separately:
```
pip install wheel cython numpy
pip install -r requirements.txt
```

* Every time you want to run the notebook, change to prediction-histling directory, and run the Jupyter notebook as a normal user:
   ```
   cd PATH\TO\PREDICTION-HISTLING
   jupyter notebook
   ```
   A browser window will open. Now, click `Word prediction in historical linguistics.ipynb`.

