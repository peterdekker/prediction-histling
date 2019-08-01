# Word prediction in historical linguistics
This is a Jupyter notebook and Python library to demonstrate the use of word prediction using deep learning as an aid in historical linguistics.

## Installation
### Linux/Mac

* Chaining search is a Jupyter notebook, which depends on Python 3, pip (PyPi) and venv. Please first install Python 3, pip and libxml2 development hearders via your package management system. E.g. for Ubuntu:
 ```
 sudo apt install python3-pip python3-venv libxml2-dev
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
   A browser window will open. Now, click `Sandbox.ipynb`. The first time you use it, pick the kernel `env` from menu `Kernel > Change kernel > env`.


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
* A browser window will open. Now, click `Sandbox.ipynb`. The first time you use it, pick the kernel `env` from menu `Kernel > Change kernel > env`.

## General workflow
Execute python3 pipeline.py

Modes:
-prediction: dat wil je bijna altijd. Achteraf wordt er een pickle opgeslagen met precies de namen van de argumenten, behalve de modus. Dus als je die pickle wilt gebruiken, roep je het programma in een andere modus aan met precies dezelfde argumenten

Prediction zonder verdere argumenten: encoder-decoder
Prediction met --seq: structured perceptron
Prediction met --phyl: phylogenetic word prediction, phylogenetic encoder-decoder

Corpus instellen:
corpus train
corpus valtest

Output naar dist.*options*.txt file, wordt geappend.


Workflow:
 - Input: data/northeuralex-0.9-lingpy.tsv
 - Convert to tsv, change encoding (ASJP or IPA) and tokenize. Output to northeuralex-asjp.tsv. Als deze al bestaat, gebruik bestaand bestand.
 Kolom IPA wordt vervangen door woordvorm in gewenste formaat (meestal ASJP).
 Kolom tokens wordt toegevoegd met losse tokens.
 
## Different tasks
The application can be invoked with different arguments, for different tasks. For most tasks, the prediction tasks serves as a basis and must be performed first.
 --cluster: Build tree. This can later be compared to ground truth with qdist command line tool
 --visualize: Show table substitutions of sounds during prediction
 --visualize_weights: Show plot of context vectors, and input/target words for comparison
 First, you have to do a prediction run with the same model options with the --export_weights flag on
 --visualize_encoding: Show plot of encoding matrix: to compare phonetic feature matrix and learned embedding encoding
 --cognate_detection
 
 All model option flags are off by default. When an option is on by default, the flag has been formulated negatively.
 
## Phylogenetic word prediction
 Phylogenetic word prediction: python3 pipeline.py --prediction --phyl --languages nld deu eng
 A tree is assumed, where the first two languages are closely connected, and the third language is more remotely connected on a higher level. By changing the order of the languages on the command line, different trees for the three languages are evaluated.
