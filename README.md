# wordprediction: Use neural networks to reconstruct language ancestry

## Installation
This program should be executed with Python 3. Before running it, you should install the dependencies.

If you want to install the dependencies only for this program, create a virtual environment
```
virtualenv env -p python3
source env/bin/activate
```

Regardless of whether you created a virtual environment or not, install the dependencies
```
pip3 install -r requirements.txt
```
Invoke this command as root, if you want to do a system-wide installation of the dependencies (instead of only the current user).

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
