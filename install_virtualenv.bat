

echo Install the Python package manager

python -m pip install -U pip /quiet



echo Create the virtual environment

pip install --user virtualenv

virtualenv ph-env



echo Activate the virtual environment

call .\ph-env\Scripts\activate.bat


echo Install dependencies
pip install wheel cython numpy
ipython kernel install --user --name=ph-env
pip install -r requirements.txt
python -m ipykernel install --user --name ph-env  /quiet



echo Done. Type 'run' to start your jupyter notebook
