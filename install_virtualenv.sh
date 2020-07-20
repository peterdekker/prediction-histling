#This guide assumes you have a Python 3 interpreter and pip (pypi) on your system. If not, install from your distribution's package management system.

# We install all required Python packages in a virtual environment, so packages do not interfere with the systemwide installation.
# Install virtualenv locally, for this user
pip3 install --user virtualenv
export PATH=$PATH:~/.local/bin
# Create virtual environment
python3 -m venv ph-env
# Activate virtual environment
source ph-env/bin/activate
# All required packages are instlled via pip in the virtual environment
#First, install some packages separately, they are needed for building the rest
pip3 install wheel cython numpy
pip3 install -r requirements.txt
# Jupyter Notebook extensions are set up
#jupyter contrib nbextension install --sys-prefix
#jupyter nbextensions_configurator enable --sys-prefix
# Collapsible headings extension is enabled
#jupyter nbextension enable collapsible_headings/main
# Remove (possibly) existing Jupyter kernel with name ph-env
rm -rf ~/.local/share/jupyter/kernels/ph-env
# Kernel is configured to work with the virtual environment
python3 -m ipykernel install --user --name ph-env
# Copy .theanorc file, so gpu is used
# Compile documentation
#cd doc
#make html