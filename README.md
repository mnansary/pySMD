# pySMD

**TPU support and memory efficient tfrecord for Small Molecule Drug Generative Models** 

        version:0.0.1
        python:3.7
        conda:4.7.12
        rdkit:2020.03.1.0
        tensorflow:2.1.0

# Qucik Setup
* Install **Anaconda** following instructions from [here](https://docs.anaconda.com/anaconda/install/)
* Create a virtual environment in conda (example: ```conda create -n smdenv python=3.7 anaconda pip```)
* Activate (```conda activate smdenv```)
* Install Rdkit: ```conda install -c rdkit rdkit```
* Install tensorflow-2.1.0: ```pip install tensorflow==2.1.0```
* Install OpenCV : ```pip install opencv-python```
* Add kernel to jupyter: ```python -m ipykernel install --user --name smdenv```

# Smiles Data to tfrecord
**Check data.ipynb for details**

# Example Data Creation
The provided **sample_dataset_c_34_128.smi** in **data** folder contains ~450k unique smiles, with the process described by [mattroconnor](https://github.com/mattroconnor)
The dataset is created from two sources: 
1. [Moses data set](https://github.com/molecularsets/moses) 
2. [ChEMBL data set](https://www.ebi.ac.uk/chembl/). 

**Together these two data sets represented about 4 millions smiles**.The smi file is created after cleaning the smiles  only retaining smiles between 34 to 128 characters in length. 
