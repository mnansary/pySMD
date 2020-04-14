# pySMD

**TPU support and memory efficient tfrecord for Small Molecule Drug Generative Models** 

        version:0.0.2
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

# Example Data Creation
The provided **sample_dataset_c_34_128.smi** in **data** folder contains ~450k unique smiles, with the process described by [mattroconnor](https://github.com/mattroconnor)
The dataset is created from two sources: 
1. [Moses data set](https://github.com/molecularsets/moses) 
2. [ChEMBL data set](https://www.ebi.ac.uk/chembl/). 

**Together these two data sets represented about 4 millions smiles**.The smi file is created after cleaning the smiles  only retaining smiles between 34 to 128 characters in length. 

**Check data.ipynb for example of tfrecord generation**

# Training For Molecule Generation
* After Data Creation is completed Upload the tfrecord folder to  **GCS BUCKET**
* Make Sure that you have permission to access the data by setting the **storage** permission properly in Google Cloud Bucket
* Authenticate yourself from **colab** for data access
> If the Bucket is made public then no auth is needed
* Git Clone the repo to your **google drive**
* Change the working directory accordingly in **train_xxxx.ipynb**
* Train and generate the **smi** data in **colab** (Faster Generation) 
