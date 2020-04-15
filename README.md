# pySMD

**TPU support and memory efficient tfrecord for Small Molecule Drug Generative Models** 

        version:0.0.2
        python:3.7
        conda:4.7.12
        rdkit:2020.03.1.0
        tensorflow:2.1.0
        pymol:2.3.5 

# Qucik Setup
* Install **Anaconda** following instructions from [here](https://docs.anaconda.com/anaconda/install/)
* Create a virtual environment in conda (example: ```conda create -n smdenv python=3.7 anaconda pip```)
* Activate (```conda activate smdenv```)
* Install Rdkit: ```conda install -c rdkit rdkit```
* Install tensorflow-2.1.0: ```pip install tensorflow==2.1.0```
* Install OpenCV : ```pip install opencv-python```
* Install pymol: ```conda install -c schrodinger pymol```
* Install AutoDock-Vina: ```sudo apt install autodock-vina```
* Install mgltools : ```sudo apt install mgltools-pmv```
* Add kernel to jupyter: ```python -m ipykernel install --user --name smdenv```

**USED ENVIRONMENT**  

    OS          : Ubuntu 18.04.3 LTS (64-bit) Bionic Beaver        
    Memory      : 7.7 GiB  
    Processor   : Intel® Core™ i5-8250U CPU @ 1.60GHz × 8    
    Graphics    : Intel® UHD Graphics 620 (Kabylake GT2)  
    Gnome       : 3.28.2  

# Data
## Example Data Creation
The provided **sample_dataset_c_34_128.smi** in **data** folder contains ~450k unique smiles(source:0)
The dataset is created from two sources: 
1. [Moses data set](https://github.com/molecularsets/moses) 
2. [ChEMBL data set](https://www.ebi.ac.uk/chembl/). 

**Together these two data sets represented about 4 millions smiles**.The smi file is created after cleaning the smiles  only retaining smiles between 34 to 128 characters in length. 

## Zinc-15 DataSet [in progress]
* [A short Intoduction from Youtube](https://www.youtube.com/watch?v=TVf5eCO4p8Q)
* Zinc-15 database: [ZINC 15 – Ligand Discovery for Everyone](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4658288/)
* Download zinc-15 DataSet from [here](https://zinc15.docking.org/tranches/home/)
* An example download config is as follows:

![](/src_img/z1.png)

![](/src_img/z2.png)

* use **moretools** or **parallal** for the  **.wget** file 
* Alternatively you can use **data/zinc15.sh** with **execution permission** *(chmod +x zinc15.sh)*
> After Download **format smi to tfrecords** 
> Finding relevant compounds with activities

# Training with TPU(Tensor Processing Unit)
![](//src_img/tpu.ico?raw=true)*TPU’s have been recently added to the Google Colab portfolio making it even more attractive for quick-and-dirty machine learning projects when your own local processing units are just not fast enough. While the **Tesla K80** available in Google Colab delivers respectable **1.87 TFlops** and has **12GB RAM**, the **TPUv2** available from within Google Colab comes with a whopping **180 TFlops**, give or take. It also comes with **64 GB** High Bandwidth Memory **(HBM)**.*
[Visit This For More Info](https://medium.com/@jannik.zuern/using-a-tpu-in-google-colab-54257328d7da)  

# Models
## LSTM_Chem
* Based ON:[Generative Recurrent Networks for De Novo Drug Design](https://onlinelibrary.wiley.com/doi/full/10.1002/minf.201700111)
* [github](https://github.com/topazape/LSTM_Chem) 
* **data_lstm_chem.ipynb**: smi data to memory efficient tfrecords creation
* **train_lstm_chem.ipynb**:  *upload the **weights** folder and **train_lstm_chem.ipynb** in the same directory in **google drive** to train the model*
* **gen_lstm_chem.ipynb**: generate news molecules with **valid chemical structre** and saves them for **docking**

### IMPROVEMENTS TO ORIGINAL IMPLEMENTATION
* trains the model **~15** times faster
* efficient storing of **tfrecords**
### Notes
* see **.ipynb** markdowns for specific instructions

## CGVA [in progress]
* Based ON:[Constrained Graph Variational Autoencoders for Molecule Design](https://arxiv.org/abs/1805.09076)
## MT-DTI [in progress]
* Based ON: [Predicting commercially available antiviral drugs that may act on the novel coronavirus (SARS-CoV-2) through a drug-target interaction deep learning model](https://www.sciencedirect.com/science/article/pii/S2001037020300490)
> BERT-Transformer

## Bioactivity Prediction - Edge Memory Neural Network [in progress]
* Based ON:[Building Attention and Edge Convolution Neural Networks for Bioactivity and Physical-Chemical Property Prediction](https://chemrxiv.org/articles/Building_Attention_and_Edge_Convolution_Neural_Networks_for_Bioactivity_and_Physical-Chemical_Property_Prediction/9873599)

# nCoV Section (Sampled from various sources)
* The genomes of several isolates of the virus are available, it is a ~30kB genome and can be found [here](https://www.ncbi.nlm.nih.gov/nuccore/NC_045512). Basic Local Alignment Search Tool (BLAST) results show close homology to the bat Coronavirus.(source:1)
* A crystal structure of the main protease of the virus can be found [here](https://www.rcsb.org/structure/6LU7). The structure is complexed with a ligand called N3, which serves as an excellent starting point for new drug candidate investigations.(source:1)

# Sources and Acknowledgements [To be Given with proper and clear mentions]
* [source:0]((https://github.com/mattroconnor))
* [source:1](https://github.com/tmacdou4/2019-nCov/blob/master/Report_Thomas_MacDougall.ipynb)
