# MutiPep-DLCL
MultiPep-DLCL:Recognition of Multi-functional Therapeutic Peptides through Deep Learning with Label-Sequence Contrastive Learning


## Introduction
In this work, the MultiPep-DLCL model is proposed to predict MFTP. This work has the following advantages over existing methods:  
(1) Learning sequence features from both global and local features.<br />
(2) Introducing label semantic information and using Transformer to establish the correlation between local features of sequences, global features and label embeddings, and obtaining sequence-related label embeddings.<br />(3) By introducing convenient label-sequence contrastive learning to further guide the related sequence and label feature expressions closer together.<br />
(4) Combining MLFDL, which deals with the dataset imbalance problem, with CEL to train the model, resulting in further improvement of the model performance.<br />

## Related Files

#### MutiPep-DLCL

| FILE NAME         | DESCRIPTION                                            |
|:------------------|:-------------------------------------------------------|
| main.py           | the main file of MutiPep-DLCL               |
| train.py          | train and predict model                                |
| models            | model construction                                     |
| DataLoad.py       | data reading and encoding                              |
| loss_functions.py | loss functions used to train models                    |
| evaluation.py     | evaluation metrics (for evaluating prediction results) |
| dataset           | data:text.txt is for test set,train.txt is train set   |
| result            | Models and results preserved during training         |
| saved_models      | Trained weights of our MutiPep-DLCL model              |
| config            | Some of the defined model parameters                   |
| demo.ipynb        | A demonstration file that can load the trained parameters of the MutiPep-DLCL model  |



## Installation

### Requirements

#### Operating System:
- Windows: Windows 10 or later
- Linux: Ubuntu 16.04 LTS or later

#### GPU:
- NVIDIA GeForce RTX 3050 Ti GPU

#### Python Libraries:
Ensure your environment is compatible with the following Python library versions:
- PyTorch: 1.12.1 (Python 3.9, CUDA 11.6)
- NumPy: 1.26.2
- Pandas: 1.2.4

### Download MutiPep-DLCL
```bash
git clone https://github.com/xialab-ahu/MultiPep-DLCL
  ```
## Training and Testing MutiPep-DLCL model
### Training
After configuring an identical training environment to ours, including the GPU, operating system, and Python libraries, you can proceed with training our model as follows:
```shell
cd "./MutiPep-DLCL"
python main.py
```

### Testing
We have saved the trained model parameters in the file `savemodels/model.pth`. If you are unable to fully replicate our training environment during your training process, you can separately download the pre-trained parameters provided by us from the `savemodels/model.pth` in this GitHub project and use our model directly. Additionally, we have provided a Jupyter Notebook file named `demo.ipynb`, which includes a demonstration of the process of loading our model weights and making predictions, as well as showcasing the prediction results of our model.

## Contact
Please feel free to contact us if you need any help.

