# OSA-Net : An Optimized Scale Aggregation Network with Attention Mechanism
The official resources for paper:
>A Compact _Litopenaeus Vannamei_ Fry Counting Network with Optimized Scale Aggregation Network [pending]
>ZDM Fasya, AI Gunawan, BSB Dewantara

OSA-Net architecure:
![alt text](https://github.com/FIKARDAVBI/OSA-NET/blob/main/assests/architecture/architecture%20OSA-Net.jpg?raw=true)
OSA-Module:
![alt text](https://github.com/FIKARDAVBI/OSA-NET/blob/main/assests/architecture/OSA-Module.jpg?raw=true)

## Contributions
1. OSA-Net evolved from original Scale Aggregation Network (SANet) that have been channels trimmed as the model is designed for edge devices. (size of parameter: 0.66 MB).
2. OSA-Module adapt from Scale Aggregation Module in SANet with additional squeeze and excitation networks to refine the features (MAE: 1.99 and MSE: 2.69).
3. Dataset PENSLV-1 consist of Litopenaeus Vannamei post larvae aged 5 and 8 days is created for training and testing the model performance compare to other density map regression model.

## Preparation
1. clone this repo
   ```
   git clone https://github.com/FIKARDAVBI/OSA-NET.git
   ```
2. we recommend to use python 3.9.15 and cuda toolkit 12.4
3. install dependencies
   ```
   cd {Repo_ROOT}
   pip install -r requirements.txt
   ```
4. we encourage to use crowd counting framework [C-3-Framework](https://github.com/gjy3035/C-3-Framework) by ghy3035 for training (the repo is not longer maintain)

## Pre-Trained Model
download our pretrained model here [OSA-Net](https://drive.google.com/file/d/1r0-AiytPWX9jgVoTKhuZ3xodpYKG4G-R/view?usp=sharing)

## Dataset
download our PENSLV1 dataset samples [here]()
PENSLV-1 dataset content:
