# OSA-Net : An Optimized Scale Aggregation Network with Attention Mechanism
The official resources for paper:
> [pending]
>A Compact _Litopenaeus Vannamei_ post-larvae Fry Counting Network with Optimized Scale Aggregation Network
>ZDM Fasya, AI Gunawan, BSB Dewantara

### OSA-Net architecure:
![alt text](https://github.com/FIKARDAVBI/OSA-NET/blob/main/assests/architecture/architecture%20OSA-Net.jpg?raw=true)
### OSA-Module:
![alt text](https://github.com/FIKARDAVBI/OSA-NET/blob/main/assests/architecture/OSA-Module.jpg?raw=true)

## Contributions
1. A compact network called OSA-Net, designed for edge devices to count Litopenaeus Vannamei post-larvae (size of parameter: 0.66 MB).
2. Optimized Scale Aggregation Module (OSA-Module) with squeeze-and-excitation networks embedding to generate high quality density maps (MAE: 1.99 and MSE: 2.69).
3. PENSLV-1 dataset consists of varied aged and density of Litopenaeus Vannamei post-larvae to train model, and additional PENSLV-2 dataset to test model on unseen data.

> Contact Information : davbi@pasca.student.pens.ac.id or fikardavbi12@gmail.com

## Preparation
1. clone this repo
   ```
   git clone https://github.com/FIKARDAVBI/OSA-NET.git
   ```
2. we recommend to use python 3.9.15 and cuda toolkit 12.4
3. install dependencies
   ```
   cd {Repo_ROOT}
   pip install -r src/requirements.txt
   ```
4. we encourage to use crowd counting framework [C-3-Framework](https://github.com/gjy3035/C-3-Framework) by ghy3035 for training, thanks for their works (use requirement in this repo because C-3-Framework repo is no longer maintained)

## Pre-Trained Model
download our pretrained model here [OSA-Net](https://drive.google.com/file/d/1r0-AiytPWX9jgVoTKhuZ3xodpYKG4G-R/view?usp=sharing)

|Model|Parameter|Dataset|MAE|MSE|
|---|---|---|---|---|
|OSA-Net|660 KB|PENSLV-1|1.99|2.69|

### result:
![alt text](https://github.com/FIKARDAVBI/OSA-NET/blob/main/assests/output/OSA-Net%20result.jpg)

## Dataset
download our Politeknik Elektronika Negeri Surabaya Litopenaeus Vannamei 1 (PENSLV-1) dataset samples [here](https://drive.google.com/file/d/1P0Fh4hCIMQyhZSjrfbFfjcwlZMheVUsP/view?usp=sharing)

PENSLV-1 dataset varied by the litopenaeus vannamei post-larvae(PL) ages (in days) and density:
|PL5 Sparse|PL5 Slightly Dense|PL8 Slightly Dense|
|---|---|---|
|<img src="https://github.com/FIKARDAVBI/OSA-NET/blob/main/assests/dataset/PL5%20Sparse.jpg" width="300">|<img src="https://github.com/FIKARDAVBI/OSA-NET/blob/main/assests/dataset/PL5%20Slightly%20Dense.jpg" width="300">|<img src="https://github.com/FIKARDAVBI/OSA-NET/blob/main/assests/dataset/PL8%20Slightly%20Dense.jpg" width="300">|

PENSLV-1 dataset samples content:
|Class|Image|Class|Image
|---|---|---|---|
|Original|<img src="https://github.com/FIKARDAVBI/OSA-NET/blob/main/assests/dataset/Original.jpg" width="300">|Increase Brightness|<img src="https://github.com/FIKARDAVBI/OSA-NET/blob/main/assests/dataset/Increase%20Brightness.jpg" width="300">|
|Gaussian Blurred|<img src="https://github.com/FIKARDAVBI/OSA-NET/blob/main/assests/dataset/Gaussian%20Blur.jpg" width="300">|Increase Contrast|<img src="https://github.com/FIKARDAVBI/OSA-NET/blob/main/assests/dataset/Increase%20Contrast.jpg" width="300">|
|Gaussian Noise|<img src="https://github.com/FIKARDAVBI/OSA-NET/blob/main/assests/dataset/Gaussian%20Noise.jpg" width="300">|Decrease Brightness|<img src="https://github.com/FIKARDAVBI/OSA-NET/blob/main/assests/dataset/Decrease%20Brightness.jpg" width="300">|

## Citation
If you find this work useful, please cite:
```
publication pending
```
