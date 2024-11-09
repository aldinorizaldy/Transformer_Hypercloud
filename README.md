# Transformer_Hypercloud

We used a Point Cloud Transformer (PCT) network as the backbone and tested different Self-Attention modules for 3D Hypercloud data in Geological application.

<img width="450" alt="Screenshot 2024-11-09 at 02 19 08" src="https://github.com/user-attachments/assets/327e3cc1-a4dd-4590-8520-6ed5b16140f2">

<img width="750" alt="Screenshot 2024-11-09 at 02 19 37" src="https://github.com/user-attachments/assets/885cb7b8-6a44-47ac-997b-cd5aee682cd1">

Download Tinto data from here: https://rodare.hzdr.de/record/2256

First prepare KNN points:
```
python find_kNN_for_training.py
python find_kNN_for_testing.py
```
Then train and test all models:
```
Train_and_Test.sh
```
Cite the paper here:
> A. Rizaldy, A. J. Afifi, P. Ghamisi and R. Gloaguen, "Transformer-Based Models for Hyperspectral Point Clouds Segmentation," 2023 13th Workshop on Hyperspectral Imaging and Signal Processing: Evolution in Remote Sensing (WHISPERS), Athens, Greece, 2023, pp. 1-5, doi: 10.1109/WHISPERS61460.2023.10431346. keywords: {Point cloud compression;Geology;Benchmark testing;Signal processing;Transformers;Noise measurement;Hyperspectral imaging;Point cloud;Hyperspectral;Transformer;Attention;Classification},
