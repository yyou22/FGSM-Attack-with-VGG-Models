# FGSM Attack on CIFAR10 Dataset with VGG Models

For our implementation, we use pretrained PyTorch models by [huyvnphan](https://github.com/huyvnphan/PyTorch_CIFAR10)

## Download Pretrained models
```python
python train.py --download_weights 1
```

## Sample a subset of CIFAR10 dataset
```python
python select_data.py
```
This will download the raw CIFAR10 testing data (10,000 entries) via PyTorch within the `cifar10` directory, and then randomly sample 10 images from each class (100 images in total) and save them as numpy arrays in `../data/`. 

`X.npy` includes all 100 images as normalized matrices, while `Y.npy` includes the ground truth labels as a numpy array.