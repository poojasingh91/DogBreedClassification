# Precision-Driven Biometric Recognition for Dog Breed Identification

## Overview

This project implements a deep learning pipeline using Convolutional Neural Networks (CNNs) to classify dog breeds from images. It uses Keras with TensorFlow backend and includes 5-fold stratified cross-validation for robust model evaluation. Preprocessing steps include bounding box cropping from annotation XML files, data augmentation, and class balancing.

## Project Structure

```
├── dataset/
│   └── filtered/ImageCropped/             # Image dataset organized by class folders
├── notebooks/
│   └── DogClassification                  # Main training, validation, and evaluation code
├── plot-image/
│   │── train_val_loss.png                 # Plot of training vs validation loss
│   │── train_val_loss.png                 # Plot of training vs validation loss
│   │── train_val_acc.png                  # Plot of training vs validation accuracy
│   │── conf_matrix_all_folds.png          # Aggregated confusion matrix from all folds
│   │── conf_matrix_fold_*.png             # Per-fold confusion matrix images
│   │── class_samples_grid.png             # Visual examples (2 per class)
│   │── train_data.png                     # Histogram of training set distribution
│   │── val_data.png                       # Histogram of validation set distribution
│   └── roc_auc.png                        # Multi-class ROC-AUC curve
└── README.md
```

## Features

Dataset preprocessed using bounding box coordinates from XML annotations

Augmentation using ImageDataGenerator (flip, zoom, shear, etc.)

CNN architecture using pretrained InceptionV3 with custom classification head

Flexible model builder for switching between InceptionV3 and NASNetLarge

5-Fold stratified cross-validation using StratifiedKFold

Evaluation metrics:

Accuracy

Precision

Recall

F1-score

Confusion matrices for each fold + overall

Multi-class ROC-AUC visualization

Publication-quality plots (Times New Roman, clean layout)


## How to Run

Install requirements:

pip install tensorflow matplotlib seaborn scikit-learn pandas pillow

Place your dataset in dataset/filtered/Image/, and XML annotations in dataset/filtered/Annotation/.

Run the notebook:

jupyter lab
# or
jupyter notebook

Output images will be saved in the current working directory.


## Results Summary (5-Fold Cross-Validation)

Metric

Average (%)

Accuracy

99.64

Precision

99.65

Recall

99.63

F1-Score

99.63

Metrics are macro-averaged over 12 dog breed classes.


## Visualizations

Training vs. validation loss and accuracy curves

Class distribution histograms (train/val)

Per-fold and overall confusion matrices

Grid of sample images (2 per class)

Multi-class ROC-AUC curves for all 12 classes

