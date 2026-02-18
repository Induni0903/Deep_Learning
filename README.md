# Flower Species Classification: CNN Architectures & Transfer Learning
## Deep Learning | | Master’s Degree Project (Google Colab)
This repository contains an end-to-end computer vision pipeline developed to classify five species of flowers. The project explores the evolution of model performance through architectural changes (Batch Normalization, Dropout) and concludes with a benchmark against ResNet50.

## Project OverviewDataset: 3,670 RGB images across five flower classes.
* Input Pipeline: Images resized to $227 \times 227$ pixels and processed in batches of 32.
* Data Augmentation: Implemented to reduce overfitting using:Random Crop ($224 \times 224$)Horizontal FlipRandom Contrast (0.25)Random Zoom (0.3)

## Data Splitting 
StrategyTo ensure robust evaluation, the dataset was partitioned with a fixed seed for reproducibility:

* Training (80%): Used for parameter optimization.
* Validation (13%): Used for hyperparameter tuning.
* Test (7%): Held-out set for final performance reporting.

## Model Architectures
Developed four custom CNN models with increasing complexity, culminating in a transfer learning comparison:

* Custom CNN (Final Iteration - Model 4)
* Feature Extraction: Three Conv2D layers (16, 32, 64 filters) with ReLU activation.
* Regularization: Batch Normalization on all layers and Dropout (0.3) to improve generalization.
* Pooling: MaxPooling2D ($3 \times 3$, strides=3) and GlobalMaxPooling2D for spatial dimensionality reduction.
* Optimization: RMSprop (Learning Rate: 0.001) with a Softmax output layer.
