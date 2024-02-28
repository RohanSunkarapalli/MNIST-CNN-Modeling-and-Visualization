# MNIST CNN Modeling and Visualization

## Overview
The repository contains Python code for MNIST dataset in the Jupyter notebook, and raw data files from the source.

The project aims to create a predictive model to classify handwritten images of size 28x28 pixels between 0-9 digits. We utilized 2d-CNN model architecture and achieved test accuracy of over 99.18 percent. We also created visualization filters and feature maps to understand the weights across the 2d-CNN layers. 

## Instructions

### Libraries
- TensorFlow
- scikit-learn
- NumPy
- pandas
- matplotlib.pyplot
- seaborn

### Files:
- Train images and corresponding labels
    - data/train-images.idx1-ubyte
    - data/train-labels.idx1-ubyte
- Test images and corresponding labels
    - data/t10k-images.idx1-ubyte
    - data/t10k-labels.idx1-ubyte
- 5-fold cross-validation
    - data/train_folds.csv

### Execution
- file: mnist.ipynb
- The jupyter notebook is equipped with necessary libraries needed for smooth run.
- Run all cells in the notebook to process the following in sequential order:
    - Data loading
    - Data preprocessing
    - Create 5-fold cross-validation using scikit-learn's StratifiedKFold module
    - Design and implement model architecture.
    - Loop over the folds to train and validate to investigate optimum hyperparameters. Next, generate performance metrics of the cross-validation.
    - Use the optimum hyperparameters to train the model with all samples in the train dataset and make prediction for the test dataset. Next, generate a report on the train and test accuracy and loss.
    - Visualize filters used in the CNN architecture
    - Visualize features maps of CNN layers and Max-Pooling layers.

## Contributing

Feel free to contribute to the project by opening issues or submitting pull requests. Please follow coding standards and provide clear documentation for any changes.

## License

This project is licensed under the [MIT License] (LICENSE).
