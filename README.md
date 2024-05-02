# Language Classification with MFCCs

## Overview

This project demonstrates language classification using Mel-Frequency Cepstral Coefficients (MFCCs) and a neural network model. The model is trained to classify spoken languages based on audio features extracted using MFCCs.

## Dataset

The dataset used for this project is stored in the `processed_data.csv` file. It includes audio features (MFCCs) and corresponding language labels.
The original data/audios is taken from Kaggle : https://www.kaggle.com/datasets/hbchaitanyabharadwaj/audio-dataset-with-10-indian-languages

## Prerequisites

Ensure you have the following dependencies installed:

- Python 3.x
- pandas
- scikit-learn
- keras
- numpy
- Tensorflow
- pickle
- uvicorn
- fastapi
- typing 
- pydantic


Drive link for datasets : https://drive.google.com/drive/u/0/folders/1Aejhj03uv2XZUfwkrXfuJa6CuE1Tzy4e


# Model Details
The neural network model consists of three layers:

Input layer: Dense layer with ReLU activation
Hidden layer: Dropout layer with 50% dropout rate
Output layer: Dense layer with softmax activation (for multiclass classification)
The model is trained using the categorical crossentropy loss function and the Adam optimizer.

# Results
After training for 20 epochs, the model achieved an accuracy of approximately 88.33% on the test set.
Successfully, predicts the language using input features.