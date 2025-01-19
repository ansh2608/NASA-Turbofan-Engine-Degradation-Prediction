# NASA Turbofan Engine Degradation Prediction

This project aims to predict the Remaining Useful Life (RUL) of turbofan engines using sensor data from the NASA Turbofan Engine Degradation Dataset. The goal is to build a predictive model that can help in proactive maintenance by predicting when an engine is likely to fail, thus improving operational efficiency and reducing downtime.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Model Training](#model-training)
- [Results](#results)
  - [Actual vs Predicted RUL](#actual-vs-predicted-rul)
  - [Feature Importance Analysis](#feature-importance-analysis)
- [Installation](#installation)
- [License](#license)

## Introduction

The NASA Turbofan Engine Degradation Dataset consists of several sensor readings from turbofan engines under different operational conditions. By predicting the Remaining Useful Life (RUL) of the engines, we can estimate the lifespan of the engine and help to take maintenance actions beforehand, which will save costs and improve system reliability.

In this project, I have implemented a machine learning solution using a **Random Forest Regressor** to predict the RUL of engines.

## Dataset

The dataset includes the following files:

- **train_FD001.txt**: Training data with sensor readings and engine cycles.
- **test_FD001.txt**: Test data with sensor readings and engine cycles.
- **RUL_FD001.txt**: Remaining Useful Life (RUL) for each engine in the test dataset.


## Methodology

The project follows the steps below:

1. **Data Loading**: The dataset is loaded into the environment, including training, test, and RUL data.
2. **Data Preprocessing**: 
   - Columns are renamed for clarity.
   - The Remaining Useful Life (RUL) is computed by subtracting the cycle number from the maximum cycle for each engine.
   - Low-variance sensors are dropped to improve model efficiency.
3. **Data Normalization**: Sensor values are scaled between 0 and 1 using MinMaxScaler to improve model performance.
4. **Model Training**: A Random Forest Regressor is trained on the preprocessed data to predict the RUL of engines.
5. **Evaluation**: The model’s performance is evaluated using metrics such as **Root Mean Squared Error (RMSE)** and **R^2 Score**.

## Model Training

The model used in this project is the **Random Forest Regressor**, which is an ensemble machine learning algorithm. It has the following configuration:

- **n_estimators**: 100 (number of trees in the forest)
- **random_state**: 42 (for reproducibility)

The model is trained on the processed sensor data to predict the RUL of the engines.

## Results

### Actual vs Predicted RUL

The plot below shows a comparison between the actual and predicted Remaining Useful Life (RUL) of engines in the test dataset. The closer the predicted values are to the actual values, the better the model’s performance.

![Actual vs Predicted RUL](https://github.com/ansh2608/NASA-Turbofan-Engine-Degradation-Prediction/blob/933f8668e47842e20f4eb61801a7d9d6041be6ab/Actual%20vs%20Predicted.png)

### Feature Importance Analysis

The plot below shows the importance of each sensor feature in predicting the Remaining Useful Life (RUL). Sensor features with higher importance contribute more to the model’s predictions.

![Feature Importance Analysis](https://github.com/ansh2608/NASA-Turbofan-Engine-Degradation-Prediction/blob/933f8668e47842e20f4eb61801a7d9d6041be6ab/Feature%20Importance%20Analysis.png)

## Installation

To run this project locally, you need to clone the repository and install the required dependencies.

1. Clone the repository:
   ```bash
   git clone https://github.com/ansh2608/NASA-Turbofan-Engine-Degradation-Prediction.git
