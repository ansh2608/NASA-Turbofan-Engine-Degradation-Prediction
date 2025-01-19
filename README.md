# NASA Turbofan Engine Degradation Prediction

## Overview
This project aims to predict the Remaining Useful Life (RUL) of turbofan engines using machine learning models. The dataset used is the [NASA Turbofan Engine Degradation Dataset](https://data.nasa.gov/dataset/Turbofan-Engine-Degradation-Simulation-Data-Set/ks2z-gb2f), which contains data on sensors from engines and their corresponding failure cycles.

## Objective
The primary objective is to build a machine learning model that predicts the remaining useful life (RUL) of an engine based on sensor data. The steps include data preprocessing, feature engineering, model training, evaluation, and visualization of results.

## Dataset Description
The dataset includes the following files:
- `train_FD001.txt`: Training data containing sensor measurements for different engines.
- `test_FD001.txt`: Test data containing sensor measurements for different engines.
- `RUL_FD001.txt`: Ground truth for the RUL values of the engines.

## Technologies Used
- **Python 3.x**
- **Google Colab** (for running the code)
- **Libraries**:
  - `pandas` for data manipulation
  - `numpy` for numerical operations
  - `matplotlib` and `seaborn` for data visualization
  - `sklearn` for machine learning

## How to Run the Code
1. **Clone or Download the Repository**:
   - If you have the project in a GitHub repository, you can clone it using the following command:
     ```bash
     git clone <repository-url>
     ```
   - Alternatively, you can download the repository as a ZIP file and extract it.

2. **Set Up Google Colab**:
   - Open Google Colab and upload your dataset to Google Drive.
   - Mount your Google Drive using the following code in a Colab cell:
     ```python
     from google.colab import drive
     drive.mount('/content/drive')
     ```

3. **Data Preprocessing**:
   - Load the dataset using pandas and clean the data (drop columns with low variance, rename columns, and normalize sensor data).
   - The following code loads and preprocesses the dataset:
     ```python
     train_data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/train_FD001.txt', sep=' ', header=None).iloc[:, :-1]
     test_data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/test_FD001.txt', sep=' ', header=None).iloc[:, :-1]
     rul_data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/RUL_FD001.txt', header=None)
     ```

4. **Model Training**:
   - Train a Random Forest Regressor to predict the RUL of engines based on sensor data:
     ```python
     model = RandomForestRegressor(n_estimators=100, random_state=42)
     model.fit(X_train, y_train)
     ```

5. **Prediction and Evaluation**:
   - Make predictions using the trained model and evaluate its performance:
     ```python
     y_pred = model.predict(X_test)
     ```

   - Evaluate the model using metrics like RMSE and R^2 score:
     ```python
     rmse = np.sqrt(mean_squared_error(y_test, y_pred))
     r2 = r2_score(y_test, y_pred)
     ```

6. **Visualizations**:
   - Visualize actual vs predicted RUL values:
     ```python
     plt.figure(figsize=(10, 6))
     plt.scatter(y_test, y_pred, alpha=0.7, color='b')
     plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
     plt.xlabel('Actual RUL')
     plt.ylabel('Predicted RUL')
     plt.title('Actual vs Predicted RUL')
     plt.grid(True)
     plt.show()
     ```

   - Visualize feature importance:
     ```python
     feature_importance = model.feature_importances_
     plt.figure(figsize=(12, 6))
     sns.barplot(x=feature_importance, y=X_train.columns)
     plt.title('Feature Importance from Random Forest')
     plt.show()
     ```

## Evaluation Metrics
The model performance is evaluated using the following metrics:
- **Root Mean Squared Error (RMSE)**: Measures the average magnitude of errors in the predictions.
- **R^2 Score**: Indicates how well the model is able to predict the RUL.

## Future Improvements
- Try other models (e.g., XGBoost, LSTM) for potentially better performance.
- Perform hyperparameter tuning using techniques like Grid Search or Random Search.
- Experiment with more feature engineering (e.g., adding rolling statistics, additional features from the sensor data).

## Conclusion
This project demonstrates how machine learning models can be applied to predict the remaining useful life (RUL) of turbofan engines. Predicting RUL can help in predictive maintenance, reducing downtime, and avoiding catastrophic engine failures.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
