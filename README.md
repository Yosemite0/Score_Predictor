# IPL Score Predictor

**Authors:** 
(Yash Chordia - 2024201029)
(Nilay Vatsal - 2024202003)
(Y Kalyan Neeraj - 2024202013)

This project implements a neural network-based model for predicting IPL cricket scores based on per-over match data. The project is organized into three main notebooks:

- **nn2.ipynb** (Training and Eval)– Data processing, model architecture, training, and saving the best model.
- **inference.ipynb** – Loading the trained model and making predictions on new match data based on `./inference/inference.csv`.
- **eval.ipynb** (Running and Basic Eval of Test) – Evaluating the model on test data and visualizing performance metrics.

## Project Structure

- `nn2.ipynb`
  - Sets up the environment and imports necessary libraries.
  - Implements data preprocessing functions (e.g., calculating weighted run rate, scaling numerical features, and encoding teams).
  - Creates Train and Test folder structure for saving processed data.
  - Defines the LSTM-based neural network architecture in the `IPLScorePredictor` class.
  - Uses a custom combined loss function that applies MSE loss for runs and BCE loss for wicket predictions.
  - Trains the model with early stopping and saves the best checkpoint along with preprocessing components.
  - Evaluates the model on the train and test set and visualizes results.

- `inference.ipynb`
  - Loads the saved model, scaler, and team encoder.
  - Reads and preprocesses new match data for inference.
  - Prepares inputs (both as sequences and single instances) and makes predictions.
  - Compares predictions with actual values (if available), saves results to CSV, and visualizes prediction outcomes.

- `eval.ipynb`
  - Loads the trained model along with scaler and encoders.
  - Prepares test data from CSV.
  - Evaluates the model performance by computing metrics such as Run MAE and Wicket Prediction Accuracy.
  - Provides visualizations including scatter plots comparing actual versus predicted runs and confusion matrices for wicket predictions.

## Data Processing

- **Weighted Run Rate Calculation:**  
  Functions in the notebooks calculate an exponentially weighted run rate for each over to provide a smoothed estimate.
  
- **Feature Scaling and Encoding:**  
  Numerical features are scaled using a `StandardScaler`, and team names are encoded to numerical values using `LabelEncoder`.

- **Dataset Splitting:**  
  The data is split by match IDs to ensure sequential integrity. Train and test datasets are saved for reproducibility.

## Model Architecture

- **LSTM-based Predictor:**  
  The `IPLScorePredictor` class defines the model architecture consisting of:
  - LSTM layers to capture sequential dependencies.
  - Fully connected layers with dropout and ReLU activation.
  - The model outputs a 2-dimensional prediction vector:  
    • The first element represents runs (a continuous value).  
    • The second element is a logit used to predict wicket occurrence (binary classification).

## Training and Loss Function

- **Custom Combined Loss Function:**  
  Combines Mean Squared Error (MSE) loss for runs with Binary Cross Entropy (BCE) loss for wicket predictions.  
- **Early Stopping:**  
  The training loop monitors validation loss and stops training if there is no improvement for a specified number of epochs.
- **Logging:**  
  Detailed logging is used during training to track progress and debug.

## Inference

- **Preprocessing New Data:**  
  Before inference, input data is preprocessed (teams encoded, features scaled) using the saved scaler and encoder.
- **Prediction Pipeline:**  
  The inference notebook groups data by match and inning (for sequence mode) or processes individual overs to generate predictions.
- **Output and Visualization:**  
  Results, including predicted runs and wicket probabilities, are saved to CSV and plotted to facilitate comparison with actual match outcomes.

## Evaluation

- **Performance Metrics:**  
  The evaluation notebook computes:
  - Mean Absolute Error (MAE) for run predictions.
  - Accuracy of wicket predictions.
- **Visualization:**  
  Results are visualized via scatter plots (predicted vs actual runs) and confusion matrices for wicket predictions.

## Setup and Dependencies

Ensure you have the following libraries installed:
- Python 3.x
- torch
- numpy
- pandas
- matplotlib
- scikit-learn

You can install the dependencies with:
```bash
pip install torch numpy pandas matplotlib scikit-learn
```

## How to Run

1. **Training:**  
   Open and run `nn2.ipynb` to process the data, train the model, and save the model checkpoint along with preprocessing components.

2. **Inference:**  
   Open and run `inference.ipynb` to load the trained model and make predictions on new input data.

3. **Evaluation:**  
   Open and run `eval.ipynb` to assess the model on test data, calculate performance metrics, and generate evaluation plots.
