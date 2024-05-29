# Nifty 50 Stock Price Prediction

This project aims to predict the closing prices of stocks in the Nifty 50 index using a Long Short-Term Memory (LSTM) neural network model. The predictions are visualized through a user-friendly web application built with Streamlit.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Prediction](#prediction)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Stock price prediction is a challenging task due to the volatile nature of the stock market. This project leverages the capabilities of LSTM neural networks to forecast future stock prices based on historical data. The web application provides an interactive way to visualize past trends and the predicted closing prices.

## Features
- **Data Retrieval**: Fetches historical stock price data from Yahoo Finance.
- **Data Preprocessing**: Normalizes the data using MinMaxScaler for better model performance.
- **LSTM Model**: Utilizes a deep learning LSTM model to predict future stock prices.
- **Interactive Visualization**: Displays stock price trends and predictions using Plotly and Streamlit.
- **Real-time Prediction**: Provides real-time stock price predictions based on the latest available data.

## Requirements
- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- yfinance
- scikit-learn
- tensorflow
- keras
- joblib
- streamlit
- plotly

## Installation
1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/nifty-prediction.git
   cd nifty-prediction
   ```

2. Create a virtual environment:

    ``` bash
    python -m venv venv
    venv\Scripts\activate
    ```

3. Install the required packages:
    ``` bash
        pip install -r requirements.txt
    ```

## Usage

1. Train the Model:

    - Run the training script to train the LSTM model on historical data:

    ``` bash
      python train_model.py
    ```

2. Run the Streamlit Application:

    - Start the Streamlit application to visualize and predict stock prices:

    ``` bash
        streamlit run nifty_predictor.py
    ```

3. Access the Application:
    - Open your web browser and navigate to http://localhost:8501 to interact with the application.

## Model Training

The model is trained using historical stock price data. The LSTM network is designed to capture temporal dependencies in the data. The training script includes data preprocessing, model architecture definition, and training routines.

## Prediction

The Streamlit application allows users to predict future stock prices based on the latest available data. The application displays the historical trend, predicted values, and highlights the prediction points on the chart.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License.
