import yfinance as yf
import numpy as np
import plotly.graph_objs as go
import joblib
from keras.api.models import load_model
import datetime as dt
import streamlit as st

def predict_next_price(model, scaler, new_data, sequence_length=60):
    """
    Function to predict the next close price using the trained model.

    Args:
    - model: Trained LSTM model
    - scaler: Scaler used for preprocessing data during training
    - new_data: Latest stock price data
    - sequence_length: Length of input sequence used during training

    Returns:
    - predicted_value: Predicted next close price
    """
    # Ensure the new data is a numpy array
    new_data = np.array(new_data).reshape(-1, 1)

    # Scale the new data using the same scaler used during training
    scaled_new_data = scaler.transform(new_data)

    # Prepare the input sequence
    input_sequence = scaled_new_data[-sequence_length:].reshape(1, sequence_length, 1)

    # Predict the next value
    predicted_scaled_value = model.predict(input_sequence)

    # Denormalize the predicted value
    predicted_value = scaler.inverse_transform(predicted_scaled_value)[0][0]

    return predicted_value

def main():
    """
    Main function to setup Streamlit app and display the last 2 months Nifty closing price and next trading day close(predicted).
    """
    # Set Streamlit app title
    st.title('Nifty 50 Prediction')

    # Fetch historical data for Nifty 50 index (^NSEI)
    nifty = yf.Ticker("^NSEI")
    df = nifty.history(period='max')

    # Get current date and time
    current_time = dt.datetime.now()

    # If current time is less than 15:31, get data from last 61 to yesterday, else get last 60
    if current_time.time() < dt.time(15, 31):
        # Retrieve data from last 61 to yesterday
        new_data = df['Close'][-61:-1].values
    else:
        # Retrieve data from last 60
        new_data = df['Close'][-60:].values

    # Load the trained model
    loaded_model = load_model('model5_n.keras')

    # Load the scaler used during training
    loaded_scaler = joblib.load('scaler_n.pkl')

    # Add a button to trigger model prediction
    if st.button('Predict'):
        # Perform prediction using the model
        next_price = predict_next_price(loaded_model, loaded_scaler, new_data)
        # Display the prediction
        st.write('Nifty 50 Closing Prediction:', round(next_price, 2))
        
        # Determine prediction(note) date based on current time
        if current_time.time() < dt.time(15, 31):
            st.info("Note: The above prediction is for today")
            prediction_date = dt.datetime.today().date()
        else:
            st.info("Note: The above prediction is for next trading session")
            prediction_date = dt.datetime.today().date() + dt.timedelta(days=1)  

        # Plot the stock chart with prediction
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index[-90:], y=df['Close'][-90:], mode='lines', name='Close Price'))
        fig.add_trace(go.Scatter(x=[prediction_date], y=[next_price], mode='markers', name='Prediction', marker=dict(color='red', size=10)))
        fig.add_annotation(x=prediction_date, y=next_price, text=f'Last Close: {df["Close"][-1]:.2f} \nNext Close: {next_price:.2f}', showarrow=True, arrowhead=1)
        fig.update_layout(title=f'Nifty 50 Chart', xaxis_title='Date', yaxis_title='Price')
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
        st.plotly_chart(fig)

if __name__ == "__main__":
    main()