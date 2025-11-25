# Real-Time Stock Predictor

This project is a Real-Time Stock Predictor that utilizes advanced machine learning techniques to predict stock prices in real-time. The project is structured to provide a clear separation of concerns, making it modular and easy to maintain.

## Purpose

The primary goal of this project is to demonstrate the capabilities of LSTM and Transformer models in predicting stock prices. It serves as an educational demo to showcase how to build a complete machine learning application with real-time capabilities.

## Components

- **Models**: 
  - `lstm_model.py`: Contains the architecture for the LSTM model.
  - `transformer_model.py`: Contains the architecture for the Transformer model.

- **Training**:
  - `pipeline.py`: Implements the training pipeline, including data loading, preprocessing, and model training.
  - `utils.py`: Provides utility functions for data preprocessing and feature engineering.

- **Server**:
  - `main.py`: Initializes the FastAPI application and sets up the necessary routes.
  - `websocket.py`: Implements the WebSocket endpoint for live predictions.

- **Dashboard**:
  - `app.py`: A Streamlit application that provides a user interface for interacting with the model and visualizing predictions.

## Setup

To run this project, you will need to install the required dependencies listed in `requirements.txt`. The project can be easily containerized using Docker, with configurations provided in the `Dockerfile` and `docker-compose.yml`.

## Note

This project is intended for educational purposes and may require further enhancements for production use.