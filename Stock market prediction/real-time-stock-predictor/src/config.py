# src/config.py

class Config:
    MODEL_PATH_LSTM = "models/lstm_model.h5"
    MODEL_PATH_TRANSFORMER = "models/transformer_model.h5"
    API_KEY = "your_api_key_here"
    WEBSOCKET_URL = "ws://localhost:8000/ws"
    DATA_SOURCE = "your_data_source_here"
    PREDICTION_INTERVAL = 60  # seconds
    MAX_TICKERS = 10  # maximum number of tickers to track
    LOG_LEVEL = "INFO"  # logging level for the application

config = Config()