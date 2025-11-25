from src.models.lstm_model import build_lstm
from src.models.transformer_model import build_transformer
from src.training.utils import add_technical_indicators, create_sequences
import pandas as pd
import os
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data, time_steps=60):
    data = add_technical_indicators(data)
    # Drop rows with NaNs introduced by rolling indicators
    data = data.dropna().reset_index(drop=True)
    sequences, labels = create_sequences(data, time_steps=time_steps)
    return sequences, labels

def train_lstm(sequences, labels, input_shape, epochs=50, batch_size=32):
    os.makedirs('models', exist_ok=True)
    model = build_lstm(input_shape)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    model_checkpoint = ModelCheckpoint(os.path.join('models', 'lstm_model.h5'), save_best_only=True)
    model.fit(sequences, labels, validation_split=0.2, epochs=epochs, batch_size=batch_size, callbacks=[early_stopping, model_checkpoint])
    return model

def train_transformer(sequences, labels, input_shape, epochs=50, batch_size=32):
    os.makedirs('models', exist_ok=True)
    model = build_transformer(input_shape)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    model_checkpoint = ModelCheckpoint(os.path.join('models', 'transformer_model.h5'), save_best_only=True)
    model.fit(sequences, labels, validation_split=0.2, epochs=epochs, batch_size=batch_size, callbacks=[early_stopping, model_checkpoint])
    return model

def train(data_file, model_type='lstm', epochs=50, batch_size=32):
    data = load_data(data_file)
    sequences, labels = preprocess_data(data, time_steps=60)
    input_shape = (sequences.shape[1], sequences.shape[2])
    if model_type == 'lstm':
        model = train_lstm(sequences, labels, input_shape, epochs, batch_size)
    elif model_type == 'transformer':
        model = train_transformer(sequences, labels, input_shape, epochs, batch_size)
    else:
        raise ValueError("Invalid model type. Choose 'lstm' or 'transformer'.")
    model.save(os.path.join('models', f'{model_type}_model.h5'))
    return model