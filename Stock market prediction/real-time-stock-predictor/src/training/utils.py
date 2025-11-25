import numpy as np

def add_technical_indicators(data):
    # Example: Adding a simple moving average (SMA) as a technical indicator
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    return data

def create_sequences(data, time_steps=60):
    sequences = []
    labels = []
    for i in range(len(data) - time_steps):
        window = data.iloc[i:i + time_steps]
        sequences.append(window.values)
        labels.append(data['Close'].iloc[i + time_steps])
    return np.array(sequences), np.array(labels)