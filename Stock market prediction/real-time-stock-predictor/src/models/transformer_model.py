from tensorflow import keras
from tensorflow.keras import layers

def build_transformer(input_shape, num_heads=8, ff_dim=32, dropout_rate=0.1):
    inputs = layers.Input(shape=input_shape)
    
    # Transformer block
    x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=input_shape[-1])(inputs, inputs)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(inputs + x)
    
    x_ff = layers.Dense(ff_dim, activation='relu')(x)
    x_ff = layers.Dense(input_shape[-1])(x_ff)
    x = layers.Dropout(dropout_rate)(x_ff)
    x = layers.LayerNormalization(epsilon=1e-6)(x + x_ff)
    
    # Output layer
    outputs = layers.Dense(1)(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model