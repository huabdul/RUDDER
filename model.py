import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#%%
def create_model():
    s = layers.Input(shape=(50,1))
    a = layers.Input(shape=(50,1))
    
    inputs = layers.concatenate([s, a], axis=2)
    
    lstm = layers.LSTM(64, return_sequences=True)(inputs)
    # lstm2 = layers.LSTM(16, return_sequences=True)(lstm)
    fc = layers.Dense(1)(lstm)
    
        
    model = keras.Model(inputs=[s, a], outputs=fc)
    
    return model