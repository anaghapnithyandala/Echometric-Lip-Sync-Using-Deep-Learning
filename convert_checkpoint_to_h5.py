import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, SpatialDropout3D, BatchNormalization, TimeDistributed, Flatten
import os

# Path to TensorFlow checkpoint (prefix, without extension)
ckpt_prefix = os.path.join('models', 'ckpt96', 'checkpoint')

# Path to output h5 file
output_h5 = os.path.join('models', 'model_weights.h5')

# Build the model structure (must match exactly)
def build_model():
    model = Sequential()
    model.add(Conv3D(128, 3, input_shape=(75,46,140,1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1,2,2)))
    model.add(Conv3D(256, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1,2,2)))
    model.add(Conv3D(75, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1,2,2)))
    model.add(TimeDistributed(Flatten()))
    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(.5))
    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(.5))
    model.add(Dense(41, kernel_initializer='he_normal', activation='softmax'))
    return model

model = build_model()

# Load weights from TensorFlow checkpoint (old format, must match structure!
model.load_weights(ckpt_prefix)

# Save weights as HDF5
model.save_weights(output_h5)
print(f"Weights saved to {output_h5}")
