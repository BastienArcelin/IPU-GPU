import tensorflow as tf
from tensorflow.python import ipu
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.optimizer_v2.adam import Adam
import numpy as np
import sys

def get_dataset(only_features=False):

    x_train = np.random.normal(0, 1, (100, 1, 10)).astype("float32")
    if only_features:
        ds = tf.data.Dataset.from_tensor_slices((x_train))
    else:
        y_train = np.random.randint(0, 10, (100, 1)).astype("float32")
        ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    
    ds = ds.cache().prefetch(tf.data.experimental.AUTOTUNE)
    return ds

def get_model():
    return ipu.keras.Sequential(
        [Dense(128),
        Dense(10, activation='sigmoid')])

# Configure IPUs
cfg = ipu.utils.create_ipu_config()
cfg = ipu.utils.auto_select_ipus(cfg, 1)
ipu.utils.configure_ipu_system(cfg)

# Set up IPU strategy
strategy = ipu.ipu_strategy.IPUStrategy()

with strategy.scope():
    model = get_model()
    model.compile(loss='categorical_crossentropy', optimizer=Adam(0.005))

    # Train on dataset
    noise_data = get_dataset(only_features=False)
    model.fit(noise_data, steps_per_epoch=20, epochs=4)

    print("weights:", model.layers[0].get_weights()[0])

    # Save weights
    model.save_weights("test")

    # Reload weights
    if (len(sys.argv) > 1 and sys.argv[1] == "true"):
        print("Reloading weights")
        model.load_weights("test")

    # Evaluate
    output = model.evaluate(noise_data)
    
    # Predict on dataset with only features
    noise_data = get_dataset(only_features=True)
    output = model.predict(noise_data)

    print("weights:", model.layers[0].get_weights()[0])
