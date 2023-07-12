import json
import numpy as np
import tensorflow as tf

with open('raw_data/record.json', 'r') as jf:
    data = json.load(jf)

counts = []
letters = []

for k, v in data.items():
    if k.isalnum():
        counts.append(len(v))
        letters.append(k)

counts = np.array(counts)
letters = np.array(letters)
print("Minimum number of samples:", counts.min())
print("Maximum number of samples:", counts.max())
print("Average number of samples:", counts.mean())
print("Total number of samples:", counts.sum())

import visualkeras

model = tf.keras.models.load_model('models_saved/v200bs1024/ocr_model_200e.hdf5')
visualkeras.layered_view(model, to_file='ocr_model_layer_diagram.png', legend=True) # write to disk
