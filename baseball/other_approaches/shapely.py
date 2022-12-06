import numpy as np
import pickle
from shap import PermutationExplainer
from tensorflow.keras.models import load_model

data_path = '../data/iaat_baseball_training_data.pickle'
data = np.array(pickle.load(open(data_path, 'rb')))
x, y = data[:, 0:-1], data[:, -1]

model = load_model('../data/trained_nn')

explainer = PermutationExplainer(model.predict, x[:50])
shap_values = explainer(x[:5])

print(shap_values.mean(axis=0))


