import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# import seaborn as sns
# from tqdm import tqdm_notebook as tqdm
# from keras.models import Model
# from keras.layers import Input, Reshape
# from keras.layers.core import Dense, Activation, Dropout, Flatten
# from keras.layers.normalization import BatchNormalization
# from keras.layers.convolutional import UpSampling1D, Conv1D
# from keras.layers.advanced_activations import LeakyReLU
# from keras.optimizers import Adam, SGD
# from keras.callbacks import TensorBoard

def sample_data(n_samples=10000, x_vals=np.arange(0, 5, .1), max_offset=100, mul_range=[1, 2]):
    vectors = []
    for i in range(n_samples):
        offset = np.random.random() * max_offset
        mul = mul_range[0] + np.random.random() * (mul_range[1] - mul_range[0])
        vectors.append(
            np.sin(offset + x_vals * mul) / 2 + .5
        )
    return np.array(vectors)


ax = pd.DataFrame(np.transpose(sample_data(5))).plot()
