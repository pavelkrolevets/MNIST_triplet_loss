import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt

embed_train = np.load('./np_embeddings/embeddings_train.npy')
embed_test = np.load('./np_embeddings/embeddings_test.npy')
y_train = np.load('./np_embeddings/labels_train.npy')
y_test = np.load('./np_embeddings/labels_test.npy')

matrix  = np.zeros((embed_train.shape[0]))

"""Get embeddings for test"""
number_test = embed_test[5,:]
number_label = y_test[5]


"""Compute distances"""
for i in range(embed_train.shape[0]):

    dist = np.sqrt(np.sum(np.square(np.subtract(embed_train[i, :], number_test))))
    matrix[i] = dist

"""Sort distances from lowest to largest"""
matrix_pd = pd.DataFrame(matrix, index=y_train).sort_values(by=[0])


"""Show 10 labels with lowest distances, basically its a predicted class"""
print("Closest labels: ", matrix_pd.iloc[0:10], "Real label: ", number_label)

