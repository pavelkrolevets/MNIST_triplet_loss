# Classifying MNIST with Triplet Loss
Simple example of triples loss tested on MNIST dataset.
You can use this pipeline for any images loaded as numpy arrays.
You can find more information about triplet loss in a great paper: [["FaceNet: A Unified Embedding for Face Recognition and Clustering"](http://arxiv.org/abs/1503.03832)](https://arxiv.org/abs/1503.03832)
## Dependencies
```
tensorflow == 1.7
keras
numpy
```
## Default parameters
* ``/model_logs/`` - path to save the trained model and load it. Don`t forget to crate it!
* ``./np_embeddings/`` - path to save 128 vectors from all of the training samples. Don`t forget to crate it!
*  ``training_epochs = 5``
* ``batch_size = 128``
* ``margin = 1.0`` - margin to keep in the triples loss for separation between negatives and positives.
## How to
1. Run `` Mnist_triplet.py`` to train triplet loss and save the model to .
2. Run ``Create_128_vect.py`` to load the model and get 128 vectors.
3. Run ``make_prediction.py`` to compute distances between a test example and all of the train examples. The lowest distance represents similarity!
#### I hope it was helpful! Cheers!

> Written with [StackEdit](https://stackedit.io/).