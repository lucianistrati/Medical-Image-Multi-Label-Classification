Images are 64 by 64 by 3, but since they are grayscale we can use a 64 by 64 by 3 or 64 by 64 by 1 images for models training.

Each image will have 3 binary labels associated with it.

One quarter of the images are black (4985 out of 15.000). A removal of all black images except for one image was performed.

1. Model convolutional neural network

2. Model recurrent neural network

3. Model fully connected neural network

Efficient Net feature extraction was tried out as well by obtaining a 1000 values embedding (but overall it was not that succesful)

EFFICIENT NET embeddings with SVC over them:
9533 9533 2396 2396
PRE FIT
POST FIT
0.7886666666666667
9533 9533 2396 2396
PRE FIT
POST FIT
0.677547770700637
9533 9533 2396 2396
PRE FIT
POST FIT
0.49814126394052044

requiremenets for documentation:

data analysis 

models description(CNN, RNN), data preprocessing/augmentation, hyperparameterization 
(learning rate, regularization, performance function) document all tried approaches (even unsuccesful ones)

at least 3 pages 

at least two different deep learning models

average precision per class on validation set with various models (tables/figures for these)

python code with explanatory comments

folder named ISTRATI_LUCIAN

ONLY PY FILES 

single zip file in the end