Images are 64 by 64 by 3, but since they are grayscale we can use a 64 by 64 by 3 or 64 by 64 by 1 images for models training

Each image will have 3 binary labels associated with it.

At a first close look through the data many images that were totally black were observed, because of this I decided to investigate how many of these black images were there in total. In total, 5000 black images were found. Since there were this many black ones I also searched for other possible duplicates in the dataset between normal images, but none were found. Trainings were done both with those 5000 black images and without them. Also, this quarter of the dataset was labelled with both 
zeroes and ones.

One quarter of the images are black (4985 out of 15.000). A removal of all black images except for one image was performed.

1. Convolutional neural network model. Five architectures were tried out for this type of model as this one proved to yield the best results from the start out of the 3 types of models:
- 1.a 
- 1.b 
- 1.c 
- 1.d 
- 1.e 
2. Recurrent neural network model. Only one architecture was tried out for this type of model.

3. Fully connected neural network model. Two architectures were tried out for this type of model:
- 3.a 
- 3.b 

cnn 2 is also 89% smth, as rnn lstm is 89% smth as well
        
requiremenets for documentation:

data analysis 

Number of epochs choices:
- 5 
- 10

Learning rate choices: 
- 1e-3
- 1e-2
- 1e-1

Optimizer choices:
- Adam
- SGD (Stochastic Gradient Descent)

Loss function choices: 
- Binary Crossentropy
- Categorical Crossentropy

Metrics function choices:
- F1Score
- Accuracy

models description(CNN, RNN), data preprocessing/augmentation, hyperparameterization 
(learning rate, regularization, performance function) document all tried approaches (even unsuccesful ones)

at least 3 pages 

at least two different deep learning models

average precision per class on validation set with various models (tables/figures for these)

python code with explanatory comments

folder named ISTRATI_LUCIAN

ONLY PY FILES 

single zip file in the end

#### 

Your grade will be given only if you provide the code and a technical report for the proposed approaches. The documentation should include:

    The description of your deep learning approaches including the tested models (CNN, RNN, etc.). 
    Details should also include data preprocessing / augmentation (if present) and 
    hyperparameter choices (learning rate, performance function, regularization, etc.).   
    A minimum of 3 pages (excluding tables and figures) is expected. Documenting all the tried 
    approaches (even unsuccessful ones) is important.


Documenting and using at least two different deep learning models is required.
    Average Precision per class on the validation set with various models (use tables / figures to report these).
    The python code of your model should include explanatory comments.
    The *.py files must be placed in a separate folder (from the rest of the documentation) named '{family_name}_{first_name}'.
    Competitors must convert *.ipynb files to *.py (failure to comply will result in disqualification).
    The documentation and the folder containing the *.py files must be archived in a single *.zip file.

Important notes regarding the submitted code and documentation:

    The deadline for submitting the documentation is 07.01.2023 23:59 EET.
    Code and documentation is subject to plagiarism verification and can lead to disqualification.
    Do not include data with your submission (only code and documentation files).
