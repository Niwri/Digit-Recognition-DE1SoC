# Digit-Recognition-DE1SoC
The software allows the user to draw on a 28x28 canvas and attempts to recognize the digit that the user drew via a Linear Neural Network model. The graphics is rendered with VGA and uses double-buffering to maximize rendering speed. A PS/2 mouse is used to enable interrupts and track the user's mouse positions for button interactions and drawing. The user may choose between "Training" and "Load" at the Start Page. Further information is described in the attached PDF document.

The software is entirely programmed in C. 

The software was programmed in a week and used to submit for the Final Project submission in ECE243.

## Hardware
- DE1-SoC Board
- PS/2 Mouse

## Neural Network
1) Input Layer (784 Nodes, Flattened array of 28x28 pixels)
2) Linear Layer (84 Nodes, Leaky ReLu Activation)
3) Linear Layer (10 Nodes, Softmax Activation)
4) Output (10 Nodes)

Loss Function: Multi-classification Cross Entropy Loss

## Next Steps
The linear model has its short-comings. The main problem is that due to the nature of a linear model, it does not accurately recognize its image but rather finds how similar the drawn digit is to the trained MNIST data in terms of pixel placement in the 28x28 canvas.

To improve on this, we can instead use a Convoluted Neural Network model to better recognize the digit drawn based on features rather than just pixel placements. Due to the short time-frame of 1 week and being the first time to create a neural network, much less in C, we were unable to get time to improve on the model architecture.

## What each file does
The following describes the use-case of each file, and how to edit them if needed
#### model.c
This file contains the neural network code. Under the "Datasets" section, it includes a training set of 3000 examples and test set of 200 examples. To load it onto CPUlator, a smaller dataset is recommended. You may replace it with the files under "smallDataHeaders" directory and define the "NUM_TEST" and "NUM_TRAIN" as 5 and 100 respectively. Note that these datasets are enough to give approximately 75% and 20% test accuracy. 

To train on a local computer, you may include "mnist.h" and remove all the pre-loaded dataset header files and defined NUM_TEST and NUM_TRAIN. In the "main()", you may include "load_mnist()" to load all 60000 training examples and 10000 test examples. However, this has already been done and the weights and biases are saved in "modelData.h" where it is pre-loaded upon clicking the "Load Model" button on the Start Page.

#### graphics.c
This file contains the graphics, user interactions, and PS/2 mouse controls tied to the DE1-SoC board. Nothing needs to be edited here.

#### main.py
This python script helps replace all "include header file" tags with their respective file content in model.c and graphics.c, and compiles them into "combined.c". It is also partly used to translate png files to an RGB array used for DE1-SoC rendering. 

#### modelData.h
This file stores the saved model run in model.c. It does so when the "saveModel()" line is uncommented. However, the "loadSavedModel()" function in main.c loads the model weights and biases sasved here. Note that the weights and biases are "hard-coded" in terms of the model structure.

#### updatedModel.h
This file is the compiled file of model.c and is used to be compiled into graphics.c. Note that the "main()" function from model.c is commented out for this very reason.

#### mnist.h
The majority of this file was written by Takafumi Horiuchi, in which is code was publically shared here: https://github.com/takafumihoriuchi/MNIST_for_C

The file was edited and new functions were added to support the training and development of the Linear Neural Network model, such as dividing the dataset into smaller datasets to be used on the board.


