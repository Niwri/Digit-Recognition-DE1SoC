#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define NUM_TEST 5
#define NUM_TRAIN 100

#define SIZE 784

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

/************************************************************************************
*                                                                                   *
*   Datasets                                                                        *
*                                                                                   *
*************************************************************************************/

#include "smallDataHeaders/test_image.h"
#include "smallDataHeaders/test_label.h"
#include "smallDataHeaders/train_image.h"
#include "smallDataHeaders/train_label.h"


/************************************************************************************
*                                                                                   *
*   Structures                                                                      *
*                                                                                   *
*************************************************************************************/

typedef double* (*activation_func_vector_ptr)(double*, int);
typedef double (*activation_func_ptr)(double);
typedef double (*gradient_func_ptr)(double);
typedef double (*gradient_loss_func_ptr)(double, double);
typedef double* (*weight_func_ptr)(int);


typedef enum  {
    LINEAR = 0
} Layers;

typedef enum {
    RELU = 0,
    LEAK_RELU = 1,
    SOFTMAX = 2
} Activations;

typedef struct {
    int numOfLayers;
    int* numOfNodes;
    Layers* layerType;
    double*** weights;
    double** bias;
    activation_func_vector_ptr* activation_funcs;
    gradient_func_ptr* gradient_funcs;
    gradient_loss_func_ptr gradient_loss_func;
} Model;


/************************************************************************************
*                                                                                   *
*   HELPER FUNCTIONS                                                                *
*                                                                                   *
*************************************************************************************/

void freeDoublePointers(void** pointerToFree, int size) {
    for(int i = 0; i < size; i++)
        free(pointerToFree[i]);
    
    free(pointerToFree);
}

void freeTriplePointers(void*** pointerToFree, int sizeFirst, int *sizeSecond) {
    for(int i = 0; i < sizeFirst; i++) {
        for(int j = 0; j < sizeSecond[i]; j++) 
            free(pointerToFree[i][j]);
        free(pointerToFree[i]);
    }    

    free(pointerToFree);
}


/************************************************************************************
*                                                                                   *
*   Model Methods                                                                   *
*                                                                                   *
*************************************************************************************/

void clearModel(Model* model) {
    free(model->activation_funcs);
    free(model->gradient_funcs);
    free(model->layerType);
    freeDoublePointers((void**)(model->bias), model->numOfLayers);
    freeTriplePointers((void***)(model->weights), model->numOfLayers, model->numOfNodes);
    free(model->numOfNodes);
}


/************************************************************************************
*                                                                                   *
*   ERROR FUNCTIONS                                                                 *
*                                                                                   *
*************************************************************************************/

double crossEntropy(double* predictedProbabilities, double* targetProbabilities, int numOfClasses) {
    double loss = 0;

    for(int i = 0; i < numOfClasses; i++) 
        loss -= targetProbabilities[i] * log(predictedProbabilities[i]);

    return loss;
}

/************************************************************************************
*                                                                                   *
*   ACTIVATION FUNCTIONS VECTOR FORMAT (ReLu, Softmax)                              *
*                                                                                   *
*************************************************************************************/

// No Activation Function. Returns the same vector.
double* noActivationVector(double* vector, int vectorSize) {
    return vector;
}


// ReLu Activation Function. Returns a vector of max(0, x)
double* reLuVector(double* vector, int vectorSize) {

    double* output = (double*)malloc(vectorSize * sizeof(double));

    for(int i = 0; i < vectorSize; i++) 
        output[i] = vector[i] > 0 ? vector[i] : 0;

    return output;
}

// Leaky ReLu Activation Function. Returns a vector of max(0.01, x)
double* leakReLuVector(double* vector, int vectorSize) {

    double* output = (double*)malloc(vectorSize * sizeof(double));

    for(int i = 0; i < vectorSize; i++) 
        output[i] = vector[i] > 0 ? vector[i] : 0.0001 * vector[i];

    return output;
}

// Softmax Activation Function. Returns a vector of e^x / sum(e^x_i)
double* softMaxVector(double* vector, int vectorSize) {

    double maxElem = vector[0];

    for(int i = 0; i < vectorSize; i++)
        if(maxElem < vector[i])
            maxElem = vector[i];

    double exp_sum = 0;
    double* output = (double*)malloc(vectorSize * sizeof(double));

    for(int i = 0; i < vectorSize; i++) {
        double x = exp(vector[i] - maxElem);
        output[i] = x;
        exp_sum += x;  // WARNING, BECOMES NAN IF BECOMES TOO BIG
    }

    for(int i = 0; i < vectorSize; i++) 
        output[i] /= exp_sum;
    
    return output;
}

/************************************************************************************
*                                                                                   *
*   ACTIVATION FUNCTIONS  (ReLu)                                                    *
*                                                                                   *
*************************************************************************************/

// ReLu Activation Function. Returns max(0, x)
double reLu(double x) {
    return x > 0 ? x : 0;
}

/************************************************************************************
*                                                                                   *
*   DECLARATION OF NODES                                                            *
*   (Linear)                                                                        *
*                                                                                   *
*************************************************************************************/

double linearNode(int inputSize, double input[inputSize], double* weights, int weightSize, double bias) {
    
    // Check if inputted inputSize matches weightSize
    if(inputSize != weightSize) {
        fprintf(stderr, "Error in Linear Node: Input size does not match weight size, %d vs. %d", inputSize, weightSize);
        exit(1);
    }

    double outputNum = bias;
    for(int i = 0; i < inputSize; i++) 
        outputNum += input[i] * weights[i];

    return outputNum;
}


/************************************************************************************
*                                                                                   *
*   DECLARATION OF LAYERS                                                           *
*   (Linear, Convolution2D, MaxPooling2D, Dense, Flatten, Dropout)                  *
*                                                                                   *
*************************************************************************************/

double* linearLayer(int inputSize, double input[inputSize], double** weightArrays, int weightSize, int numOfWeightArrays, 
                    double* bias, int biasSize, int numOfNodes, activation_func_vector_ptr activation_func) {

    // Check if inputted inputSize matches weightSize
    if(inputSize != weightSize) {
        fprintf(stderr, "Error in Linear Layer: Input size does not match weight size, %d vs. %d", inputSize, weightSize);
        exit(1);
    }
    
    // Check if inputted number of weight arrays matches number of nodes
    if(numOfWeightArrays != numOfNodes) {
        fprintf(stderr, "Error in Linear Layer: Number of Weight Arrays does not match Number of Nodes, %d vs. %d", numOfWeightArrays, numOfNodes);
        exit(1);
    }

    // Check if inputted biasSize matches number of nodes
    if(biasSize != numOfNodes) {
        fprintf(stderr, "Error in Linear Layer: Number of Biases does not match Number of Nodes, %d vs. %d", biasSize, numOfNodes);
        exit(1);
    }

    /*
        Note: At this point, numOfWeightArrays == numOfNodes == biasSize.
    */

    double* output = (double*)malloc(numOfNodes * sizeof(double));

    for(int i = 0; i < numOfNodes; i++)
        output[i] = linearNode(inputSize, input, weightArrays[i], weightSize, bias[i]);

    output = activation_func(output, numOfNodes);

    return output;

}


/************************************************************************************
*                                                                                   *
*   WEIGHT INITIALIZATION                                                           *
*                                                                                   *
*************************************************************************************/

// He Initialization Method of Weights. Uses Box-Muller Transform to generate a normal distribution with mean of 0 and std of sqrt(2/n)
double* HeInitialization(int numOfPreviousNeurons) {
    double* weights = (double*)malloc(numOfPreviousNeurons * sizeof(double));
    
    double std = sqrt(2.0 / numOfPreviousNeurons);
    for(int i = 0; i < numOfPreviousNeurons; i+=2) {

        double u1 = (double)rand() / RAND_MAX;
        double u2 = (double)rand() / RAND_MAX;
        double z1 = std * sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
        double z2 = std * sqrt(-2.0 * log(u1)) * sin(2.0 * M_PI * u2);
        weights[i] = z1;

        if(i+1 < numOfPreviousNeurons)
            weights[i+1] = z2;

    }
    
    return weights;
}

// Random initialization
double* RandomInitialization(int numOfPreviousNeurons) {
    double* weights = (double*)malloc(numOfPreviousNeurons * sizeof(double));
    
    for(int i = 0; i < numOfPreviousNeurons; i++) 
        weights[i] = (double)rand() / RAND_MAX;
    
    return weights;
}


/************************************************************************************
*                                                                                   *
*   GRADIENT FUNCTIONS                                                              *
*                                                                                   *
*************************************************************************************/

double reLuGradient(double x) {
    return x > 0 ? 1.0 : 0;
}

double leakReLuGradient(double x) {
    return x > 0 ? 1.0 : 0.0001;
}

double crossEntropyGradientWithSoftmax(double y, double s) {
    return s - y;
}

double softMaxGradient(double x) {
    return x;
}


/************************************************************************************
*                                                                                   *
*   TRAINING FUNCTIONS                                                              *
*                                                                                   *
*************************************************************************************/

// Allocate arrays for batches of examples and layers
void allocateBatch(double*** batchExample, int** batchLabel, int batchSize,
                   int exampleSize, double features[][exampleSize],
                   int iterationNum, int numOfExamples, int labels[numOfExamples]) {

    *batchExample = (double**)malloc(batchSize * sizeof(double*));
    *batchLabel = (int*)malloc(batchSize * sizeof(int));

    for(int j = 0; j < batchSize; j++) {
        int exampleIndex = (iterationNum*batchSize + j) % numOfExamples;
        (*batchExample)[j] = (double*)malloc(exampleSize * sizeof(double));

        memcpy((*batchExample)[j], features[exampleIndex], exampleSize * sizeof(double));
        (*batchLabel)[j] = labels[exampleIndex];
    }
     
}

// Encodes the given label into a one-hot-encoded vector
void oneHotEncoded(int* labelVector, double*** encodedVector, int numOfLabels, int numOfCodes) {
    (*encodedVector) = (double**)malloc(numOfLabels * sizeof(double*));

    for(int i = 0; i < numOfLabels; i++) {
        int label = labelVector[i];
        (*encodedVector)[i] = (double*)malloc(numOfCodes * sizeof(double));

        for(int j = 0; j < numOfCodes; j++)
            (*encodedVector)[i][j] = 0;

        (*encodedVector)[i][label] = 1;
    }
}

/*  Forward Propagation 
    Parameters: 1D Example Input, Example Size,
                3D Weight Array, Number of Weights Array, Weights Array Size,
                2D Biases, Number of Bias Arrays,
    
    Returns: 1D Output Vector 
*/
int predictModel(Model* model, int exampleSize, double features[exampleSize]) {
    // Initialize Outputs after each layer
    double** outputLayers = (double**)malloc((model->numOfLayers+1) * sizeof(double*));

    outputLayers[0] = (double*)malloc(model->numOfNodes[0] * sizeof(double));

    for(int j = 0; j < exampleSize; j++) 
        outputLayers[0][j] = features[j];                

    // FORWARD PROPAGATION 
    for(int j = 0; j < model->numOfLayers; j++) {
        int size = model->numOfNodes[j];
        
        switch(model->layerType[j]) {
            case LINEAR:
                outputLayers[j+1] = linearLayer(size, outputLayers[j], model->weights[j], size, model->numOfNodes[j+1], 
                                                model->bias[j], model->numOfNodes[j+1], model->numOfNodes[j+1], 
                                                model->activation_funcs[j]);
                break;
            default:
                fprintf(stderr, "Error: Unknown Layer Type");
                exit(1);
                break;
        }
    }

    // Calculate accuracy
    int outputMaxLabel = 0;
    int outputMaxNum = outputLayers[model->numOfLayers][0];

    for(int i = 1; i < 10; i++) {
        if(outputMaxNum < outputLayers[model->numOfLayers][i]) {
            outputMaxNum = outputLayers[model->numOfLayers][i];
            outputMaxLabel = i;
        }
    }

    freeDoublePointers(outputLayers, model->numOfLayers+1);

    return outputMaxLabel;
}

/*
    Train Model Function.
    Parameters: 2D Input of multiple 1D Feature Array, Number of examples, Example size, 
                Empty 3D Weights array, Number of weights array, weights array size, 
                Empty 2D Biases, Number of bias arrays,
                1D Label (Same number as number of examples)
                Batch size, Number of epochs, Learning Rate
    Returns:    Updated Weights and Bias Arrays

    Input[Example][Index]
    Weights[Layer][Node][Index]
    Bias[Layer][Node]

    NOTE: THE MODEL LAYERS ARE PRE-DEFINED (Possible to "Add" layers before training?)
*/

// Initializes the input size of the model and must be done first
void initializeModel(Model* model, int inputSize) {
    if(inputSize <= 0) {
        fprintf(stderr, "Error in initializeModel: Input is not > 0");
        exit(1);
    }

    model->numOfNodes = (int*)malloc(1 * sizeof(int));

    model->numOfNodes[0] = inputSize;

    model->numOfLayers = 0;

    model->layerType = (Layers*)malloc(0);

    model->activation_funcs = (activation_func_vector_ptr*)malloc(0);

    model->gradient_funcs = (gradient_func_ptr*)malloc(0);

}

void addLayer(Model* model, Layers layer, int numOfNodes, activation_func_vector_ptr activation_func, gradient_func_ptr gradient_func) {
    
    model->numOfLayers += 1;

    // Increase memory of pointer arrays for nodes, layers, and activation functions
    model->numOfNodes = realloc(model->numOfNodes, sizeof(int) * (model->numOfLayers + 1));
    model->layerType = realloc(model->layerType, sizeof(Layers*) * model->numOfLayers);
    model->activation_funcs = realloc(model->activation_funcs, sizeof(activation_func_vector_ptr*) * model->numOfLayers);
    model->gradient_funcs = realloc(model->gradient_funcs, sizeof(gradient_func_ptr) * model->numOfLayers);

    model->numOfNodes[model->numOfLayers] = numOfNodes;
    model->layerType[model->numOfLayers - 1] = layer;
    model->activation_funcs[model->numOfLayers - 1] = activation_func;
    model->gradient_funcs[model->numOfLayers - 1] = gradient_func;

}

void setupModel(Model* model, weight_func_ptr weight_func, gradient_loss_func_ptr gradient_loss_func) {
    model->weights = (double***)malloc(model->numOfLayers * sizeof(double**));

    // Iterate through every layer and allocate weights according to passed weight initialization function
    for(int i = 0; i < model->numOfLayers; i++) {
        model->weights[i] = (double**)malloc(model->numOfNodes[i+1] * sizeof(double*));
        for(int j = 0; j < model->numOfNodes[i+1]; j++)
            model->weights[i][j] = weight_func(model->numOfNodes[i]);
    }

    model->bias = (double**)malloc(model->numOfLayers * sizeof(double*));

    // Iterate through every layer and allocate bias as 0
    for(int i = 0; i < model->numOfLayers; i++) {
        model->bias[i] = (double*)malloc(model->numOfNodes[i+1] * sizeof(double));

        for(int j = 0; j < model->numOfNodes[i+1]; j++) 
            model->bias[i][j] = 0;
        
    }    

    model->gradient_loss_func = gradient_loss_func;

}

void trainModel(Model* model,
                int numOfExamples, int exampleSize, double features[][exampleSize],  
                int labels[numOfExamples], 
                int batchSize, int epochs, double learningRate,
                int numOfTests, double testFeatures[][exampleSize], int testLabels[numOfTests]) {

    /*
        Model Structure:
        Input Layer (784 Features)
        Linear Layer (50 Nodes)
        ReLu Layer (50 Nodes)
        Linear Layer (10 Nodes)
        Softmax Layer (10 Nodes) (Output)
    */

    int numOfIterations = ((double)numOfExamples / (double)batchSize + 0.5); 

    printf("Number of Iterations: %d\n", numOfIterations);

    while(epochs-- > 0) {
        printf("Epoch Num: %d\n", epochs);

        double totalAccuracy = 0;

        for(int iter = 0; iter < numOfIterations; iter++) {
            double** batchImages;
            int* batchLabel;
            double** batchLabelOneHot;

            double accuracy = 0;
        
            // Allocate features and labels into batches (batchImages, batchLabelOneHot)
            allocateBatch(&batchImages, &batchLabel, batchSize, exampleSize, features, iter, numOfExamples, labels);

            // Convert label into one-hot-encoded vector 
            oneHotEncoded(batchLabel, &batchLabelOneHot, batchSize, 10);

            
            // Begin iterating through the batches (batchImages)
            for(int i = 0; i < batchSize; i++) {

                // Initialize Outputs after each layer
                double** outputLayers = (double**)malloc((model->numOfLayers+1) * sizeof(double*));

                outputLayers[0] = (double*)malloc(model->numOfNodes[0] * sizeof(double));

                for(int j = 0; j < exampleSize; j++) 
                    outputLayers[0][j] = batchImages[i][j];                
            
                // FORWARD PROPAGATION 
                for(int j = 0; j < model->numOfLayers; j++) {
                    int size = model->numOfNodes[j];
                    
                    switch(model->layerType[j]) {
                        case LINEAR:
                            outputLayers[j+1] = linearLayer(size, outputLayers[j], model->weights[j], size, model->numOfNodes[j+1], 
                                                            model->bias[j], model->numOfNodes[j+1], model->numOfNodes[j+1], 
                                                            model->activation_funcs[j]);
                            break;
                        default:
                            fprintf(stderr, "Error: Unknown Layer Type");
                            exit(1);
                            break;
                    }
                }

                // Calculate accuracy
                int outputMaxLabel = 0;
                int outputMaxNum = outputLayers[model->numOfLayers][0];
                for(int i = 1; i < 10; i++) {
                    if(outputMaxNum < outputLayers[model->numOfLayers][i]) {
                        outputMaxNum = outputLayers[model->numOfLayers][i];
                        outputMaxLabel = i;
                    }
                }
                if(outputMaxLabel == batchLabel[i])
                    accuracy++;

                // BACK PROPAGATION
                // Initialize gradient memory
                double** gradient = (double**)malloc(model->numOfLayers * sizeof(double*));

                for(int j = 0; j < model->numOfLayers; j++)
                    gradient[j] = (double*)malloc(model->numOfNodes[j+1] * sizeof(double));

                for(int j = model->numOfLayers-1; j >= 0; j--) {
                    for(int k = 0; k < model->numOfNodes[j+1]; k++) {
                        double gradientSum = 0;

                        if(j < model->numOfLayers-1) {
                            
                            for(int l = 0; l < model->numOfNodes[j+2]; l++)
                                gradientSum += model->weights[j][l][k] * gradient[j+1][l];
                        }

                        if(j == model->numOfLayers-1)
                            gradient[j][k] = model->gradient_loss_func(batchLabelOneHot[i][k], outputLayers[j+1][k]);

                        else   
                            gradient[j][k] = gradientSum * model->gradient_funcs[j](outputLayers[j+1][k]);
                    }
                }
                
                // UPDATE WEIGHTS & BIASES
                // Last Linear Layer


                for(int j = 0; j < model->numOfLayers; j++) {
                    for(int k = 0; k < model->numOfNodes[j+1]; k++) {

                        for(int l = 0; l < model->numOfNodes[j]; l++)
                            model->weights[j][k][l] -= gradient[j][k] * learningRate * outputLayers[j][l];

                        model->bias[j][k] -= gradient[j][k] * learningRate;
                    }
                }

                freeDoublePointers((void**)gradient, model->numOfLayers);
                freeDoublePointers((void**)outputLayers, model->numOfLayers+1);

            }
            
            accuracy /= batchSize;
            totalAccuracy += accuracy;
            
            free(batchLabel);
            freeDoublePointers((void**)batchLabelOneHot, batchSize);
            freeDoublePointers((void**)batchImages, batchSize);

        }
        
        totalAccuracy /= numOfIterations;

        printf("Accuracy: %lf\n", totalAccuracy);

        double testAccuracy = 0;
        for(int i = 0; i < numOfTests; i++) {

            if(predictModel(model, exampleSize, testFeatures[i]) == testLabels[i])

                testAccuracy++;
        }

        testAccuracy /= numOfTests;

        printf("Test Accuracy: %f\n", testAccuracy);

        learningRate /= 2;

        printf("Epoch done.\n");
    }
    
    printf("Train Ended\n");


}


/************************************************************************************
*                                                                                   *
*   MAIN FUNCTION                                                                   *
*                                                                                   *
*************************************************************************************/
/*
int main() {

    srand(time(0));

    Model model;

    initializeModel(&model, SIZE);

    addLayer(&model, LINEAR, 84, leakReLuVector, leakReLuGradient);

    addLayer(&model, LINEAR, 10, softMaxVector, softMaxGradient);

    setupModel(&model, RandomInitialization, crossEntropyGradientWithSoftmax);

    int batchSize = 100;
    int epochs = 13;
    double learningRate = 0.05;

    trainModel(&model,
                NUM_TRAIN, SIZE, train_image, train_label, 
                batchSize, epochs, learningRate,
                NUM_TEST, test_image, test_label);

    
    char temp;
    for(int testNum = 0; testNum < NUM_TEST; testNum++) {
        for (int i=0; i<784; i++) {
            printf("%1.1f ", test_image[testNum][i]);
            if ((i+1) % 28 == 0) putchar('\n');
        }

        printf("Predicted Label: %d\nTrue Label: %d\n", predictModel(&model, SIZE, test_image[testNum]), test_label[testNum]);
        printf("Continue: ");
        scanf("%c", &temp);
        printf("\n");
    }


    // Free remaining vectors
    clearModel(&model);
    printf("\nEnd.");

    return 0;
}*/