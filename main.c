#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "mnist.h"

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

typedef double* (*activation_func_vector_ptr)(double*, int);
typedef double (*activation_func_ptr)(double);

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

void freeTriplePointers(void*** pointerToFree, int sizeFirst, int sizeSecond) {
    for(int i = 0; i < sizeFirst; i++) {
        for(int j = 0; j < sizeSecond; j++) 
            free(pointerToFree[i][j]);
        free(pointerToFree[i]);
    }    

    free(pointerToFree);
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

// Softmax Activation Function. Returns a vector of e^x / sum(e^x_i)
double* softMaxVector(double* vector, int vectorSize) {
    double exp_sum = 0;

    for(int i = 0; i < vectorSize; i++)
        exp_sum += exp(vector[i]);

    double* output = (double*)malloc(vectorSize * sizeof(double));

    for(int i = 0; i < vectorSize; i++) 
        output[i] = exp(vector[i]) / exp_sum;

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
                    double* bias, int biasSize, int numOfNodes) {

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
void forwardPropagation(int exampleSize, double* features[exampleSize], int a) {

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
void trainModel(int numOfExamples, int exampleSize, double features[][exampleSize],  
                double**** weights, int* numOfWeightLayers, int** numOfWeightArrays, int** weightSize,
                double*** bias, int* numOfBiasArrays, int** biasSize,
                int labels[exampleSize], 
                int batchSize, int epochs, double learningRate) {

    /*
        Model Structure:
        Input Layer (784 Features)
        Linear Layer (50 Nodes)
        ReLu Layer (50 Nodes)
        Linear Layer (10 Nodes)
        Softmax Layer (10 Nodes) (Output)
    */

    *numOfWeightLayers = 2;
    int numOfLayers = 4;  

    // Initialize number of weights, corresponding to number of input nodes at each layer
    (*weightSize) = (int*)malloc((*numOfWeightLayers) * sizeof(int));
    (*weightSize)[0] = SIZE;
    (*weightSize)[1] = 50;

    *numOfBiasArrays = (*numOfWeightLayers);

    // Initialize number of weight arrays, corresponding to number of nodes pointed to by the weights at each layer
    (*numOfWeightArrays) = (int*)malloc((*numOfWeightLayers) * sizeof(int));
    (*numOfWeightArrays)[0] = 50;
    (*numOfWeightArrays)[1] = 10;

    // Initialize weights based on He Initialization
    (*weights) = (double***)malloc((*numOfWeightLayers) * sizeof(double**));

    (*weights)[0] = (double**)malloc(50 * sizeof(double*));
    for(int j = 0; j < 50; j++) 
        (*weights)[0][j] = HeInitialization(exampleSize);

    (*weights)[1] = (double**)malloc(10 * sizeof(double**));    
    for(int j = 0; j < 10; j++) 
        (*weights)[1][j] = HeInitialization(50);


    // Initialize biases based on He Initialization (essentially set them to 0)
    (*biasSize) = (int*)malloc((*numOfWeightLayers) * sizeof(int));
    (*biasSize)[0] = 50;
    (*biasSize)[1] = 10;

    (*bias) = (double**)malloc((*numOfBiasArrays) * sizeof(double*));

    (*bias)[0] = (double*)malloc(50 * sizeof(double));
    for(int i = 0; i < 50; i++)
        (*bias)[0][i] = 0;
    
    (*bias)[1] = (double*)malloc(10 * sizeof(double));
    for(int i = 0; i < 10; i++)
        (*bias)[1][i] = 0;
    

    int numOfIterations = ((double)numOfExamples / (double)batchSize + 0.5); 
    printf("Number of Iterations: %d\n", numOfIterations);
    while(epochs-- > 0) {
        printf("Epoch Num: %d\n", epochs);
        for(int iter = 0; iter < numOfIterations; iter++) {
            double** batchImages;
            int* batchLabelTemp;
            double** batchLabelOneHot;

            // Allocate features and labels into batches (batchImages, batchLabelOneHot)
            allocateBatch(&batchImages, &batchLabelTemp, batchSize, exampleSize, features, iter, numOfExamples, labels);

            // Convert label into one-hot-encoded vector 
            oneHotEncoded(batchLabelTemp, &batchLabelOneHot, batchSize, 10);


            // Begin iterating through the batches (batchImages)
            for(int i = 0; i < batchSize; i++) {

                // Initialize Outputs after each layer
                double** outputLayers = (double**)malloc(numOfLayers * sizeof(double*));

                // Number of Nodes: 50
                outputLayers[0] = linearLayer(SIZE, batchImages[i], (*weights)[0], (*weightSize)[0], (*numOfWeightArrays)[0], 
                                             (*bias)[0], (*biasSize)[0], 50);

                outputLayers[1] = reLuVector(outputLayers[0], 50);

                // Number of Nodes: 10
                outputLayers[2] = linearLayer(50, outputLayers[1], (*weights)[1], (*weightSize)[1], (*numOfWeightArrays)[1], 
                                             (*bias)[1], (*biasSize)[1], 10);

                outputLayers[3] = softMaxVector(outputLayers[2], 10);

                
                freeDoublePointers((void**)outputLayers, numOfLayers);
            }

            
            free(batchLabelTemp);
            freeDoublePointers((void**)batchImages, batchSize);
           
        }

        printf("Epoch done.\n");
    }
    
    printf("Train Ended\n");


}



/************************************************************************************
*                                                                                   *
*   TEST FUNCTIONS                                                                  *
*                                                                                   *
*************************************************************************************/

void testLinearNode() {

    double input[] = {1, 2, 3, 4, 5, 6, 7, 8};
    double weights[] = {1, 2, 3, 4, 5, 6, 7, 8};
    double bias = 5;

    int inputSize = sizeof(input) / sizeof(input[0]);
    int weightSize = sizeof(weights) / sizeof(weights[0]);

    double output = linearNode(inputSize, input, weights, weightSize, bias);
    printf("Linear Node Test:\n");
    printf("\t Input size: %d\n", inputSize);
    printf("\t Weight size: %d\n", weightSize);
    printf("\t Output: %lf\n", output);
    printf("\t Expected: 209.0");
}

void testLinearLayer() {
    
    int inputSize = 4;
    double input[] = {1, 2, 3, 4}; 
    
    int numOfWeightArrays = 3;
    int weightSize = 4;

    double weightArraysToCopy[3][4] = {{1, 2, 3, -4}, {1, 1, 1, 1,}, {0.5, 0.5, 0.5, 0.5}};

    double **weightArrays = (double**)malloc(numOfWeightArrays * sizeof(double*));
    for(int i = 0; i < numOfWeightArrays; i++) {
        weightArrays[i] = (double*)malloc(weightSize * sizeof(double));
        for(int j = 0; j < weightSize; j++)
            weightArrays[i][j] = weightArraysToCopy[i][j];
    }
    
    int biasSize = 3;
    double bias[] = {1, 2, 3};

    int numOfNodes = 3;

    double* output = linearLayer(inputSize, input, weightArrays, weightSize, numOfWeightArrays, bias, biasSize, numOfNodes);

    printf("Linear Linear Test:\n");
    printf("\t Input size: %d\n", inputSize);
    printf("\t Number of Weights: %d\n", numOfWeightArrays);
    printf("\t Weight size: %d\n", weightSize);
    printf("\t Bias size: %d\n", biasSize);
    printf("\t Number of Nodes: %d\n", numOfNodes);
    printf("\t Output size: %d\n", sizeof(output)/sizeof(output[0]));
    printf("\t Output: \n\t\t");
    for(int i = 0; i < numOfNodes; i++) {
        printf(" %lf ", output[i]);
    }
    printf("\n\t Expected Output: [-1, 12, 8]");
    printf("\n\t Expected ReLu: [0, 12, 8]");
    printf("\n\t Expected Softmax: [0.000002, 0.982012, 0.017986]\n");

    freeDoublePointers((void**)weightArrays, numOfWeightArrays);
    free(output);
}

void testReLuLayer() {
    int inputSize = 4;
    double input[] = {1, 2, 3, 4}; 
    
    int numOfWeightArrays = 3;
    int weightSize = 4;

    double weightArraysToCopy[3][4] = {{1, 2, 3, -4}, {1, 1, 1, 1,}, {0.5, 0.5, 0.5, 0.5}};

    double **weightArrays = (double**)malloc(numOfWeightArrays * sizeof(double*));
    for(int i = 0; i < numOfWeightArrays; i++) {
        weightArrays[i] = (double*)malloc(weightSize * sizeof(double));
        for(int j = 0; j < weightSize; j++)
            weightArrays[i][j] = weightArraysToCopy[i][j];
    }
    
    int biasSize = 3;
    double bias[] = {1, 2, 3};

    int numOfNodes = 3;

    double* output = linearLayer(inputSize, input, weightArrays, weightSize, numOfWeightArrays, bias, biasSize, numOfNodes);
    output = reLuVector(output, numOfNodes);

    printf("Linear Linear Test:\n");
    printf("\t Input size: %d\n", inputSize);
    printf("\t Number of Weights: %d\n", numOfWeightArrays);
    printf("\t Weight size: %d\n", weightSize);
    printf("\t Bias size: %d\n", biasSize);
    printf("\t Number of Nodes: %d\n", numOfNodes);
    printf("\t Output size: %d\n", sizeof(output)/sizeof(output[0]));
    printf("\t Output: \n\t\t");
    for(int i = 0; i < numOfNodes; i++) {
        printf(" %lf ", output[i]);
    }
    printf("\n\t Expected Output: [-1, 12, 8]");
    printf("\n\t Expected ReLu: [0, 12, 8]");
    printf("\n\t Expected Softmax: [0.000002, 0.982012, 0.017986]\n");

    freeDoublePointers((void**)weightArrays, numOfWeightArrays);
    free(output);
}

void testSoftmaxLayer() {
    int inputSize = 4;
    double input[] = {1, 2, 3, 4}; 
    
    int numOfWeightArrays = 3;
    int weightSize = 4;

    double weightArraysToCopy[3][4] = {{1, 2, 3, -4}, {1, 1, 1, 1,}, {0.5, 0.5, 0.5, 0.5}};

    double **weightArrays = (double**)malloc(numOfWeightArrays * sizeof(double*));
    for(int i = 0; i < numOfWeightArrays; i++) {
        weightArrays[i] = (double*)malloc(weightSize * sizeof(double));
        for(int j = 0; j < weightSize; j++)
            weightArrays[i][j] = weightArraysToCopy[i][j];
    }
    
    int biasSize = 3;
    double bias[] = {1, 2, 3};

    int numOfNodes = 3;

    double* output = linearLayer(inputSize, input, weightArrays, weightSize, numOfWeightArrays, bias, biasSize, numOfNodes);
    output = softMaxVector(output, numOfNodes);

    printf("Linear Linear Test:\n");
    printf("\t Input size: %d\n", inputSize);
    printf("\t Number of Weights: %d\n", numOfWeightArrays);
    printf("\t Weight size: %d\n", weightSize);
    printf("\t Bias size: %d\n", biasSize);
    printf("\t Number of Nodes: %d\n", numOfNodes);
    printf("\t Output size: %d\n", sizeof(output)/sizeof(output[0]));
    printf("\t Output: \n\t\t");
    for(int i = 0; i < numOfNodes; i++) {
        printf(" %lf ", output[i]);
    }
    printf("\n\t Expected Output: [-1, 12, 8]");
    printf("\n\t Expected ReLu: [0, 12, 8]");
    printf("\n\t Expected Softmax: [0.000002, 0.982012, 0.017986]\n");

    freeDoublePointers((void**)weightArrays, numOfWeightArrays);
    free(output);
}

void testMNIST() {
    // print pixels of first data in test dataset
    int i;
    for (i=0; i<784; i++) {
        printf("%1.1f ", test_image[4][i]);
        if ((i+1) % 28 == 0) putchar('\n');
    }

    // print first label in test dataset
    printf("label: %d\n", test_label[4]);

    char dummy;
    // save image of first data in test dataset as .pgm file
    printf("Continue: ");
    scanf("%c", &dummy);

    // show all pixels and labels in test dataset
    print_mnist_pixel(test_image, NUM_TEST);
    //print_mnist_label(test_label, NUM_TEST);
}

void testBatchAllocation(int numOfExamples, int exampleSize, double input[][exampleSize], int batchSize, int label[numOfExamples]) {

    int numOfIterations = ((double)numOfExamples / (double)batchSize + 0.5); 
        
    for(int i = 0; i < numOfIterations; i++) {

        // Allocate examples and labels in batches of (batchSize);
        double** inputBatchImage;
        int* inputBatchLabel;

        allocateBatch(&inputBatchImage, &inputBatchLabel, batchSize, exampleSize, input, i, numOfExamples, label);

        
        freeDoublePointers((void**)inputBatchImage, batchSize);
        free(inputBatchLabel);
    }  
    printf("Allocated properly!");
}

void testOneHotEncoded() {
    int* vectorToTest = (int*)malloc(sizeof(int) * 3);
    vectorToTest[0] = 0;
    vectorToTest[1] = 2;
    vectorToTest[2] = 9;

    double** vectorEncoded;

    oneHotEncoded(vectorToTest, &vectorEncoded, 3, 10);

    for(int i = 0; i < 3; i++) {
        printf("Label: %d\nVector: [", vectorToTest[i]);
        for(int j = 0; j < 10; j++)
            if(j < 9)
                printf("%f, ", vectorEncoded[i][j]);
            else
                printf("%f]\n", vectorEncoded[i][j]);

    }
}


/************************************************************************************
*                                                                                   *
*   MAIN FUNCTION                                                                   *
*                                                                                   *
*************************************************************************************/

int main() {
    /* 
        double train_image[NUM_TRAIN][SIZE];
        double test_image[NUM_TEST][SIZE];
        int  train_label[NUM_TRAIN];
        int test_label[NUM_TEST];
    */
    load_mnist();

    /* Testing Conv2D [ WORKS ]*/
    //testConv2D();

    /* Testing Linear Node [ WORKS ]*/
    //testLinearNode();

    /* Testing Linear Layer [ WORKS ]*/
    //testLinearLayer();

    /* Testing ReLu Layer [ WORKS ]*/
    //testReLuLayer();

    /* Testing Softmax Layer [ WORKS ]*/
    //testSoftmaxLayer();
    
    /* Testing MNIST Data [ WORKS ]*/
    //testMNIST();
    
    /* Testing Batch Allocation [ WORKS ]*/
    //testBatchAllocation( NUM_TEST, 784, test_image, 10, test_label);

    /* Testing one-hot-encoded vector [ WORKS ]*/
    //testOneHotEncoded();

    double*** weights = NULL;
    double** bias = NULL;
    int numOfWeightLayers;
    int numOfBiasArrays;
    int* weightSize;
    int* biasSize;
    int* numOfWeightArrays;

    int batchSize = 100;
    int epochs = 20;
    double learningRate = 0.01;

    trainModel(NUM_TEST, SIZE, test_image,  
                &weights, &numOfWeightLayers, &numOfWeightArrays, &weightSize, 
                &bias, &numOfBiasArrays, &biasSize,
                test_label, 
                batchSize, epochs, learningRate);

    printf("\nEnd.");

    return 0;
}












/************************************************************************************
*                                                                                   *
*   DEPRECATED FUNCTIONS                                                            *
*                                                                                   *
*************************************************************************************/

// Convolution2D Layer.
// Performs element-wise dot-product between input and kernels with a stride of 1 and a bias
// Uses an activation function if not NULL for every element in the output
double*** conv2D(double** input, double inputSize, double*** filter, int numFilters, int filterSize, double* bias, activation_func_ptr activation_func) {

    // Allocate memory for output matrix (input[y][x]) with assumption of stride length of 1 
    int outputHeight = inputSize - filterSize + 1;
    int outputWidth = inputSize - filterSize + 1;

    double ***output = (double***)malloc(numFilters * sizeof(double**));

    for(int i = 0; i < numFilters; i++) {
        output[i] = (double**)malloc(outputHeight * sizeof(double*));
        for(int j = 0; j < outputHeight; j++) 
            output[i][j] = (double*)malloc(outputWidth * sizeof(double));
    }

    // Iterate through every filter and use dot-product operation between filter elements and local input elements

    for(int filterNum = 0; filterNum < numFilters; filterNum++) {
        
        for(int i = 0; i < outputHeight; i++) {
            for(int j = 0; j < outputWidth; j++) {
                
                double product = 0;

                for(int filterY = 0; filterY < filterSize; filterY++) 
                    for(int filterX = 0; filterX < filterSize; filterX++) 
                        product += filter[filterNum][filterY][filterX] * input[i+filterY][j+filterX];
                
                output[filterNum][i][j] = activation_func(product + bias[filterNum]);
            }

        }

    }

    return output;
}

// MaxPooling2D Layer. 
// Takes the largest number in the local input region and maps it to a new output matrix, given the maxPooling size
// WARNING: THIS LAYER DOES NOT PAD. ASSUMES ALL ARE SQUARE MATRICES
double** maxPool2D(double** input, int inputSize, int maxPoolSize, int stride) {

    int outputWidth = (inputSize - (maxPoolSize/2) + 1) / stride;
    int outputHeight = outputWidth;

    double** output = (double**)malloc(sizeof(double*) * outputHeight);
    for(int i = 0; i < outputHeight; i++)
        output[i] = (double*)malloc(sizeof(double) * outputWidth);

    for(int i = 0; i < outputHeight; i++) {
        for(int j = 0; j < outputWidth; j++) {
            output[i][j] = input[i*stride][j*stride];
            
            for(int k = 0; k < stride; k++)
                for(int l = 0; l < stride; l++)
                    output[i][j] = output[i][j] > input[i*stride + k][j*stride + l] ? output[i][j] : input[i*stride + k][j*stride + l];
            
        }
    }

    return output;
    
}


void testConv2D() {
    int numFilters = 1;
    int filterSize = 5;
    double*** filter = (double***)malloc(numFilters * sizeof(double**));

    for(int i = 0; i < numFilters; i++) {
        filter[i] = (double**)malloc(filterSize * sizeof(double*));
        for(int j = 0; j < filterSize; j++) {
            filter[i][j] = (double*)malloc(filterSize * sizeof(double));
            for(int k = 0; k < filterSize; k++)
                filter[i][j][k] = 3;
        }
    }   

    int inputSize = 28;
    double** input = (double**)malloc(inputSize * sizeof(double*));
    
    for(int i = 0; i < inputSize; i++) {
        input[i] = (double*)malloc(inputSize * sizeof(double));
        for(int j = 0; j < inputSize; j++) 
            input[i][j] = 1;
    }

    double* bias = (double*)malloc(numFilters * sizeof(double));

    for(int i = 0; i < numFilters; i++) 
        bias[i] = 1;


    double*** output = conv2D(input, inputSize, filter, numFilters, filterSize, bias, reLu);

    int outputLayers = numFilters;
    int outputHeight = inputSize - filterSize + 1;
    int outputWidth = inputSize - filterSize + 1;

    for(int i = 0; i < outputLayers; i++) {
        printf("Output Layer %d:\n", i);

        for(int j = 0; j < outputHeight; j++) {
            for(int k = 0; k < outputWidth; k++) {
                printf("%.2lf ", output[i][j][k]);
            }
            printf("\n");
        }
        printf("\n\n");
    }
    freeTriplePointers((void***)output, numFilters, outputHeight);
    freeDoublePointers((void**)input, inputSize);
    freeTriplePointers((void***)filter, numFilters, filterSize);
}
