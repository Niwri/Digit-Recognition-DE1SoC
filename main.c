#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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
        loss -= predictedProbabilities[i] * log(targetProbabilities[i]);

    return loss;
}

/************************************************************************************
*                                                                                   *
*   ACTIVATION FUNCTIONS VECTOR FORMAT (ReLu, Softmax)                              *
*                                                                                   *
*************************************************************************************/

// No Activation Function. Returns the same vector.
double* noActivation(double* vector, int vectorSize) {
    return vector;
}


// ReLu Activation Function. Returns a vector of max(0, x)
double* reLu(double* vector, int vectorSize) {

    double* output = (double*)malloc(vectorSize * sizeof(double));

    for(int i = 0; i < vectorSize; i++) 
        output[i] = vector[i] > 0 ? vector[i] : 0;

    return output;
}

// Softmax Activation Function. Returns a vector of e^x / sum(e^x_i)
double* softMax(double* vector, int vectorSize) {
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

double linearNode(double* input, int inputSize, double* weights, int weightSize, double bias) {
    
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

double* linearLayer(double* input, int inputSize, double** weightArrays, int weightSize, int numOfWeightArrays, 
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

    double* outputWithoutActivation = (double*)malloc(numOfNodes * sizeof(double));

    for(int i = 0; i < numOfNodes; i++)
        outputWithoutActivation[i] = linearNode(input, inputSize, weightArrays[i], weightSize, bias[i]);
    
    double* outputWithActivation = activation_func(outputWithoutActivation, numOfNodes);

    free(outputWithoutActivation);

    return outputWithActivation;

}

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


/************************************************************************************
*                                                                                   *
*   TEST FUNCTIONS                                                                  *
*                                                                                   *
*************************************************************************************/

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

void testLinearNode() {

    double input[] = {1, 2, 3, 4, 5, 6, 7, 8};
    double weights[] = {1, 2, 3, 4, 5, 6, 7, 8};
    double bias = 5;

    int inputSize = sizeof(input) / sizeof(input[0]);
    int weightSize = sizeof(weights) / sizeof(weights[0]);

    double output = linearNode(input, inputSize, weights, weightSize, bias);
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

    double* output = linearLayer(input, inputSize, weightArrays, weightSize, numOfWeightArrays, bias, biasSize, numOfNodes, softMax);

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
    printf("\n\t Expected ReLu: [0, 12, 8]");
    printf("\n\t Expected Softmax: [0.000002, 0.982012, 0.017986]\n");

    freeDoublePointers((void**)weightArrays, numOfWeightArrays);
}


/************************************************************************************
*                                                                                   *
*   MAIN FUNCTION                                                                   *
*                                                                                   *
*************************************************************************************/

int main() {
    /* Testing Conv2D [ WORKS ]*/
    //testConv2D();

    /* Testing Linear Node [ WORKS ]*/
    //testLinearNode();

    /* Testing Linear Layer [ WORKS ]*/
    //testLinearLayer();

    printf("\nEnd.");

    return 0;
}