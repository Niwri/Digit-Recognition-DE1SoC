#include <stdio.h>
#include <stdlib.h>

typedef double (*activation_func_ptr)(double);

/************************************************************************************
*                                                                                   *
*   DECLARATION OF ACTIVATION FUNCTIONS (ReLu)                                      *
*                                                                                   *
*************************************************************************************/

// ReLu Activation Function. Returns max(0, x)
double reLu(double x) {
    return x > 0 ? x : 0;
}

/************************************************************************************
*                                                                                   *
*   DECLARATION OF LAYERS (Convolution2D, MaxPooling2D, Dense, Flatten, Dropout)    *
*                                                                                   *
*************************************************************************************/

// Convolution2D Layer.
// Performs element-wise dot-product between input and kernels with a stride of 1 and a bias
// Uses an activation function if not NULL for every element in the output
int*** conv2D(int** input, int*** filter, int numFilters, int filterSize, int* bias, activation_func_ptr activation_func) {

    // Check for any possible errors with the inputs
    if(sizeof(filter) / sizeof(int**) != numFilters) {
        fprintf(stderr, "Error: Number of Filters do not match: %d vs. %d", sizeof(filter) / sizeof(int**), numFilters);
        exit(1);
    }

    if(sizeof(filter[0]) / sizeof(int*) != filterSize  || sizeof(filter[0][0]) / sizeof(int) != filterSize) {
        fprintf(stderr, "Error: Filter size does not match: %dx%d vs. %dx%d", sizeof(filter[0]) / sizeof(int*), sizeof(filter[0][0]) / sizeof(int), filterSize, filterSize);
        exit(1);
    }

    if(sizeof(input[0]) / sizeof(int) < filterSize || sizeof(input) / sizeof(int*) < filterSize) {
        fprintf(stderr, "Error: Input matrix is smaller than filter: %dx%d vs. %dx%d", sizeof(input) / sizeof(int*), sizeof(input[0]) / sizeof(int), filterSize, filterSize);
        exit(1);
    }

    // Allocate memory for output matrix (input[y][x]) with assumption of stride length of 1 
    int outputHeight = sizeof(input) / sizeof(int*) - filterSize + 1;
    int outputWidth = sizeof(input[0]) / sizeof(int) - filterSize + 1;

    int ***output = (int***)malloc(numFilters * sizeof(int**));

    for(int i = 0; i < numFilters; i++) {
        output[i] = (int**)malloc(outputHeight * sizeof(int*));
        for(int j = 0; j < outputHeight; j++) 
            output[i][j] = (int*)malloc(outputWidth * sizeof(int));
    }

    // Iterate through every filter and use dot-product operation between filter elements and local input elements

    for(int i = 0; i < numFilters; i++) {

    }

    
    return output;
}


int main() {

    printf("Test");

    return 0;
}