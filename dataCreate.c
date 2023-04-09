#include "mnist.h"

int main() {
    load_mnist();

    int testNum = 200;
    int trainNum = 3000;

    write_2D_image_to_file("test_image.h", "test_image", testNum, 5000-testNum/2, test_image);
    write_2D_image_to_file("train_image.h", "train_image", trainNum, 35000 - trainNum/2, train_image);
    write_1D_label_to_file("test_label.h", "test_label", testNum, 5000-testNum/2, NUM_TEST, test_label);
    write_1D_label_to_file("train_label.h", "train_label", trainNum, 35000 - trainNum/2, NUM_TRAIN, train_label);

}