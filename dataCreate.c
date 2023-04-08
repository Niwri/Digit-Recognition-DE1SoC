#include "mnist.h"

int main() {
    load_mnist();

    write_2D_image_to_file("test_image.h", "test_image", 5, 4500, test_image);
    write_2D_image_to_file("train_image.h", "train_image", 20, 33500, train_image);
    write_1D_label_to_file("test_label.h", "test_label", 5, 4500, NUM_TEST, test_label);
    write_1D_label_to_file("train_label.h", "train_label", 20, 33500, NUM_TRAIN, train_label);

}