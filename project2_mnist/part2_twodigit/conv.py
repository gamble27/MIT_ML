import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from project2_mnist.part2_twodigit.train_utils import batchify_data, run_epoch, train_model, Flatten
import project2_mnist.part2_twodigit.utils_multiMNIST as U
path_to_data_dir = '../Datasets/'
use_mini_dataset = True

batch_size = 64
nb_classes = 10
nb_epoch = 30
num_classes = 10
img_rows, img_cols = 42, 28 # input image dimensions



class CNN(nn.Module):

    def __init__(self, input_dimension):
        super(CNN, self).__init__()

        self.conv_size1 = 32
        self.conv_size2 = 64
        self.linear_size = 128

        self.conv2d1 = nn.Conv2d(1, self.conv_size1, (3, 5))
        self.conv2d2 = nn.Conv2d(
            self.conv_size1, self.conv_size2, (3, 5)
        )

        self.pool2d1 = nn.MaxPool2d((3, 3))
        self.pool2d2 = nn.MaxPool2d((3, 3))

        self.linear1 = nn.Linear(192, self.linear_size)
        self.linear2 = nn.Linear(self.linear_size, 20)

        self.flatten = Flatten()
        self.ReLU = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        xf = self.conv2d1(x)
        xf = self.ReLU(xf)
        xf = self.pool2d1(xf)

        xf = self.conv2d2(xf)
        xf = self.ReLU(xf)
        xf = self.pool2d2(xf)

        xf = self.flatten(xf)

        xf = self.linear1(xf)
        xf = self.dropout(xf)

        xf = self.linear2(xf)

        out_first_digit = xf[:, :10]
        out_second_digit = xf[:, 10:]

        return out_first_digit, out_second_digit


def main():
    X_train, y_train, X_test, y_test = U.get_data(path_to_data_dir, use_mini_dataset)

    # Split into train and dev
    dev_split_index = int(9 * len(X_train) / 10)
    X_dev = X_train[dev_split_index:]
    y_dev = [y_train[0][dev_split_index:], y_train[1][dev_split_index:]]
    X_train = X_train[:dev_split_index]
    y_train = [y_train[0][:dev_split_index], y_train[1][:dev_split_index]]

    permutation = np.array([i for i in range(len(X_train))])
    np.random.shuffle(permutation)
    X_train = [X_train[i] for i in permutation]
    y_train = [[y_train[0][i] for i in permutation], [y_train[1][i] for i in permutation]]

    # Split dataset into batches
    train_batches = batchify_data(X_train, y_train, batch_size)
    dev_batches = batchify_data(X_dev, y_dev, batch_size)
    test_batches = batchify_data(X_test, y_test, batch_size)

    # Load model
    input_dimension = img_rows * img_cols
    model = CNN(input_dimension) # TODO add proper layers to CNN class above

    # Train
    train_model(train_batches, dev_batches, model, nesterov=True, n_epochs=14)

    ## Evaluate the model on test data
    loss, acc = run_epoch(test_batches, model.eval(), None)
    print('Test loss1: {:.6f}  accuracy1: {:.6f}  loss2: {:.6f}   accuracy2: {:.6f}'.format(loss[0], acc[0], loss[1], acc[1]))

if __name__ == '__main__':
    # Specify seed for deterministic behavior, then shuffle. Do not change seed for official submissions to edx
    np.random.seed(12321)  # for reproducibility
    torch.manual_seed(12321)  # for reproducibility
    main()
