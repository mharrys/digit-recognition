# Digit Recognition

Logistic regression to classify digits with a convolutional neural network
(CNN) or a multilayer perceptron (MLP) using Torch7.

![demo](https://github.com/mharrys/digit-recognition/raw/master/demo.gif)

# Dependencies

Training and using the model depends on [MNIST Dataset for
Torch](https://github.com/wb14123/mnist) and
[Net-Toolkit](https://github.com/Atcold/net-toolkit).

The GUI requires CMake, Qt5 and Lua development files to build.

# Building the GUI

Execute the following commands from the root directory:

    $ mkdir build
    $ cd build
    $ cmake ..
    $ make

# Testing the model with the GUI

The network must first be trained and tested. From the root directory, execute
the following command to see the options:

    $ cd net
    $ th train.lua --help

Every option has a default value set, but for demonstration let us train a
convolutional neural network with the following command:

    $ th train.lua -model cnn -eta 0.1 -bsize 5 -epochs 4

At current time and date the model will be saved as
`cnn_model-160214_151433.t7` where the numbers denotes the date and time when
the model was created.

The model can be tested interactively in the GUI with the following command:

    $ ../build/gui/gui cnn_model-160214_151433.t7

# Observations

Getting good performance on the test data is easy but living up to what the
user expects when using the GUI is more difficult. The MLP model is very
sensitive to the size and placement of the digit, and while the CNN model is
more robust against such affine transformations it can still show
difficulties. One possible solution is most likely to apply random rotations,
scaling and translations to the training data and attempt to center and scale
the digit drawn by the user before running the classifier.
