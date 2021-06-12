# Neural networks practice

Following Michael Nielsen's amazingly accessible (and free) online book *[Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)*, and 3Blue1Brown's brilliant (and also free) YouTube playlist *[Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)*.

I will be writing a neural network classifier for the [MNIST data set](http://yann.lecun.com/exdb/mnist/).

- [x] Created a decoder class for the MNIST dataset files (`DataLoader.py`).
    - The dataset files are in a particular binary format (described on their website). In order to read the images you have to read the bytes in order with different offsets as specified by the website. Read the `DataLoader.py` code for more information on how to do this. 
