# transferlearning
This repository explains the implementation of transfer learning technique using resnet 50 v1.

The bottom 4 layers of the residual network resnet 50 and their weights are used for training the neural network to achieve better accuracy and learn complex problems.

This is done with the help of using command Trainable = 'True' for using the weights of the pretrained network and changing their state

And then using the model as a normal neural netowrk with softmax activation function with probability output for 2 states, we can use the argmax function and fetch the label with maximum probability. 
