A convolutional neural network which is trained on data collected from my mouse finding program. The mouse finding program involves generating a random matrix of 0's, 1's, 2's and 3's. The grid represents the layout of a ship, wiht 0's representing open cells, 1's representing closed off cells, 2 representing the bot and 3 representing the mouse. The object of the program is for the bot to make the right moves in order to find the mouse based on sensor data. The bot senses by receiving a response, the probability of which depends on the bot's proximity to the mouse. A grid representing the probability of the mouse occupying each cell of the ship is maintained based off of this data, and the bot travels towards the cell with the highest probability before sensing again. The goal of the torch AI approach to this problem was to train a model to find the mouse based off of data received from the bot program. 
My input consists of two tensors, x_train and y_train. x_train is structured as a 6000 x 3 x
40 x 40 tensor. Each data point consists of three planes; one for the layout of the ship as
represented by zeros and ones, one for the probability grid of the ship, and one for the
position of the bot in the ship, as represented as the number 1. y_train is a 6000 x 1
tensor with five possible values for each data point, each representing the action my bot
took when it was in the state described by the corresponding input data point. The output
of my model is a (batch_size) x 5 vector of logits, with each row representing a different
input data point. Each column represents the likelihood that the bot will make the action
associated with that index given the input state. For the indices, 0 represents moving up,
1 represents moving down, 2 represents moving left, 3 represents moving right, and 4
represents sensing.
My model consists of three convolution layers and one dense linear layer. In the forward
function, I apply the convolutional layers in order to the input tensor and apply Tanh as
an activation function each time. I then flatten the input and apply the dense linear layer,
which results in a logits array. I then traverse each data point in the input tensor and
search for adjacent blocked cells, setting the respective logits for these actions to
negative infinity.

Files: 
TorchGenerator.ipynb: My code for gathering input tensors from bot 3. Their outputs are saved
as the pt files included in the zip folder, which are to be used in the
MouseFinder.ipynb notebook.
MouseFinder.ipynb: The code for my model, as well as its training.
