import numpy as np
import pandas as pd
import sys

class Fully_Connected_Layer:
    def __init__(self, learning_rate):
        self.InputDim = 784
        self.HiddenDim = 128
        self.OutputDim = 10
        self.learning_rate = learning_rate
        
        '''Weight Initialization'''
        self.W1 = np.random.randn(self.InputDim, self.HiddenDim)
        self.W2 = np.random.randn(self.HiddenDim, self.OutputDim)
        # print('W1',self.W1)
        # print('W2',self.W2)
    
    def sigmoid(self, x):
        return 1 / (1+np.exp(-x))
   
    def sigmoid_derivative(self,x):
        return x * (1-x)
    
    def Forward(self, Input):
        '''Implement forward propagation'''
        self.HiddenLayerInput = np.dot(Input, self.W1)
        self.HiddenLayerOutput = self.sigmoid(self.HiddenLayerInput)
        self.OutputLayerInput = np.dot(self.HiddenLayerOutput, self.W2)
        self.Output = self.sigmoid(self.OutputLayerInput)
        
        return self.Output
    
    def Backward(self, Input, Label, Output,i):
        '''Implement backward propagation'''
        '''Update parameters using gradient descent'''
        error = Label - Output
        loss = np.sum(error) ** 2
        # if i % 5000 == 0 :
        #     print(f'i: {i}, Loss: {loss}')
        output_delta = error * self.sigmoid_derivative(Output)
        hidden_layer_error = output_delta.dot(self.W2.T)
        hidden_layer_delta = hidden_layer_error * self.sigmoid_derivative(self.HiddenLayerOutput)
        self.W2 += self.HiddenLayerOutput.T.dot(output_delta) * self.learning_rate
        self.W1 += Input.T.dot(hidden_layer_delta) * self.learning_rate
        
        
    def Train(self, Input, Label,i):
        Output = self.Forward(Input)
        self.Backward(Input, Label, Output,i)        
        
def one_hot_encoding(labels, num_classes):
    num_classes = len(np.unique(labels))
    encoded_labels = np.eye(num_classes)[labels]
    return encoded_labels

def load_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        data = [list(map(float, line.strip().split(','))) for line in lines]
    return np.array(data)

train_data_path = sys.argv[1]
test_data_path  = sys.argv[2]

train_data = load_data(train_data_path)[:,:-1]
# print(train_data.shape)
test_data = load_data(test_data_path)[:,:-1]

train_labels = load_data(train_data_path)[:, -1].astype(int)
test_labels = load_data(test_data_path)[:, -1].astype(int)

num_classes = 10
train_labels_one_hot = one_hot_encoding(train_labels, num_classes)
test_labels_one_hot = one_hot_encoding(test_labels, num_classes)

# train_data /= 255.0
# test_data /= 255.0

learning_rate = 0.01
iteration = 10000

np.random.seed(1)
'''Construct a fully-connected network'''        
Network = Fully_Connected_Layer(learning_rate)

'''Train the network for the number of iterations'''
'''Implement function to measure the accuracy'''
for i in range(iteration):
    Network.Train(train_data, train_labels_one_hot,i)

def accuracy(predictions, labels):
    predicted_labels = np.argmax(predictions, axis=1)
    # print(predicted_labels)
    # print(labels)
    correct_predictions = np.sum(predicted_labels == labels)
    total_samples = len(predictions)
    acc = correct_predictions / total_samples
    return acc

train_predictions = Network.Forward(train_data)
train_acc = accuracy(train_predictions, train_labels)

test_predictions = Network.Forward(test_data)
test_acc = accuracy(test_predictions, test_labels)

print(train_acc)
print(f'{test_acc}')
print(iteration)
print(learning_rate)


