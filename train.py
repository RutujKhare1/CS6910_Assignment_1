from keras.datasets import fashion_mnist
from keras.datasets import mnist
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import main as Main
import argparse
import wandb


parser = argparse.ArgumentParser()

parser.add_argument('-wp','--wandb_project',default='cs22m074-dl-project',
                help='Project name used to track experiments in Weights & Biases dashboard')
parser.add_argument('-we','--wandb_entity',default='team_exe',
                help='wandb entity name')
parser.add_argument('-d','--dataset',default='fashion_mnist',choices=['fashion_mnist','mnist'],
                help='Dataset to be used - choices: ["mnist", "fashion_mnist"]')
parser.add_argument('-e','--epochs',default=15,type=int,
                help='Number of epochs to train neural network')
parser.add_argument('-b','--batch_size',default=64, type=int,
                help='Batch size used to train neural network')
parser.add_argument('-eps','--epsilon',default=1e-8, type=float,
                help='Epsilon used by optimizers')
parser.add_argument('-w_d','--weight_decay',default=0.0001, type=float,
                help='Weight decay used by optimizers')
parser.add_argument('-w_i','--weight_init',default='xavier', choices = ["random", "xavier"],
                help='choices: ["random", "xavier"]')
parser.add_argument('-nhl','--num_layers',default=4, type=int,
                help='Number of hidden layers used in feedforward neural network')
parser.add_argument('-sz','--hidden_size',default=128, type=int,
                help='Number of hidden neurons in a feedforward layer')
parser.add_argument('-a','--activation',default='tanh', choices = ["sigmoid", "tanh", "relu"],
                help='choices: ["sigmoid", "tanh", "relu"]')
parser.add_argument('-l','--loss',default='crossEntropy',choices = ["MSE", "crossEntropy"],
                help='choices: ["MSE", "crossEntropy"]')
parser.add_argument('-o','--optimizer',default='adam', choices = ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"],
                help='choices: ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]')
parser.add_argument('-lr','--learning_rate',default=0.001, type=float,
                help='Learning rate used to optimize model parameters')
parser.add_argument('-m','--momentum',default=0.9, type=float,
                help='Momentum used by momentum and nag optimizers')
parser.add_argument('-beta','--beta',default=0.9, type=float,
                help='Beta used by rmsprop optimizer')
parser.add_argument('-beta1','--beta1',default=0.9, type=float,
                help='Beta1 used by adam and nadam optimizers.')
parser.add_argument('-beta2','--beta2',default=0.999, type=float,
                help='Beta2 used by adam and nadam optimizers')

args = parser.parse_args()


num_layers = args.num_layers
hidden_size = args.hidden_size
activation = args.activation
epochs = args.epochs
optimizer = args.optimizer
neta = args.learning_rate
batch_size = args.batch_size
weight_init = args.weight_init
weight_decay = args.weight_decay
loss = args.loss

wandb.login(key = '5b3ff6cba361172038b8948f6dace9286a5bbfa0')
wandb.init(project=args.wandb_project, entity = args.wandb_entity)
wandb.run.name = f'hln_{num_layers}_hls_{hidden_size}_op_{optimizer}_hla_{activation}_lr_{neta}_ep_{epochs}_bs_{batch_size}_winit_{weight_init}_wtdecay_{weight_decay}_lossfn_{loss}'


if(args.dataset == 'fashion_mnist'):
	(X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
else:
	(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
num_label = 10
num_samples = len(X_train)

X_mini = np.reshape(X_train, (num_samples,784))/255.0
num_data_points = num_samples
num_labels = 10
num_val_points = num_data_points//10
num_train_points = num_data_points - num_val_points
Xtrain = X_mini[:num_train_points]
Ytrain = Y_train[:num_train_points]
X_valid = X_mini[num_train_points:]
Y_valid =  Y_train[num_train_points:]
train_loss = list() 
train_acc = list()
val_loss = list()
val_acc = list()

F = Main.feedforwardNeuralNetwork(neta = neta, hidden_size = hidden_size, num_layers = num_layers, X_train = Xtrain, Y_train = Ytrain, X_val = X_valid, Y_val = Y_valid, activation=activation, batch = batch_size, epochs = epochs, wt_decay = weight_decay, loss_fn = loss)
if(args.optimizer == 'sgd'):
	train_loss, train_acc, val_loss, val_acc = F.stochasticGD()

elif (args.optimizer == 'momentum'):
	train_loss, train_acc, val_loss, val_acc = F.momentumGD(beta = args.momentum)

elif (args.optimizer == 'nag'):
	train_loss, train_acc, val_loss, val_acc = F.nagGD(beta = args.momentum)

elif (args.optimizer == 'rmsprop'):
	train_loss, train_acc, val_loss, val_acc = F.RMSprop(beta = args.beta, eps = args.epsilon)

elif (args.optimizer == 'adam'):
	train_loss, train_acc, val_loss, val_acc = F.adam(beta1 = args.beta1, beta2 = args.beta2, eps = args.epsilon)
elif (args.optimizer == 'nadam'):
	train_loss, train_acc, val_loss, val_acc = F.nAdam(beta1 = args.beta1, beta2 = args.beta2, eps = args.epsilon)
else:
	print("Invalid Optimizer")
	exit()

Xtest = np.reshape(X_test, (len(X_test),784))/255
y_hat = F.feedForward(Xtest)
print("Test Accuracy : {}".format(F.calculateAccuracy(y_hat, Y_test)))

for i in range(len(train_loss)):
    wandb.log({'training_loss': train_loss[i],
              'training_accuracy': train_acc[i],
              'validation_loss': val_loss[i],
              'validation_accuracy': val_acc[i]
              })
