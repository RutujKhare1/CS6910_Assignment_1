# Deep-Learning  Assignment 1
In this project, a neural network is implemented from scratch in Python and wandb is used to log metrics like accuracy and losses. You can build neural network, train it using the mnist or fashion mnist dataset with various parameter configurations.

### Dependencies
 - python
 - wandb library
 - numpy library
 - keras library
 - matplotlib

### Usage

 The table below displays various possible arguments and their corresponding values for the different parameters

 | Name | Default Value | Description |
| :---: | :-------------: | :----------- |
| `-wp`, `--wandb_project` | cs22m074-dl-a1 | Project name used to track experiments in Weights & Biases dashboard |
| `-we`, `--wandb_entity` | team_exe  | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
| `-d`, `--dataset` | fashion_mnist | choices:  ["mnist", "fashion_mnist"] |
| `-e`, `--epochs` | 15 |  Number of epochs to train neural network.|
| `-b`, `--batch_size` | 64 | Batch size used to train neural network. | 
| `-l`, `--loss` | crossEntropy | choices:  ["MSE", "crossEntropy"] |
| `-o`, `--optimizer` | adam | choices:  ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"] | 
| `-lr`, `--learning_rate` | 0.001 | Learning rate used to optimize model parameters | 
| `-m`, `--momentum` | 0.9 | Momentum used by momentum and nag optimizers. |
| `-beta`, `--beta` | 0.9 | Beta used by rmsprop optimizer | 
| `-beta1`, `--beta1` | 0.9 | Beta1 used by adam and nadam optimizers. | 
| `-beta2`, `--beta2` | 0.99 | Beta2 used by adam and nadam optimizers. |
| `-eps`, `--epsilon` | 0.000001 | Epsilon used by optimizers. |
| `-w_d`, `--weight_decay` | 0.0001 | Weight decay used by optimizers. |
| `-w_i`, `--weight_init` | xavier | choices:  ["random", "xavier"] | 
| `-nhl`, `--num_layers` | 4 | Number of hidden layers used in feedforward neural network. | 
| `-sz`, `--hidden_size` | 128 | Number of hidden neurons in a feedforward layer. |
| `-a`, `--activation` | tanh | choices:  ["identity", "sigmoid", "tanh", "relu"] |


Visit `https://api.wandb.ai/links/team_exe/c5uytpu7` to view all the sweep information for selecting the hyperparameters, runs, sample photos, and accompanying visualisations as it contains the whole report.
You can refer to the python notebook for detailed implementation of code along with all of the questions that were given in assignment. 
