import torch
from torch import nn
import lightning as L
import torch.nn.functional as F

    
class MNISTModel(L.LightningModule):
    def __init__(self, class_dict, input_size: int=784, hidden_dim: int=256, output_size: int=10):
        super().__init__()
        self.save_hyperparameters()

        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        self.class_dict = class_dict

        self.fc1 = nn.Linear(input_size, hidden_dim*2, bias=True)
        self.fc2 = nn.Linear(hidden_dim*2, hidden_dim, bias=True)
        self.fc3 = nn.Linear(hidden_dim, output_size, bias=True)

        self.learning_rate = 1e-3

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.softmax(x, dim=1)

        return x

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        y_hat = self.forward(x)
        train_loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", train_loss)
        return train_loss
    
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        y_hat = self.forward(x)
        val_loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", val_loss)
        return val_loss
    
    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        y_hat = self.forward(x)
        test_loss = F.cross_entropy(y_hat, y)
        self.log("test_loss", test_loss)
        return test_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer