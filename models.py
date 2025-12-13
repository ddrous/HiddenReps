import torch
import torch.nn as nn
import pytorch_lightning as pl

class RNNRegressor(pl.LightningModule):
    def __init__(self, rnn_type, input_size, hidden_size, output_size=1, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        # dynamic model selection
        rnn_cls = {'RNN': nn.RNN, 'LSTM': nn.LSTM, 'GRU': nn.GRU}[rnn_type]
        self.rnn = rnn_cls(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        # out: (batch, seq, hidden), h_n: (num_layers, batch, hidden)
        out, _ = self.rnn(x) 
        # Extract last time step for prediction
        last_out = out[:, -1, :] 
        return self.fc(last_out), out # Return full hidden sequence for analysis

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    


class RNNRegressorSigmoid(pl.LightningModule):
    def __init__(self, rnn_type, input_size, hidden_size, output_size=1, lr=1e-4):
        """ The intiial idea is to sigmoids applied to each hidden state, gradually extending it as the tranning evolves. 
        In this simple case, let use a single sigmoid on the last hidden state !
        """
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        # dynamic model selection
        rnn_cls = {'RNN': nn.RNN, 'LSTM': nn.LSTM, 'GRU': nn.GRU}[rnn_type]
        self.rnn = rnn_cls(input_size, hidden_size, batch_first=True)
        self.sigparams = nn.Parameter(torch.zeros((1,)))  # Actually, just one param, we want absolute steepness
        self.fc = nn.Linear(hidden_size, output_size)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        # out: (batch, seq, hidden), h_n: (num_layers, batch, hidden)
        out, _ = self.rnn(x) 

        # Extract last time step for prediction
        _, seq_len, hidden_dim = out.shape
        sigpos = torch.linspace(0, hidden_dim-1, steps=hidden_dim).to(out.device)
        sigdrop = self.sigparams[0] * hidden_dim
        sig_fun = 1.0 - torch.sigmoid(10 * (sigpos - sigdrop))  # Steepness fixed at 100 for now

        last_out = out[:, -1, :] * sig_fun  # Apply sigmoid mask 
        return self.fc(last_out), out # Return full hidden sequence for analysis

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self(x)
        # loss = self.criterion(y_hat, y)
        loss = self.criterion(y_hat, y) + 0.1 * self.sigparams[0]**2  # Regularize to encourage smaller hidden usage
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    




#%% Plot this sigmoid

if __name__ == "__main__":

    def inverted_sigmoid(
        x,
        x_start=0.0,        # starts at 0
        x_end=50.0,         # ends at 50
        x_drop=20.0,        # shoots down around 20
        y_start=1.0,        # value at x=0
        y_end=0.0,          # value at x=50
        steepness=1.0       # controls how sharp the drop is
    ):
        """
        Inverted sigmoid spanning [x_start, x_end]
        """
        return y_end + (y_start - y_end) * (
            1.0 - torch.sigmoid(steepness * (x - x_drop))
        )

    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    data = torch.linspace(0, 50, steps=500)

    plt.figure(figsize=(8, 4))
    plt.plot(data.numpy(), inverted_sigmoid(data, x_drop=15.0, steepness=0.5).numpy(), label='Steepness=0.5, Drop=15')
    plt.plot(data.numpy(), inverted_sigmoid(data, x_drop=25.0, steepness=1.0).numpy(), label='Steepness=1.0, Drop=25')
    plt.plot(data.numpy(), inverted_sigmoid(data, x_drop=35.0, steepness=10.0).numpy(), label='Steepness=2.0, Drop=35')
    plt.title('Inverted Sigmoid Function Variations')
    plt.xlabel('x')
    plt.ylabel('inverted_sigmoid(x)')
    plt.legend()
    plt.grid(True)
    plt.show()
