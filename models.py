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