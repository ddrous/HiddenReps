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




class AutoencoderSigmoid(pl.LightningModule):
    def __init__(self, input_dim: int, latent_dim: int, lr: float = 1e-3):
        """
        Autoencoder with a sigmoid-masked latent code for intrinsic dimensionality exploration.

        Args:
            input_dim (int): Dimension of the input data (3 for the spiral data).
            latent_dim (int): Maximum latent dimension (e.g., 2 or 3).
            lr (float): Learning rate for the optimizer.
        """
        super().__init__()
        # Save all arguments as hyperparameters for checkpointing and logging
        self.save_hyperparameters()
        
        # 1. Encoder: Maps input_dim (3) to latent_dim (e.g., 3)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim) # Output: latent code z
        )
        
        # 2. Decoder: Maps latent_dim (e.g., 3) back to input_dim (3)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim) # Output: reconstructed data x_hat
        )
        
        # 3. Sigmoid Drop-off Parameter (The key for tuning ID)
        # We want the drop-off position, normalized [0, 1]. Initialized to 0.5.
        # self.sigparams = nn.Parameter(torch.tensor(0.5)) 
        self.sigparams = nn.Parameter(torch.tensor(0.0)) 
        
        # Loss and Optimizer
        self.criterion = nn.MSELoss()

    def forward(self, x):
        z = self.encoder(x)
        
        # --- Sigmoid Masking Logic ---
        latent_dim = self.hparams.latent_dim
        
        # Create a linear tensor from 0 to latent_dim-1 (position along the latent vector)
        sigpos = torch.linspace(0, latent_dim - 1, steps=latent_dim).to(z.device)
        
        # sigdrop scales the normalized self.sigparams (e.g., 0.5) to the actual dimension index (e.g., 1.5)
        sigdrop_index = self.sigparams * latent_dim
        
        # Sigmoid function (1 - sigmoid gives the drop-off shape)
        # Steepness fixed at 10 (can be a hyperparameter if needed)
        sig_mask = 1.0 - torch.sigmoid(100 * (sigpos - sigdrop_index)) 
        
        # Apply the mask: only the first few latent dimensions will be active
        z_masked = z * sig_mask 

        x_hat = self.decoder(z_masked)
        
        return x_hat, z, z_masked # Return all three for potential analysis

    def _shared_step(self, batch):
        x, y = batch # x is input, y is target (x for reconstruction)
        x_hat, z, z_masked = self(x)
        
        # Reconstruction Loss (L_rec)
        loss_rec = self.criterion(x_hat, y)
        
        # Regularization Loss (L_reg): Encourages the sigmoid drop-off parameter to be small
        # This penalizes the model for using all available latent dimensions.
        # We penalize using a small value of the squared normalized drop-off position.
        loss_reg = 0.1 * (self.sigparams)**2
        
        # Total Loss
        loss = loss_rec + loss_reg
        
        return loss, loss_rec, self.sigparams.item()

    def training_step(self, batch, batch_idx):
        loss, loss_rec, sig_val = self._shared_step(batch)
        self.log_dict({'train_loss': loss, 'train_loss_rec': loss_rec, 'sig_param': sig_val}, 
                      prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_rec, sig_val = self._shared_step(batch)
        self.log_dict({'val_loss': loss, 'val_loss_rec': loss_rec, 'sig_param': sig_val}, 
                      prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)




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
