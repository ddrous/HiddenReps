#%% How to tell if the hidden size of an RNN is sufficient?
import pytorch_lightning as pl
from data import SineWaveDataModule
from models import RNNRegressor
from analysis import analyze_hidden_states, plot_loss_curves, LossHistory
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='ticks')

def run_experiment():
    # Setup Data
    dm = SineWaveDataModule(seq_len=50, batch_size=64)
    dm.setup()
    

    ## PLot a few data samples
    x_sample, y_sample = next(iter(dm.train_dataloader()))
    plt.figure(figsize=(10, 4))
    for i in range(5):
        plt.plot(x_sample[i].squeeze().numpy(), label=f'Sample {i+1}')
        plt.scatter(len(x_sample[i]), y_sample[i].item(), c='r', marker='x') # Target point
    plt.title('Sample Sine Wave Sequences with Targets')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

    # Sizes to compare: 1 (Too small), 4 (Likely too small), 32 (Sufficient)
    hidden_sizes = [2, 4, 32, 128, 512] 
    # hidden_sizes = []

    results = {}

    for h_size in hidden_sizes:
        print(f"\n--- Training RNN with Hidden Size: {h_size} ---")
        
        model = RNNRegressor(rnn_type='LSTM', input_size=1, hidden_size=h_size, lr=1e-3)
        
        # Train
        history = LossHistory()
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor='val_loss',  # Name of the logged metric
            mode='min',          # Minimize the loss
            save_top_k=1,        # Keep only the best model
            dirpath='checkpoints/', # Directory to save checkpoints (optional)
            filename='best-model-{epoch:02d}-{val_loss:.2f}' # Optional custom filename
        )

        trainer = pl.Trainer(
            max_epochs=500, 
            callbacks=[history, checkpoint_callback], # <--- Add here
            enable_progress_bar=False, 
            logger=False, 
            # enable_checkpointing=False
        )
        trainer.fit(model, dm)
        
        plot_loss_curves(history, title=f"Training vs Val Loss (Hidden Size {h_size})")

        # Store final val loss
        val_loss = trainer.validate(model, dm, verbose=False)[0]['val_loss']
        results[h_size] = val_loss
        
        # Visualize
        print(f"Final Val Loss: {val_loss:.5f}")
        analyze_hidden_states(model, dm.val_dataloader(), title_suffix=f"(Size {h_size})")

    # Final Loss Comparison
    plt.figure(figsize=(6, 4))
    plt.bar([str(k) for k in results.keys()], results.values(), color='skyblue')
    plt.xlabel('Hidden Size'); plt.ylabel('Validation MSE')
    plt.yscale('log')
    plt.title('Loss vs Hidden Size')
    plt.show()

if __name__ == "__main__":
    run_experiment()
