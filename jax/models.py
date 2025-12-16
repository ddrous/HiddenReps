import jax
import jax.numpy as jnp
import equinox as eqx

class Autoencoder(eqx.Module):
    encoder: eqx.nn.Sequential
    decoder: eqx.nn.Sequential
    input_dim: int
    latent_dim: int

    def __init__(self, input_dim, latent_dim, key):
        key1, key2 = jax.random.split(key)
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = eqx.nn.Sequential([
            eqx.nn.Linear(input_dim, 64, key=key1),
            eqx.nn.Lambda(jax.nn.relu),
            eqx.nn.Linear(64, 32, key=jax.random.split(key1)[0]), 
            eqx.nn.Lambda(jax.nn.relu),
            eqx.nn.Linear(32, latent_dim, key=jax.random.split(key1)[1])
        ])

        self.decoder = eqx.nn.Sequential([
            eqx.nn.Linear(latent_dim, 32, key=key2),
            eqx.nn.Lambda(jax.nn.relu),
            eqx.nn.Linear(32, 64, key=jax.random.split(key2)[0]),
            eqx.nn.Lambda(jax.nn.relu),
            eqx.nn.Linear(64, input_dim, key=jax.random.split(key2)[1])
        ])

    def __call__(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def encode(self, x):
        return self.encoder(x)
