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
        # z = self.encoder(x)
        # x_hat = self.decoder(z)
        z = eqx.filter_vmap(self.encoder)(x)
        x_hat = eqx.filter_vmap(self.decoder)(z)
        return x_hat, z

    def encode(self, x):
        # return self.encoder(x)
        return eqx.filter_vmap(self.encoder)(x)



class TaylorAutoencoder(eqx.Module):
    """Autoencoder with Taylor expansion."""
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

    def taylor_encode(self, x, x0):
        # Encode x0
        z0, grad_z0 = eqx.filter_jvp(self.encoder, (x0,), (x-x0,))
        return z0 + grad_z0

    def __call__(self, xs):

        ## For each x in xs, get the closest point x0 from training data
        ## Then compute Taylor expansion around x0

        # 1. First we get th distance between xs and all training points
        dists = jnp.linalg.norm(xs[:, None, :] - xs[None, :, :], axis=-1)
        closest_indices = jnp.argmin(dists + jnp.eye(dists.shape[0]) * 1e6, axis=1) # Eye (identity) avoids self-distances
        x0s = xs[closest_indices]

        # # 2. Encode all xs around their x0s
        def taylor_encode(x, x0):
            # Encode x0
            z0, grad_z0 = eqx.filter_jvp(self.encoder, (x0,), (x-x0,))
            return z0 + grad_z0

        zs = eqx.filter_vmap(taylor_encode)(xs, x0s)

        ## 3. Decode zs
        x_hats = eqx.filter_vmap(self.decoder)(zs)

        return x_hats, zs


    def encode(self, xs):
        """ No taylor expansion at inference time """
        return eqx.filter_vmap(self.encoder)(xs)
