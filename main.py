import jax
import jax.numpy as jnp
from jax import grad, vmap
import flax.linen as nn
import optax
import wandb
import ml_collections
import scipy.io
import numpy as np
from functools import partial
from flax.training import train_state
import matplotlib.pyplot as plt


# --- MLP model --- #

class MLP(nn.Module):
    hidden_dim: int
    num_layers: int

    @nn.compact
    def __call__(self, x):
        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.tanh(x)
        x = nn.Dense(1)(x)
        return x.squeeze(-1)


# --- PINNs helpers --- #

def net_u(params, model, t, x):
    inputs = jnp.stack([t, x], axis=-1)
    return model.apply(params, inputs)


def pde_residual(params, model, t, x, nu):
    t = jnp.asarray(t, dtype=jnp.float32)
    x = jnp.asarray(x, dtype=jnp.float32)
    u = lambda t, x: net_u(params, model, t, x)
    u_t = grad(u, argnums=0)(t, x)
    u_x = grad(u, argnums=1)(t, x)
    u_xx = grad(grad(u, argnums=1), argnums=1)(t, x)
    return u_t + u(t, x) * u_x - nu * u_xx


def loss_fn(params, model, t_ic, x_ic, u_ic, t_bc, x_bc, u_bc, t_r, x_r, nu):
    u_ic_pred = vmap(lambda t, x: net_u(params, model, t, x))(t_ic, x_ic)
    loss_ic = jnp.mean((u_ic_pred - u_ic) ** 2)

    u_bc_pred = vmap(lambda t, x: net_u(params, model, t, x))(t_bc, x_bc)
    loss_bc = jnp.mean((u_bc_pred - u_bc) ** 2)

    f_pred = vmap(lambda t, x: pde_residual(params, model, t, x, nu))(t_r, x_r)
    loss_r = jnp.mean(f_pred ** 2)

    return loss_ic + loss_bc + loss_r, (loss_ic, loss_bc, loss_r)

@partial(jax.jit, static_argnums=(1,))
def train_step(state, model, batch, nu):
    t_ic, x_ic, u_ic, t_bc, x_bc, u_bc, t_r, x_r = batch
    params = state.params

    (loss, (loss_ic, loss_bc, loss_r)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        params, model, t_ic, x_ic, u_ic, t_bc, x_bc, u_bc, t_r, x_r, nu
    )

    state = state.apply_gradients(grads=grads)
    return state, loss, loss_ic, loss_bc, loss_r


def get_config():
    config = ml_collections.ConfigDict()
    config.seed = 42
    config.nu = 0.01 / jnp.pi
    config.hidden_dim = 64
    config.num_layers = 4
    config.learning_rate = 1e-3
    config.max_steps = 10000
    config.batch_size = 10000
    config.logging_interval = 500

    config.wandb = ml_collections.ConfigDict()
    config.wandb.project = "PINN-Burgers"
    config.wandb.name = "jax-flax"

    return config


def train(config):
    # Load data
    data = scipy.io.loadmat("burgers.mat")
    u_ref = data["usol"]
    x_star = data["x"].squeeze()
    t_star = data["t"].squeeze()
    u0 = u_ref[0, :]

    # Initial condition points
    t_ic = jnp.zeros_like(x_star, dtype=jnp.float32)
    x_ic = jnp.array(x_star, dtype=jnp.float32)
    u_ic = jnp.array(u0, dtype=jnp.float32)

    # Boundary condition points
    t_bc = jnp.array(np.random.uniform(0, 1, size=100), dtype=jnp.float32)
    x_bc = jnp.array(np.random.choice([-1.0, 1.0], size=100), dtype=jnp.float32)
    u_bc = jnp.zeros_like(x_bc, dtype=jnp.float32)

    # Collocation points
    t_r = jnp.array(np.random.uniform(0, 1, size=config.batch_size), dtype=jnp.float32)
    x_r = jnp.array(np.random.uniform(-1, 1, size=config.batch_size), dtype=jnp.float32)

    # Init model and params
    model = MLP(hidden_dim=config.hidden_dim, num_layers=config.num_layers)
    rng = jax.random.PRNGKey(config.seed)
    dummy_input = jnp.ones((1, 2))
    params = model.init(rng, dummy_input)

    # Create TrainState
    tx = optax.adam(config.learning_rate)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    # Init wandb
    wandb.init(project=config.wandb.project, name=config.wandb.name, config=config)

    # Training loop
    batch = (t_ic, x_ic, u_ic, t_bc, x_bc, u_bc, t_r, x_r)
    for step in range(config.max_steps):
      state, loss, loss_ic, loss_bc, loss_r = train_step(state, model, batch, config.nu)

      if step % config.logging_interval == 0:
          print(
              f"Step {step}: total = {loss:.3e}, "
              f"IC = {loss_ic:.3e}, BC = {loss_bc:.3e}, Residual = {loss_r:.3e}"
          )
          wandb.log({
              "loss": float(loss),
              "loss_ic": float(loss_ic),
              "loss_bc": float(loss_bc),
              "loss_r": float(loss_r),
          }, step=step)

    return state.params, model, (t_star, x_star, u_ref)


# --- Plotting --- #

def plot_results(params, model, t_star, x_star, u_ref):
    T, X = jnp.meshgrid(t_star, x_star, indexing="ij")
    t_flat = T.flatten()
    x_flat = X.flatten()

    @jax.jit
    def predict(t, x):
        inputs = jnp.stack([t, x], axis=-1)
        return model.apply(params, inputs)

    u_pred_flat = vmap(predict)(t_flat, x_flat)
    u_pred = u_pred_flat.reshape(T.shape)

    error = jnp.abs(u_pred - u_ref)

    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    cmap = "viridis"

    im0 = axs[0].imshow(u_ref, extent=[x_star.min(), x_star.max(), t_star.max(), t_star.min()],
                        aspect='auto', cmap=cmap)
    axs[0].set_title("True Solution $u(t,x)$")
    plt.colorbar(im0, ax=axs[0])

    im1 = axs[1].imshow(u_pred, extent=[x_star.min(), x_star.max(), t_star.max(), t_star.min()],
                        aspect='auto', cmap=cmap)
    axs[1].set_title("Predicted Solution $\hat{u}(t,x)$")
    plt.colorbar(im1, ax=axs[1])

    im2 = axs[2].imshow(error, extent=[x_star.min(), x_star.max(), t_star.max(), t_star.min()],
                        aspect='auto', cmap="inferno")
    axs[2].set_title("Absolute Error $|u - \hat{u}|$")
    plt.colorbar(im2, ax=axs[2])

    for ax in axs:
        ax.set_xlabel("x")
        ax.set_ylabel("t")

    plt.tight_layout()
    plt.show()


def main():
    config = get_config()
    params, model, (t_star, x_star, u_ref) = train(config)
    plot_results(params, model, t_star, x_star, u_ref)


if __name__ == "__main__":
    main()

from google.colab import files
uploaded = files.upload()