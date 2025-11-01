import os
import jax
import jax.numpy as jnp
from flax import nnx
from flax.training import checkpoints
import shutil
import optax
import unet
import numpy as np
from functools import partial

print("Available devices:", jax.devices())
device = jax.devices()[0]
print("Using device:", device)
if device.platform == 'gpu':
    print("Running on GPU")
else:
    print("Running on CPU")

# Download MNIST
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
print(f"Training samples: {len(train_dataset)}  | Test samples: {len(test_dataset)}")

def loader_to_jax_arrays(loader):
    """
        Convertit un DataLoader PyTorch en arrays JAX.
        On veut pas de python haut niveau en gros pour utiliser jax.jit dans l'entrainement et le rendre ainsi plus performant
    """
    xs, ys = [], []
    for xb, yb in loader:
        x = np.asarray(xb)
        x = np.transpose(x, (0, 2, 3, 1))
        xs.append(x)
        ys.append(np.asarray(yb))
    X = jnp.asarray(np.stack(xs))
    Y = jnp.asarray(np.stack(ys))
    return X, Y

X_train, Y_train = loader_to_jax_arrays(train_loader)
X_test, Y_test = loader_to_jax_arrays(test_loader)

#Definition du modèle
model = unet.UNet(in_channels=1, time_dim=40, rngs=nnx.Rngs(0))
optimizer = optax.adam(learning_rate=1e-3)
optim = nnx.Optimizer(model, optimizer)
key = jax.random.PRNGKey(0)

def lossdef(model, batchX, key):
    """
        Fonction de loss pour l'entrainement du modèle. On se colle à la definition de la loss dans le papier Flow Matching.
    """
    x1 = batchX
    key, key_x0 = jax.random.split(key)
    key, key_t = jax.random.split(key)
    x0 = jax.random.normal(key_x0, x1.shape)
    t = jax.random.uniform(key_t, (x1.shape[0], 1, 1, 1))
    sigma_min = 0.002 #Je l'ai pris un peu arbitrairement 

    #Equation 20
    mu_t_x1 = t * x1
    sigma_t_x1 = 1 - (1 - sigma_min) * t 
    #Equation 11
    trident_t_x0 = sigma_t_x1 * x0 + mu_t_x1
    #Equations 22 et 23
    v_trident_t_xo = model(trident_t_x0, t)
    u_t_trident_t_x0  = x1 - (1 - sigma_min)*x0
    
    loss = jnp.mean((v_trident_t_xo - u_t_trident_t_x0) ** 2)
    return loss, key

#Boucle d'entrainement    
@nnx.jit
def train_step(model, optim, batchX, key) :
    def loss_fn(model):
        return lossdef(model, batchX, key)
    (loss, key), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optim.update(grads)
    return loss, key

num_epochs = 5
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for i in range(X_train.shape[0]):
        batchX = X_train[i]
        loss, key = train_step(model, optim, batchX, key)
        epoch_loss += float(loss)
    print(f"Epoch {epoch+1}: loss={epoch_loss / X_train.shape[0]:.4f}")

#On sauvegarde le modèle entrainé
model_dir =  os.path.abspath("./saved_models")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
params, state = nnx.split(model)
pure = nnx.State.to_pure_dict(state)
checkpoints.save_checkpoint(model_dir, pure, step=0, overwrite=True)