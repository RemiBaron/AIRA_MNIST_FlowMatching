import jax
import jax.numpy as jnp
from flax import nnx
import optax
import unet
import numpy as np
from functools import partial


# Je n'ai pas de gpu nvidia... Aïe aïe aïe.
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
model = unet.UNet(in_channels=32, time_dim=40, rngs=nnx.Rngs(0))
optimizer = optax.adam(learning_rate=1e-3)
params, state = nnx.split(model)
opt_state = optimizer.init(params)

#Boucle d'entrainement
def lossdef(params, batchX):
    x = batchX
    #TODO : blablabla on doit créer la même loss que dans le papier flow matching
    loss = jnp.array(0.0)
    return loss
    
@jax.jit
def train_step(carry, batchX) :
        params, opt_state, running_loss, k = carry["params"], carry["opt_state"], carry["running_loss"], carry["k"]
        loss, grads = jax.value_and_grad(lossdef)(params, batchX)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return {"params": params, "opt_state": opt_state, "running_loss": running_loss + loss, "k": k + 1}, loss
        
        
@jax.jit
def train_epoch(params, opt_state, batchesX):
    
    init = {"params": params, "opt_state": opt_state, "running_loss": jnp.array(0.0), "k": jnp.array(0)}
    carry, _ = jax.lax.scan(train_step, init, batchesX)
    running_loss, total = carry["running_loss"], carry["k"]
    average_loss = running_loss / total
    return carry["params"], carry["opt_state"], average_loss
    

#Easter egg : ABEEEEEEEEEEEEEEEEEEEEEELLLLLLLL C'EST TROP DUR JAX !! GRAAAAAAAAHHHHH >:(
    
@partial(jax.jit, static_argnames=('num_epochs',))
def train(params, opt_state, X_train, num_epochs: int):
    def epoch_step(carry, X):
        params, opt_state, batchesX = carry["params"], carry["opt_state"], carry["X_train"]
        params, opt_state, average_loss = train_epoch(params, opt_state, batchesX)
        jax.debug.print("Epoch loss = {l}", l=average_loss)
        return {"params": params, "opt_state": opt_state, "X_train": batchesX}, average_loss
    init = {"params": params, "opt_state": opt_state, "X_train": X_train}
    carry, train_loss = jax.lax.scan(epoch_step, init, None, length=num_epochs)
    params, opt_state = carry["params"], carry["opt_state"]
    return params, opt_state, train_loss


#Et on lance l'entrainement
num_epochs = 10
train(params, opt_state, X_train, num_epochs)