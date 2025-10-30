import jax
import jax.numpy as jnp
from flax import nnx
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
params, state = nnx.split(model)
opt_state = optimizer.init(params)
key = jax.random.PRNGKey(0)

def lossdef(params, state, batchX, key):
    """
        Fonction de loss pour l'entrainement du modèle. On se colle à la definition de la loss dans le papier Flow Matching.
    """
    x1 = batchX
    key, key_x0 = jax.random.split(key)
    key, key_t = jax.random.split(key)
    x0 = jax.random.normal(key_x0, x1.shape)
    t = jax.random.uniform(key_t, (1, 1, 1, 1))
    sigma_min = 0.002 #Je l'ai pris un peu arbitrairement 
    model = nnx.merge(params, state)
    
    #Equation 20
    mu_t_x1 = t * x1
    sigma_t_x1 = 1 - (1 - sigma_min) * t 
    #Equation 11
    trident_t_x0 = sigma_t_x1 * x0 + mu_t_x1
    #Equations 22 et 23
    v_trident_t_xo = model(trident_t_x0, t)
    u_t_trident_t_x0  = x1 - (1 - sigma_min)*x0    
    jax.debug.print("v_trident_t_xo shape:", v_trident_t_xo.shape)
    jax.debug.print("u_t_trident_t_x0 shape:", u_t_trident_t_x0.shape)
    
    loss = jnp.mean((v_trident_t_xo - u_t_trident_t_x0) ** 2)
    return loss, key

#Boucle d'entrainement    
@jax.jit
def train_step(carry, batchX) :
        params, state, opt_state, running_loss, k, key = carry["params"], carry["state"], carry["opt_state"], carry["running_loss"], carry["k"], carry["key"]
        (loss,key), grads = jax.value_and_grad(lossdef, has_aux=True)(params, state, batchX, key)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return {"params": params, "state": state, "opt_state": opt_state, "running_loss": running_loss + loss, "k": k + 1, "key": key}, loss

@jax.jit
def train_epoch(params, state, opt_state, batchesX, key):   
    init = {"params": params, "state": state,"opt_state": opt_state, "running_loss": jnp.array(0.0), "k": jnp.array(0), "key": key}
    carry, _ = jax.lax.scan(train_step, init, batchesX)
    running_loss, total = carry["running_loss"], carry["k"]
    average_loss = running_loss / total
    return carry["params"], carry["opt_state"], average_loss, carry["key"]

@partial(jax.jit, static_argnames=('num_epochs',))
def train(params, state, opt_state, X_train, num_epochs: int, key):
    def epoch_step(carry, X):
        params, state, opt_state, batchesX, key = carry["params"], carry["state"], carry["opt_state"], carry["X_train"], carry["key"]
        params, opt_state, average_loss, key = train_epoch(params, state, opt_state, batchesX, key)
        jax.debug.print("Epoch loss = {l}", l=average_loss)
        return {"params": params, "state": state, "opt_state": opt_state, "X_train": batchesX, "key": key}, average_loss
    init = {"params": params, "state": state, "opt_state": opt_state, "X_train": X_train, "key": key}
    carry, train_loss = jax.lax.scan(epoch_step, init, None, length=num_epochs)
    params, opt_state = carry["params"], carry["opt_state"]
    return params, opt_state, train_loss


#Et on lance l'entrainement
num_epochs = 10
train(params, state, opt_state, X_train, num_epochs, key)