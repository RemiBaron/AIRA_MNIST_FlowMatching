#En gros faut qu'ici on puisse utiliser notre modèle entrainé pour générer des images
#Faut utiliser l'algo truc pour transformer du bruit et notre champs de vecteur appris par le modèle en une image
from flax.training import checkpoints
from flax import nnx
import unet
import os

model_dir =  os.path.abspath("./saved_models")
restored = checkpoints.restore_checkpoint(model_dir, target=None)

abstract_model = nnx.eval_shape(lambda: unet.UNet(in_channels=1, time_dim=40, rngs=nnx.Rngs(0)))
params_abs, state_abs = nnx.split(abstract_model)
nnx.replace_by_pure_dict(state_abs, restored)
model = nnx.merge(params_abs, state_abs)

