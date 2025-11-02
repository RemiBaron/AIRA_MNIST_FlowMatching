from flax import nnx
import jax
import jax.numpy as jnp
from functools import partial

class DoubleConvolution(nnx.Module):
    """
        Bloc de deux convolutions avec ajout du time embedding entre les deux convolutions. C'est le bloc que y a dans chaque étape de l'UNet, entre chaque downsample/upsample.
    """
    def __init__(self, in_channels, time_dim, label_dim, rngs: nnx.Rngs):
        self.conv1 = nnx.Sequential(nnx.Conv(in_channels, in_channels, kernel_size=(3,3), padding="SAME", rngs=rngs), nnx.relu)
        self.conv2 = nnx.Sequential(nnx.Conv(in_channels, in_channels, kernel_size=(3,3), padding="SAME", rngs=rngs), nnx.relu)
        self.time_adapter = nnx.Sequential(nnx.Linear(time_dim, time_dim, rngs=rngs), 
                                           nnx.silu,
                                           nnx.Linear(time_dim, in_channels, rngs=rngs))
        self.label_adapter = nnx.Sequential(nnx.Linear(label_dim, label_dim, rngs=rngs),
                                           nnx.silu,
                                           nnx.Linear(label_dim, in_channels, rngs=rngs))

    def __call__(self, x, t, y):
        #Première convolution
        x = self.conv1(x)
        #On ajoute le time embedding
        t = self.time_adapter(t)
        t = t[:, None, None, :]
        x = x + t
        #On ajoute le label embedding
        y = self.label_adapter(y)
        y = y[:, None, None, :]
        x = x + y
        #Deuxième convolution
        x = self.conv2(x)
        return x
        
    
class EncoderBlock(nnx.Module):
    """
        Le bloc de l'encodeur ou downsample. En gros il faut DoubleConvolution puis maxpooler.
    """
    
    def __init__(self, in_channels, time_dim, label_dim, rngs: nnx.Rngs):
        self.proj = nnx.Conv(in_channels, in_channels*2, kernel_size=(1,1), padding="SAME", rngs=rngs)
        self.double_conv = DoubleConvolution(in_channels*2, time_dim, label_dim, rngs=rngs)
        self.maxpool =  partial(nnx.max_pool, window_shape=(2,2), strides=(2,2), padding="SAME"
)
    
    def __call__(self, x, t, y, debug=False):
        #Pour avoir les bonnes dimensions
        x = self.proj(x)
        #D'abord la double convolution
        x = self.double_conv(x, t, y)
        #On garde en mémoire ce résultat, pour la fameuse flèche grise "Copy and crop" du Unet, qui va de l'encodeur au décodeur sans passer par le bottleneck
        copy_and_crop = x 
        #Puis le maxpool
        x = self.maxpool(x)
        if debug:
            print("EncoderBlock output shape:", x.shape)
        return x, copy_and_crop

class Bottleneck(nnx.Module):
    """
        Le bloc du milieu, entre l'encodeur/downsamples et le décodeur/upsamples. C'est juste une DoubleConvolution.
    """
    def __init__(self, in_channels, time_dim, label_dim, rngs: nnx.Rngs):
        self.double_conv = DoubleConvolution(in_channels, time_dim, label_dim, rngs=rngs)
    
    def __call__(self, x, t, y):
        return self.double_conv(x, t, y)

class DecoderBlock(nnx.Module):
    """
        Le bloc du décodeur ou upsample. En gros il faut upsample puis DoubleConvolution.
    """
    
    def __init__(self, in_channels, time_dim, label_dim, rngs: nnx.Rngs):
        self.upsample = nnx.ConvTranspose(in_channels, in_channels//2, kernel_size=(2,2), strides=(2,2), padding="SAME", rngs=rngs)
        self.reduce = nnx.Conv(in_channels+in_channels//2, in_channels//2, kernel_size=(1,1), padding="SAME", rngs=rngs)
        self.double_conv = DoubleConvolution(in_channels//2, time_dim, label_dim, rngs=rngs)
        
    def __call__(self, x, t, y, copy_and_crop,debug=False):
        #D'abord l'upsample
        x = self.upsample(x)
        #On concatène avec le copy_and_crop de l'encodeur
        x = jnp.concatenate([x, copy_and_crop], axis=-1)
        #Pour les dimensions
        x = self.reduce(x)
        #Puis la double convolution
        x = self.double_conv(x, t, y)
        if debug:
            print("DecoderBlock output shape:", x.shape)
        return x

@partial(jax.jit, static_argnums=(1,))
def Fourier_embeding(t, time_dim):
    """
        Embedding de t en utilisant Fourier.
    """
    t = jnp.asarray(t, dtype=jnp.float32)
    if t.ndim == 0:                
        t = t[None]                     
    elif t.ndim > 1:
        t = t.reshape((t.shape[0],))
    k = jnp.arange(time_dim//2, dtype=jnp.float32) 
    cos = jnp.cos((2 * jnp.pi) * t[:, None] * (2.0 ** k)[None, :])
    sin = jnp.sin((2 * jnp.pi) * t[:, None] * (2.0 ** k)[None, :])
    return jnp.concatenate([cos, sin], axis=-1)

class UNet(nnx.Module):
    """
        Le modèle UNet complet, avec encodeur, bottleneck et décodeur.
    """
    def __init__(self, in_channels, time_dim, label_dim, rngs: nnx.Rngs):
        self.y_embedder = nnx.Embed(num_embeddings=11, features=label_dim, rngs=rngs)
        self.stem = nnx.Conv(1, in_channels, (3,3), padding="SAME", rngs=rngs)
        self.enc1 = EncoderBlock(in_channels, time_dim, label_dim, rngs=rngs)
        self.enc2 = EncoderBlock(in_channels*2, time_dim, label_dim, rngs=rngs)
        self.enc3 = EncoderBlock(in_channels*4, time_dim, label_dim, rngs=rngs)
        self.bottleneck = Bottleneck(in_channels*8, time_dim, label_dim, rngs=rngs)
        self.dec1 = DecoderBlock(in_channels*8, time_dim, label_dim, rngs=rngs)
        self.dec2 = DecoderBlock(in_channels*4, time_dim, label_dim, rngs=rngs)
        self.dec3 = DecoderBlock(in_channels*2, time_dim, label_dim, rngs=rngs)  
        self.out = nnx.Conv(in_channels, 1, kernel_size=(1,1), padding="SAME", rngs=rngs)
        self.time_dim = time_dim      
        
    def __call__(self, x, t, y, debug=False):
        #On embed tout d'abord t en time_dim dimensions en utilisant Fourier
        t = Fourier_embeding(t, time_dim=self.time_dim)
        #On embed aussi le label en label_dim dimensions
        y = self.y_embedder(y)    
        #On pad l'image pour éviter les problèmes de dimensions lors des downsample/upsample
        x = jnp.pad(x, ((0,0),(2,2),(2,2),(0,0))) 
        x = self.stem(x) 
        if debug:
            print("Time embedding shape:", t.shape)
        #Encodeur
        x, copy_and_crop1 = self.enc1(x, t, y, debug=debug)
        x, copy_and_crop2 = self.enc2(x, t, y, debug=debug)
        x, copy_and_crop3 = self.enc3(x, t, y, debug=debug)        
        #Bottleneck
        x = self.bottleneck(x, t, y)        
        #Décodeur
        x = self.dec1(x, t, y, copy_and_crop3, debug=debug)
        x = self.dec2(x, t, y, copy_and_crop2, debug=debug)
        x = self.dec3(x, t, y, copy_and_crop1, debug=debug)
        x = self.out(x)
        x = x[:,2:-2,2:-2,:] 
                
        return x


