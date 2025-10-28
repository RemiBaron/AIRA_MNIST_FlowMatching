from flax import nnx
import jax.numpy as jnp
from functools import partial

class DoubleConvolution(nnx.Module):
    """
        Bloc de deux convolutions avec ajout du time embedding entre les deux convolutions. C'est le bloc que y a dans chaque étape de l'UNet, entre chaque downsample/upsample.
    """
    def __init__(self, in_channels, time_dim, rngs: nnx.Rngs):
        self.conv1 = nnx.Sequential(nnx.Conv(in_channels, in_channels, kernel_size=(3,3), padding="SAME", rngs=rngs), nnx.relu)
        self.conv2 = nnx.Sequential(nnx.Conv(in_channels, in_channels, kernel_size=(3,3), padding="SAME", rngs=rngs), nnx.relu)
        self.time_adapter = nnx.Sequential(nnx.Linear(time_dim, time_dim, rngs=rngs), 
                                           nnx.silu,
                                           nnx.Linear(time_dim, in_channels, rngs=rngs))

    def __call__(self, x, t):
        #Première convolution
        x = self.conv1(x)
        #On ajoute le time embedding
        t = self.time_adapter(t)
        t = t[:, None, None, :]
        x = x + t
        #Deuxième convolution
        x = self.conv2(x)
        return x
        
    
class EncoderBlock(nnx.Module):
    """
        Le bloc de l'encodeur ou downsample. En gros il faut DoubleConvolution puis maxpooler.
    """
    
    def __init__(self, in_channels, time_dim, rngs: nnx.Rngs):
        self.proj = nnx.Conv(in_channels, in_channels*2, kernel_size=(1,1), padding="SAME", rngs=rngs)
        self.double_conv = DoubleConvolution(in_channels*2, time_dim, rngs=rngs)
        self.maxpool =  partial(nnx.max_pool, window_shape=(2,2), strides=(2,2), padding="SAME"
)
    
    def __call__(self, x, t, debug=False):
        #Pour avoir les bonnes dimensions
        x = self.proj(x)
        #D'abord la double convolution
        x = self.double_conv(x, t)
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
    def __init__(self, in_channels, time_dim, rngs: nnx.Rngs):
        self.double_conv = DoubleConvolution(in_channels, time_dim, rngs=rngs)
    
    def __call__(self, x, t):
        return self.double_conv(x, t)

class DecoderBlock(nnx.Module):
    """
        Le bloc du décodeur ou upsample. En gros il faut upsample puis DoubleConvolution.
    """
    
    def __init__(self, in_channels, time_dim, rngs: nnx.Rngs):
        self.upsample = nnx.ConvTranspose(in_channels, in_channels//2, kernel_size=(2,2), strides=(2,2), padding="SAME", rngs=rngs)
        self.reduce = nnx.Conv(in_channels+in_channels//2, in_channels//2, kernel_size=(1,1),
                               padding="SAME", rngs=rngs)
        self.double_conv = DoubleConvolution(in_channels//2, time_dim, rngs=rngs)
        
    def __call__(self, x, t, copy_and_crop,debug=False):
        #D'abord l'upsample
        x = self.upsample(x)
        #On concatène avec le copy_and_crop de l'encodeur
        x = jnp.concatenate([x, copy_and_crop], axis=-1)
        #Pour les dimensions
        x = self.reduce(x)
        #Puis la double convolution
        x = self.double_conv(x, t)
        if debug:
            print("DecoderBlock output shape:", x.shape)
        return x


class UNet(nnx.Module):
    """
        Le modèle UNet complet, avec encodeur, bottleneck et décodeur.
    """
    def __init__(self, in_channels, time_dim, rngs: nnx.Rngs):
        self.enc1 = EncoderBlock(in_channels, time_dim, rngs=rngs)
        self.enc2 = EncoderBlock(in_channels*2, time_dim, rngs=rngs)
        self.enc3 = EncoderBlock(in_channels*4, time_dim, rngs=rngs)
        self.bottleneck = Bottleneck(in_channels*8, time_dim, rngs=rngs)
        self.dec1 = DecoderBlock(in_channels*8, time_dim, rngs=rngs)
        self.dec2 = DecoderBlock(in_channels*4, time_dim, rngs=rngs)
        self.dec3 = DecoderBlock(in_channels*2, time_dim, rngs=rngs)
        
    def __call__(self, x, t, debug=False):
        #Encodeur
        x, copy_and_crop1 = self.enc1(x, t, debug=debug)
        x, copy_and_crop2 = self.enc2(x, t, debug=debug)
        x, copy_and_crop3 = self.enc3(x, t, debug=debug)
        
        #Bottleneck
        x = self.bottleneck(x, t)
        
        #Décodeur
        x = self.dec1(x, t, copy_and_crop3, debug=debug)
        x = self.dec2(x, t, copy_and_crop2, debug=debug)
        x = self.dec3(x, t, copy_and_crop1, debug=debug)
                
        return x

toto = UNet(in_channels=1, time_dim=10, rngs=nnx.Rngs(0))
x, t = jnp.ones((1,32,32,1)), jnp.ones((1,10))
print("x shape:", x.shape)
print("t shape:", t.shape)
output = toto(x, t, debug=False)
print(output.shape)  # Devrait afficher (1, 32, 32, 1)

