import torch
import torch.nn as nn

class DoubleConvolution(nn.Module):
    #IMPORTANT : todo ajouter le temps comme entrée, probablement en faisant comme le boug du MIT ?
    def __init__(self, in_channels, out_channels):
        super(DoubleConvolution, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
class EncoderBlock(nn.Module):
    #On peut appeler ça soit encoder soit downsample, comme tu veux. En gros il faut DoubleConvolution puis Maxpool
    pass

class DecoderBlock(nn.Module):
    #IDem, soit decoder soit upsample. En gros il faut UpConv(typiquement nn.ConvTranspose2D (ou eux ils font nn.UpSample + nn.Conv2D) je crois) puis DoubleConvolution
    pass
    
class MidcoderBlock(nn.Module):
    #Le bloc du milieu, normalement ils appellent ça le Bottleneck. C'est juste une bête DoubleConvolution normalement mais j'ai pas trop compris (ni trop regardé en vrai) ce qu'ils font dans le repo du MIT.
    pass

class UNet(nn.Module):
    #EncoderBlockS + MidcoderBlock + DecoderBlockS
    pass