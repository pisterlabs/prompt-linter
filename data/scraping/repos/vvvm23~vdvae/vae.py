"""
    Encoder Components:
        - Encoder, contains all the EncoderBlocks and manages data flow through them.
        - EncoderBlock, contains sub-blocks of residual units and a pooling layer.
        - ResidualBlock, contains a block of residual connections, as described in the paper (1x1,3x3,3x3,1x1)
            - We could slightly adapt, and make it a ReZero connection. Needs some testing.
        
    Decoder Components:
        - Decoder, contains all DecoderBlocks and manages data flow through them.
        - DecoderBlock, contains sub-blocks of top-down units and an unpool layer.
        - TopDownBlock, implements the topdown block from the original paper.

    All is encapsulated in the main VAE class.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

"""
    Some helper functions for common constructs
"""
class ConvBuilder:
    def _bconv(in_dim, out_dim, kernel_size, stride, padding):
        conv = nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding)
        return conv
    def b1x1(in_dim, out_dim):
        return ConvBuilder._bconv(in_dim, out_dim, 1, 1, 0)
    def b3x3(in_dim, out_dim):
        return ConvBuilder._bconv(in_dim, out_dim, 3, 1, 1)

"""
    Diagonal Gaussian Distribution and loss.
    Taken directly from OpenAI implementation 
    Decorators means these functions will be compiled as TorchScript
"""
@torch.jit.script
def gaussian_analytical_kl(mu1, mu2, logsigma1, logsigma2):
    return -0.5 + logsigma2 - logsigma1 + 0.5 * (logsigma1.exp() ** 2 + (mu1 - mu2) ** 2) / (logsigma2.exp() ** 2)
@torch.jit.script
def draw_gaussian_diag_samples(mu, logsigma):
    eps = torch.empty_like(mu).normal_(0., 1.)
    return torch.exp(logsigma) * eps + mu

"""
    Helper module to call super().__init__() for us
"""
class HelperModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.build(*args, **kwargs)

    def build(self, *args, **kwargs):
        raise NotImplementedError

"""
    Encoder Components
"""
class ResidualBlock(HelperModule):
    def build(self, in_width, hidden_width, rezero=False): # hidden_width should function as a bottleneck!
        self.conv = nn.ModuleList([
            ConvBuilder.b1x1(in_width, hidden_width),
            ConvBuilder.b3x3(hidden_width, hidden_width),
            ConvBuilder.b3x3(hidden_width, hidden_width),
            ConvBuilder.b1x1(hidden_width, in_width)
        ])
        if rezero:
            self.gate = nn.Parameter(torch.tensor(0.0))
        else:
            self.gate = 1.0

    def forward(self, x):
        xh = x
        for l in self.conv:
            xh = l(F.gelu(xh))
        y = x + self.gate*xh
        return y

class EncoderBlock(HelperModule):
    def build(self, in_dim, middle_width, nb_r_blocks, downscale_rate):
        self.downscale_rate = downscale_rate
        self.res_blocks = nn.ModuleList([
            ResidualBlock(in_dim, middle_width)
        for _ in range(nb_r_blocks)])
        
    def forward(self, x):
        y = x
        for l in self.res_blocks:
            y = l(y)
        a = y
        y = F.avg_pool2d(y, kernel_size=self.downscale_rate, stride=self.downscale_rate)
        return y, a # y is input to next block, a is activations to topdown layer

class Encoder(HelperModule):
    def build(self, in_dim, hidden_width, middle_width, nb_encoder_blocks, nb_res_blocks=3, downscale_rate=2):
        self.in_conv = ConvBuilder.b3x3(in_dim, hidden_width)
        self.enc_blocks = nn.ModuleList([
            EncoderBlock(hidden_width, middle_width, nb_res_blocks, 1 if i==(nb_encoder_blocks-1) else downscale_rate)
        for i in range(nb_encoder_blocks)])

        # TODO: could just pass np.sqrt( ... ) value to EncoderBlock, rather than this weird loop
        # it is the same in every block.
        for be in self.enc_blocks:
            for br in be.res_blocks:
                br.conv[-1].weight.data *= np.sqrt(1 / (nb_encoder_blocks*nb_res_blocks))

    def forward(self, x):
        x = self.in_conv(x)
        activations = [x]
        for b in self.enc_blocks:
            x, a = b(x)
            activations.append(a)
        return activations

"""
    Decoder Components
"""
class Block(HelperModule):
    def build(self, in_width, hidden_width, out_width): # hidden_width should function as a bottleneck!
        self.conv = nn.ModuleList([
            ConvBuilder.b1x1(in_width, hidden_width),
            ConvBuilder.b3x3(hidden_width, hidden_width),
            ConvBuilder.b3x3(hidden_width, hidden_width),
            ConvBuilder.b1x1(hidden_width, out_width)
        ])

    def forward(self, x):
        for l in self.conv:
            x = l(F.gelu(x))
        return x

class TopDownBlock(HelperModule):
    def build(self, in_width, middle_width, z_dim):
        self.cat_conv = Block(in_width*2, middle_width, z_dim*2) # parameterises mean and variance
        self.prior = Block(in_width, middle_width, z_dim*2 + in_width) # parameterises mean, variance and xh
        self.out_res = ResidualBlock(in_width, middle_width)
        self.z_conv = ConvBuilder.b1x1(z_dim, in_width)
        self.z_dim = z_dim

    def forward(self, x, a):
        xa = torch.cat([x,a], dim=1)
        qm, qv = self.cat_conv(xa).chunk(2, dim=1) # Calculate q distribution parameters. Chunk into 2 (first z_dim is mean, second is variance)
        pfeat = self.prior(x)
        pm, pv, px = pfeat[:, :self.z_dim], pfeat[:, self.z_dim:self.z_dim*2], pfeat[:, self.z_dim*2:]
        x = x + px

        z = draw_gaussian_diag_samples(qm, qv)
        kl = gaussian_analytical_kl(qm, pm, qv, pv)

        z = self.z_conv(z)
        x = x + z
        x = self.out_res(x)

        return x, kl

    def sample(self, x):
        pfeat = self.prior(x)
        pm, pv, px = pfeat[:, :self.z_dim], pfeat[:, self.z_dim:self.z_dim*2], pfeat[:, self.z_dim*2:]
        x = x + px

        z = draw_gaussian_diag_samples(pm, pv)

        x = x + self.z_conv(z)
        x = self.out_res(x)
        return x

class DecoderBlock(HelperModule):
    def build(self, in_dim, middle_width, z_dim, nb_td_blocks, upscale_rate):
        self.upscale_rate = upscale_rate
        self.td_blocks = nn.ModuleList([
            TopDownBlock(in_dim, middle_width, z_dim)
        for _ in range(nb_td_blocks)])

    def forward(self, x, a):
        x = F.interpolate(x, scale_factor=self.upscale_rate)
        block_kl = []
        for b in self.td_blocks:
            x, kl = b(x, a)
            block_kl.append(kl)
        return x, block_kl

    def sample(self, x):
        x = F.interpolate(x, scale_factor=self.upscale_rate)
        for b in self.td_blocks:
            x = b.sample(x)
        return x

class Decoder(HelperModule):
    def build(self, in_dim, middle_width, out_dim, z_dim, nb_decoder_blocks, nb_td_blocks=3, upscale_rate=2):
        self.dec_blocks = nn.ModuleList([
            DecoderBlock(in_dim, middle_width, z_dim, nb_td_blocks, 1 if i == 0 else upscale_rate)
         for i in range(nb_decoder_blocks)])
        self.in_dim = in_dim
        self.out_conv = ConvBuilder.b3x3(in_dim, out_dim)

        for bd in self.dec_blocks:
            for bt in bd.td_blocks:
                bt.z_conv.weight.data *= np.sqrt(1 / (nb_decoder_blocks*nb_td_blocks))
                bt.out_res.conv[-1].weight.data *= np.sqrt(1 / (nb_decoder_blocks*nb_td_blocks))

    def forward(self, activations):
        activations = activations[::-1]
        x = None
        decoder_kl = []
        for i, b in enumerate(self.dec_blocks):
            a = activations[i]
            if x == None:
                x = torch.zeros_like(a)
            x, block_kl = b(x, a)
            decoder_kl.extend(block_kl)

        x = self.out_conv(x)
        return x, decoder_kl

    def sample(self, nb_samples):
        x = None
        for b in self.dec_blocks:
            if x == None:
                x = torch.zeros(nb_samples, self.in_dim, 4, 4).to('cuda') # TODO: Variable device and size
            x = b.sample(x)
        x = self.out_conv(x)
        return x

"""
    Main VAE class
"""
class VAE(HelperModule):
    def build(self, in_dim, hidden_width, middle_width, z_dim, nb_blocks=4, nb_res_blocks=3, scale_rate=2):
        self.encoder = Encoder(in_dim, hidden_width, middle_width, nb_blocks, nb_res_blocks=nb_res_blocks, downscale_rate=scale_rate)
        self.decoder = Decoder(hidden_width, middle_width, in_dim, z_dim, nb_blocks, nb_td_blocks=nb_res_blocks, upscale_rate=scale_rate)
    def forward(self, x):
        activations = self.encoder(x)
        y, decoder_kl = self.decoder(activations)
        return y, decoder_kl
    def sample(self, nb_samples):
        return self.decoder.sample(nb_samples)

if __name__ == "__main__":
    import torchvision
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae = VAE(3, 64, 32, 32, nb_blocks=6).to(device)
    x = torch.randn(1, 3, 256, 256).to(device)
    y, kls = vae(x)
