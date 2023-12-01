import torch
import torch.nn as nn
from Utils import split_feature
from Utils import ActFun
import torch.distributions as td
import torch.nn.functional as F


## Glow Modules
class ActNorm(nn.Module):

    def __init__(self, num_channels):
        super().__init__()

        size = [1, num_channels, 1, 1]
        self.register_parameter("bias", nn.Parameter(torch.zeros(*size), requires_grad=True))
        self.register_parameter("logs", nn.Parameter(torch.zeros(*size), requires_grad=True))
        
        # Buffer to register if initialization has been performed
        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
      if not self.training:
        return

      with torch.no_grad():
          bias = input.clone().mean(dim=[0, 2, 3], keepdim=True)
          std_input = input.clone().std(dim=[0, 2, 3], keepdim=True)
          logs = (1.0 / (std_input + 1e-6)).log()
          self.bias.data.copy_(-bias)
          self.logs.data.copy_(logs)

    def forward(self, input, logdet, reverse):
        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)
            
        dims = input.size(2) * input.size(3)

        if reverse == False:
            input = input + self.bias
            input = input * self.logs.exp()
            dlogdet = torch.sum(self.logs) * dims
            if logdet is not None:
              logdet = logdet + dlogdet

        if reverse == True:
            input = input * self.logs.mul(-1).exp()
            input = input - self.bias
            dlogdet = - torch.sum(self.logs) * dims
            if logdet is not None:
              logdet = logdet + dlogdet

        return input, logdet

class BatchNormFlow(nn.Module):
    """ An implementation of a batch normalization layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """
    def __init__(self, x_size, momentum=0.1, eps=1e-5):
        super(BatchNormFlow, self).__init__()
        Bx, Cx, Hx, Wx = x_size
        size = [1, Cx, Hx, Wx]
        self.log_gamma = nn.Parameter(torch.zeros(size))
        self.beta = nn.Parameter(torch.zeros(size))
        self.momentum = momentum
        self.eps = eps

        self.register_buffer('running_mean', torch.zeros(size))
        self.register_buffer('running_var', torch.ones(size))

    def forward(self, input, logdet, reverse):
        ## The reverse == False is only as failsafe, as we are only training
        ## in the reverse == False direction.  
        if self.training and reverse == False: 
            self.batch_mean = input.mean(0)
            self.batch_var = (
                input - self.batch_mean).pow(2).mean(0) + self.eps
    
            self.running_mean.mul_(self.momentum)
            self.running_var.mul_(self.momentum)
    
            self.running_mean.add_(self.batch_mean.data *
                                   (1 - self.momentum))
            self.running_var.add_(self.batch_var.data *
                                  (1 - self.momentum))
            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var
        dlogdet = torch.sum(self.log_gamma - 0.5 * torch.log(var))
        if reverse == False:
            x_hat = (input - mean) / var.sqrt()
            z = torch.exp(self.log_gamma) * x_hat + self.beta
            if logdet is not None:
                logdet = logdet + dlogdet
        else:
            x_hat = (input - self.beta) / torch.exp(self.log_gamma)
            z = x_hat * var.sqrt() + mean
            if logdet is not None:
                logdet = logdet - dlogdet
        return z, logdet

class Conv2dZeros(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=[3,3], stride=[1,1]):
        super().__init__()
        
        padding = (kernel_size[0] - 1) // 2
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        
        self.logscale_factor = 3
        self.register_parameter("logs", nn.Parameter(torch.zeros(out_channel, 1, 1)))
        
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

    def forward(self, input):
      output = self.conv(input) * torch.exp(self.logs * self.logscale_factor)
      return output 

class Conv2dNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=[3, 3], stride=[1, 1], norm = "actnorm"):
        super().__init__()

        padding = [(kernel_size[0]-1)//2, (kernel_size[1]-1)//2]

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=(norm != "actnorm"))
        self.conv.weight.data.normal_(mean=0.0, std=0.05)
        
        self.norm = norm
        if self.norm == "actnorm":
          self.norm_type = ActNorm(out_channels)
        elif self.norm=="batchnorm":
          self.conv.bias.data.zero_()
          self.norm_type = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        output = self.conv(input)
        if self.norm == "actnorm":
          output,_ = self.norm_type(output, logdet=0.0, reverse=False)
        elif self.norm == "batchnorm":
          output = self.norm_type(output)
        else:
          return output
        return output


class InvConv(nn.Module):
    def __init__(self, num_channels, LU_decomposed):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = torch.qr(torch.randn(*w_shape))[0]

        if not LU_decomposed:
            self.weight = nn.Parameter(torch.Tensor(w_init))
        else:
            p, lower, upper = torch.lu_unpack(*torch.lu(w_init))
            s = torch.diag(upper)
            sign_s = torch.sign(s)
            log_s = torch.log(torch.abs(s))
            upper = torch.triu(upper, 1)
            l_mask = torch.tril(torch.ones(w_shape), -1)
            eye = torch.eye(*w_shape)

            self.register_buffer("p", p)
            self.register_buffer("sign_s", sign_s)
            self.lower = nn.Parameter(lower)
            self.log_s = nn.Parameter(log_s)
            self.upper = nn.Parameter(upper)
            self.l_mask = l_mask
            self.eye = eye

        self.w_shape = w_shape
        self.LU_decomposed = LU_decomposed

    def get_weight(self, input, reverse):
        b, c, h, w = input.shape

        if not self.LU_decomposed:
            dlogdet = torch.slogdet(self.weight)[1] * h * w
            if reverse:
                weight = torch.inverse(self.weight)
            else:
                weight = self.weight
        else:
            self.l_mask = self.l_mask.to(input.device)
            self.eye = self.eye.to(input.device)

            lower = self.lower * self.l_mask + self.eye

            u = self.upper * self.l_mask.transpose(0, 1).contiguous()
            u = u + torch.diag(self.sign_s * torch.exp(self.log_s))

            dlogdet = torch.sum(self.log_s) * h * w

            if reverse:
                u_inv = torch.inverse(u)
                l_inv = torch.inverse(lower)
                p_inv = torch.inverse(self.p)

                weight = torch.matmul(u_inv, torch.matmul(l_inv, p_inv))
            else:
                weight = torch.matmul(self.p, torch.matmul(lower, u))

        return weight.view(self.w_shape[0], self.w_shape[1], 1, 1), dlogdet

    def forward(self, input, logdet, reverse):
        weight, dlogdet = self.get_weight(input, reverse)

        if not reverse:
            z = F.conv2d(input, weight)
            if logdet is not None:
              logdet = logdet + dlogdet
            return z, logdet
        else:
            z = F.conv2d(input, weight)
            if logdet is not None:
              logdet = logdet - dlogdet
            return z, logdet

class AffineCoupling(nn.Module):
    def __init__(self, x_size, condition_size, hidden_units=256, non_lin = 'relu', clamp_type = "realnvp"):
        super(AffineCoupling, self).__init__()
        
        Bx, Cx, Hx, Wx = x_size
        
        B, C, H, W = condition_size
        channels = Cx // 2 + C

        self.net = nn.Sequential(
            Conv2dNorm(channels, hidden_units),
            ActFun(non_lin),
            Conv2dNorm(hidden_units, hidden_units, kernel_size=[1, 1]),
            ActFun(non_lin),
            Conv2dZeros(hidden_units, Cx),
        )
        
        self.clamp_type = clamp_type
        if clamp_type == "glow":
            self.clamper = self.glow_clamp
        elif clamp_type == "softclamp":
            self.clamper = self.s_clamp
        elif clamp_type =="realnvp":
            self.scale = nn.Parameter(torch.zeros(Cx//2, 1, 1), requires_grad=True) #try all dimensions
            self.scale_shift = nn.Parameter(torch.zeros(Cx//2, 1, 1), requires_grad=True)
            self.clamper = self.realnvp_clamp
        else:
            self.clamper = self.none_clamp
        
    def s_clamp(self, s):
        #soft clamp from arXiv:1907.02392v3
        clamp = 2.5
        log_scale_clamped = clamp * 0.636 * torch.atan(s / clamp)
        return log_scale_clamped
    
    def glow_clamp(self, s):
        # Glow clamp from Openai code
        scale = torch.log(torch.sigmoid(s + 2.))
        return scale
    
    def realnvp_clamp(self, s):
        log_scale_clamped = self.scale * torch.tanh(s) + self.scale_shift
        return log_scale_clamped
    
    def none_clamp(self, s):
        return s

    def forward(self, x, condition, logdet, reverse): 
        z1, z2 = split_feature(x, "split")
        assert condition.shape[2:4] == x.shape[2:4], "condition and x in affine needs to match"
        h = torch.cat([z1, condition], dim=1)

        shift, log_scale = split_feature(self.net(h), "cross")

        log_scale_clamped = self.clamper(log_scale)
        
        if reverse == False:
            z2 = z2 + shift
            z2 = z2 * (log_scale_clamped.exp())
            if logdet is not None:
              logdet = logdet + torch.sum(log_scale_clamped, dim=[1, 2, 3])
        else:
            z2 = z2 * (log_scale_clamped.mul(-1).exp())
            z2 = z2 - shift
            if logdet is not None:
              logdet = logdet - torch.sum(log_scale_clamped, dim=[1, 2, 3]) 

        output = torch.cat((z1, z2), dim=1)
        return output, logdet


class Squeeze2d(nn.Module):
    def __init__(self):
        super(Squeeze2d, self).__init__()
        
    def forward(self, x, undo_squeeze):
      B, C, H, W = x.shape
      if undo_squeeze == False:
        # C x H x W -> 4C x H/2 x W/2
        x = x.reshape(B, C, H // 2, 2, W // 2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.reshape(B, C * 4, H // 2, W // 2)
      else:
        # 4C x H/2 x W/2  ->  C x H x W
        x = x.reshape(B, C // 4, 2, 2, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.reshape(B, C // 4, H * 2, W * 2)
      return x

class Split2d(nn.Module):
    def __init__(self, x_size, condition_size, make_conditional = True, clamp_function = 'softplus'):
      super(Split2d, self).__init__()
      self.make_conditional = make_conditional
      Bx, Cx, Hx, Wx = x_size
      non_lin = 'relu'
      
      if make_conditional:
        B, C, H, W = condition_size
        channels = Cx // 2 + C
        self.convcond = nn.Sequential(
            Conv2dNorm(C, C),
            ActFun(non_lin),
            Conv2dNorm(C, C, kernel_size=[1, 1]),
            ActFun(non_lin),
            )
      else:
        channels = Cx // 2
        
      self.conv = nn.Sequential(Conv2dZeros(channels, Cx),)
    
      if clamp_function == 'softplus': 
          self.clamper = self.softplus
      elif clamp_function == 'exp':
          self.clamper = self.exp_x
      else:
          assert False, 'Please specify a clamp function for the split2d from the set {softplus, exp}'
    
    def softplus(self,x):
        return nn.Softplus()(x) + 1e-8
    
    def exp_x(self,x):
        return x.exp()
    
    def forward(self, x, condition, logdet, reverse, temperature = None):
        
        if reverse == False:
            z1, z2 = split_feature(x, "split")
        else:
            z1 = x
        
        if self.make_conditional:
            condition = self.convcond(condition)
            h = torch.cat([z1, condition], dim=1)
        else:
            h = z1
        
        out = self.conv(h)
        mean, log_scale = split_feature(out, "cross")  
        
        if reverse == False:
            if logdet is not None:
              logdet = logdet + torch.sum(td.Normal(mean, self.clamper(log_scale)).log_prob(z2), dim=(1,2,3))
            return z1, logdet
        else:
            z2 = td.Normal(mean, self.clamper(log_scale)*temperature).sample()
            z = torch.cat((z1, z2), dim=1)
            return z, logdet
