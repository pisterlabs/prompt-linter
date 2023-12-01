import torch
from torch.nn import functional as F
import einops
import numpy as np
from som_utils import SOMGrid
from utils import tuple_checker, approximate_square_root

# Module for quantizers. Lots of influence from OpenAI's Jukebox VQ-VAE, rosinality's VQ-VAE, and the OG paper:
# https://arxiv.org/pdf/1711.00937.pdf

# TODO : Add support for different types of quantizers, e.g. Gumbel-Softmax, etc.

class BaseQuantizer(torch.nn.Module):
    """Base class for quantizers.
    Parameters
    ----------
    dim : int
        Dimension of the input.
    codebook_size : int
        Number of entries in the codebook.
    cut_freq : int
        The cutoff frequency for removing codebook vectors and replacing with inputs.
    alpha : float
        The EMA decay rate for the codebook frequency.
    replace_with_obs : bool
        Whether to replace codebook entries with observations.
    init_scale : float
        The scale of the initial codebook. Does not apply to cosine distance.
    new_code_noise : float
        The amount of noise to add to new codebook entries when the stale entries outnumber replacements.
    use_som : bool
        Whether to use a SOM in the codebook update.
    som_neighbor_distance : int
        The distance for neighbors in the SOM update.
    dist_type : str
        The distance type to use for the codebook. Either "cos" or "euclidean".
    in_rvq : bool
        Whether this quantizer is in an RVQ. Used to avoid some redundancies.
    """
    def __init__(self, 
                 dim : int, 
                 codebook_size : int,
                 cut_freq : int = 1,
                 alpha : float = 0.95,
                 replace_with_obs : bool = True,
                 init_scale = 1.0,
                 new_code_noise : float = 1e-9,
                 use_som : bool = False,
                 som_neighbor_distance : int = 2,
                 som_kernel_type = "hard",
                 dist_type : str = "e",
                 in_rvq : bool = False,
                 precreated_som : torch.nn.Module = None,):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.alpha = alpha
        self.cut_freq = cut_freq
        self.replace_with_obs = replace_with_obs
        self.new_code_noise = new_code_noise
        self.use_som = use_som
        self.ema = False # flag for the occaisonal conditional update
        self.dist_type = dist_type
        self.in_rvq = in_rvq

        self.stale_clusters = None


        # initialize codebook as parameter to enable optimization
        self.codebook = torch.nn.Parameter(torch.randn(dim, codebook_size) * init_scale)
        # initalize a count that each codebook entry appears
        self.register_buffer("cluster_frequency", torch.ones(codebook_size))

        if self.use_som:
            if in_rvq and precreated_som is not None:
                self.som = precreated_som
            else:
                h, w = approximate_square_root(self.codebook_size)
                self.som = SOMGrid(h, w,
                                   kernel_type = som_kernel_type,
                                   neighbor_distance = som_neighbor_distance,)

    def codebook_euclidean_d(self, x_flat):

        # distances, note this is ~~ (flatten - codebook)T @ (flatten - codebook), where T is the transpose
        # TODO : look into replacing this with torch.cdist
        dist = (
            x_flat.pow(2).sum(-1, keepdim=True)
            - 2 * x_flat @ self.codebook
            + self.codebook.pow(2).sum(0, keepdim=True)
        )

        # get the closest codebook entry
        min_dist, codebook_index = torch.min(dist, dim = -1)
        return codebook_index
    
    def codebook_cosine_d(self, x_flat):
        """Distance with respect to cosine similarity. Note that we are only normalizing x and the codebook for
        the metric, otherwise, they are allowed to stay unnormalized. Constant renormalization works poorly with RVQ."""
        x_mag = torch.linalg.vector_norm(x_flat.detach(), dim = -1, keepdim = True)
        codebook_mag = torch.linalg.vector_norm(self.codebook.detach(), dim = 0, keepdim = True)
        dist = torch.einsum("bld,dn -> bln", 
                            x_flat / x_mag, 
                            self.codebook / codebook_mag)
        min_dist, codebook_index = torch.max(dist, dim = -1)
        return codebook_index

    def quantize(self, input):
        """Quantize the input. Returns the index of the closest codebook entry for each input element."""
        flatten = einops.rearrange(input, "b ... d -> b (...) d")

        if self.dist_type == "cos":
            codebook_index = self.codebook_cosine_d(flatten)
        else:
            codebook_index = self.codebook_euclidean_d(flatten)

        return codebook_index, flatten

    def dequantize(self, codebook_index):
        """Given the symbol/index of the codebook entry, return the corresponding vector in the codebook."""
        return F.embedding(codebook_index, self.codebook.T)

    def codebook_update_function(self, x, x_quantized, codebook_index, x_flat, codebook_onehot):
        """Default way to update the codebook is by gradient descent on a codebook loss. Accepts extra arguments for
        other types of quantizers that may need them."""
        
        if self.use_som:
            assert self.som.kernel_type == "hard", "SOM kernel type must be 'hard' for base quantizer"
            # codebook loss includes all codebook entries near the chosen one
            # undo one hot
            codebook_indices = codebook_onehot.nonzero() # note from update_codebook_count that this already has SOM blending
            indices, cb_index = codebook_indices.split([2, 1], dim = -1)
            cb_vectors = self.dequantize(cb_index.squeeze(-1))
            x_vectors = x[indices[:, 0], indices[:, 1], :]
            # get distance between all surrounding codebook entries and the input
            codebook_loss = (cb_vectors - x_vectors.detach()).pow(2).mean()
        else:
            # codebook loss, must stop/detach the gradient of the non-quantized input
            codebook_loss = (x_quantized - x.detach()).pow(2).mean()

        return codebook_loss
    
    def update_codebook_count(self, codebook_index, x_flat, verbose = False, update_codebook = True):
        """Update the count of how many times each codebook entry has been used, useful for avoiding
        codebook collapse."""
        with torch.no_grad():
            # get a one-hot encoding of the codebook index
            codebook_onehot = F.one_hot(codebook_index, self.codebook_size).type(x_flat.dtype)
            if self.use_som:
                codebook_onehot = self.som(codebook_onehot, 
                                           update_t = False if (not self.ema) else update_codebook)

            # use that to get the count that each codebook entry has been used
            codebook_count = codebook_onehot.sum((0, 1))

            # take ema weighted sum of current code/symbol use frequencies
            self.cluster_frequency.data.mul_(self.alpha).add_(codebook_count, alpha = 1 - self.alpha)

            # replace the underutilized codebook entries with new ones
            self.replace_low(codebook_count, x_flat, verbose, update_codebook)

        return codebook_onehot
    
    def _replace_with_high(self, low_clusters, num_low_clusters):
        if num_low_clusters <= self.codebook_size // 2:

            high_clusters = torch.topk(self.cluster_frequency, 
                                    num_low_clusters,
                                    largest = True, sorted = False)[1]
        else:
            # note that these are indices and not a bool vector like low_clusters
            high_clusters = (~low_clusters).nonzero().squeeze(1)
            num_high_clusters = high_clusters.shape[0]
            # if there are many low clusters, use the highest multiple times
            n_repeats = num_low_clusters // num_high_clusters
            remainder = num_low_clusters % num_high_clusters

            high_clusters = high_clusters.repeat(n_repeats)
            if remainder > 0:
                # repeat the remainder
                high_clusters = torch.cat([high_clusters, high_clusters[:remainder]])

        high_vectors = self.codebook[:, high_clusters].detach().clone()

        # get the high vectors, jitter them
        high_vectors += torch.randn_like(high_vectors) * self.new_code_noise
        return high_vectors
    
    def replace_low(self, codebook_count, x_flat, verbose, update_codebook):
        """Replaces underutilize codebook entries."""
        # when determining underused clusters, take into account history and recent use
        comparison_clusters = torch.maximum(self.cluster_frequency, codebook_count)

        # replace the codebook entries that have been used less than the cut frequency
        low_clusters = comparison_clusters < self.cut_freq
        num_low_clusters = int(low_clusters.sum().item())
        self.stale_clusters = num_low_clusters
        # all clusters will be low clusters on the first run, so only update in between
        if update_codebook and ((low_clusters.any()) and not (low_clusters.all())):
            
            if verbose:
                print(f"{num_low_clusters} clusters are poorly represented. Updating...")
            if not self.replace_with_obs:
                high_vectors = self._replace_with_high(low_clusters, num_low_clusters)
            else:
                # get a sample from x_flat with size num_low_clusters
                high_vectors = einops.rearrange(x_flat.detach().clone(), "b l d -> (b l) d")
                if high_vectors.shape[0] < num_low_clusters:
                    # repeat and add some scaled noise
                    high_vectors = high_vectors.repeat((num_low_clusters // high_vectors.shape[0]) + 1, 1)
                    high_vectors += torch.randn_like(high_vectors) * self.new_code_noise
                # shuffle them  - don't want to add position-based bias - and select num_low_clusters of them
                high_vectors = high_vectors[torch.randperm(high_vectors.shape[0]), :][:num_low_clusters].T

            # convert low clusters to indices
            low_clusters = low_clusters.nonzero().squeeze(1)
                
            # replace the low clusters with the new clusters
            self.codebook[:, low_clusters] = high_vectors
            # update cluster frequency
            self.cluster_frequency[low_clusters] += 1
            if self.ema:
                self.ema_codebook[:, low_clusters] = high_vectors
        

    def forward(self, x_in, update_codebook : bool = False):
        """Quantize and dequantize the input. Returns the quantized input, the index of the codebook entry for each
        input element, and the commitment loss. Note that update_codebook means to reassign input vectors to 
        codebook entries that are poorly represented; the codebook updates via gradient descent (or
        k-means for the EMAQuantizer) regardless of this flag."""
        x = x_in.clone()

        codebook_index, x_flat = self.quantize(x)
        x_quantized = self.dequantize(codebook_index)

        # the commitment loss, stop/detach the gradient of the quantized input
        inner_loss = (x_quantized.detach() - x_in).pow(2).mean()
        if self.training:
            codebook_onehot = self.update_codebook_count(codebook_index, x_flat, update_codebook = update_codebook)
            codebook_loss = self.codebook_update_function(x_in, x_quantized, codebook_index, x_flat, codebook_onehot)
            if codebook_loss is not None:
                inner_loss += codebook_loss

        # passes the gradient through the quantization for the reconstruction loss
        x_quantized = x_in + (x_quantized - x_in).detach()

        return x_quantized, codebook_index, inner_loss
    
    def get_stale_clusters(self):
        return self.stale_clusters
    
    def update_cutoff(self, new_cutoff : int = None, ratio : float = None):
        if new_cutoff is not None:
            self.cut_freq = new_cutoff
        elif ratio is not None:
            self.cut_freq = self.cut_freq * ratio
        else:
            raise ValueError("Must specify either new cutoff or ratio")

class EMAQuantizer(BaseQuantizer):
    """Quantizer that uses an exponential moving average to update the codebook, 
    based on the Appendix of the VQ-VAE paper.
    Parameters
    ----------
    dim : int
        Dimension of the input.
    codebook_size : int
        Number of entries in the codebook.
    alpha : float
        The EMA decay rate for the codebook frequency.
    smoothing_alpha : float
        The value used in Laplace smoothing of the codebook frequency.
    cut_freq : int
        The cutoff frequency for removing codebook vectors and replacing with inputs.
    replace_with_obs : bool
        Whether to replace codebook entries with observations.
    init_scale : float
        The scale of the initial codebook. Does not apply to cosine distance.
    new_code_noise : float
        The amount of noise to add to new codebook entries when the stale entries outnumber replacements.
    use_som : bool
        Whether to use a SOM in the codebook update.
    som_neighbor_distance : int
        The distance for neighbors in the SOM update.
    dist_type : str
        The distance type to use for the codebook. Either "cos" or "euclidean".
    in_rvq : bool
        Whether this quantizer is in an RVQ. Used to avoid some redundancies.
    """
    def __init__(self, 
                 dim : int, 
                 codebook_size : int, 
                 alpha : float = 0.96, 
                 smoothing_alpha : float = 1,
                 cut_freq : int = 2,
                 replace_with_obs : bool = True,
                 init_scale : float = 1.0,
                 use_som : bool = True,
                 som_neighbor_distance : int = 2,
                 som_kernel_type = "hard",
                 dist_type : str = "e",
                 in_rvq : bool = False,
                 precreated_som : torch.nn.Module = None,):
        super().__init__(dim, 
                         codebook_size, 
                         alpha = alpha, 
                         cut_freq = cut_freq, 
                         replace_with_obs = replace_with_obs,
                         init_scale = init_scale,
                         use_som = use_som,
                         som_neighbor_distance = som_neighbor_distance,
                         som_kernel_type = som_kernel_type,
                         dist_type = dist_type,
                         in_rvq = in_rvq,
                         precreated_som = precreated_som)
        
        self.smoothing_alpha = smoothing_alpha
        self.ema = True # helper bool

        # disable grad for the codebook
        self.codebook.requires_grad = False
        
        self.register_buffer("ema_codebook", self.codebook.clone())

    def codebook_update_function(self, x, x_quantized, codebook_index, x_flat, codebook_onehot):
        """Update the codebook using an exponential moving average. In this case, the vectors in the codebook are updated by 
        averaging the input vectors that are closest to them. This makes the codebook better represent the encoding of the data.
        The description can be found in Appendix A.1 here: https://arxiv.org/pdf/1711.00937.pdf"""

        # projects the input onto the closest codebook entry, taking a mean along the batch and length
        with torch.no_grad():
            codebook_sum = torch.einsum("b l d, b l c -> d c", x_flat, codebook_onehot)

            # update ema codebook with the input vectors
            self.ema_codebook.data.mul_(self.alpha).add_(codebook_sum, alpha = 1 - self.alpha)

            # normalize the codebook with laplace smoothing
            n = self.cluster_frequency.sum()
            cluster_frequency_normalized = ((self.cluster_frequency + self.smoothing_alpha) / (n + self.codebook_size * self.smoothing_alpha) * n)
            codebook_normalized = self.ema_codebook / cluster_frequency_normalized.unsqueeze(0)

            # overwrite codebook
            self.codebook.data.copy_(codebook_normalized)
    

class ResidualQuantizer(torch.nn.Module):
    """Residual vector quantization, as here:
    https://arxiv.org/pdf/2107.03312.pdf
    
    Essentially, we quantize the encoder output, and then quantize the residual iteratively.
    """
    def __init__(self,
                 num_quantizers,
                 dim,
                 codebook_sizes,
                 quantizer_class = "ema",
                 scale_factor = 4.0,
                 priority_n = 24,
                 vq_cutoff_freq = 2,
                 decorr_loss_weight : float = 0.0,
                 use_som = True,
                 som_kernel_type = "hard",
                 som_neighbor_distance = 2,
                 dist_type = "e"):
        
        super().__init__()

        self.num_quantizers = num_quantizers
        self.dim = dim
        self.codebook_sizes = tuple_checker(codebook_sizes, num_quantizers)
        self.priority_n = priority_n
        self.decorr_loss_weight = decorr_loss_weight
        self.use_som = use_som
        self.dist_type = dist_type

        # residual gets smaller at each step, so can be helpful to have small quantizer vectors
        scale_factors = [1 / (scale_factor ** i) for i in range(num_quantizers)]

        quantizer_type = EMAQuantizer if quantizer_class == "ema" else BaseQuantizer
        print(f"Initializing residual quantizer with class {quantizer_class}")

        if self.use_som:
            h, w = approximate_square_root(self.codebook_sizes[0])
            self.som = SOMGrid(h, w,
                               kernel_type = som_kernel_type,
                               neighbor_distance = som_neighbor_distance)
        else:
            self.som = None

        quantizers = [quantizer_type(self.dim, 
                                     codebook_size, 
                                     init_scale = scale,
                                     cut_freq = vq_cutoff_freq,
                                     use_som = use_som,
                                     dist_type = dist_type,
                                     in_rvq = True,
                                     precreated_som = self.som) for codebook_size, scale in zip(self.codebook_sizes, scale_factors)]

        self.quantizers = torch.nn.ModuleList(quantizers)

    def forward(self, 
                x, 
                n = None, 
                update_codebook : bool = False, 
                prioritize_early : bool = False):
        # can limit to first n quantizers, they call this bitrate dropout in the paper
        # if n is None, use all quantizers, for training n will typically be sampled uniformly from [1, num_quantizers]
        if n is None:
            n = self.num_quantizers

        x_hat = 0
        residual = x
        inner_loss = 0

        indices = []

        for i in range(n):

            x_i, index, inner_loss_i = self.quantizers[i](residual, update_codebook = update_codebook)

            if (self.decorr_loss_weight > 0) and (i > 0):
                # loss term to encourage decorrelation between quantizers
                decorr_loss = self.decorr_loss_weight * torch.nn.functional.cosine_similarity(x_i, prev_x_i, dim = -1).mean()
                decorr_loss /= (self.num_quantizers - 1)
                inner_loss_i += decorr_loss
            
            prev_x_i = x_i.detach().clone()

            stale_clusters = self.quantizers[i].stale_clusters

            # prioritize early quantizers, switch off update_codebook if stale clusters exceeds threshold
            if prioritize_early and update_codebook and (stale_clusters > self.priority_n):
                update_codebook = False

            x_hat += x_i
            residual -= x_i.detach()
            inner_loss += inner_loss_i
            indices.append(index)

        return x_hat, torch.stack(indices, dim = -1), inner_loss
    
    def get_stale_clusters(self):
        """Get the number of stale clusters from all quantizers"""
        stale_clusters = []
        for quantizer in self.quantizers:
            stale_clusters.append(quantizer.stale_clusters)
        return stale_clusters
    
    def update_cutoff(self, new_cutoff = None, ratio = None):
        """Update the cutoff frequency for all quantizers"""
        if new_cutoff is not None:
            for quantizer in self.quantizers:
                quantizer.cut_freq = new_cutoff
        elif ratio is not None:
            for quantizer in self.quantizers:
                quantizer.cut_freq = quantizer.cut_freq * ratio
        else:
            raise ValueError("Must specify either new cutoff or ratio")


if __name__ == "__main__":
    # extensive test on CIFAR - takes ~11 mins on RTX 3090
    from tqdm import tqdm
    import os
    import torchvision
    from itertools import chain
    from einops import rearrange
    from einops.layers.torch import Rearrange
    import matplotlib.pyplot as plt
    from utils import losses_to_running_loss
    # helpful functions
    im2tensor = torchvision.transforms.ToTensor()

    def collate(x, im2tensor = im2tensor):
        x = [im2tensor(x_i[0]) for x_i in x]
        return torch.stack(x, dim = 0)

    def tensor2im(x):
        return torchvision.transforms.ToPILImage()(x)

    def save_im(x, path):
        tensor2im(x).save(path)

    def get_networks(case, patch_dim, embed_dim, device):
        if case == "linear":
            # feedforward layers to convert the last dim of the patched images to an embedding dimension
            patch_embedder = torch.nn.Sequential(
                torch.nn.Linear(patch_dim, embed_dim),
                torch.nn.LayerNorm(embed_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(embed_dim, embed_dim)).to(device)
            patch_deembedder = torch.nn.Sequential(
                torch.nn.Linear(embed_dim, embed_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(embed_dim, patch_dim)).to(device)
            
            # these break a batch of images into patches and the converse, respectively
            patcher = Rearrange("... c (h p1) (w p2) -> ... (h w) (p1 p2 c)", 
                                p1 = patch_size, 
                                p2 = patch_size).requires_grad_(False).to(device)

            depatcher = Rearrange("... (h w) (p1 p2 c) -> ... c (h p1) (w p2)", 
                                  p1 = patch_size, 
                                  p2 = patch_size, 
                                  h = h // patch_size, 
                                  w = w // patch_size).requires_grad_(False).to(device)
        elif case == "conv":
            layers = []
            n_steps = 3
            curr_channel = 3
            prev_channel = 3
            for i in range(n_steps):
                curr_channel = prev_channel * 4
                layers.append(torch.nn.Conv2d(prev_channel, 
                                              curr_channel,
                                              kernel_size = 3,
                                              stride = 2,
                                              padding = 1))
                if i < n_steps - 1:
                    layers.append(torch.nn.ReLU())
                prev_channel = curr_channel
            patcher = torch.nn.Sequential(*layers).to(device)

            embed_dim = curr_channel
            layers = []
            for i in range(n_steps):
                curr_channel = prev_channel // 4
                layers.append(torch.nn.Upsample(scale_factor = 2))
                layers.append(torch.nn.Conv2d(prev_channel,
                                              curr_channel,
                                              kernel_size = 3,
                                              padding = "same"))
                
                if i < n_steps - 1:
                    layers.append(torch.nn.ReLU())
                prev_channel = curr_channel
            depatcher = torch.nn.Sequential(*layers).to(device)

            patch_embedder = Rearrange("... c h w -> ... (h w) c").requires_grad_(False)
            patch_deembedder = Rearrange("... (h w) c -> ... c h w",
                                  h = 4,
                                  w = 4).requires_grad_(False)
        else:
            raise ValueError("Case must be either 'linear' or 'conv'")
        
        embedder = torch.nn.Sequential(patcher, patch_embedder).to(device)
        deembedder = torch.nn.Sequential(patch_deembedder, depatcher).to(device)

        return embedder, deembedder, embed_dim

    def plot_som_codebook(path, quantizer, deembedder, case):
        """Plot the codebook as a SOM."""
        with torch.no_grad():
            if not isinstance(quantizer, ResidualQuantizer):
                quantizers = [quantizer]
            else:
                quantizers = quantizer.quantizers
            codebooks = []
            # we use this so that deeper entries are shown in a realistic setting
            max_entry = torch.zeros(quantizers[0].codebook.shape[0], 1, device = device)
            for i, quantizer in enumerate(quantizers):
                codebook = quantizer.codebook.clone() #dim, codebook_size

                # build a realistic setting by using the most common codebook entries
                argmax_entry = quantizer.cluster_frequency.argmax()
                max_entry_i = codebook[:, argmax_entry]

                codebook += max_entry
                max_entry += max_entry_i.unsqueeze(-1)

                codebook = quantizer.som.codebook_to_grid(codebook) #dim, h, w

                codebook = codebook.permute(1, 2, 0) #h, w, dim

                if case == "conv":
                    codebook = deembedder(codebook) #c, h, w
                else:
                    codebook = deembedder[0](codebook) #h, w, dim'
                    codebook = rearrange(codebook, "h w (p1 p2 c) -> c (h p1) (w p2)", p1 = patch_size, p2 = patch_size) #c, h, w

                codebooks.append(codebook)
            codebooks = torch.concat(codebooks, dim = -1)
            codebooks = torchvision.utils.make_grid(codebooks, nrow = i + 1, padding = 2, pad_value = 1)
            save_im(codebooks, path + ".png")

    # making a tmp folder to store the images
    os.makedirs("tmp/", exist_ok = True)

    # data root
    data_root = "D:/Projects/" # you should only need to change this
    # image export frequency
    output_every = 100

    # data and patcher params
    h, w = 32, 32
    patch_size = 4
    patch_dim = patch_size**2 * 3
    embed_dim = patch_dim // 5 # 5x compression 
    n_patches = (h // patch_size) * (w // patch_size)
    batch_size = 32
    residual = True
    som_neighbor_distance = 2
    som_kernel = "hard"
    case = "linear"
    update_codebook = True
    residual_count = 4
    codebook_size = 512

    # training params
    n_epochs = 5
    lr = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cifar = torchvision.datasets.CIFAR100(root = data_root, train = True, download = True)
       
    comparison_losses = []
    for dist_type in ["cos",
                      "e"
                      ]:
        for test_base in [True,
                          False
                          ]:

            embedder, deembedder, embed_dim = get_networks(case, 
                                                           patch_dim, 
                                                           embed_dim, 
                                                           device)

            # test the quantizers
            if residual:
                quantizer = ResidualQuantizer(residual_count, 
                                              embed_dim, 
                                              codebook_size, 
                                              quantizer_class = "base" if test_base else "ema",
                                              dist_type = dist_type,
                                              som_neighbor_distance = som_neighbor_distance,
                                              som_kernel_type = som_kernel)
            elif test_base:
                quantizer = BaseQuantizer(embed_dim, 
                                        codebook_size,
                                        use_som = True,
                                        dist_type = dist_type)

            else:
                quantizer = EMAQuantizer(embed_dim, 
                                            codebook_size, 
                                            use_som = True,
                                            dist_type = dist_type, 
                                            som_neighbor_distance = som_neighbor_distance,
                                            som_kernel_type = som_kernel)
            if test_base:
                chainz = chain(embedder.parameters(),
                               deembedder.parameters(),
                               quantizer.parameters())
            else:
                chainz = chain(embedder.parameters(),
                               deembedder.parameters())
                
            quantizer.to(device)

            optimizer = torch.optim.Adam(chainz, 
                                        lr = lr)

            criterion = torch.nn.MSELoss()

            losses = []
            cb_1_mean = []

            for epoch in range(n_epochs):
                print(f"Epoch {epoch}")
                dataloader = torch.utils.data.DataLoader(cifar, 
                                                    batch_size = batch_size, 
                                                    shuffle = True,
                                                    collate_fn = collate)
                for i, x in enumerate(tqdm(dataloader)):
                    optimizer.zero_grad()
                    x = x.to(device)
                    x_orig = x.clone().detach()

                    if (i == 0) and (epoch == 0):
                        x_copy = x.clone().detach()

                    x = embedder(x)          

                    x, codebook_index, inner_loss = quantizer(x, 
                                                              update_codebook = update_codebook)

                    x = deembedder(x)
  
                    loss = criterion(x, x_orig) + inner_loss
                    loss.backward()
                    optimizer.step()

                    losses.append(loss.item())
                    cb_1_mean.append(quantizer.quantizers[0].codebook.mean().item())
                    
                    if i % output_every == 0:
                        str_i = str(i).zfill(5)
                        b, c, h, w = x.shape

                        with torch.no_grad():
                            x = embedder(x_copy)

                            x, codebook_index, inner_loss = quantizer(x, update_codebook = update_codebook)
                            x = deembedder(x)

                        x_out = torch.stack([x_copy, x], dim = 0)
                        x_out = rearrange(x_out, "n b c h w -> (b n) c h w")

                        x_out = torchvision.utils.make_grid(x_out, nrow = 8, padding = 2, pad_value = 1)
                        save_im(x_out, f"tmp/epoch_{epoch}_{str_i}.png")
                        if quantizer.use_som:
                            plot_som_codebook(f"tmp/codebook_epoch_{epoch}_{str_i}", quantizer, deembedder, case)
                quantizer.update_cutoff(ratio = 2/3)
                print(f"Stale codebook entries: {quantizer.get_stale_clusters()}")
            comparison_losses.append(losses)

    labels = ["Base, cos", "EMA, cos", "Base, euclidean", "EMA, euclidean"]
    for i, label in enumerate(labels):
        plt.plot(np.log(losses_to_running_loss(comparison_losses[i])), label = label)
    plt.legend()