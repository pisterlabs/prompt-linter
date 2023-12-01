# Sourced directly from OpenAI's CLIP repo
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from typing import Iterable, Optional

from typing import Dict
from torch import Tensor

class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float())#.type(x.dtype)

class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x, self.weight, None if self.bias is None else self.bias
        )

def sinusoids(length: int, channels: int, max_timescale: int=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None
    ):
        q = self.query(x)

        k = self.key(x if xa is None else xa)
        v = self.value(x if xa is None else xa)

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        #qk = qk.float()

        w = F.softmax(qk, dim=-1)#.to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = MultiHeadAttention(n_state, n_head) if cross_attention else None
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state))
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None
    ):
        x = x + self.attn(self.attn_ln(x), mask=mask)[0]
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x

class TextDecoder(nn.Module):
    def __init__(self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head, cross_attention=True) for _ in range(n_layer)]
        )
        self.ln = LayerNorm(n_state)

        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: Tensor, xa: Tensor):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_state)
            the encoded audio features to be attended on
        """
        x = x

        # reshape such that our visual embeddings matches the sequence length
        # note that every text token now obtains (the same) information from the visual embeddings
        xa = xa.repeat(x.shape[1],1,1).permute(1,0,2)

        for block in self.blocks:
            x = block(x, xa, mask=self.mask)

        x = self.ln(x)
        
        logits = (x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0,1))
        return logits

class WSIDecoder(nn.Module):
    def __init__(self, vision_dim: int, n_ctx: int, n_state: int, n_head: int = 2, n_layer: int = 3):
        super().__init__()

        self.input_linear = nn.Linear(vision_dim, n_state)

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head, cross_attention=True) for _ in range(n_layer)]
        )
        self.ln = LayerNorm(vision_dim)
        self.ln_post = LayerNorm(n_state)

        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

        self.output_linear = nn.Linear(n_state, n_state)
        
    def forward(self, x: Tensor):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_state)
            the encoded audio features to be attended on
        """
        x = self.input_linear(self.ln(x))#.to(xa.dtype)

        # reshape such that our visual embeddings matches the sequence length
        # note that every text token now obtains (the same) information from the visual embeddings

        for block in self.blocks:
            x = block(x, xa=None, mask=self.mask)
        
        x = self.ln_post(x)
        # get the embedding of the embeddings
        x = self.output_linear(x[:,0,:])
        return x


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 context_length: int,
                 ):
        super().__init__()

        self.context_length = context_length                                  
        
        # HIPT encodes 4096x4096 regions to a vector of 192
        self.visual = WSIDecoder(192, 32, embed_dim, 2, 3)
        # We have diagnostic report projection from 768 to embed_dim to match the visual embeddings
        self.text_projection = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Dropout(0.1),
                                             nn.Linear(768, embed_dim), nn.ReLU(), nn.Dropout(0.1), 
                                             nn.Linear(embed_dim, embed_dim))


        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)) 


class WSICaptioner(nn.Module):
    def __init__(self,
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int    
                 ):
        super().__init__()

        self.context_length = context_length                                   

        self.decoder = TextDecoder(vocab_size,
                    context_length,
                    transformer_width,
                    transformer_heads,
                    transformer_layers,
                )                                    
    
    def forward(self, lm_hiddens: Tensor, visual_clip_embeddings: Tensor) -> Dict[str, Tensor]:
        
        lm_logits = self.decoder(lm_hiddens, visual_clip_embeddings)
        
        return lm_logits            