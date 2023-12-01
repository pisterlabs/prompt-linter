import torch
import torch.nn.functional as F
import torch.distributed as dist
import math
import deepspeed

from torch import nn
from torch.nn import Parameter
from megatron import get_args, mpu
from megatron.model.fused_bias_gelu import bias_gelu_impl
from megatron.model.fused_layer_norm import MixedFusedLayerNorm as LayerNorm
from megatron.model.utils import openai_gelu, erf_gelu, attention_mask_func
from megatron.model.fused_softmax import FusedScaleMaskSoftmax
from .positional_embeddings import (
    RotaryEmbedding,
    apply_rotary_pos_emb_torch,
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_index_torch,
    apply_rotary_pos_emb_index,
)
from megatron.enums import PositionEmbeddingType, LayerType
from .module import MegatronModule

@torch.jit.script
def apply_scale_offset(x, gamma, beta):
    return x * gamma + beta


class ScaleOffset(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.gamma = Parameter(torch.Tensor(hidden_size))
        self.beta = Parameter(torch.Tensor(hidden_size))
        self.reset_parameters()

        self.gamma.tied_tensor_model_parallel = True
        self.beta.tied_tensor_model_parallel = True

    def reset_parameters(self):
        torch.nn.init.ones_(self.gamma)
        torch.nn.init.zeros_(self.beta)

    def forward(self, x):
        with torch.enable_grad():
            x = apply_scale_offset(x, self.gamma, self.beta)
        return x


class SharedLinear(nn.Module):
    """Shared Linear Layer across mp group, with bias gelu fusion"""

    def __init__(self, input_size, output_size, init_method):
        args = get_args()
        super(SharedLinear, self).__init__()

        self.weight = Parameter(torch.empty(
            output_size,
            input_size,
            device=torch.cuda.current_device(),
            dtype=args.params_dtype,
        ))
        self.bias = Parameter(
            torch.empty(
                output_size, device=torch.cuda.current_device(), dtype=args.params_dtype
            )
        )

        self.bias_gelu_fusion = args.bias_gelu_fusion
        self.activation_func = F.gelu
        if args.openai_gelu:
            self.activation_func = openai_gelu
        elif args.onnx_safe:
            self.activation_func = erf_gelu

        self.init_parameters(init_method)

        self.weight.tied_tensor_model_parallel = True
        self.bias.tied_tensor_model_parallel = True

    def init_parameters(self, init_method):
        with torch.no_grad():
            init_method(self.weight)
            # sync weight across mp group
            dist.broadcast(
                self.weight,
                mpu.get_tensor_model_parallel_src_rank(),
                group=mpu.get_tensor_model_parallel_group(),
            )
            # zero bias
            torch.nn.init.zeros_(self.bias)

    def forward(self, x):
        x = F.linear(x, self.weight)
        x = (
            bias_gelu_impl(x, self.bias)
            if self.bias_gelu_fusion
            else self.activation_func(x + self.bias)
        )
        return x


def bias_dropout_add(x, bias, residual, prob, training):
    # type: (Tensor, Tensor, Tensor, float, bool) -> Tensor
    out = torch.nn.functional.dropout(x + bias, p=prob, training=training)
    out = residual + out
    return out


def get_bias_dropout_add(training):
    def _bias_dropout_add(x, bias, residual, prob):
        return bias_dropout_add(x, bias, residual, prob, training)

    return _bias_dropout_add


@torch.jit.script
def bias_dropout_add_fused_train(x, bias, residual, prob):
    # type: (Tensor, Tensor, Tensor, float) -> Tensor
    return bias_dropout_add(x, bias, residual, prob, True)


@torch.jit.script
def bias_dropout_add_fused_inference(x, bias, residual, prob):
    # type: (Tensor, Tensor, Tensor, float) -> Tensor
    return bias_dropout_add(x, bias, residual, prob, False)


def debug(point, tensor, printed=False):
    if mpu.get_data_parallel_rank() == 0:
        print(f"[MP={mpu.get_tensor_model_parallel_rank()}] {point}: {tensor.float().abs().sum()} {tensor if printed else ''}", flush=True)


class GatedAttentionUnit(MegatronModule):
    def __init__(
        self,
        init_method,
        output_layer_init_method,
        layer_number,
        self_attn_mask_type,
        layer_type=LayerType.encoder,
    ):
        args = get_args()

        super().__init__()

        # Assertions for simplicity
        assert args.position_embedding_type != PositionEmbeddingType.alibi
        assert args.fp32_residual_connection is False

        self.fp16 = args.fp16
        self.bf16 = args.bf16
        # Pre-LN/Post-LN
        self.apply_residual_connection_post_layernorm = (
            args.apply_residual_connection_post_layernorm
        )
        self.hidden_dropout = args.hidden_dropout
        # Bias fusion
        self.bias_dropout_fusion = args.bias_dropout_fusion
        self.bias_gelu_fusion = args.bias_gelu_fusion

        self.apply_query_key_layer_scaling = args.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = args.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)
        self.position_embedding_type = args.position_embedding_type
        self.attn_mask_type = self_attn_mask_type
        # PB-Relax
        self.apply_pb_relax = args.apply_pb_relax
        self.pb_relax_alpha = args.pb_relax_alpha
        # Sandwich-LN
        self.apply_scale_normalization = args.sandwich_ln

        self.key_size = args.gated_attention_unit_key_size

        self.input_layernorm = LayerNorm(args.hidden_size, eps=args.layernorm_epsilon)
        if self.apply_scale_normalization:
            self.mlp_layernorm = LayerNorm(args.hidden_size, eps=args.layernorm_epsilon)

        self.dense_uv = mpu.ColumnParallelLinear(
            args.hidden_size,
            2 * args.ffn_hidden_size,
            gather_output=False,
            init_method=init_method,
            skip_bias_add=True,
        )

        # dense_k is shared across model parallel group
        self.dense_qk = SharedLinear(
            args.hidden_size, self.key_size, init_method=init_method
        )

        self.dense_w = mpu.RowParallelLinear(
            args.ffn_hidden_size,
            args.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True,
        )

        self.scale_offset_q = ScaleOffset(self.key_size)
        self.scale_offset_k = ScaleOffset(self.key_size)

        if self.position_embedding_type == PositionEmbeddingType.rotary:
            self.rotary_emb = RotaryEmbedding(
                self.key_size,
                base=10000,
                precision=args.params_dtype,
                learnable=args.learnable_rotary_embedding,
            )

        coeff = None
        self.norm_factor = math.sqrt(self.key_size)
        if args.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff
        self.scale_mask_softmax = FusedScaleMaskSoftmax(
            self.fp16,
            self.bf16,
            self.attn_mask_type,
            args.masked_softmax_fusion,
            attention_mask_func,
            self.attention_softmax_in_fp32,
            coeff,
        )

        self.attention_dropout = torch.nn.Dropout(args.attention_dropout)

        if deepspeed.checkpointing.is_configured():
            global get_cuda_rng_tracker, checkpoint
            get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
            checkpoint = deepspeed.checkpointing.checkpoint

    def get_uv(self, x):
        mixed_uv, mixed_uv_bias = self.dense_uv(x)
        if self.bias_gelu_fusion:
            mixed_uv = bias_gelu_impl(mixed_uv, mixed_uv_bias)
        else:
            mixed_uv = F.gelu(mixed_uv + mixed_uv_bias)
        return mixed_uv

    def attention(self, x, attention_mask, position_ids=None):
        # x: [s, b, h], v: [s, b, e]
        qk = self.dense_qk(x)
        # q, k: [s, b, key_size]
        q, k = self.scale_offset_q(qk), self.scale_offset_k(qk)
        # all-reduce grad
        q = mpu.copy_to_tensor_model_parallel_region(q)
        k = mpu.copy_to_tensor_model_parallel_region(k)

        # Rotary embeddings
        if self.position_embedding_type == PositionEmbeddingType.rotary:
            if position_ids is not None:  # GLM
                apply_rotary_fn = (
                    apply_rotary_pos_emb_index_torch
                    if self.bf16
                    else apply_rotary_pos_emb_index
                )
                # [b, sq] -> [sq, b]
                position_ids = position_ids.transpose(0, 1)
                cos, sin = self.rotary_emb(k, seq_len=position_ids.max() + 1)
                q, k = apply_rotary_fn(q, k, cos, sin, position_ids)
            else:
                apply_rotary_fn = (
                    apply_rotary_pos_emb_torch if self.bf16 else apply_rotary_pos_emb
                )
                cos, sin = self.rotary_emb(k, seq_len=k.size(0))
                q, k = apply_rotary_fn(q, k, cos, sin)

        s, b = q.size(0), q.size(1)
        matmul_result = torch.empty(
            b, s, s, dtype=q.dtype, device=torch.cuda.current_device()
        )
        # [b, s, s]
        attention_scores = torch.baddbmm(
            matmul_result,
            (q.transpose(0, 1) / self.norm_factor).contiguous(),
            (k.transpose(0, 1).transpose(1, 2)).contiguous()
            / (self.pb_relax_alpha if self.apply_pb_relax else 1.0),
            beta=0.0,
            alpha=1.0,
        )

        if self.apply_pb_relax:
            attention_scores = (
                attention_scores
                - attention_scores.view(b, -1).abs().max(dim=-1).values.view(b, 1, 1)
            ) * self.pb_relax_alpha

        # softmax need [b, np, s, s], here np always = 1
        attention_probs = self.scale_mask_softmax(
            attention_scores.unsqueeze(1), attention_mask
        )

        # We shouldn't fork cuda rng tracker since we want same prob in mp group
        attention_probs = self.attention_dropout(attention_probs.squeeze(1))

        return attention_probs

    def final_dense(self, x, residual):
        x, x_bias = self.dense_w(x)

        if self.apply_scale_normalization:
            x = self.mlp_layernorm(x)

        if self.bias_dropout_fusion:
            if self.training:
                bias_dropout_add_func = bias_dropout_add_fused_train
            else:
                bias_dropout_add_func = bias_dropout_add_fused_inference
        else:
            bias_dropout_add_func = get_bias_dropout_add(self.training)

        # re-enable torch grad to enable fused optimization.
        with torch.enable_grad():
            output = bias_dropout_add_func(
                x, x_bias.expand_as(residual), residual, self.hidden_dropout
            )

        return output

    @staticmethod
    @torch.jit.script
    def gau_fused(uv, a):
        # [s, b, 2 * e] -> [s, b, e] * 2
        u, v = uv.chunk(2, dim=(uv.ndim - 1))
        # [b, s, s] x ([s, b, e] -> [b, s, e]) -> [b, s, e]
        av = torch.bmm(a, v.transpose(0, 1))
        # [b, s, e] -> [s, b, e]
        av = av.transpose(0, 1)
        return u * av

    def forward(self, hidden_states, attention_mask, encoder_output=None, enc_dec_attn_mask=None,
                layer_past=None, get_key_value=False, position_ids=None):
        assert encoder_output is None and enc_dec_attn_mask is None and layer_past is None and get_key_value is False
        # hidden_states: [s, b, h]
        ln_out = self.input_layernorm(hidden_states)
        # a: [b, s, s]
        a = self.attention(ln_out, attention_mask=attention_mask, position_ids=position_ids)
        # uv: [s, b, 2 * e]
        uv = self.get_uv(ln_out)
        # x: [s, b, e]
        x = self.gau_fused(uv, a)
        residual = (
            ln_out if self.apply_residual_connection_post_layernorm else hidden_states
        )
        output = self.final_dense(x, residual)
        return output
