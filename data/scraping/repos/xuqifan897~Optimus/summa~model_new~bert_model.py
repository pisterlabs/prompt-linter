from summa.module import OptimusModule
from summa import get_args
from summa.model.utils import init_method_normal
from summa.model.utils import scaled_init_method_normal
from summa.model_new.language_model import get_language_model
from summa.model.utils import openai_gelu, erf_gelu
from summa.mpu.layers import SUMMA_ABT, _initialize_affine_weight_gpu
import summa.mpu as mpu
from summa.mpu.layers import _initialize_affine_weight_gpu
from torch.nn.parameter import Parameter
import torch

from .BertLMHead import DENSE

def bert_attention_mask_func(attention_scores, attention_mask):
    attention_scores = attention_scores + attention_mask
    return attention_scores


def bert_extended_attention_mask(attention_mask, dtype):
    # We create a 3D attention mask from a 2D tensor mask.
    # [b, 1, s]
    attention_mask_b1s = attention_mask.unsqueeze(1)
    # [b, s, 1]
    attention_mask_bs1 = attention_mask.unsqueeze(2)
    # [b, s, s]
    attention_mask_bss = attention_mask_b1s * attention_mask_bs1
    # [b, 1, s, s]
    extended_attention_mask = attention_mask_bss.unsqueeze(1)
    # Since attention_mask is 1.0 for positions we want to attend and 0.0
    # for masked positions, this operation will create a tensor which is
    # 0.0 for positions we want to attend and -10000.0 for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    # fp16 compatibility
    extended_attention_mask = extended_attention_mask.to(dtype=dtype)
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

    return extended_attention_mask


class BertLMHead(OptimusModule):
    """Masked LM head for Bert

    Arguments:
        mpu_vocab_size: model parallel size of vocabulary.
        hidden_size: hidden size
        init_method: init method for weight initialization
        layernorm_epsilon: tolerance for layer norm divisions
        parallel_output: whether output logits being distributed or not.
    """

    def __init__(self, vocab_size, hidden_size, init_method, bias=False):
        super(BertLMHead, self).__init__()
        args = get_args()
        self.hidden_pp = mpu.divide(hidden_size, args.summa_dim)
        self.vocab_pp = mpu.divide(vocab_size, args.summa_dim)
        self.row_rank = torch.distributed.get_rank(group=mpu.get_summa_col_group())
        self.col_rank = torch.distributed.get_rank(group=mpu.get_summa_row_group())
        self.ddp_rank = torch.distributed.get_rank(group=mpu.get_data_parallel_group())
        self.summa_dim = args.summa_dim
        self.model_parallel_size = args.model_parallel_size
        self.bias_flag = bias
        if self.bias_flag:
            if self.row_rank == 0:
                self.bias = Parameter(torch.zeros(
                    self.vocab_pp,
                    dtype=args.params_dtype,
                    device=torch.cuda.current_device()))
            else:
                self.bias = None
        self.dense_weight = Parameter(torch.empty((
            self.hidden_pp, self.hidden_pp),
            dtype=args.params_dtype,
            device=torch.cuda.current_device()))
        _initialize_affine_weight_gpu(self.dense_weight, init_method)

        self.layernorm = mpu.LayerNorm_summa(self.hidden_pp)
        self.gelu = torch.nn.functional.gelu
        if args.openai_gelu:
            self.gelu = openai_gelu
        elif args.onnx_safe:
            self.gelu = erf_gelu

        self.forward_buffer = mpu.get_h4h_forward_buffer()
        self.backward_buffer = mpu.get_backward_buffer()
        # self.forward_buffer = None
        # self.backward_buffer = None
        self.parameter_gradient_buffer = mpu.get_parameter_gradient_buffer()

    def forward(self, hidden_states, word_embeddings_weight):
        # hidden_states: [b/q, s, h/q]
        # word_embeddings_weight: [v/q, h/q]
        hidden_states = DENSE.apply(
            hidden_states, self.dense_weight, hidden_states.shape,
            self.row_rank, self.col_rank, self.ddp_rank,
            self.summa_dim, self.model_parallel_size)
        hidden_states = self.gelu(hidden_states)
        hidden_states = self.layernorm(hidden_states)
        output_shape = hidden_states.shape[:-1] + (word_embeddings_weight.shape[0], )
        self.forward_buffer.reset()
        self.backward_buffer.reset()
        output = SUMMA_ABT.apply(
            hidden_states, word_embeddings_weight, output_shape,
            self.row_rank, self.col_rank, self.ddp_rank,
            self.summa_dim, self.model_parallel_size,
            self.forward_buffer, self.backward_buffer,
            self.parameter_gradient_buffer)
        if self.bias_flag:
            output = mpu.SUMMAbias.apply(
                output, self.bias, self.vocab_pp, self.row_rank,
                self.col_rank, self.ddp_rank, self.model_parallel_size,
                False, output.dtype)
        return output

class BertModel(OptimusModule):
    """Bert Language Model."""

    def __init__(self, num_tokentypes=2,
                 add_binary_head=True):
        super(BertModel, self).__init__()
        args = get_args()

        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy
        self.add_binary_head = add_binary_head
        vocab_pp = mpu.divide(args.padded_vocab_size, args.summa_dim)
        col_rank = torch.distributed.get_rank(group=mpu.get_summa_row_group())
        self.vocab_start = vocab_pp * col_rank
        self.vocab_end = self.vocab_start + vocab_pp
        init_method = init_method_normal(args.init_method_std)
        scaled_init_method = scaled_init_method_normal(args.init_method_std,
                                                       args.num_layers)

        self.language_model, self._language_model_key = get_language_model(
            attention_mask_func=bert_attention_mask_func,
            num_tokentypes=num_tokentypes,
            add_pooler=self.add_binary_head,
            init_method=init_method,
            scaled_init_method=scaled_init_method)

        self.lm_head = BertLMHead(
            args.padded_vocab_size, args.hidden_size, init_method)
        self._lm_head_key = 'lm_head'

        if self.add_binary_head:
            self.binary_head = BinaryHead(
                args.hidden_size, 2, init_method)
            self._binary_head_key = 'binary_head'

        self.parameter_gradient_buffer = mpu.get_parameter_gradient_buffer()
        self.activation_checkpoint_buffer = mpu.get_checkpoint_activation_buffer()

    def forward(self, input_ids, attention_mask,
                tokentype_ids=None, lm_labels=None):

        extended_attention_mask = bert_extended_attention_mask(
            attention_mask, next(self.language_model.parameters()).dtype)

        self.parameter_gradient_buffer.reset()
        self.activation_checkpoint_buffer.reset()
        if self.add_binary_head:
            lm_output, pooled_output = self.language_model(
                input_ids,
                None,
                extended_attention_mask,
                tokentype_ids=tokentype_ids)
        else:
            lm_output = self.language_model(
                input_ids,
                None,
                extended_attention_mask,
                tokentype_ids=tokentype_ids)

        # Output.
        lm_logits = self.lm_head(
            lm_output, self.language_model.embedding.vocab_weight)

        binary_logits = None
        if self.add_binary_head:
            binary_logits = self.binary_head(pooled_output)

        if lm_labels is None:
            return lm_logits, binary_logits
        else:
            lm_loss = mpu.SUMMA_CrossEntropy.apply(
                lm_logits, lm_labels, self.vocab_start, self.vocab_end)
            return lm_loss, binary_logits


class BinaryHead(OptimusModule):
    def __init__(self, hidden_size, num_classes, init_method):
        super(BinaryHead, self).__init__()
        args = get_args()
        self.col_rank = torch.distributed.get_rank(group=mpu.get_summa_row_group())
        self.row_rank = torch.distributed.get_rank(group=mpu.get_summa_col_group())
        self.ddp_rank = torch.distributed.get_rank(group=mpu.get_data_parallel_group())
        self.model_parallel_size = args.model_parallel_size
        self.hidden_pp = mpu.divide(hidden_size, args.summa_dim)
        self.dim = (self.hidden_pp, num_classes)
        self.dtype = args.params_dtype
        if self.row_rank == 0:
            self.weight = Parameter(torch.empty(
                self.dim, dtype=args.params_dtype,
                device=torch.cuda.current_device()))
            _initialize_affine_weight_gpu(self.weight, init_method)
        else:
            self.weight = None

    def forward(self, hidden_states):
        binary_logits = SUMMA_BinaryHead.apply(
            hidden_states, self.weight, self.row_rank, self.col_rank,
            self.ddp_rank, self.model_parallel_size, self.dim, self.dtype)
        return binary_logits


class SUMMA_BinaryHead(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, weight, row_rank, col_rank,
                ddp_rank, model_parallel_size, dim, dtype):
        # hidden_states: [b/q, s, h/q]
        # weight: [h/q, 2]
        with torch.no_grad():
            if row_rank == 0:
                weight_temp = weight.clone()
            else:
                weight_temp = torch.zeros(
                    dim,
                    dtype=dtype,
                    device=torch.cuda.current_device())
            torch.distributed.broadcast(
                weight_temp,
                src=col_rank+ddp_rank*model_parallel_size,
                group=mpu.get_summa_col_group())
            # [b/q, s, 2]
            logits = torch.matmul(hidden_states, weight_temp)
            torch.distributed.all_reduce(
                logits, group=mpu.get_summa_row_group())
        ctx.save_for_backward(hidden_states, weight_temp)
        ctx.row_rank = row_rank
        ctx.col_rank = col_rank
        ctx.ddp_rank = ddp_rank
        ctx.model_parallel_size = model_parallel_size
        return logits

    @staticmethod
    def backward(ctx, output_grad):
        # output_grad: [b/q, s, 2]
        # [b/q, s, h/q], [h/q, 2]
        hidden_states, weight_temp = ctx.saved_tensors
        hidden_grad = torch.matmul(output_grad, weight_temp.transpose(0, 1))

        output_grad_shape = output_grad.shape
        # [bs/q, 2]
        output_grad = output_grad.view((-1, output_grad_shape[-1]))

        hidden_states_shape = hidden_states.shape
        # [bs/q, h/q]
        hidden_states = hidden_states.view((-1, hidden_states_shape[-1]))

        # [h/q, 2]
        weight_grad = torch.matmul(hidden_states.transpose(0, 1), output_grad)
        torch.distributed.reduce(
            weight_grad,
            dst=ctx.col_rank+ctx.ddp_rank*ctx.model_parallel_size,
            group=mpu.get_summa_col_group())

        if ctx.row_rank == 0:
            return hidden_grad, weight_grad, None, None, None, None, None, None
        else:
            return hidden_grad, None, None, None, None, None, None, None