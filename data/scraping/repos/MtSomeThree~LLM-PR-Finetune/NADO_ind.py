import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Model, GPT2Config
from transformers import OpenAIGPTTokenizer, OpenAIGPTModel, OpenAIGPTLMHeadModel, OpenAIGPTPreTrainedModel, OpenAIGPTConfig
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions, CausalLMOutput

from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from packaging import version
from torch import nn
from torch.nn.functional import softmax
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, Sigmoid, LogSigmoid

import numpy as np

class NADOInd(GPT2LMHeadModel):
    _keys_to_ignore_on_load_missing = [r"NADO"]
    def __init__(self, config):
        super().__init__(config)
        self.NADO = GPT2Model.from_pretrained("gpt2")
        self.NADO_linear = nn.Linear(config.n_embd, config.vocab_size, bias=True)
        self.constraint_factor = 1.0
        self.temperature = 1.0
        self.log_sigmoid_fct = LogSigmoid()

    def save_NADO_to_cache(self, filename):
        torch.save((self.NADO.state_dict(), self.NADO_linear.state_dict()), filename)

    def load_NADO_from_cache(self, filename, device='cpu'):
        NADO_state_dict, NADO_linear_state_dict = torch.load(filename, map_location=device)
        self.NADO.load_state_dict(NADO_state_dict)
        self.NADO_linear.load_state_dict(NADO_linear_state_dict)

    def set_NADO(self, NADO, NADO_linear):
        self.NADO = NADO
        self.NADO_linear = NADO_linear

    def set_constraint_factor(self, factor):
        self.constraint_factor = factor

    def get_NADO(self):
        return self.NADO, self.NADO_linear

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        rc_labels=None,
        rc_weights=None,
        use_cache=None,
        use_temperature=False,
        output_attentions=None,
        output_hidden_states=None,
        prompt_ppl=False,
        prompt_length=1,
        return_dict=None,
    ):

        #return_dict is always None
        #copy from huggingface docs
        if past_key_values is not None:
            past1, past2 = past_key_values
        else:
            past1 = None
            past2 = None

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past1,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        hidden_states = transformer_outputs.last_hidden_state
        return_pkv1 = transformer_outputs.past_key_values
        # if past_key_values is not None and head_mask is not None:
        #     print ("Shape Check:")
        #     print (input_ids.shape)
        #     print (attention_mask.shape)
        #     print (len(past_key_values))
        #     print (len(past_key_values[0]))
        #     print (past_key_values[0][0].shape)
        #     print (head_mask.shape)
        #     print (head_mask.sum())
        lm_logits = self.lm_head(hidden_states)
        if self.constraint_factor == 0.0:
            pred_logits = lm_logits
            return_pkv2 = None
        else:
            NADO_outputs = self.NADO(
                input_ids,
                past_key_values=past2,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
            NADO_hidden_states = NADO_outputs.last_hidden_state
            constr_logits = self.log_sigmoid_fct(self.NADO_linear(NADO_hidden_states)) * self.constraint_factor

            pred_logits = lm_logits + constr_logits
            return_pkv2 = NADO_outputs.past_key_values

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., prompt_length - 1: -1, :].contiguous()
            shift_labels = labels[..., prompt_length:].contiguous()
            loss_fct = CrossEntropyLoss()
            # Flatten the tokens
            if rc_labels is not None:
                rc_labels = rc_labels.long()
                shift_logits = shift_logits.index_select(0, torch.where(rc_labels == 1)[0])
                shift_labels = shift_labels.index_select(0, torch.where(rc_labels == 1)[0])

            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        elif rc_labels is not None:
            length = input_ids.shape[1]
            if len(rc_labels.shape) == 1:
                rc_labels = rc_labels.unsqueeze(1)
            for i in range(prompt_length - 1, length - 1):
                pred_logits = torch.gather(constr_logits[:, i, :], 1, input_ids[:, i + 1].unsqueeze(1))

                # if rc_weights is not None:
                #     weights = softmax(rc_weights * (1.0 - self.temperature), dim=0)
                #     loss_fct = BCEWithLogitsLoss(weight=
                #         (attention_mask[:, i + 1] * weights).unsqueeze(1), reduction='sum')
                # else:
                loss_fct = BCEWithLogitsLoss(weight=attention_mask[:, i + 1].unsqueeze(1))

                cur_loss = loss_fct(pred_logits, rc_labels)
                if loss is not None:
                    loss += cur_loss
                else:
                    loss = cur_loss

        elif prompt_ppl: # evaluate PPL
            pred_logits = lm_logits[:, prompt_length - 1: -1, :].contiguous()
            pred_labels = input_ids[:, prompt_length:].contiguous()

            #loss_fct = BCEWithLogitsLoss(weight=
            #            attention_mask[:, self.prompt_length:].to(constr_logits.device))
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(pred_logits.view(-1, pred_logits.size(-1)), pred_labels.view(-1))

        return CausalLMOutputWithCrossAttentions(
                loss=loss,
                logits=pred_logits,
                past_key_values=(return_pkv1, return_pkv2),
                hidden_states=transformer_outputs.hidden_states,
                attentions=transformer_outputs.attentions,
                cross_attentions=transformer_outputs.cross_attentions,
            )
        
    @staticmethod
    def _reorder_cache(past, beam_idx):
        past1, past2 = past

        reordered_past1 = ()
        for layer_past in past1:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past1 += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )

        reordered_past2 = ()
        for layer_past in past2:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past2 += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )

        return (reordered_past1, reordered_past2)

if __name__ == "__main__":
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = ConstrainedLM.from_pretrained("gpt2")

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    print(sum([np.prod(p.size()) for p in model_parameters]))

    sentence_prefix = "I play piano every"

    input_ids = tokenizer.encode(
        sentence_prefix,
        add_special_tokens=False,
        return_tensors="pt",
    )

    #model(input_ids=input_ids)

    for i in range(5):
        output_ids = model.generate(
            input_ids=input_ids,
            do_sample=True,
            max_length=30,  # desired output sentence length
            pad_token_id=model.config.eos_token_id,
        )[0].tolist()
         
        generated_text = tokenizer.decode(
            output_ids,
            clean_up_tokenization_spaces=True)
         
        print(generated_text)

