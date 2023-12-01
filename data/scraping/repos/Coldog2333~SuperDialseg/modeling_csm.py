import numpy as np
import torch
import torch.nn as nn
from transformers import BertForNextSentencePrediction, AutoTokenizer, BertTokenizer

from ...modeling_utils import BaseSegmenter
from ...models.texttiling.utils_texttiling import depth_score_cal, cutoff_threshold
from ...models.csm.utils_csm import CoherenceScoringModel


class TexttilingNSPSegmenter(BaseSegmenter):
    def __init__(
        self,
        backbone: str = 'bert-base-uncased',
        max_utterance_len: int = 50,
        cut_rate: float = 0.5,
    ):
        super(TexttilingNSPSegmenter, self).__init__()

        self.backbone = backbone
        self.max_utterance_len = max_utterance_len
        self.cut_rate = cut_rate

        self.device = 'cpu'

        self.model, self.tokenizer = self.build_model()
        self.model.eval()

    def build_model(self):
        model = BertForNextSentencePrediction.from_pretrained(
            self.backbone,
            num_labels=2,
            output_attentions=False,
            output_hidden_states=False,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.backbone)

        return model, tokenizer

    def get_predictions_from_logits(self, logits):
        scores = torch.sigmoid(logits)
        depth_scores, mean, std = depth_score_cal(scores.cpu().detach().numpy())
        #### top-k startegy
        # predictions = [0] * (len(depth_scores) + 1)
        #
        # for i in boundary_indice:
        #     predictions[i] = 1

        ##### by threshold
        threshold = cutoff_threshold(
            cut_rate=self.cut_rate,
            mean=np.mean(depth_scores),
            std=np.std(depth_scores)
        )

        # threshold = miu - sigma / 2
        predictions = []
        for depth_score in depth_scores:
            if depth_score > threshold:
                predictions.append(1)
            else:
                predictions.append(0)
        predictions.append(1)

        return predictions

    def forward(self, inputs):
        utterances = inputs['utterances']

        utterances_a = utterances[:-1]
        utterances_b = utterances[1:]

        batch_inputs = self.tokenizer(
            text=utterances_a,
            text_pair=utterances_b,
            return_tensors='pt',
            max_length=self.max_utterance_len,
            add_special_tokens=True,
            truncation=True,
            padding='longest',
        )

        outputs = self.model(
            batch_inputs['input_ids'].to(self.device),
            attention_mask=batch_inputs['attention_mask'].to(self.device),
            token_type_ids=batch_inputs['token_type_ids'].to(self.device),
            return_dict=True
        )

        logits = outputs['logits'][:, 0]
        # coherence_scores = torch.sigmoid(logits)
        predictions = self.get_predictions_from_logits(logits)

        predictions[-1] = 0

        return predictions

    def to(self, device):
        self.device = device
        self.model.to(device)


class CSMSegmenter(TexttilingNSPSegmenter):
    def __init__(
        self,
        backbone: str = 'bert-base-uncased',
        max_utterance_len: int = 50,
        cut_rate: float = 1.,
    ):
        super(CSMSegmenter, self).__init__(
            backbone=backbone,
            max_utterance_len=max_utterance_len,
            cut_rate=cut_rate,
        )

    def build_model(self):
        model = CoherenceScoringModel(backbone=self.backbone)
        tokenizer = AutoTokenizer.from_pretrained(self.backbone)
        return model, tokenizer

    def load_state_dict(self, ckpt_path):
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.model.eval()
