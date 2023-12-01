import os
import hydra
import torch
from torchmetrics import Accuracy, Precision, Recall, F1Score
import transformers
import pandas as pd
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import rank_zero_only
from typing import List, Any, Dict, Optional
from transformers.trainer_pt_utils import get_parameter_names
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

import src.utils as utils
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from src.utils import general_helpers
from src.models.collators import MetaCollator
from src.utils.evaluation_utils import EvaluationUtils
import numpy as np
import torch.nn.functional as F
from src.models.meta_models import EstimationModel
from src.models.meta_models import OpenAIModelInfo

log = utils.get_pylogger(__name__)


class MetaModelForSequenceClassification(LightningModule, EstimationModel):
    def __init__(
            self,
            pretrained_model_name_or_path: str,
            models_info: List[OpenAIModelInfo] = None,
            tokenizer_name_or_path: str = None,
            config: Dict[str, Any] = None,
            num_labels: int = None,
            metrics_parameters: Dict[str, Any] = None,
            default_collator_parameters: Dict[str, Any] = None,
            hparams_overrides=None,
            **kwargs,
    ):
        super().__init__()
        EstimationModel.__init__(self, models_info)
        self.save_hyperparameters(
            logger=False,
            ignore=[
                "collator",
                "hparams_overrides",
                "datamodule",
            ],
        )

        if hparams_overrides is not None:
            self._override_checkpoint_hparams(hparams_overrides)

        if self.hparams.tokenizer_name_or_path is None:
            self.hparams.tokenizer_name_or_path = self.hparams.pretrained_model_name_or_path

        if self.hparams.num_labels is None:
            self.hparams.num_labels = 1

        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer_name_or_path)

        hf_config = None
        if self.hparams.get("hf_config", None):
            hf_config = AutoConfig.from_pretrained(self.hparams.pretrained_model_name_or_path)
            log.info("HF model config:")
            log.info(hf_config)

        if hf_config is not None:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.hparams.pretrained_model_name_or_path, config=hf_config)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.hparams.pretrained_model_name_or_path, ignore_mismatched_sizes=True, num_labels=self.hparams.num_labels
            )

        self.collator = kwargs.get("collator", None)
        if self.collator is None:
            if self.hparams.default_collator_parameters is None:
                self.hparams.default_collator_parameters = {'max_length': 512, 'padding': "longest", 'truncation': True, 'num_outputs': None}
            self.collator = self._get_default_collator()
        else:
            self.collator.set_tokenizer(self.tokenizer)

        if self.hparams.metrics_parameters is None:
            self.hparams.metrics_parameters = {'task': 'binary', 'average': 'macro', 'threshold': 0.5}
        metrics_params = self.hparams.metrics_parameters

        self.acc = Accuracy(task=metrics_params['task'], average=metrics_params['average'], threshold=metrics_params['threshold'])
        self.prec = Precision(task=metrics_params['task'], average=metrics_params['average'], threshold=metrics_params['threshold'])
        self.rec = Recall(task=metrics_params['task'], average=metrics_params['average'], threshold=metrics_params['threshold'])
        self.f1 = F1Score(task=metrics_params['task'], average=metrics_params['average'], threshold=metrics_params['threshold'])
        self.output_dir = None

    def _override_checkpoint_hparams(self, hparams_overrides: dict):
        """
        Overrides the hyperparameters of a checkpoint at an arbitrary depth
        :param hparams_overrides:
        :return:
        """
        general_helpers.rec_dict_update(self.hparams, hparams_overrides)
        log.info("Some values of the original hparams were overridden")
        log.info("Hyper-parameters:")
        log.info(self.hparams)

    def _get_default_collator(self):
        return MetaCollator(tokenizer=self.tokenizer, **self.hparams.default_collator_parameters)

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=None,
            **kwargs
        )

        return output

    def training_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        self.log("train loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        self.log("val loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"val_loss": loss}

    def _compute_loss(self, batch):
        model_output = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        logits = model_output.logits
        if self.model.config.num_labels == 1:
            criterion = torch.nn.BCEWithLogitsLoss()
        else:
            criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(logits, batch['labels'])
        return loss

    def _get_predictions_for_batch(self, batch):
        hf_inference_params = self.hparams.inference["hf_inference_params"].copy()
        hf_inference_params.update(
            {
                "input_is_processed_batch": True,
            }
        )
        sample_output = self.sample(
            batch,
            **hf_inference_params,
        )

        return sample_output

    def test_step(self, batch, batch_idx): # todo writing other things to the output file
        raw_input = [sample["queries"] for sample in batch["raw"]]
        raw_target = [sample["labels"] for sample in batch["raw"]]
        raw_id = [sample["ids"] for sample in batch["raw"]]
        if "datasets" in batch["raw"][0]:
            raw_datasets = [sample["datasets"] for sample in batch["raw"]]
        else:
            raw_datasets = None

        if "models" in batch["raw"][0]:
            raw_models = [sample["models"] for sample in batch["raw"]]
        else:
            raw_models = None

        if "completions" in batch["raw"][0]:
            raw_completions = [sample["completions"] for sample in batch["raw"]]
        else:
            raw_completions = None

        sample_output = self._get_predictions_for_batch(batch)
        self._write_step_output(raw_input=raw_input, raw_target=raw_target, raw_id=raw_id, sample_output=sample_output, raw_datasets=raw_datasets, raw_models=raw_models, raw_completions=raw_completions)

        return_object = {
            "inputs": raw_input,
            "targets": raw_target,
            "predictions": sample_output
        }

        return return_object

    def on_test_batch_end(self, outputs, batch: Any = None, batch_idx: int = None, dataloader_idx: int = 0):
        targets = torch.tensor(outputs["targets"])
        if self.model.config.num_labels == 1:
            predictions = F.sigmoid(outputs["predictions"].logits.squeeze().cpu())
        else:
            predictions = F.softmax(outputs["predictions"].logits.squeeze().cpu(), dim=1)
        acc = self.acc(predictions, targets)
        p = self.prec(predictions, targets)
        r = self.rec(predictions, targets)
        f1 = self.f1(predictions, targets)
        self.log("test/accuracy", acc, on_step=True, on_epoch=False, prog_bar=True)
        self.log("test/precision", p, on_step=True, on_epoch=False, prog_bar=True)
        self.log("test/recall", r, on_step=True, on_epoch=False, prog_bar=True)
        self.log("test/f1", f1, on_step=True, on_epoch=False, prog_bar=True)

    def _write_step_output(
            self,
            raw_input,
            raw_target,
            raw_id,
            sample_output,
            raw_datasets,
            raw_models,
            raw_completions,
    ):
        if self.model.config.num_labels == 1:
            inference_output = F.sigmoid(sample_output["logits"].squeeze().cpu().detach()).numpy().astype(np.float64)
        else:
            inference_output = F.softmax(sample_output["logits"].squeeze().cpu().detach(), dim=1).numpy().astype(np.float64)
        prediction_outputs = {
            "input": raw_input,
            "target": raw_target,
            "id": raw_id,
            "inference":  inference_output
        }

        if raw_datasets is not None:
            prediction_outputs["dataset"] = raw_datasets

        if raw_models is not None:
            prediction_outputs["model"] = raw_models

        if raw_completions is not None:
            prediction_outputs["completion"] = raw_completions

        prediction_outputs_path = os.path.join(
            EvaluationUtils.get_predictions_dir_path(self.output_dir),
            f"testing_output_{self.global_rank}.prediction.jsonl.gz",
        )

        prediction_outputs_summary = general_helpers.get_list_of_dicts(prediction_outputs)
        general_helpers.write_gzipped_jsonlines(prediction_outputs_path, prediction_outputs_summary, mode="a+")

    def on_test_epoch_end(self):
        acc = self.acc.compute()
        prec = self.prec.compute()
        rec = self.rec.compute()
        f1 = self.f1.compute()
        self.log("test/accuracy", acc)
        self.log("test/precision", prec)
        self.log("test/recall", rec)
        self.log("test/f1", f1)

        if hasattr(torch.distributed, "is_initialized") and torch.distributed.is_initialized():
            torch.distributed.barrier()

        general_helpers._move_predictions_for_subprocesses(
            EvaluationUtils.get_predictions_dir_path(os.getcwd()),
            EvaluationUtils.get_predictions_dir_path(self.output_dir),
        )

        EvaluationUtils.upload_outputs_to_wandb(
            getattr(self, "hparams_to_log", {}),
            EvaluationUtils.get_predictions_dir_path(self.output_dir),
            logger=self.logger,
        )
        return {
            "test/acc": acc,
            "test/precision": prec,
            "test/recall": rec,
            "test/f1": f1
        }

    @torch.no_grad()
    def sample(
            self,
            input_data,
            input_is_processed_batch=False,
            seed=None,
            **kwargs,
    ):
        training = self.training
        if training:
            self.eval()

        if seed is None:
            seed = self.hparams.inference.get("seed", None)
        if seed:
            transformers.trainer_utils.set_seed(seed)

        if input_is_processed_batch:
            input_ids = input_data["input_ids"].to(self.device)
            attention_mask = input_data["attention_mask"].to(self.device)
        else:
            tokenizer_output = self.tokenize(input_data)
            input_ids = tokenizer_output["input_ids"]
            attention_mask = tokenizer_output["attention_mask"]

        inference_kwargs = {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
        }

        inference_outputs = self.model(**inference_kwargs)
        if training:
            self.train()

        return inference_outputs

    def configure_optimizers(self):
        decay_parameters = get_parameter_names(self.model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if n in decay_parameters],
                "weight_decay": self.hparams.optimizer.weight_decay,
                "betas": (0.9, 0.999),
                "eps": self.hparams.optimizer.eps,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters],
                "weight_decay": 0.0,
                "betas": (0.9, 0.999),
                "eps": self.hparams.optimizer.eps,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.optimizer.lr,
            weight_decay=self.hparams.optimizer.weight_decay,
        )

        if self.hparams.scheduler.name == "linear":
            scheduler = transformers.get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.scheduler.warmup_updates,
                num_training_steps=self.hparams.scheduler.total_num_updates,
            )
        elif self.hparams.scheduler.name == "polynomial":
            scheduler = transformers.get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.scheduler.warmup_updates,
                num_training_steps=self.hparams.scheduler.total_num_updates,
                lr_end=self.hparams.scheduler.lr_end,
            )
        elif self.hparams.scheduler.name is not None:
            raise ValueError("Unknown scheduler name {}".format(self.hparams.scheduler.name))

        lr_dict = {
            "scheduler": scheduler,  # scheduler instance
            "interval": "step",  # The unit of the scheduler's step size. 'step' or 'epoch
            "frequency": 1,  # corresponds to updating the learning rate after every `frequency` epoch/step
            # used by the LearningRateMonitor callback
            "name": f"LearningRateScheduler-{self.hparams.scheduler.name}",
        }

        return [optimizer], [lr_dict]

    def _prepare_inputs(self, batch: List[str], model: str):
        batch = ["<" + model + ">" + sample for sample in batch]
        processed_batch = {}
        tokenizer_output = self.tokenizer(
            batch,
            return_tensors="pt",
            return_attention_mask=True,
            padding=self.collator.params["padding"],
            max_length=self.collator.params["max_length"],
            truncation=self.collator.params["truncation"],
        )
        for k, v in tokenizer_output.items():
            processed_batch[k] = v
        return processed_batch

    def test_batch(self, batch: List[str]):
        output = {}
        models = [model.model_name for model in self.models_info]
        model_prefixes = [model.model_prefix for model in self.models_info]
        for model, model_prefix in zip(models, model_prefixes):
            inputs = self._prepare_inputs(batch, model_prefix)
            outputs = self.model(**inputs)
            if self.model.config.num_labels == 1:
                output[model] = F.sigmoid(outputs.logits.squeeze().detach()).tolist()
            else:
                output[model] = F.softmax(outputs.logits.squeeze().detach(), dim=1).tolist()
        if isinstance(list(output.values())[0], list):
            result = [dict(zip(output.keys(), values)) for values in zip(*output.values())]
        else:
            result = [output]
        return result
