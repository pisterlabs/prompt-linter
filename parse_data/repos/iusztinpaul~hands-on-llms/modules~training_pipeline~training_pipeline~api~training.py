import logging
from pathlib import Path
from typing import Optional, Tuple

import comet_ml
from datasets import Dataset
from peft import PeftConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EvalPrediction,
    TrainingArguments,
)
from trl import SFTTrainer

from training_pipeline import constants, metrics, models
from training_pipeline.configs import TrainingConfig
from training_pipeline.data import qa

logger = logging.getLogger(__name__)


class ToModelRegistryMixin:
    def __init__(self, model_id: str):
        self._model_id = model_id

    @property
    def model_name(self) -> str:
        return f"financial_assistant/{self._model_id}"

    def to_model_registry(self, checkpoint_dir: Path):
        checkpoint_dir = checkpoint_dir.resolve()

        assert (
            checkpoint_dir.exists()
        ), f"Checkpoint directory {checkpoint_dir} does not exist"

        experiment = comet_ml.Experiment()
        logger.debug(f"Starting logging model checkpoint @ {self.model_name}")
        experiment.log_model(self.model_name, str(checkpoint_dir))
        logger.debug(f"Finished logging model checkpoint @ {self.model_name}")


class ModelRegistryAPI(ToModelRegistryMixin):
    pass


class TrainingAPI(ToModelRegistryMixin):
    def __init__(
        self,
        root_dataset_dir: Path,
        model_id: str,
        template_name: str,
        training_arguments: TrainingArguments,
        name: str = "training-api",
        max_seq_length: int = 1024,
        debug: bool = False,
        model_cache_dir: Path = constants.CACHE_DIR,
    ):
        super().__init__(model_id=model_id)

        self._root_dataset_dir = root_dataset_dir
        self._model_id = model_id
        self._template_name = template_name
        self._training_arguments = training_arguments
        self._name = name
        self._max_seq_length = max_seq_length
        self._debug = debug
        self._model_cache_dir = model_cache_dir

        self._training_dataset, self._validation_dataset = self.load_data()
        self._model, self._tokenizer, self._peft_config = self.load_model()

    @classmethod
    def from_config(
        cls,
        config: TrainingConfig,
        root_dataset_dir: Path,
        model_cache_dir: Optional[Path] = None,
    ):
        return cls(
            root_dataset_dir=root_dataset_dir,
            model_id=config.model["id"],
            template_name=config.model["template"],
            training_arguments=config.training,
            max_seq_length=config.model["max_seq_length"],
            debug=config.setup["debug"],
            model_cache_dir=model_cache_dir,
        )

    def load_data(self) -> Tuple[Dataset, Dataset]:
        logger.info(f"Loading QA datasets from {self._root_dataset_dir=}")

        if self._debug:
            logger.info("Debug mode enabled. Truncating datasets...")

            training_max_samples = 12
            validation_max_samples = 6
        else:
            training_max_samples = None
            # To avoid waiting an eternity to run the evaluation we will only use a subset of the validation dataset.
            validation_max_samples = 75

        training_dataset = qa.FinanceDataset(
            data_path=self._root_dataset_dir / "training_data.json",
            template=self._template_name,
            scope=constants.Scope.TRAINING,
            max_samples=training_max_samples,
        ).to_huggingface()
        validation_dataset = qa.FinanceDataset(
            data_path=self._root_dataset_dir / "testing_data.json",
            template=self._template_name,
            scope=constants.Scope.TRAINING,
            max_samples=validation_max_samples,
        ).to_huggingface()

        logger.info(f"Training dataset size: {len(training_dataset)}")
        logger.info(f"Validation dataset size: {len(validation_dataset)}")

        return training_dataset, validation_dataset

    def load_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer, PeftConfig]:
        logger.info(f"Loading model using {self._model_id=}")
        model, tokenizer, peft_config = models.build_qlora_model(
            pretrained_model_name_or_path=self._model_id,
            gradient_checkpointing=True,
            cache_dir=self._model_cache_dir,
        )

        return model, tokenizer, peft_config

    def train(self) -> SFTTrainer:
        logger.info("Training model...")

        # TODO: Handle this error: "Token indices sequence length is longer than the specified maximum sequence length
        # for this model (2302 > 2048). Running this sequence through the model will result in indexing errors"
        trainer = SFTTrainer(
            model=self._model,
            train_dataset=self._training_dataset,
            eval_dataset=self._validation_dataset,
            peft_config=self._peft_config,
            dataset_text_field="prompt",
            max_seq_length=self._max_seq_length,
            tokenizer=self._tokenizer,
            args=self._training_arguments,
            packing=True,
            compute_metrics=self.compute_metrics,
        )
        trainer.train()

        best_model_checkpoint = trainer.state.best_model_checkpoint
        has_best_model_checkpoint = best_model_checkpoint is not None
        if has_best_model_checkpoint:
            best_model_checkpoint = Path(best_model_checkpoint)
            logger.info(
                f"Logging best model from {best_model_checkpoint} to the model registry..."
            )

            self.to_model_registry(best_model_checkpoint)
        else:
            logger.warning(
                "No best model checkpoint found. Skipping logging it to the model registry..."
            )

        return trainer

    def compute_metrics(self, eval_pred: EvalPrediction):
        return {"perplexity": metrics.compute_perplexity(eval_pred.predictions)}
