from django.contrib.postgres.fields import ArrayField
from django.db import models
from django.apps import apps
from simple_history.models import HistoricalRecords
from back.apps.language_model.tasks import generate_embeddings_task

from back.apps.language_model.models.data import KnowledgeItem, KnowledgeBase
from back.common.models import ChangesMixin

from logging import getLogger

logger = getLogger(__name__)

LLM_CHOICES = (
    ('local_cpu', 'Local CPU Model'),  # GGML models optimized for CPU inference
    ('local_gpu', 'Local GPU Model'),  # Use locally VLLM or HuggingFace for GPU inference.
    ('vllm', 'VLLM Client'),  # Access VLLM engine remotely
    ('openai', 'OpenAI Model'),  # ChatGPT models from OpenAI
    ('claude', 'Claude Model')  # Claude models from Anthropic
)


class RAGConfig(ChangesMixin):
    """
    It relates the different elements to create a RAG (Retrieval Augmented Generation) pipeline
    """
    name = models.CharField(max_length=255, unique=True)
    knowledge_base = models.ForeignKey(KnowledgeBase, on_delete=models.CASCADE)
    llm_config = models.ForeignKey("LLMConfig", on_delete=models.PROTECT)
    prompt_config = models.ForeignKey("PromptConfig", on_delete=models.PROTECT)
    generation_config = models.ForeignKey("GenerationConfig", on_delete=models.PROTECT)
    retriever_config = models.ForeignKey("RetrieverConfig", on_delete=models.PROTECT)

    def __str__(self):
        return self.name if self.name is not None else f"{self.llm_config.name} - {self.knowledge_base.name}"

    # When saving we want to check if the knowledge_base or the retriever_config has changed and in that case regenerate
    # all the embeddings for it.
    def save(self, *args, **kwargs):
        generate_and_load = False # flag to indicate if we need to generate the embeddings and load a new RAG

        if self.pk is None: # New rag config
            generate_and_load = True
            logger.info(f"New RAG config {self.name} being created...")
        else:
            old = RAGConfig.objects.get(pk=self.pk)
            if self.knowledge_base != old.knowledge_base:
                generate_and_load = True
                logger.info(f"RAG config {self.name} changed knowledge base...")
            if self.retriever_config.model_name != old.retriever_config.model_name:
                generate_and_load = True
                logger.info(f"RAG config {self.name} changed retriever model...")

        super().save(*args, **kwargs)

        if generate_and_load:
            self.trigger_generate_embeddings()

    def trigger_generate_embeddings(self):
        logger.info('Triggering generate embeddings task')
        # remove all the embeddings for this rag config
        Embedding = apps.get_model("language_model", "Embedding")
        Embedding.objects.filter(rag_config=self).delete()

        generate_embeddings_task.delay(
            list(KnowledgeItem.objects.filter(knowledge_base=self.knowledge_base).values_list("pk", flat=True)),
            self.pk,
            recache_models=True
        )


class RetrieverConfig(ChangesMixin):
    """
    A config with all the settings to configure the retriever.
    name: str
        Just a name for the retriever.
    model_name: str
        The name of the retriever model to use. It must be a HuggingFace repo id.
    batch_size: int
        The batch size to use for the retriever.
    """
    DEVICE_CHOICES = (
        ('cpu', 'CPU'),
        ('cuda', 'GPU'),
    )

    name = models.CharField(max_length=255, unique=True)
    model_name = models.CharField(max_length=255, default="intfloat/e5-small-v2") # For dev and demo purposes.
    batch_size = models.IntegerField(default=1) # batch size 1 for better default cpu generation
    device = models.CharField(max_length=10, choices=DEVICE_CHOICES, default="cpu")

    def __str__(self):
        return self.name

    # When saving we want to check if the model_name has changed and in that case regenerate all the embeddings for the
    # knowledge bases that uses this retriever.
    def save(self, *args, **kwargs):
        logger.info('Checking if we need to generate embeddings because of a retriever config change')
        generated_embeddings = False
        if self.pk is not None:
            old_retriever = RetrieverConfig.objects.get(pk=self.pk)
            if self.model_name != old_retriever.model_name:
                generated_embeddings = True
        super().save(*args, **kwargs)
        if generated_embeddings:
            self.trigger_generate_embeddings()

    def trigger_generate_embeddings(self):
        rag_configs = RAGConfig.objects.filter(retriever_config=self)
        Embeddings = apps.get_model("language_model", "Embedding")

        last_i = rag_configs.count() - 1
        for i, rag_config in enumerate(rag_configs.all()):
            # check the rag configs that use this retriever
            if rag_config.retriever_config == self:
                logger.info(f"Triggering generate embeddings task for RAG config {rag_config} because of a retriever config change")
                # remove all the embeddings for this rag config
                Embeddings.objects.filter(rag_config=rag_config).delete()
                generate_embeddings_task.delay(
                    list(
                        KnowledgeItem.objects.filter(knowledge_base=rag_config.knowledge_base).values_list("pk", flat=True)
                    ),
                    rag_config.pk,
                    recache_models=(i == last_i) # recache models if we are in the last iteration
                )


class LLMConfig(ChangesMixin):
    """
    A model config with all the settings to configure an LLM.
    name: str
        Just a name for the model.
    llm_type: str
        The type of LLM to use.
    llm_name: str
        The name of the LLM to use. It can be a HuggingFace repo id, an OpenAI model id, etc.
    ggml_model_filename: str
        The GGML filename of the model, if it is a GGML model.
    model_config: str
        The huggingface model config of the model, needed for GGML models.
    load_in_8bit: bool
        Whether to load the model in 8bit or not.
    use_fast_tokenizer: bool
        Whether to use the fast tokenizer or not.
    trust_remote_code_tokenizer: bool
        Whether to trust the remote code for the tokenizer or not.
    trust_remote_code_model: bool
        Whether to trust the remote code for the model or not.
    revision: str
        The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a git-based system for storing models
    model_max_length: int
        The maximum length of the model.
    """

    name = models.CharField(max_length=255, unique=True)
    llm_type = models.CharField(max_length=10, choices=LLM_CHOICES, default="local_gpu")
    llm_name = models.CharField(max_length=100, default="gpt2")
    ggml_llm_filename = models.CharField(max_length=255, blank=True, null=True)
    model_config = models.CharField(max_length=255, blank=True, null=True)
    load_in_8bit = models.BooleanField(default=False)
    use_fast_tokenizer = models.BooleanField(default=True)
    trust_remote_code_tokenizer = models.BooleanField(default=False)
    trust_remote_code_model = models.BooleanField(default=False)
    revision = models.CharField(max_length=255, blank=True, null=True, default="main")
    model_max_length = models.IntegerField(blank=True, null=True)

    def __str__(self):
        return self.name


class PromptConfig(ChangesMixin):
    """
    Defines the structure of the prompt for a model.
    system_prefix : str
        The prefix to indicate instructions for the LLM.
    system_tag : str
        The tag to indicate the start of the system prefix for the LLM.
    system_end : str
        The tag to indicate the end of the system prefix for the LLM.
    user_tag : str
        The tag to indicate the start of the user input.
    user_end : str
        The tag to indicate the end of the user input.
    assistant_tag : str
        The tag to indicate the start of the assistant output.
    assistant_end : str
        The tag to indicate the end of the assistant output.
    n_contexts_to_use : int, optional
        The number of contexts to use, by default 3
    """
    name = models.CharField(max_length=255, unique=True)
    system_prefix = models.TextField(blank=True, default="")
    system_tag = models.CharField(max_length=255, blank=True, default="")
    system_end = models.CharField(max_length=255, blank=True, default="")
    user_tag = models.CharField(max_length=255, blank=True, default="<|prompt|>")
    user_end = models.CharField(max_length=255, blank=True, default="")
    assistant_tag = models.CharField(max_length=255, blank=True, default="<|answer|>")
    assistant_end = models.CharField(max_length=255, blank=True, default="")
    n_contexts_to_use = models.IntegerField(default=3)
    history = HistoricalRecords()

    def __str__(self):
        return self.name


class GenerationConfig(ChangesMixin):
    """
    Defines the generation configuration for a model.
    top_k : int, optional
        The number of tokens to consider for the top-k sampling, by default 50
    top_p : float, optional
        The cumulative probability for the top-p sampling, by default 1.0
    temperature : float, optional
        The temperature for the sampling, by default 0.2
    repetition_penalty : float, optional
        The repetition penalty for the sampling, by default 1.0
    seed : int, optional
        The seed for the sampling, by default 42
    max_new_tokens : int, optional
        The maximum number of new tokens to generate, by default 256
    model : Model
        The model this generation configuration belongs to.
    """
    name = models.CharField(max_length=255, unique=True)
    top_k = models.IntegerField(default=50)
    top_p = models.FloatField(default=1.0)
    temperature = models.FloatField(default=0.2)
    repetition_penalty = models.FloatField(default=1.0)
    seed = models.IntegerField(default=42)
    max_new_tokens = models.IntegerField(default=256)

    def __str__(self):
        return self.name
