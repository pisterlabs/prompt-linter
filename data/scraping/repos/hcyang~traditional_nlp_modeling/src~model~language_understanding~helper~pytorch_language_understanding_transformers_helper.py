# -*- coding: utf-8 -*-
# ---- NOTE-OPTIONAL-CODING ----
# -*- coding: latin-1 -*-
"""
This module implements helper function to manage Pytorch Transformer models.
"""
# ---- NOTE-PYLINT ---- C0302: Too many lines in module
# pylint: disable=C0302
# ---- NOTE-PYLINT ---- E0401: Unable to import 'transformers.optimization' (import-error)
# pylint: disable=E0401

from typing import Dict
from typing import List
from typing import Set
from typing import Tuple
from typing import Pattern

import regex

# ---- NOTE-conditional-for-FP16 ---- try:
# ---- NOTE-conditional-for-FP16 ----     from apex import amp
# ---- NOTE-conditional-for-FP16 ---- except ImportError:
# ---- NOTE-conditional-for-FP16 ----     # ---- NOTE-PYLINT ---- C0301: Line too long
# ---- NOTE-conditional-for-FP16 ----     # pylint: disable=C0301
# ---- NOTE-conditional-for-FP16 ----     raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use FP16 training.")

import transformers

from transformers \
    import DataCollatorForLanguageModeling

from transformers \
    import PreTrainedTokenizer

# ---- from transformers import PretrainedConfig

from transformers import BertConfig
# ---- from transformers import BertEmbeddings
# ---- from transformers import BertSelfAttention
# ---- from transformers import BertSelfOutput
# ---- from transformers import BertAttention
# ---- from transformers import BertIntermediate
# ---- from transformers import BertOutput
# ---- from transformers import BertLayer
# ---- from transformers import BertEncoder
# ---- from transformers import BertPooler
# ---- from transformers import BertPredictionHeadTransform
# ---- from transformers import BertLMPredictionHead
# ---- from transformers import BertOnlyMLMHead
# ---- from transformers import BertOnlyNSPHead
# ---- from transformers import BertPreTrainingHeads
# ---- from transformers import BertPreTrainedModel
# ---- from transformers import BertForPreTrainingOutput
# ---- from transformers import BertLMHeadModel
# ==== NOTE-FOR-SPECIFIC-NLP-TASKS ====
from transformers import BertPreTrainedModel
from transformers import BertModel
from transformers import BertForPreTraining
from transformers import BertForMaskedLM
from transformers import BertForNextSentencePrediction
from transformers import BertForSequenceClassification
from transformers import BertForMultipleChoice
from transformers import BertForTokenClassification
from transformers import BertForQuestionAnswering

from transformers import RobertaConfig
# ---- from transformers import RobertaEmbeddings
# ---- from transformers import RobertaSelfAttention
# ---- from transformers import RobertaSelfOutput
# ---- from transformers import RobertaAttention
# ---- from transformers import RobertaIntermediate
# ---- from transformers import RobertaOutput
# ---- from transformers import RobertaLayer
# ---- from transformers import RobertaEncoder
# ---- from transformers import RobertaPooler
# ---- from transformers import RobertaPreTrainedModel
# ==== NOTE-FOR-SPECIFIC-NLP-TASKS ====
from transformers import RobertaPreTrainedModel
from transformers import RobertaModel
# ---- from transformers import RobertaForCausalLM
from transformers import RobertaForMaskedLM
# ---- from transformers import RobertaLMHead
from transformers import RobertaForSequenceClassification
from transformers import RobertaForMultipleChoice
from transformers import RobertaForTokenClassification
# ---- from transformers import RobertaClassificationHead
from transformers import RobertaForQuestionAnswering
# ====     from transformers import RobertaForPreTraining
# ====     from transformers import RobertaForNextSentencePrediction

from transformers import GPT2Config
# ==== NOTE-FOR-SPECIFIC-NLP-TASKS ====
from transformers import GPT2PreTrainedModel
from transformers import GPT2Model
from transformers import GPT2LMHeadModel
from transformers import GPT2DoubleHeadsModel

from transformers import OpenAIGPTConfig
# ==== NOTE-FOR-SPECIFIC-NLP-TASKS ====
from transformers import OpenAIGPTPreTrainedModel
from transformers import OpenAIGPTModel
from transformers import OpenAIGPTLMHeadModel
from transformers import OpenAIGPTDoubleHeadsModel

from transformers import TransfoXLConfig
# ==== NOTE-FOR-SPECIFIC-NLP-TASKS ====
from transformers import TransfoXLPreTrainedModel
from transformers import TransfoXLModel
from transformers import TransfoXLLMHeadModel

from transformers import XLMConfig
# ==== NOTE-FOR-SPECIFIC-NLP-TASKS ====
from transformers import XLMPreTrainedModel
from transformers import XLMModel
from transformers import XLMWithLMHeadModel
from transformers import XLMForSequenceClassification
from transformers import XLMForQuestionAnswering

from transformers import XLNetConfig
# ---- NOTE-FOR-SPECIFIC-NLP-TASKS ====
from transformers import XLNetPreTrainedModel
from transformers import XLNetModel
from transformers import XLNetLMHeadModel
from transformers import XLNetForSequenceClassification
from transformers import XLNetForQuestionAnswering

from transformers import BertTokenizer
from transformers import RobertaTokenizer
from transformers import GPT2Tokenizer
from transformers import OpenAIGPTTokenizer
from transformers import TransfoXLTokenizer
from transformers import XLMTokenizer
from transformers import XLNetTokenizer

from transformers.models.bert.modeling_bert \
    import BERT_PRETRAINED_MODEL_ARCHIVE_LIST \
        as PytorchTransformersBert_PRETRAINED_MODEL_ARCHIVE_LIST
from transformers.models.bert.configuration_bert \
    import BERT_PRETRAINED_CONFIG_ARCHIVE_MAP \
        as PytorchTransformersBert_PRETRAINED_CONFIG_ARCHIVE_MAP

from transformers.models.roberta.modeling_roberta \
    import ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST \
        as PytorchTransformersRoberta_PRETRAINED_MODEL_ARCHIVE_LIST
from transformers.models.roberta.configuration_roberta \
    import ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP \
        as PytorchTransformersRoberta_PRETRAINED_CONFIG_ARCHIVE_MAP

from transformers.models.gpt2.modeling_gpt2 \
    import GPT2_PRETRAINED_MODEL_ARCHIVE_LIST \
        as PytorchTransformersGpt2_PRETRAINED_MODEL_ARCHIVE_LIST
from transformers.models.gpt2.configuration_gpt2 \
    import GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP \
        as PytorchTransformersGpt2_PRETRAINED_CONFIG_ARCHIVE_MAP

from transformers.models.openai.modeling_openai \
    import OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_LIST \
        as PytorchTransformersOpenai_PRETRAINED_MODEL_ARCHIVE_LIST
from transformers.models.openai.configuration_openai \
    import OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP \
        as PytorchTransformersOpenai_PRETRAINED_CONFIG_ARCHIVE_MAP

from transformers.models.transfo_xl.modeling_transfo_xl \
    import TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_LIST \
        as PytorchTransformersTransfoXl_PRETRAINED_MODEL_ARCHIVE_LIST
from transformers.models.transfo_xl.configuration_transfo_xl \
    import TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP \
        as PytorchTransformersTransfoXl_PRETRAINED_CONFIG_ARCHIVE_MAP

from transformers.models.xlm.modeling_xlm \
    import XLM_PRETRAINED_MODEL_ARCHIVE_LIST \
        as PytorchTransformersXml_PRETRAINED_MODEL_ARCHIVE_LIST
from transformers.models.xlm.configuration_xlm \
    import XLM_PRETRAINED_CONFIG_ARCHIVE_MAP \
        as PytorchTransformersXml_PRETRAINED_CONFIG_ARCHIVE_MAP

from transformers.models.xlnet.modeling_xlnet \
    import XLNET_PRETRAINED_MODEL_ARCHIVE_LIST \
        as PytorchTransformersXlnet_PRETRAINED_MODEL_ARCHIVE_LIST
from transformers.models.xlnet.configuration_xlnet \
    import XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP \
        as PytorchTransformersXlnet_PRETRAINED_CONFIG_ARCHIVE_MAP

from transformers.models.bert.tokenization_bert \
    import VOCAB_FILES_NAMES \
        as PytorchTransformersBert_VOCAB_FILES_NAMES
from transformers.models.bert.tokenization_bert \
    import PRETRAINED_VOCAB_FILES_MAP \
        as PytorchTransformersBert_PRETRAINED_VOCAB_FILES_MAP
from transformers.models.bert.tokenization_bert \
    import PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES \
        as PytorchTransformersBert_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
from transformers.models.bert.tokenization_bert \
    import PRETRAINED_INIT_CONFIGURATION \
        as PytorchTransformersBert_PRETRAINED_INIT_CONFIGURATION

from transformers.models.roberta.tokenization_roberta \
    import VOCAB_FILES_NAMES \
        as PytorchTransformersRoberta_VOCAB_FILES_NAMES
from transformers.models.roberta.tokenization_roberta \
    import PRETRAINED_VOCAB_FILES_MAP \
        as PytorchTransformersRoberta_PRETRAINED_VOCAB_FILES_MAP
from transformers.models.roberta.tokenization_roberta \
    import PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES \
        as PytorchTransformersRoberta_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
# ==== NOTE-FOR-SPECIFIC-NLP-TASKS ====
# ====     from transformers.models.roberta.tokenization_roberta \
# ====         import PRETRAINED_INIT_CONFIGURATION \
# ====             as PytorchTransformersRoberta_PRETRAINED_INIT_CONFIGURATION

from transformers.models.gpt2.tokenization_gpt2 \
    import VOCAB_FILES_NAMES \
        as PytorchTransformersGpt2_VOCAB_FILES_NAMES
from transformers.models.gpt2.tokenization_gpt2 \
    import PRETRAINED_VOCAB_FILES_MAP \
        as PytorchTransformersGpt2_PRETRAINED_VOCAB_FILES_MAP
from transformers.models.gpt2.tokenization_gpt2 \
    import PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES \
        as PytorchTransformersGpt2_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
# ==== NOTE-FOR-SPECIFIC-NLP-TASKS ====
# ====     from transformers.models.gpt2.tokenization_gpt2 \
# ====         import PRETRAINED_INIT_CONFIGURATION \
# ====             as PytorchTransformersGpt2_PRETRAINED_INIT_CONFIGURATION

from transformers.models.openai.tokenization_openai \
    import VOCAB_FILES_NAMES \
        as PytorchTransformersOpenai_VOCAB_FILES_NAMES
from transformers.models.openai.tokenization_openai \
    import PRETRAINED_VOCAB_FILES_MAP \
        as PytorchTransformersOpenai_PRETRAINED_VOCAB_FILES_MAP
from transformers.models.openai.tokenization_openai \
    import PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES \
        as PytorchTransformersOpenai_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
# ==== NOTE-FOR-SPECIFIC-NLP-TASKS ====
# ====     from transformers.models.openai.tokenization_openai \
# ====         import PRETRAINED_INIT_CONFIGURATION \
# ====             as PytorchTransformersOpenai_PRETRAINED_INIT_CONFIGURATION

from transformers.models.transfo_xl.tokenization_transfo_xl \
    import VOCAB_FILES_NAMES \
        as PytorchTransformersTransfoXl_VOCAB_FILES_NAMES
from transformers.models.transfo_xl.tokenization_transfo_xl \
    import PRETRAINED_VOCAB_FILES_MAP \
        as PytorchTransformersTransfoXl_PRETRAINED_VOCAB_FILES_MAP
from transformers.models.transfo_xl.tokenization_transfo_xl \
    import PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES \
        as PytorchTransformersTransfoXl_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
# ==== NOTE-FOR-SPECIFIC-NLP-TASKS ====
# ====     from transformers.models.transfo_xl.tokenization_transfo_xl \
# ====         import PRETRAINED_INIT_CONFIGURATION \
# ====             as PytorchTransformersTransfoXl_PRETRAINED_INIT_CONFIGURATION
# ---- NOTE-NOT-USED-YET ---- from transformers.models.transfo_xl.tokenization_transfo_xl \
# ---- NOTE-NOT-USED-YET ----     import PRETRAINED_CORPUS_ARCHIVE_MAP \
# ---- NOTE-NOT-USED-YET ----         as PytorchTransformersTransfoXl_PRETRAINED_CORPUS_ARCHIVE_MAP

from transformers.models.xlm.tokenization_xlm \
    import VOCAB_FILES_NAMES \
        as PytorchTransformersXml_VOCAB_FILES_NAMES
from transformers.models.xlm.tokenization_xlm \
    import PRETRAINED_VOCAB_FILES_MAP \
        as PytorchTransformersXml_PRETRAINED_VOCAB_FILES_MAP
from transformers.models.xlm.tokenization_xlm \
    import PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES \
        as PytorchTransformersXml_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
from transformers.models.xlm.tokenization_xlm \
    import PRETRAINED_INIT_CONFIGURATION \
        as PytorchTransformersXml_PRETRAINED_INIT_CONFIGURATION

from transformers.models.xlnet.tokenization_xlnet \
    import VOCAB_FILES_NAMES \
        as PytorchTransformersXlnet_VOCAB_FILES_NAMES
from transformers.models.xlnet.tokenization_xlnet \
    import PRETRAINED_VOCAB_FILES_MAP \
        as PytorchTransformersXlnet_PRETRAINED_VOCAB_FILES_MAP
from transformers.models.xlnet.tokenization_xlnet \
    import PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES \
        as PytorchTransformersXlnet_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
# from transformers.models.xlnet.tokenization_xlnet \
#     import PRETRAINED_INIT_CONFIGURATION \
#         as PytorchTransformersXlnet_PRETRAINED_INIT_CONFIGURATION

from transformers.optimization import AdamW

from model.language_understanding.featurizer.\
    data_feature_manager_single_label_ngram_subword_tokenization \
    import TokenizationNGramSubwordSingleLabelDataFeatureManager

from utility.datatype_helper.datatype_helper \
    import DatatypeHelper
from utility.debugging_helper.debugging_helper \
    import DebuggingHelper

class PytorchLanguageUnderstandingTransformersPretainedModelHelper:
    """
    This class defined helper functions and data structures
    to manage Pytorch Transformers models
    """
    # ---- NOTE-PYLINT ---- R0904: Too many public methods
    # pylint: disable=R0904
    # ---- NOTE-PYLINT ---- W1113: Keyword argument before variable positional arguments list
    # ---- NOTE-PYLINT ----        in the definition of function (keyword-arg-before-vararg)
    # pylint: disable=W1113

    @staticmethod
    def create_data_collator_for_language_modeling( \
        tokenizer: PreTrainedTokenizer, \
        mlm_probability: float = 0.2) -> DataCollatorForLanguageModeling:
        """
        create_data_collator_for_language_modeling()
        """
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=mlm_probability)
        return data_collator

    @staticmethod
    def convert_token_lists_to_ids(token_lists: List[List[str]], tokenizer) -> List[List[str]]:
        """
        Convert a list of list of tokens into vocab ids.
        """
        token_id_lists: List[List[str]] = \
            [PytorchLanguageUnderstandingTransformersPretainedModelHelper.convert_tokens_to_ids(x, tokenizer) for x in token_lists]
        return token_id_lists

    @staticmethod
    def convert_tokens_to_ids(tokens: List[str], tokenizer) -> List[str]:
        """
        Convert a list of tokens into vocab ids.
        """
        token_ids: List[List[str]] = \
            tokenizer.convert_tokens_to_ids(tokens)
        return token_ids

    @staticmethod
    def tokenize( \
        text: str, \
        tokenizer, \
        add_space_prefix: bool = False) -> List[List[str]]:
        """
        Segnment a text string first and tokenize each string individually.
        """
        return PytorchLanguageUnderstandingTransformersPretainedModelHelper.tokenize_with_transformers_segmentation_regex( \
            text, \
            tokenizer, \
            add_space_prefix)
        # return PytorchLanguageUnderstandingTransformersPretainedModelHelper.tokenize_with_punctuations(\
        #     text, \
        #     tokenizer, \
        #     add_space_prefix)

    @staticmethod
    def tokenize_with_transformers_segmentation_regex( \
        text: str, \
        tokenizer, \
        add_space_prefix: bool = False) -> List[List[str]]:
        """
        Segnment a text string using Transformers segementation regex logic first and tokenize each string individually.
        """
        token_pieces: List[str] = \
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.segment_with_transformers_segmentation_regex(text)
        return PytorchLanguageUnderstandingTransformersPretainedModelHelper.tokenize_to_token_pieces(token_pieces, tokenizer, add_space_prefix)
    @staticmethod
    def tokenize_with_punctuations( \
        text: str, \
        tokenizer, \
        add_space_prefix: bool = False) -> List[List[str]]:
        """
        Segnment a text string with printable punctuations first and tokenize each string individually.
        """
        token_pieces: List[str] = \
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.segment_with_punctuations(text)
        return PytorchLanguageUnderstandingTransformersPretainedModelHelper.tokenize_to_token_pieces(token_pieces, tokenizer, add_space_prefix)

    # _transformers_segmentation_regular_expression_definition: str = \
    #     r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
    _transformers_segmentation_regular_expression_definition: str = \
        r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    _transformers_segmentation_regular_expression: Pattern = \
        regex.compile(_transformers_segmentation_regular_expression_definition)

    @staticmethod
    def segment_with_transformers_segmentation_regex(text: str) -> List[str]:
        """
        Segnment a text string using Transformers regex logic.
        """
        segmented_pieces: List[str] = \
            PytorchLanguageUnderstandingTransformersPretainedModelHelper._transformers_segmentation_regular_expression.findall(text)
        DebuggingHelper.write_line_to_system_console_out(\
            f'segmented_pieces={segmented_pieces}')
        return segmented_pieces
    @staticmethod
    def segment_with_punctuations(text: str) -> List[str]:
        """
        Segnment a text string on punctuations.
        """
        segmented_pieces: List[str] = \
            TokenizationNGramSubwordSingleLabelDataFeatureManager.segment_input_to_components_with_language_token_punctuation_delimiters(text)
        DebuggingHelper.write_line_to_system_console_out(\
            f'segmented_pieces={segmented_pieces}')
        return segmented_pieces

    @staticmethod
    def tokenize_to_token_pieces( \
        token_pieces: List[str], \
        tokenizer, \
        add_space_prefix: bool = False) -> List[List[str]]:
        """
        Tokenize a collection of tokens individually.
        """
        if add_space_prefix:
            return [tokenizer.tokenize(PytorchLanguageUnderstandingTransformersPretainedModelHelper.text_prepend_space_prefix(tokenpiece)) for tokenpiece in token_pieces]
        return [tokenizer.tokenize(tokenpiece) for tokenpiece in token_pieces]
        # ---- NOTE-MAY-NOT-BE-CONSISTENT-WITH-WHOLE-UTTERANCE-TOKENIZATION ---- return [tokenizer.tokenize(tokenpiece.strip()) for tokenpiece in token_pieces]

    @staticmethod
    def text_prepend_space_prefix(text: str) -> str:
        """
        Add a space prefix is necessary. This is to comply with Roberta tokenizer particularity.
        """
        if text[0] != ' ':
            text = ' ' + text
        return text

    @staticmethod
    def get_pretrained_model_list_bert() -> List[str]:
        """
        Return a list of BERT pretrained model map.
        """
        model_list: List[str] = \
            PytorchTransformersBert_PRETRAINED_MODEL_ARCHIVE_LIST
        # DebuggingHelper.write_line_to_system_console_out(\
        #     f'model_list={model_list}')
        return model_list
    @staticmethod
    def get_pretrained_model_list_roberta() -> List[str]:
        """
        Return a list of ROBERTA pretrained model map.
        """
        model_list: List[str] = \
            PytorchTransformersRoberta_PRETRAINED_MODEL_ARCHIVE_LIST
        # DebuggingHelper.write_line_to_system_console_out(\
        #     f'model_list={model_list}')
        return model_list
    @staticmethod
    def get_pretrained_model_list_gpt2() -> List[str]:
        """
        Return a list of GPT2 pretrained model map.
        """
        model_list: List[str] = \
            PytorchTransformersGpt2_PRETRAINED_MODEL_ARCHIVE_LIST
        # DebuggingHelper.write_line_to_system_console_out(\
        #     f'model_list={model_list}')
        return model_list
    @staticmethod
    def get_pretrained_model_list_openai() -> List[str]:
        """
        Return a list of OPENAI pretrained model map.
        """
        model_list: List[str] = \
            PytorchTransformersOpenai_PRETRAINED_MODEL_ARCHIVE_LIST
        # DebuggingHelper.write_line_to_system_console_out(\
        #     f'model_list={model_list}')
        return model_list
    @staticmethod
    def get_pretrained_model_list_transfo_xl() -> List[str]:
        """
        Return a list of transfo-xl pretrained model map.
        """
        model_list: List[str] = \
            PytorchTransformersTransfoXl_PRETRAINED_MODEL_ARCHIVE_LIST
        # DebuggingHelper.write_line_to_system_console_out(\
        #     f'model_list={model_list}')
        return model_list
    @staticmethod
    def get_pretrained_model_list_xlm() -> List[str]:
        """
        Return a list of XLM pretrained model map.
        """
        model_list: List[str] = \
            PytorchTransformersXml_PRETRAINED_MODEL_ARCHIVE_LIST
        # DebuggingHelper.write_line_to_system_console_out(\
        #     f'model_list={model_list}')
        return model_list
    @staticmethod
    def get_pretrained_model_list_xlnet() -> List[str]:
        """
        Return a list of XLNET pretrained model map.
        """
        model_list: List[str] = \
            PytorchTransformersXlnet_PRETRAINED_MODEL_ARCHIVE_LIST
        # DebuggingHelper.write_line_to_system_console_out(\
        #     f'model_list={model_list}')
        return model_list

    _pretrained_model_list: List[str] = None
    @staticmethod
    def get_pretrained_model_list() -> List[str]:
        """
        Return a list of pretrained model map.
        """
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        if PytorchLanguageUnderstandingTransformersPretainedModelHelper._pretrained_model_list is None:
            model_list: List[str] = []
            model_list.append(PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_model_list_bert())
            model_list.append(PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_model_list_roberta())
            model_list.append(PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_model_list_gpt2())
            model_list.append(PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_model_list_openai())
            model_list.append(PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_model_list_transfo_xl())
            model_list.append(PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_model_list_xlm())
            model_list.append(PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_model_list_xlnet())
            PytorchLanguageUnderstandingTransformersPretainedModelHelper._pretrained_model_list = model_list
        return PytorchLanguageUnderstandingTransformersPretainedModelHelper._pretrained_model_list

    @staticmethod
    def get_pretrained_model_map_bert() -> Dict[str, str]:
        """
        Return a list of BERT pretrained model map.
        """
        model_map: Dict[str, str] = \
            PytorchTransformersBert_PRETRAINED_CONFIG_ARCHIVE_MAP
        # DebuggingHelper.write_line_to_system_console_out(\
        #     f'model_map={model_map}')
        return model_map
    @staticmethod
    def get_pretrained_model_map_roberta() -> Dict[str, str]:
        """
        Return a list of ROBERTA pretrained model map.
        """
        model_map: Dict[str, str] = \
            PytorchTransformersRoberta_PRETRAINED_CONFIG_ARCHIVE_MAP
        # DebuggingHelper.write_line_to_system_console_out(\
        #     f'model_map={model_map}')
        return model_map
    @staticmethod
    def get_pretrained_model_map_gpt2() -> Dict[str, str]:
        """
        Return a list of GPT2 pretrained model map.
        """
        model_map: Dict[str, str] = \
            PytorchTransformersGpt2_PRETRAINED_CONFIG_ARCHIVE_MAP
        # DebuggingHelper.write_line_to_system_console_out(\
        #     f'model_map={model_map}')
        return model_map
    @staticmethod
    def get_pretrained_model_map_openai() -> Dict[str, str]:
        """
        Return a list of OPENAI pretrained model map.
        """
        model_map: Dict[str, str] = \
            PytorchTransformersOpenai_PRETRAINED_CONFIG_ARCHIVE_MAP
        # DebuggingHelper.write_line_to_system_console_out(\
        #     f'model_map={model_map}')
        return model_map
    @staticmethod
    def get_pretrained_model_map_transfo_xl() -> Dict[str, str]:
        """
        Return a list of transfo-xl pretrained model map.
        """
        model_map: Dict[str, str] = \
            PytorchTransformersTransfoXl_PRETRAINED_CONFIG_ARCHIVE_MAP
        # DebuggingHelper.write_line_to_system_console_out(\
        #     f'model_map={model_map}')
        return model_map
    @staticmethod
    def get_pretrained_model_map_xlm() -> Dict[str, str]:
        """
        Return a list of XLM pretrained model map.
        """
        model_map: Dict[str, str] = \
            PytorchTransformersXml_PRETRAINED_CONFIG_ARCHIVE_MAP
        # DebuggingHelper.write_line_to_system_console_out(\
        #     f'model_map={model_map}')
        return model_map
    @staticmethod
    def get_pretrained_model_map_xlnet() -> Dict[str, str]:
        """
        Return a list of XLNET pretrained model map.
        """
        model_map: Dict[str, str] = \
            PytorchTransformersXlnet_PRETRAINED_CONFIG_ARCHIVE_MAP
        # DebuggingHelper.write_line_to_system_console_out(\
        #     f'model_map={model_map}')
        return model_map

    _pretrained_model_map: Dict[str, str] = None
    @staticmethod
    def get_pretrained_model_map() -> Dict[str, str]:
        """
        Return a list of pretrained model map.
        """
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        if PytorchLanguageUnderstandingTransformersPretainedModelHelper._pretrained_model_map is None:
            model_map: Dict[str, str] = {}
            model_map.update(PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_model_map_bert())
            model_map.update(PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_model_map_roberta())
            model_map.update(PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_model_map_gpt2())
            model_map.update(PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_model_map_openai())
            model_map.update(PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_model_map_transfo_xl())
            model_map.update(PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_model_map_xlm())
            model_map.update(PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_model_map_xlnet())
            PytorchLanguageUnderstandingTransformersPretainedModelHelper._pretrained_model_map = model_map
        return PytorchLanguageUnderstandingTransformersPretainedModelHelper._pretrained_model_map

    @staticmethod
    def get_pretrained_model_config_map_bert() -> Dict[str, str]:
        """
        Return a list of BERT pretrained model config map.
        """
        model_config_map: Dict[str, str] = \
            PytorchTransformersBert_PRETRAINED_CONFIG_ARCHIVE_MAP
        # DebuggingHelper.write_line_to_system_console_out(\
        #     f'model_config_map={model_config_map}')
        return model_config_map
    @staticmethod
    def get_pretrained_model_config_map_roberta() -> Dict[str, str]:
        """
        Return a list of ROBERTA pretrained model config map.
        """
        model_config_map: Dict[str, str] = \
            PytorchTransformersRoberta_PRETRAINED_CONFIG_ARCHIVE_MAP
        # DebuggingHelper.write_line_to_system_console_out(\
        #     f'model_config_map={model_config_map}')
        return model_config_map
    @staticmethod
    def get_pretrained_model_config_map_gpt2() -> Dict[str, str]:
        """
        Return a list of GPT2 pretrained model config map.
        """
        model_config_map: Dict[str, str] = \
            PytorchTransformersGpt2_PRETRAINED_CONFIG_ARCHIVE_MAP
        # DebuggingHelper.write_line_to_system_console_out(\
        #     f'model_config_map={model_config_map}')
        return model_config_map
    @staticmethod
    def get_pretrained_model_config_map_openai() -> Dict[str, str]:
        """
        Return a list of OPENAI pretrained model config map.
        """
        model_config_map: Dict[str, str] = \
            PytorchTransformersOpenai_PRETRAINED_CONFIG_ARCHIVE_MAP
        # DebuggingHelper.write_line_to_system_console_out(\
        #     f'model_config_map={model_config_map}')
        return model_config_map
    @staticmethod
    def get_pretrained_model_config_map_transfo_xl() -> Dict[str, str]:
        """
        Return a list of transfo-xl pretrained model config map.
        """
        model_config_map: Dict[str, str] = \
            PytorchTransformersTransfoXl_PRETRAINED_CONFIG_ARCHIVE_MAP
        # DebuggingHelper.write_line_to_system_console_out(\
        #     f'model_config_map={model_config_map}')
        return model_config_map
    @staticmethod
    def get_pretrained_model_config_map_xlm() -> Dict[str, str]:
        """
        Return a list of XLM pretrained model config map.
        """
        model_config_map: Dict[str, str] = \
            PytorchTransformersXml_PRETRAINED_CONFIG_ARCHIVE_MAP
        # DebuggingHelper.write_line_to_system_console_out(\
        #     f'model_config_map={model_config_map}')
        return model_config_map
    @staticmethod
    def get_pretrained_model_config_map_xlnet() -> Dict[str, str]:
        """
        Return a list of XLNET pretrained model config map.
        """
        model_config_map: Dict[str, str] = \
            PytorchTransformersXlnet_PRETRAINED_CONFIG_ARCHIVE_MAP
        # DebuggingHelper.write_line_to_system_console_out(\
        #     f'model_config_map={model_config_map}')
        return model_config_map

    _pretrained_model_config_map: Dict[str, str] = None
    @staticmethod
    def get_pretrained_model_config_map() -> Dict[str, str]:
        """
        Return a list of pretrained model config map.
        """
        if PytorchLanguageUnderstandingTransformersPretainedModelHelper._pretrained_model_config_map is None:
            model_config_map: Dict[str, str] = {}
            model_config_map.update(PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_model_config_map_bert())
            model_config_map.update(PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_model_config_map_roberta())
            model_config_map.update(PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_model_config_map_gpt2())
            model_config_map.update(PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_model_config_map_openai())
            model_config_map.update(PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_model_config_map_transfo_xl())
            model_config_map.update(PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_model_config_map_xlm())
            model_config_map.update(PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_model_config_map_xlnet())
            PytorchLanguageUnderstandingTransformersPretainedModelHelper._pretrained_model_config_map = model_config_map
        return PytorchLanguageUnderstandingTransformersPretainedModelHelper._pretrained_model_config_map

    @staticmethod
    def get_pretrained_model_keys_bert() -> List[str]:
        """
        Return a list of BERT pretrained model keys.
        """
        model_map: Dict[str, str] = \
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_model_map_bert()
        # DebuggingHelper.write_line_to_system_console_out(\
        #     f'model_map={model_map}')
        return model_map.keys()
        # ---- NOTE: R1721: Unnecessary use of a comprehension ---- return [x for x in model_map]
    @staticmethod
    def get_pretrained_model_keys_roberta() -> List[str]:
        """
        Return a list of ROBERTA pretrained model keys.
        """
        model_map: Dict[str, str] = \
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_model_map_roberta()
        # DebuggingHelper.write_line_to_system_console_out(\
        #     f'model_map={model_map}')
        return model_map.keys()
        # ---- NOTE: R1721: Unnecessary use of a comprehension ---- return [x for x in model_map]
    @staticmethod
    def get_pretrained_model_keys_gpt2() -> List[str]:
        """
        Return a list of GPT2 pretrained model keys.
        """
        model_map: Dict[str, str] = \
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_model_map_gpt2()
        # DebuggingHelper.write_line_to_system_console_out(\
        #     f'model_map={model_map}')
        return model_map.keys()
        # ---- NOTE: R1721: Unnecessary use of a comprehension ---- return [x for x in model_map]
    @staticmethod
    def get_pretrained_model_keys_openai() -> List[str]:
        """
        Return a list of OPENAI pretrained model keys.
        """
        model_map: Dict[str, str] = \
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_model_map_openai()
        # DebuggingHelper.write_line_to_system_console_out(\
        #     f'model_map={model_map}')
        return model_map.keys()
        # ---- NOTE: R1721: Unnecessary use of a comprehension ---- return [x for x in model_map]
    @staticmethod
    def get_pretrained_model_keys_transfo_xl() -> List[str]:
        """
        Return a list of transfo-xl pretrained model keys.
        """
        model_map: Dict[str, str] = \
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_model_map_transfo_xl()
        # DebuggingHelper.write_line_to_system_console_out(\
        #     f'model_map={model_map}')
        return model_map.keys()
        # ---- NOTE: R1721: Unnecessary use of a comprehension ---- return [x for x in model_map]
    @staticmethod
    def get_pretrained_model_keys_xlm() -> List[str]:
        """
        Return a list of XLM pretrained model keys.
        """
        model_map: Dict[str, str] = \
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_model_map_xlm()
        # DebuggingHelper.write_line_to_system_console_out(\
        #     f'model_map={model_map}')
        return model_map.keys()
        # ---- NOTE: R1721: Unnecessary use of a comprehension ---- return [x for x in model_map]
    @staticmethod
    def get_pretrained_model_keys_xlnet() -> List[str]:
        """
        Return a list of XLNET pretrained model keys.
        """
        model_map: Dict[str, str] = \
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_model_map_xlnet()
        # DebuggingHelper.write_line_to_system_console_out(\
        #     f'model_map={model_map}')
        return model_map.keys()
        # ---- NOTE: R1721: Unnecessary use of a comprehension ---- return [x for x in model_map]

    _pretrained_model_keys: List[str] = None
    @staticmethod
    def get_pretrained_model_keys() -> List[str]:
        """
        Return a list of pretrained model keys.
        """
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        if PytorchLanguageUnderstandingTransformersPretainedModelHelper._pretrained_model_keys is None:
            model_keys: List[str] = []
            model_keys.extend(PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_model_keys_bert())
            model_keys.extend(PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_model_keys_roberta())
            model_keys.extend(PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_model_keys_gpt2())
            model_keys.extend(PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_model_keys_openai())
            model_keys.extend(PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_model_keys_transfo_xl())
            model_keys.extend(PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_model_keys_xlm())
            model_keys.extend(PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_model_keys_xlnet())
            PytorchLanguageUnderstandingTransformersPretainedModelHelper._pretrained_model_keys = model_keys
        return PytorchLanguageUnderstandingTransformersPretainedModelHelper._pretrained_model_keys
    _pretrained_model_key_set: Set[str] = None
    @staticmethod
    def get_pretrained_model_key_set() -> Set[str]:
        """
        Return a list of pretrained model key set.
        """
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        if PytorchLanguageUnderstandingTransformersPretainedModelHelper._pretrained_model_key_set is None:
            pretrained_model_keys = PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_model_keys()
            PytorchLanguageUnderstandingTransformersPretainedModelHelper._pretrained_model_key_set = set(pretrained_model_keys)
        return PytorchLanguageUnderstandingTransformersPretainedModelHelper._pretrained_model_key_set

    @staticmethod
    def get_pretrained_vocabulary_file_names_bert() -> List[str]:
        """
        Return a list of BERT pretrained vocabulary file map.
        """
        vocabulary_file_names: List[str] = \
            PytorchTransformersBert_VOCAB_FILES_NAMES
        # DebuggingHelper.write_line_to_system_console_out(\
        #     f'vocabulary_file_names={vocabulary_file_names}')
        return vocabulary_file_names
    @staticmethod
    def get_pretrained_vocabulary_file_names_roberta() -> List[str]:
        """
        Return a list of ROBERTA pretrained vocabulary file map.
        """
        vocabulary_file_names: List[str] = \
            PytorchTransformersRoberta_VOCAB_FILES_NAMES
        # DebuggingHelper.write_line_to_system_console_out(\
        #     f'vocabulary_file_names={vocabulary_file_names}')
        return vocabulary_file_names
    @staticmethod
    def get_pretrained_vocabulary_file_names_gpt2() -> List[str]:
        """
        Return a list of GPT2 pretrained vocabulary file map.
        """
        vocabulary_file_names: List[str] = \
            PytorchTransformersGpt2_VOCAB_FILES_NAMES
        # DebuggingHelper.write_line_to_system_console_out(\
        #     f'vocabulary_file_names={vocabulary_file_names}')
        return vocabulary_file_names
    @staticmethod
    def get_pretrained_vocabulary_file_names_openai() -> List[str]:
        """
        Return a list of OPENAI pretrained vocabulary file map.
        """
        vocabulary_file_names: List[str] = \
            PytorchTransformersOpenai_VOCAB_FILES_NAMES
        # DebuggingHelper.write_line_to_system_console_out(\
        #     f'vocabulary_file_names={vocabulary_file_names}')
        return vocabulary_file_names
    @staticmethod
    def get_pretrained_vocabulary_file_names_transfo_xl() -> List[str]:
        """
        Return a list of transfo-xl pretrained vocabulary file map.
        """
        vocabulary_file_names: List[str] = \
            PytorchTransformersTransfoXl_VOCAB_FILES_NAMES
        # DebuggingHelper.write_line_to_system_console_out(\
        #     f'vocabulary_file_names={vocabulary_file_names}')
        return vocabulary_file_names
    @staticmethod
    def get_pretrained_vocabulary_file_names_xlm() -> List[str]:
        """
        Return a list of XLM pretrained vocabulary file map.
        """
        vocabulary_file_names: List[str] = \
            PytorchTransformersXml_VOCAB_FILES_NAMES
        # DebuggingHelper.write_line_to_system_console_out(\
        #     f'vocabulary_file_names={vocabulary_file_names}')
        return vocabulary_file_names
    @staticmethod
    def get_pretrained_vocabulary_file_names_xlnet() -> List[str]:
        """
        Return a list of XLNET pretrained vocabulary file map.
        """
        vocabulary_file_names: List[str] = \
            PytorchTransformersXlnet_VOCAB_FILES_NAMES
        # DebuggingHelper.write_line_to_system_console_out(\
        #     f'vocabulary_file_names={vocabulary_file_names}')
        return vocabulary_file_names

    @staticmethod
    def get_pretrained_vocabulary_file_map_bert() -> Dict[str, str]:
        """
        Return a list of BERT pretrained vocabulary file map.
        """
        vocabulary_file_map: Dict[str, str] = \
            PytorchTransformersBert_PRETRAINED_VOCAB_FILES_MAP['vocab_file']
        # DebuggingHelper.write_line_to_system_console_out(\
        #     f'vocabulary_file_map={vocabulary_file_map}')
        return vocabulary_file_map
    @staticmethod
    def get_pretrained_vocabulary_file_map_roberta() -> Dict[str, str]:
        """
        Return a list of ROBERTA pretrained vocabulary file map.
        """
        vocabulary_file_map: Dict[str, str] = \
            PytorchTransformersRoberta_PRETRAINED_VOCAB_FILES_MAP['vocab_file']
        # DebuggingHelper.write_line_to_system_console_out(\
        #     f'vocabulary_file_map={vocabulary_file_map}')
        return vocabulary_file_map
    @staticmethod
    def get_pretrained_vocabulary_file_map_gpt2() -> Dict[str, str]:
        """
        Return a list of GPT2 pretrained vocabulary file map.
        """
        vocabulary_file_map: Dict[str, str] = \
            PytorchTransformersGpt2_PRETRAINED_VOCAB_FILES_MAP['vocab_file']
        # DebuggingHelper.write_line_to_system_console_out(\
        #     f'vocabulary_file_map={vocabulary_file_map}')
        return vocabulary_file_map
    @staticmethod
    def get_pretrained_vocabulary_file_map_openai() -> Dict[str, str]:
        """
        Return a list of OPENAI pretrained vocabulary file map.
        """
        vocabulary_file_map: Dict[str, str] = \
            PytorchTransformersOpenai_PRETRAINED_VOCAB_FILES_MAP['vocab_file']
        # DebuggingHelper.write_line_to_system_console_out(\
        #     f'vocabulary_file_map={vocabulary_file_map}')
        return vocabulary_file_map
    @staticmethod
    def get_pretrained_vocabulary_file_map_transfo_xl() -> Dict[str, str]:
        """
        Return a list of transfo-xl pretrained vocabulary file map.
        """
        vocabulary_file_map: Dict[str, str] = \
            PytorchTransformersTransfoXl_PRETRAINED_VOCAB_FILES_MAP['pretrained_vocab_file']
        # DebuggingHelper.write_line_to_system_console_out(\
        #     f'vocabulary_file_map={vocabulary_file_map}')
        return vocabulary_file_map
    @staticmethod
    def get_pretrained_vocabulary_file_map_xlm() -> Dict[str, str]:
        """
        Return a list of XLM pretrained vocabulary file map.
        """
        vocabulary_file_map: Dict[str, str] = \
            PytorchTransformersXml_PRETRAINED_VOCAB_FILES_MAP['vocab_file']
        # DebuggingHelper.write_line_to_system_console_out(\
        #     f'vocabulary_file_map={vocabulary_file_map}')
        return vocabulary_file_map
    @staticmethod
    def get_pretrained_vocabulary_file_map_xlnet() -> Dict[str, str]:
        """
        Return a list of XLNET pretrained vocabulary file map.
        """
        vocabulary_file_map: Dict[str, str] = \
            PytorchTransformersXlnet_PRETRAINED_VOCAB_FILES_MAP['vocab_file']
        # DebuggingHelper.write_line_to_system_console_out(\
        #     f'vocabulary_file_map={vocabulary_file_map}')
        return vocabulary_file_map

    _pretrained_vocabulary_file_map: Dict[str, str] = None
    @staticmethod
    def get_pretrained_vocabulary_file_map() -> Dict[str, str]:
        """
        Return a list of pretrained vocabulary file map.
        """
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        if PytorchLanguageUnderstandingTransformersPretainedModelHelper._pretrained_vocabulary_file_map is None:
            vocabulary_file_map: Dict[str, str] = {}
            vocabulary_file_map.update(PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_vocabulary_file_map_bert())
            vocabulary_file_map.update(PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_vocabulary_file_map_roberta())
            vocabulary_file_map.update(PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_vocabulary_file_map_gpt2())
            vocabulary_file_map.update(PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_vocabulary_file_map_openai())
            vocabulary_file_map.update(PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_vocabulary_file_map_transfo_xl())
            vocabulary_file_map.update(PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_vocabulary_file_map_xlm())
            vocabulary_file_map.update(PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_vocabulary_file_map_xlnet())
            PytorchLanguageUnderstandingTransformersPretainedModelHelper._pretrained_vocabulary_file_map = vocabulary_file_map
        return PytorchLanguageUnderstandingTransformersPretainedModelHelper._pretrained_vocabulary_file_map

    @staticmethod
    def get_pretrained_merge_file_map_bert() -> Dict[str, str]:
        """
        Return a list of BERT pretrained vocabulary file map.
        """
        merge_file_map: Dict[str, str] = \
            PytorchTransformersBert_PRETRAINED_VOCAB_FILES_MAP['merges_file']
        # DebuggingHelper.write_line_to_system_console_out(\
        #     f'merge_file_map={merge_file_map}')
        return merge_file_map
    @staticmethod
    def get_pretrained_merge_file_map_roberta() -> Dict[str, str]:
        """
        Return a list of ROBERTA pretrained vocabulary file map.
        """
        merge_file_map: Dict[str, str] = \
            PytorchTransformersRoberta_PRETRAINED_VOCAB_FILES_MAP['merges_file']
        # DebuggingHelper.write_line_to_system_console_out(\
        #     f'merge_file_map={merge_file_map}')
        return merge_file_map
    @staticmethod
    def get_pretrained_merge_file_map_gpt2() -> Dict[str, str]:
        """
        Return a list of GPT2 pretrained vocabulary file map.
        """
        merge_file_map: Dict[str, str] = \
            PytorchTransformersGpt2_PRETRAINED_VOCAB_FILES_MAP['merges_file']
        # DebuggingHelper.write_line_to_system_console_out(\
        #     f'merge_file_map={merge_file_map}')
        return merge_file_map
    @staticmethod
    def get_pretrained_merge_file_map_openai() -> Dict[str, str]:
        """
        Return a list of OPENAI pretrained vocabulary file map.
        """
        merge_file_map: Dict[str, str] = \
            PytorchTransformersOpenai_PRETRAINED_VOCAB_FILES_MAP['merges_file']
        # DebuggingHelper.write_line_to_system_console_out(\
        #     f'merge_file_map={merge_file_map}')
        return merge_file_map
    @staticmethod
    def get_pretrained_merge_file_map_transfo_xl() -> Dict[str, str]:
        """
        Return a list of transfo-xl pretrained vocabulary file map.
        """
        merge_file_map: Dict[str, str] = \
            PytorchTransformersTransfoXl_PRETRAINED_VOCAB_FILES_MAP['pretrained_merges_file']
        # DebuggingHelper.write_line_to_system_console_out(\
        #     f'merge_file_map={merge_file_map}')
        return merge_file_map
    @staticmethod
    def get_pretrained_merge_file_map_xlm() -> Dict[str, str]:
        """
        Return a list of XLM pretrained vocabulary file map.
        """
        merge_file_map: Dict[str, str] = \
            PytorchTransformersXml_PRETRAINED_VOCAB_FILES_MAP['merges_file']
        # DebuggingHelper.write_line_to_system_console_out(\
        #     f'merge_file_map={merge_file_map}')
        return merge_file_map
    @staticmethod
    def get_pretrained_merge_file_map_xlnet() -> Dict[str, str]:
        """
        Return a list of XLNET pretrained vocabulary file map.
        """
        merge_file_map: Dict[str, str] = \
            PytorchTransformersXlnet_PRETRAINED_VOCAB_FILES_MAP['merges_file']
        # DebuggingHelper.write_line_to_system_console_out(\
        #     f'merge_file_map={merge_file_map}')
        return merge_file_map

    _pretrained_merge_file_map: Dict[str, str] = None
    @staticmethod
    def get_pretrained_merge_file_map() -> Dict[str, str]:
        """
        Return a list of pretrained vocabulary file map.
        """
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        if PytorchLanguageUnderstandingTransformersPretainedModelHelper._pretrained_merge_file_map is None:
            merge_file_map: Dict[str, str] = {}
            merge_file_map.update(PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_merge_file_map_bert())
            merge_file_map.update(PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_merge_file_map_roberta())
            merge_file_map.update(PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_merge_file_map_gpt2())
            merge_file_map.update(PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_merge_file_map_openai())
            merge_file_map.update(PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_merge_file_map_transfo_xl())
            merge_file_map.update(PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_merge_file_map_xlm())
            merge_file_map.update(PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_merge_file_map_xlnet())
            PytorchLanguageUnderstandingTransformersPretainedModelHelper._pretrained_merge_file_map = merge_file_map
        return PytorchLanguageUnderstandingTransformersPretainedModelHelper._pretrained_merge_file_map

    @staticmethod
    def get_pretrained_positional_embedding_size_map_bert() -> Dict[str, int]:
        """
        Return a list of BERT pretrained embedding size map.
        """
        positional_embedding_size_map: Dict[str, int] = \
            PytorchTransformersBert_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
        # DebuggingHelper.write_line_to_system_console_out(\
        #     f'positional_embedding_size_map={positional_embedding_size_map}')
        return positional_embedding_size_map
    @staticmethod
    def get_pretrained_positional_embedding_size_map_roberta() -> Dict[str, int]:
        """
        Return a list of ROBERTA pretrained embedding size map.
        """
        positional_embedding_size_map: Dict[str, int] = \
            PytorchTransformersRoberta_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
        # DebuggingHelper.write_line_to_system_console_out(\
        #     f'positional_embedding_size_map={positional_embedding_size_map}')
        return positional_embedding_size_map
    @staticmethod
    def get_pretrained_positional_embedding_size_map_gpt2() -> Dict[str, int]:
        """
        Return a list of GPT2 pretrained embedding size map.
        """
        positional_embedding_size_map: Dict[str, int] = \
            PytorchTransformersGpt2_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
        # DebuggingHelper.write_line_to_system_console_out(\
        #     f'positional_embedding_size_map={positional_embedding_size_map}')
        return positional_embedding_size_map
    @staticmethod
    def get_pretrained_positional_embedding_size_map_openai() -> Dict[str, int]:
        """
        Return a list of OPENAI pretrained embedding size map.
        """
        positional_embedding_size_map: Dict[str, int] = \
            PytorchTransformersOpenai_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
        # DebuggingHelper.write_line_to_system_console_out(\
        #     f'positional_embedding_size_map={positional_embedding_size_map}')
        return positional_embedding_size_map
    @staticmethod
    def get_pretrained_positional_embedding_size_map_transfo_xl() -> Dict[str, int]:
        """
        Return a list of transfo-xl pretrained embedding size map.
        """
        positional_embedding_size_map: Dict[str, int] = \
            PytorchTransformersTransfoXl_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
        # DebuggingHelper.write_line_to_system_console_out(\
        #     f'positional_embedding_size_map={positional_embedding_size_map}')
        return positional_embedding_size_map
    @staticmethod
    def get_pretrained_positional_embedding_size_map_xlm() -> Dict[str, int]:
        """
        Return a list of XLM pretrained embedding size map.
        """
        positional_embedding_size_map: Dict[str, int] = \
            PytorchTransformersXml_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
        # DebuggingHelper.write_line_to_system_console_out(\
        #     f'positional_embedding_size_map={positional_embedding_size_map}')
        return positional_embedding_size_map
    @staticmethod
    def get_pretrained_positional_embedding_size_map_xlnet() -> Dict[str, int]:
        """
        Return a list of XLNET pretrained embedding size map.
        """
        positional_embedding_size_map: Dict[str, int] = \
            PytorchTransformersXlnet_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
        # DebuggingHelper.write_line_to_system_console_out(\
        #     f'positional_embedding_size_map={positional_embedding_size_map}')
        return positional_embedding_size_map

    _pretrained_positional_embedding_size_map: Dict[str, int] = None
    @staticmethod
    def get_pretrained_positional_embedding_size_map() -> Dict[str, int]:
        """
        Return a list of pretrained embedding size map.
        """
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        if PytorchLanguageUnderstandingTransformersPretainedModelHelper._pretrained_positional_embedding_size_map is None:
            positional_embedding_size_map: Dict[str, int] = {}
            positional_embedding_size_map.update(PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_positional_embedding_size_map_bert())
            positional_embedding_size_map.update(PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_positional_embedding_size_map_roberta())
            positional_embedding_size_map.update(PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_positional_embedding_size_map_gpt2())
            positional_embedding_size_map.update(PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_positional_embedding_size_map_openai())
            positional_embedding_size_map.update(PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_positional_embedding_size_map_transfo_xl())
            positional_embedding_size_map.update(PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_positional_embedding_size_map_xlm())
            positional_embedding_size_map.update(PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_positional_embedding_size_map_xlnet())
            PytorchLanguageUnderstandingTransformersPretainedModelHelper._pretrained_positional_embedding_size_map = positional_embedding_size_map
        return PytorchLanguageUnderstandingTransformersPretainedModelHelper._pretrained_positional_embedding_size_map

    @staticmethod
    def get_pretrained_pretrained_init_configuration_map_bert():
        """
        Return a list of BERT pretrained embedding size map.
        """
        pretrained_init_configuration_map = \
            PytorchTransformersBert_PRETRAINED_INIT_CONFIGURATION
        # DebuggingHelper.write_line_to_system_console_out(\
        #     f'pretrained_init_configuration_map={pretrained_init_configuration_map}')
        return pretrained_init_configuration_map
    # @staticmethod
    # def get_pretrained_pretrained_init_configuration_map_roberta():
    #     """
    #     Return a list of ROBERTA pretrained embedding size map.
    #     """
    #     pretrained_init_configuration_map = \
    #         PytorchTransformersRoberta_PRETRAINED_INIT_CONFIGURATION
    #     # DebuggingHelper.write_line_to_system_console_out(\
    #     #     f'pretrained_init_configuration_map={pretrained_init_configuration_map}')
    #     return pretrained_init_configuration_map
    # @staticmethod
    # def get_pretrained_pretrained_init_configuration_map_gpt2():
    #     """
    #     Return a list of GPT2 pretrained embedding size map.
    #     """
    #     pretrained_init_configuration_map = \
    #         PytorchTransformersGpt2_PRETRAINED_INIT_CONFIGURATION
    #     # DebuggingHelper.write_line_to_system_console_out(\
    #     #     f'pretrained_init_configuration_map={pretrained_init_configuration_map}')
    #     return pretrained_init_configuration_map
    # @staticmethod
    # def get_pretrained_pretrained_init_configuration_map_openai():
    #     """
    #     Return a list of OPENAI pretrained embedding size map.
    #     """
    #     pretrained_init_configuration_map = \
    #         PytorchTransformersOpenai_PRETRAINED_INIT_CONFIGURATION
    #     # DebuggingHelper.write_line_to_system_console_out(\
    #     #     f'pretrained_init_configuration_map={pretrained_init_configuration_map}')
    #     return pretrained_init_configuration_map
    # @staticmethod
    # def get_pretrained_pretrained_init_configuration_map_transfo_xl():
    #     """
    #     Return a list of transfo-xl pretrained embedding size map.
    #     """
    #     pretrained_init_configuration_map = \
    #         PytorchTransformersTransfoXl_PRETRAINED_INIT_CONFIGURATION
    #     # DebuggingHelper.write_line_to_system_console_out(\
    #     #     f'pretrained_init_configuration_map={pretrained_init_configuration_map}')
    #     return pretrained_init_configuration_map
    @staticmethod
    def get_pretrained_pretrained_init_configuration_map_xlm():
        """
        Return a list of XLM pretrained embedding size map.
        """
        pretrained_init_configuration_map = \
            PytorchTransformersXml_PRETRAINED_INIT_CONFIGURATION
        # DebuggingHelper.write_line_to_system_console_out(\
        #     f'pretrained_init_configuration_map={pretrained_init_configuration_map}')
        return pretrained_init_configuration_map
    # @staticmethod
    # def get_pretrained_pretrained_init_configuration_map_xlnet():
    #     """
    #     Return a list of XLNET pretrained embedding size map.
    #     """
    #     pretrained_init_configuration_map = \
    #         PytorchTransformersXlnet_PRETRAINED_INIT_CONFIGURATION
    #     # DebuggingHelper.write_line_to_system_console_out(\
    #     #     f'pretrained_init_configuration_map={pretrained_init_configuration_map}')
    #     return pretrained_init_configuration_map

    _pretrained_pretrained_init_configuration_map = None
    @staticmethod
    def get_pretrained_pretrained_init_configuration_map():
        """
        Return a pretrained init configuration map.
        """
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        if PytorchLanguageUnderstandingTransformersPretainedModelHelper._pretrained_pretrained_init_configuration_map is None:
            pretrained_init_configuration_map = {}
            pretrained_init_configuration_map.update(PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_pretrained_init_configuration_map_bert())
            # pretrained_init_configuration_map.update(PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_pretrained_init_configuration_map_roberta())
            # pretrained_init_configuration_map.update(PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_pretrained_init_configuration_map_gpt2())
            # pretrained_init_configuration_map.update(PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_pretrained_init_configuration_map_openai())
            # pretrained_init_configuration_map.update(PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_pretrained_init_configuration_map_transfo_xl())
            pretrained_init_configuration_map.update(PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_pretrained_init_configuration_map_xlm())
            # pretrained_init_configuration_map.update(PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_pretrained_init_configuration_map_xlnet())
            PytorchLanguageUnderstandingTransformersPretainedModelHelper._pretrained_pretrained_init_configuration_map = pretrained_init_configuration_map
        return PytorchLanguageUnderstandingTransformersPretainedModelHelper._pretrained_pretrained_init_configuration_map

    @staticmethod
    def get_pretrained_vocabulary_file_keys_bert() -> List[str]:
        """
        Return a list of BERT pretrained vocabulary file keys.
        """
        vocabulary_file_map: Dict[str, str] = \
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_vocabulary_file_map_bert()
        # DebuggingHelper.write_line_to_system_console_out(\
        #     f'vocabulary_file_map={vocabulary_file_map}')
        return vocabulary_file_map.keys()
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        # ---- NOTE: R1721: Unnecessary use of a comprehension ---- return [x for x in vocabulary_file_map]
    @staticmethod
    def get_pretrained_vocabulary_file_keys_roberta() -> List[str]:
        """
        Return a list of ROBERTA pretrained vocabulary file keys.
        """
        vocabulary_file_map: Dict[str, str] = \
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_vocabulary_file_map_roberta()
        # DebuggingHelper.write_line_to_system_console_out(\
        #     f'vocabulary_file_map={vocabulary_file_map}')
        return vocabulary_file_map.keys()
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        # ---- NOTE: R1721: Unnecessary use of a comprehension ---- return [x for x in vocabulary_file_map]
    @staticmethod
    def get_pretrained_vocabulary_file_keys_gpt2() -> List[str]:
        """
        Return a list of GPT2 pretrained vocabulary file keys.
        """
        vocabulary_file_map: Dict[str, str] = \
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_vocabulary_file_map_gpt2()
        # DebuggingHelper.write_line_to_system_console_out(\
        #     f'vocabulary_file_map={vocabulary_file_map}')
        return vocabulary_file_map.keys()
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        # ---- NOTE: R1721: Unnecessary use of a comprehension ---- return [x for x in vocabulary_file_map]
    @staticmethod
    def get_pretrained_vocabulary_file_keys_openai() -> List[str]:
        """
        Return a list of OPENAI pretrained vocabulary file keys.
        """
        vocabulary_file_map: Dict[str, str] = \
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_vocabulary_file_map_openai()
        # DebuggingHelper.write_line_to_system_console_out(\
        #     f'vocabulary_file_map={vocabulary_file_map}')
        return vocabulary_file_map.keys()
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        # ---- NOTE: R1721: Unnecessary use of a comprehension ---- return [x for x in vocabulary_file_map]
    @staticmethod
    def get_pretrained_vocabulary_file_keys_transfo_xl() -> List[str]:
        """
        Return a list of transfo-xl pretrained vocabulary file keys.
        """
        vocabulary_file_map: Dict[str, str] = \
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_vocabulary_file_map_transfo_xl()
        # DebuggingHelper.write_line_to_system_console_out(\
        #     f'vocabulary_file_map={vocabulary_file_map}')
        return vocabulary_file_map.keys()
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        # ---- NOTE: R1721: Unnecessary use of a comprehension ---- return [x for x in vocabulary_file_map]
    @staticmethod
    def get_pretrained_vocabulary_file_keys_xlm() -> List[str]:
        """
        Return a list of XLM pretrained vocabulary file keys.
        """
        vocabulary_file_map: Dict[str, str] = \
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_vocabulary_file_map_xlm()
        # DebuggingHelper.write_line_to_system_console_out(\
        #     f'vocabulary_file_map={vocabulary_file_map}')
        return vocabulary_file_map.keys()
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        # ---- NOTE: R1721: Unnecessary use of a comprehension ---- return [x for x in vocabulary_file_map]
    @staticmethod
    def get_pretrained_vocabulary_file_keys_xlnet() -> List[str]:
        """
        Return a list of XLNET pretrained vocabulary file keys.
        """
        vocabulary_file_map: Dict[str, str] = \
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_vocabulary_file_map_xlnet()
        # DebuggingHelper.write_line_to_system_console_out(\
        #     f'vocabulary_file_map={vocabulary_file_map}')
        return vocabulary_file_map.keys()
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        # ---- NOTE: R1721: Unnecessary use of a comprehension ---- return [x for x in vocabulary_file_map]

    _pretrained_vocabulary_file_keys: List[str] = None
    @staticmethod
    def get_pretrained_vocabulary_file_keys() -> List[str]:
        """
        Return a list of pretrained vocabulary file keys.
        """
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        if PytorchLanguageUnderstandingTransformersPretainedModelHelper._pretrained_vocabulary_file_keys is None:
            vocabulary_file_keys: List[str] = []
            vocabulary_file_keys.extend(PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_vocabulary_file_keys_bert())
            vocabulary_file_keys.extend(PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_vocabulary_file_keys_roberta())
            vocabulary_file_keys.extend(PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_vocabulary_file_keys_gpt2())
            vocabulary_file_keys.extend(PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_vocabulary_file_keys_openai())
            vocabulary_file_keys.extend(PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_vocabulary_file_keys_transfo_xl())
            vocabulary_file_keys.extend(PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_vocabulary_file_keys_xlm())
            vocabulary_file_keys.extend(PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_vocabulary_file_keys_xlnet())
            PytorchLanguageUnderstandingTransformersPretainedModelHelper._pretrained_vocabulary_file_keys = vocabulary_file_keys
        return PytorchLanguageUnderstandingTransformersPretainedModelHelper._pretrained_vocabulary_file_keys
    _pretrained_vocabulary_file_key_set: Set[str] = None
    @staticmethod
    def get_pretrained_vocabulary_file_key_set() -> Set[str]:
        """
        Return a list of pretrained vocabulary_file key set.
        """
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        if PytorchLanguageUnderstandingTransformersPretainedModelHelper._pretrained_vocabulary_file_key_set is None:
            pretrained_vocabulary_file_keys = PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_vocabulary_file_keys()
            PytorchLanguageUnderstandingTransformersPretainedModelHelper._pretrained_vocabulary_file_key_set = set(pretrained_vocabulary_file_keys)
        return PytorchLanguageUnderstandingTransformersPretainedModelHelper._pretrained_vocabulary_file_key_set

    @staticmethod
    def get_number_training_optimization_steps( \
        optimizer_max_number_training_steps: int, \
        number_training_instances: int, \
        optimizer_gradient_accumulation_steps: int, \
        optimizer_number_training_epochs: int) -> Tuple[int, int]:
        # ---- device_gpu_cuda_local_rank: int):
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        """
        Calculate number of training optimization steps.
        REFERENCE: https://github.com/huggingface/pytorch-transformers/blob/master/examples/run_glue.py
        """
        # ---- NOTE-PYLINT ---- R0913: Too many arguments
        # pylint: disable=R0913
        DebuggingHelper.write_line_to_system_console_out(\
            f'get_number_training_optimization_steps(): INPUT: optimizer_max_number_training_steps='
            f'{optimizer_max_number_training_steps}')
        DebuggingHelper.write_line_to_system_console_out(\
            f'get_number_training_optimization_steps(): INPUT: number_training_instances='
            f'{number_training_instances}')
        DebuggingHelper.write_line_to_system_console_out(\
            f'get_number_training_optimization_steps(): INPUT: optimizer_gradient_accumulation_steps='
            f'{optimizer_gradient_accumulation_steps}')
        DebuggingHelper.write_line_to_system_console_out(\
            f'get_number_training_optimization_steps(): INPUT: optimizer_number_training_epochs='
            f'{optimizer_number_training_epochs}')
        if optimizer_max_number_training_steps > 0:
            optimizer_number_training_optimization_steps = \
                optimizer_max_number_training_steps
            optimizer_number_training_epochs = \
                optimizer_max_number_training_steps // (number_training_instances // optimizer_gradient_accumulation_steps) + 1
        else:
            optimizer_number_training_optimization_steps = \
                (number_training_instances // optimizer_gradient_accumulation_steps) * optimizer_number_training_epochs
        DebuggingHelper.write_line_to_system_console_out(\
            f'get_number_training_optimization_steps(): OUTPUT: optimizer_number_training_optimization_steps='
            f'{optimizer_number_training_optimization_steps}')
        DebuggingHelper.write_line_to_system_console_out(\
            f'get_number_training_optimization_steps(): OUTPUT: optimizer_number_training_epochs='
            f'{optimizer_number_training_epochs}')
        return (optimizer_number_training_optimization_steps, optimizer_number_training_epochs)

    MODEL_TYPE_BERT = 'bert'
    MODEL_TYPE_ROBERTA = 'roberta'
    MODEL_TYPE_GPT2 = 'gpt2'
    MODEL_TYPE_OPENAI = 'openai'
    MODEL_TYPE_TRANSFO_XI = 'transfo_xl'
    MODEL_TYPE_XLM = 'xlm'
    MODEL_TYPE_XLNET = 'xlnet'

    MODEL_TYPES: Set[str] = {
        MODEL_TYPE_BERT,
        MODEL_TYPE_ROBERTA,
        MODEL_TYPE_GPT2,
        MODEL_TYPE_OPENAI,
        MODEL_TYPE_TRANSFO_XI,
        MODEL_TYPE_XLM,
        MODEL_TYPE_XLNET}

    _pretrained_model_key_type_map: Dict[str, str] = None
    @staticmethod
    def get_pretrained_model_key_type_map() -> Dict[str, str]:
        """
        Return a dictionary of pretrained model keys mapping to their model types.
        """
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        if PytorchLanguageUnderstandingTransformersPretainedModelHelper._pretrained_model_key_type_map is None:
            model_key_map: Dict[str, str] = {}
            model_key_map.update(\
                {key : PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_BERT \
                    for key in PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_model_keys_bert()})
            model_key_map.update(\
                {key : PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_ROBERTA \
                    for key in PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_model_keys_roberta()})
            model_key_map.update(\
                {key : PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_GPT2 \
                    for key in PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_model_keys_gpt2()})
            model_key_map.update(\
                {key : PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_OPENAI \
                    for key in PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_model_keys_openai()})
            model_key_map.update(\
                {key : PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_TRANSFO_XI \
                    for key in PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_model_keys_transfo_xl()})
            model_key_map.update(\
                {key : PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_XLM \
                    for key in PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_model_keys_xlm()})
            model_key_map.update(\
                {key : PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_XLNET \
                    for key in PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_model_keys_xlnet()})
            PytorchLanguageUnderstandingTransformersPretainedModelHelper._pretrained_model_key_type_map = model_key_map
        return PytorchLanguageUnderstandingTransformersPretainedModelHelper._pretrained_model_key_type_map

    _pretrained_vocabulary_key_type_map: Dict[str, str] = None
    @staticmethod
    def get_pretrained_vocabulary_key_type_map() -> Dict[str, str]:
        """
        Return a dictionary of pretrained vocabulary keys mapping to their vocabulary types.
        """
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        if PytorchLanguageUnderstandingTransformersPretainedModelHelper._pretrained_vocabulary_key_type_map is None:
            vocabulary_key_map: Dict[str, str] = {}
            vocabulary_key_map.update(\
                {key : PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_BERT \
                    for key in PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_vocabulary_file_names_bert()})
            vocabulary_key_map.update(\
                {key : PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_ROBERTA \
                    for key in PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_vocabulary_file_names_roberta()})
            vocabulary_key_map.update(\
                {key : PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_GPT2 \
                    for key in PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_vocabulary_file_names_gpt2()})
            vocabulary_key_map.update(\
                {key : PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_OPENAI \
                    for key in PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_vocabulary_file_names_openai()})
            vocabulary_key_map.update(\
                {key : PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_TRANSFO_XI \
                    for key in PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_vocabulary_file_names_transfo_xl()})
            vocabulary_key_map.update(\
                {key : PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_XLM \
                    for key in PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_vocabulary_file_names_xlm()})
            vocabulary_key_map.update(\
                {key : PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_XLNET \
                    for key in PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_vocabulary_file_names_xlnet()})
            PytorchLanguageUnderstandingTransformersPretainedModelHelper._pretrained_vocabulary_key_type_map = vocabulary_key_map
        return PytorchLanguageUnderstandingTransformersPretainedModelHelper._pretrained_vocabulary_key_type_map

    @staticmethod
    def bert_config_from_pretrained( \
        pretrained_model_name_or_path: str, \
        *argv, **kwargs): # ---- NOTE-PLACE-HOLDER ---- -> "PretrainedConfig":
        """
        Load a pretrained BERT config.
        """
        return BertConfig.from_pretrained(
            pretrained_model_name_or_path,
            *argv, **kwargs)
    @staticmethod
    def roberta_config_from_pretrained( \
        pretrained_model_name_or_path: str, \
        *argv, **kwargs): # ---- NOTE-PLACE-HOLDER ---- -> "PretrainedConfig":
        """
        Load a pretrained ROBERTA config.
        """
        return RobertaConfig.from_pretrained(
            pretrained_model_name_or_path,
            *argv, **kwargs)
    @staticmethod
    def gpt2_config_from_pretrained( \
        pretrained_model_name_or_path: str, \
        *argv, **kwargs): # ---- NOTE-PLACE-HOLDER ---- -> "PretrainedConfig":
        """
        Load a pretrained GPT2 config.
        """
        return GPT2Config.from_pretrained(
            pretrained_model_name_or_path,
            *argv, **kwargs)
    @staticmethod
    def openai_config_from_pretrained( \
        pretrained_model_name_or_path: str, \
        *argv, **kwargs): # ---- NOTE-PLACE-HOLDER ---- -> "PretrainedConfig":
        """
        Load a pretrained OpenAI config.
        """
        return OpenAIGPTConfig.from_pretrained(
            pretrained_model_name_or_path,
            *argv, **kwargs)
    @staticmethod
    def transfo_xl_config_from_pretrained( \
        pretrained_model_name_or_path: str, \
        *argv, **kwargs): # ---- NOTE-PLACE-HOLDER ---- -> "PretrainedConfig":
        """
        Load a pretrained TransfoXL config.
        """
        return TransfoXLConfig.from_pretrained(
            pretrained_model_name_or_path,
            *argv, **kwargs)
    @staticmethod
    def xlm_config_from_pretrained( \
        pretrained_model_name_or_path: str, \
        *argv, **kwargs): # ---- NOTE-PLACE-HOLDER ---- -> "PretrainedConfig":
        """
        Load a pretrained XLM config.
        """
        return XLMConfig.from_pretrained(
            pretrained_model_name_or_path,
            *argv, **kwargs)
    @staticmethod
    def xlnet_config_from_pretrained( \
        pretrained_model_name_or_path: str, \
        *argv, **kwargs): # ---- NOTE-PLACE-HOLDER ---- -> "PretrainedConfig":
        """
        Load a pretrained XLNet config.
        """
        return XLNetConfig.from_pretrained(
            pretrained_model_name_or_path,
            *argv, **kwargs)
    @staticmethod
    def config_from_pretrained( \
        pretrained_model_name_or_path: str, \
        model_type: str = '', \
        *argv, **kwargs): # ---- NOTE-PLACE-HOLDER ---- -> "PretrainedConfig":
        """
        Load a config.
        """
        if pretrained_model_name_or_path is None:
            DebuggingHelper.throw_exception(
                'input argument, pretrained_model_name_or_path, is None')
        model_key_type_map: Dict[str, str] = \
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_model_key_type_map()
        # ---- NOTE-PYLINT ---- E1135: Value '' doesn't support membership test
        #                              (unsupported-membership-test)
        # pylint: disable=E1135
        if pretrained_model_name_or_path in model_key_type_map:
            # ---- NOTE-PYLINT ---- E1136: Value '' is unsubscriptable
            #                              ((unsubscriptable-object)
            # pylint: disable=E1136
            model_type = model_key_type_map[pretrained_model_name_or_path]
        else:
            if DatatypeHelper.is_none_empty_whitespaces_or_nan(model_type):
                DebuggingHelper.throw_exception(
                    f'pretrained_model_name_or_path={pretrained_model_name_or_path} is not in '
                    f'model_key_type_map|{model_key_type_map}|'
                    f', model_type|{model_type}| is empty')
        switcher = {
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_BERT: \
                PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
                    bert_config_from_pretrained,
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_ROBERTA: \
                PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
                   roberta_config_from_pretrained,
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_GPT2: \
                PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
                    gpt2_config_from_pretrained,
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_OPENAI: \
                PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
                    openai_config_from_pretrained,
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_TRANSFO_XI: \
                PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
                    transfo_xl_config_from_pretrained,
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_XLM: \
                PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
                    xlm_config_from_pretrained,
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_XLNET: \
                PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
                    xlnet_config_from_pretrained,
        }
        config_from_pretrained = switcher.get(
            model_type)
        return config_from_pretrained(pretrained_model_name_or_path, *argv, **kwargs)

    @staticmethod
    def bert_config_new_from_dictionary( \
        dictionary: Dict[str, any]) -> BertConfig:
        """
        Load a pretrained BERT config.
        """
        return BertConfig(
            **dictionary)
    @staticmethod
    def roberta_config_new_from_dictionary( \
        dictionary: Dict[str, any]) -> RobertaConfig:
        """
        Load a pretrained ROBERTA config.
        """
        return RobertaConfig(
            **dictionary)
    @staticmethod
    def gpt2_config_new_from_dictionary( \
        dictionary: Dict[str, any]) -> GPT2Config:
        """
        Load a pretrained GPT2 config.
        """
        return GPT2Config(
            **dictionary)
    @staticmethod
    def openai_config_new_from_dictionary( \
        dictionary: Dict[str, any]) -> OpenAIGPTConfig:
        """
        Load a pretrained OpenAI config.
        """
        return OpenAIGPTConfig(
            **dictionary)
    @staticmethod
    def transfo_xl_config_new_from_dictionary( \
        dictionary: Dict[str, any]) -> TransfoXLConfig:
        """
        Load a pretrained TransfoXL config.
        """
        return TransfoXLConfig(
            **dictionary)
    @staticmethod
    def xlm_config_new_from_dictionary( \
        dictionary: Dict[str, any]) -> XLMConfig:
        """
        Load a pretrained XLM config.
        """
        return XLMConfig(
            **dictionary)
    @staticmethod
    def xlnet_config_new_from_dictionary( \
        dictionary: Dict[str, any]) -> XLNetConfig:
        """
        Load a pretrained XLNet config.
        """
        return XLNetConfig(
            **dictionary)
    @staticmethod
    def config_new_from_dictionary( \
        model_type: str, \
        dictionary: Dict[str, any]): # ---- NOTE-PLACE-HOLDER ---- -> PretrainedConfig:
        """
        Load a config.
        """
        if model_type is None:
            DebuggingHelper.throw_exception(
                'input argument, model_type, is None')
        if model_type not in PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPES:
            DebuggingHelper.throw_exception(
                f'input argument, model_type, is not in the list {PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPES}')
        if dictionary is None:
            DebuggingHelper.throw_exception(
                'input argument, dictionary, is None')
        switcher = {
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_BERT: \
                PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
                    bert_config_new_from_dictionary,
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_ROBERTA: \
                PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
                   roberta_config_new_from_dictionary,
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_GPT2: \
                PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
                    gpt2_config_new_from_dictionary,
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_OPENAI: \
                PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
                    openai_config_new_from_dictionary,
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_TRANSFO_XI: \
                PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
                    transfo_xl_config_new_from_dictionary,
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_XLM: \
                PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
                    xlm_config_new_from_dictionary,
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_XLNET: \
                PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
                    xlnet_config_new_from_dictionary,
        }
        config_from_pretrained = switcher.get(
            model_type)
        return config_from_pretrained(dictionary)

    @staticmethod
    def bert_config_new( \
        vocab_size=30522, \
        hidden_size=768, \
        num_hidden_layers=12, \
        num_attention_heads=12, \
        intermediate_size=3072, \
        hidden_act="gelu", \
        hidden_dropout_prob=0.1, \
        attention_probs_dropout_prob=0.1, \
        max_position_embeddings=512, \
        type_vocab_size=2, \
        initializer_range=0.02, \
        layer_norm_eps=1e-12, \
        pad_token_id=0, \
        **kwargs) -> "BertConfig":
        # ---- NOTE-PYLINT ---- R0913: Too many arguments
        # pylint: disable=R0913
        """
        Load a pretrained BERT config.
        """
        return BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            pad_token_id=pad_token_id,
            **kwargs)
    @staticmethod
    def roberta_config_new( \
        pad_token_id=1, \
        bos_token_id=0, \
        eos_token_id=2, \
        **kwargs) -> RobertaConfig:
        # ---- NOTE-PYLINT ---- R0913: Too many arguments
        # pylint: disable=R0913
        """
        Load a pretrained ROBERTA config.
        """
        return RobertaConfig(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs)
    @staticmethod
    def gpt2_config_new( \
        vocab_size=50257, \
        n_positions=1024, \
        n_ctx=1024, \
        n_embd=768, \
        n_layer=12, \
        n_head=12, \
        activation_function="gelu_new", \
        resid_pdrop=0.1, \
        embd_pdrop=0.1, \
        attn_pdrop=0.1, \
        layer_norm_epsilon=1e-5, \
        initializer_range=0.02, \
        summary_type="cls_index", \
        summary_use_proj=True, \
        summary_activation=None, \
        summary_proj_to_labels=True, \
        summary_first_dropout=0.1, \
        bos_token_id=50256, \
        eos_token_id=50256, \
        **kwargs): # ---- NOTE-PLACE-HOLDER ---- -> "PretrainedConfig":
        # ---- NOTE-PYLINT ---- R0913: Too many arguments
        # pylint: disable=R0913
        # ---- NOTE-PYLINT ---- R0914: Too many local variables
        # pylint: disable=R0914
        """
        Load a pretrained GPT2 config.
        """
        return GPT2Config(
            vocab_size=vocab_size,
            n_positions=n_positions,
            n_ctx=n_ctx,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            activation_function=activation_function,
            resid_pdrop=resid_pdrop,
            embd_pdrop=embd_pdrop,
            attn_pdrop=attn_pdrop,
            layer_norm_epsilon=layer_norm_epsilon,
            initializer_range=initializer_range,
            summary_type=summary_type,
            summary_use_proj=summary_use_proj,
            summary_activation=summary_activation,
            summary_proj_to_labels=summary_proj_to_labels,
            summary_first_dropout=summary_first_dropout,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs)
    @staticmethod
    def openai_config_new( \
        vocab_size=40478, \
        n_positions=512, \
        n_ctx=512, \
        n_embd=768, \
        n_layer=12, \
        n_head=12, \
        afn="gelu", \
        resid_pdrop=0.1, \
        embd_pdrop=0.1, \
        attn_pdrop=0.1, \
        layer_norm_epsilon=1e-5, \
        initializer_range=0.02, \
        predict_special_tokens=True, \
        summary_type="cls_index", \
        summary_use_proj=True, \
        summary_activation=None, \
        summary_proj_to_labels=True, \
        summary_first_dropout=0.1, \
        **kwargs) -> "OpenAIGPTConfig":
        # ---- NOTE-PYLINT ---- R0913: Too many arguments
        # pylint: disable=R0913
        # ---- NOTE-PYLINT ---- R0914: Too many local variables
        # pylint: disable=R0914
        """
        Load a pretrained OpenAI config.
        """
        return OpenAIGPTConfig(
            vocab_size=vocab_size,
            n_positions=n_positions,
            n_ctx=n_ctx,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            afn=afn,
            resid_pdrop=resid_pdrop,
            embd_pdrop=embd_pdrop,
            attn_pdrop=attn_pdrop,
            layer_norm_epsilon=layer_norm_epsilon,
            initializer_range=initializer_range,
            predict_special_tokens=predict_special_tokens,
            summary_type=summary_type,
            summary_use_proj=summary_use_proj,
            summary_activation=summary_activation,
            summary_proj_to_labels=summary_proj_to_labels,
            summary_first_dropout=summary_first_dropout,
            **kwargs)
    @staticmethod
    def transfo_xl_config_new( \
        vocab_size=267735, \
        cutoffs=[20000, 40000, 200000], \
        d_model=1024, \
        d_embed=1024, \
        n_head=16, \
        d_head=64, \
        d_inner=4096, \
        div_val=4, \
        pre_lnorm=False, \
        n_layer=18, \
        tgt_len=128, \
        ext_len=0, \
        mem_len=1600, \
        clamp_len=1000, \
        same_length=True, \
        proj_share_all_but_first=True, \
        attn_type=0, \
        sample_softmax=-1, \
        adaptive=True, \
        tie_weight=True, \
        dropout=0.1, \
        dropatt=0.0, \
        untie_r=True, \
        init="normal", \
        init_range=0.01, \
        proj_init_std=0.01, \
        init_std=0.02, \
        layer_norm_epsilon=1e-5, \
        eos_token_id=0, \
        **kwargs) -> "TransfoXLConfig":
        # ---- NOTE-PYLINT ---- R0913: Too many arguments
        # pylint: disable=R0913
        # ---- NOTE-PYLINT ---- R0914: Too many local variables
        # pylint: disable=R0914
        # ---- NOTE-PYLINT ---- W0102: Dangerous default value [] as argument (dangerous-default-value)
        # pylint: disable=W0102
        """
        Load a pretrained TransfoXL config.
        """
        return TransfoXLConfig(
            vocab_size=vocab_size,
            cutoffs=cutoffs,
            d_model=d_model,
            d_embed=d_embed,
            n_head=n_head,
            d_head=d_head,
            d_inner=d_inner,
            div_val=div_val,
            pre_lnorm=pre_lnorm,
            n_layer=n_layer,
            tgt_len=tgt_len,
            ext_len=ext_len,
            mem_len=mem_len,
            clamp_len=clamp_len,
            same_length=same_length,
            proj_share_all_but_first=proj_share_all_but_first,
            attn_type=attn_type,
            sample_softmax=sample_softmax,
            adaptive=adaptive,
            tie_weight=tie_weight,
            dropout=dropout,
            dropatt=dropatt,
            untie_r=untie_r,
            init=init,
            init_range=init_range,
            proj_init_std=proj_init_std,
            init_std=init_std,
            layer_norm_epsilon=layer_norm_epsilon,
            eos_token_id=eos_token_id,
            **kwargs)
    @staticmethod
    def xlm_config_new( \
        vocab_size=30145, \
        emb_dim=2048, \
        n_layers=12, \
        n_heads=16, \
        dropout=0.1, \
        attention_dropout=0.1, \
        gelu_activation=True, \
        sinusoidal_embeddings=False, \
        causal=False, \
        asm=False, \
        n_langs=1, \
        use_lang_emb=True, \
        max_position_embeddings=512, \
        embed_init_std=2048 ** -0.5, \
        layer_norm_eps=1e-12, \
        init_std=0.02, \
        bos_index=0, \
        eos_index=1, \
        pad_index=2, \
        unk_index=3, \
        mask_index=5, \
        is_encoder=True, \
        summary_type="first", \
        summary_use_proj=True, \
        summary_activation=None, \
        summary_proj_to_labels=True, \
        summary_first_dropout=0.1, \
        start_n_top=5, \
        end_n_top=5, \
        mask_token_id=0, \
        lang_id=0, \
        pad_token_id=2, \
        bos_token_id=0, \
        **kwargs): # ---- NOTE-PLACE-HOLDER ---- -> "PretrainedConfig":
        # ---- NOTE-PYLINT ---- R0913: Too many arguments
        # pylint: disable=R0913
        # ---- NOTE-PYLINT ---- R0914: Too many local variables
        # pylint: disable=R0914
        """
        Load a pretrained XLM config.
        """
        return XLMConfig(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            gelu_activation=gelu_activation,
            sinusoidal_embeddings=sinusoidal_embeddings,
            causal=causal,
            asm=asm,
            n_langs=n_langs,
            use_lang_emb=use_lang_emb,
            max_position_embeddings=max_position_embeddings,
            embed_init_std=embed_init_std,
            layer_norm_eps=layer_norm_eps,
            init_std=init_std,
            bos_index=bos_index,
            eos_index=eos_index,
            pad_index=pad_index,
            unk_index=unk_index,
            mask_index=mask_index,
            is_encoder=is_encoder,
            summary_type=summary_type,
            summary_use_proj=summary_use_proj,
            summary_activation=summary_activation,
            summary_proj_to_labels=summary_proj_to_labels,
            summary_first_dropout=summary_first_dropout,
            start_n_top=start_n_top,
            end_n_top=end_n_top,
            mask_token_id=mask_token_id,
            lang_id=lang_id,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            **kwargs)
    @staticmethod
    def xlnet_config_new( \
        vocab_size=32000, \
        d_model=1024, \
        n_layer=24, \
        n_head=16, \
        d_inner=4096, \
        ff_activation="gelu", \
        untie_r=True, \
        attn_type="bi", \
        initializer_range=0.02, \
        layer_norm_eps=1e-12, \
        dropout=0.1, \
        mem_len=None, \
        reuse_len=None, \
        bi_data=False, \
        clamp_len=-1, \
        same_length=False, \
        summary_type="last", \
        summary_use_proj=True, \
        summary_activation="tanh", \
        summary_last_dropout=0.1, \
        start_n_top=5, \
        end_n_top=5, \
        pad_token_id=5, \
        bos_token_id=1, \
        eos_token_id=2, \
        **kwargs) -> "XLNetConfig":
        # ---- NOTE-PYLINT ---- R0913: Too many arguments
        # pylint: disable=R0913
        # ---- NOTE-PYLINT ---- R0914: Too many local variables
        # pylint: disable=R0914
        """
        Load a pretrained XLNet config.
        """
        return XLNetConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layer=n_layer,
            n_head=n_head,
            d_inner=d_inner,
            ff_activation=ff_activation,
            untie_r=untie_r,
            attn_type=attn_type,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            dropout=dropout,
            mem_len=mem_len,
            reuse_len=reuse_len,
            bi_data=bi_data,
            clamp_len=clamp_len,
            same_length=same_length,
            summary_type=summary_type,
            summary_use_proj=summary_use_proj,
            summary_activation=summary_activation,
            summary_last_dropout=summary_last_dropout,
            start_n_top=start_n_top,
            end_n_top=end_n_top,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs)

    @staticmethod
    def bert_model_new( \
        config):
        """
        Create a new BERT model.
        """
        return BertModel(
            config=config)
    @staticmethod
    def roberta_model_new( \
        config):
        """
        Create a new ROBERTA model.
        """
        return RobertaModel(
            config=config)
    @staticmethod
    def gpt2_model_new( \
        config):
        """
        Load a pretrained GPT2 model.
        """
        return GPT2Model(
            config=config)
    @staticmethod
    def openai_model_new( \
        config):
        """
        Load a pretrained OpenAI model.
        """
        return OpenAIGPTModel(
            config=config)
    @staticmethod
    def transfo_xl_model_new( \
        config):
        """
        Load a pretrained TransfoXL model.
        """
        return TransfoXLModel(
            config=config)
    @staticmethod
    def xlm_model_new( \
        config):
        """
        Load a pretrained XLM model.
        """
        return XLMModel(
            config=config)
    @staticmethod
    def xlnet_model_new( \
        config):
        """
        Load a pretrained XLNet model.
        """
        return XLNetModel(
            config=config)
    @staticmethod
    def model_new( \
        model_type: str, \
        config):
        """
        Load a model model.
        """
        if model_type is None:
            DebuggingHelper.throw_exception(
                'input argument, model_type, is None')
        if model_type not in PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPES:
            DebuggingHelper.throw_exception(
                f'input argument, model_type, is not in the list {PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPES}')
        switcher = {
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_BERT: \
                PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
                    bert_model_new,
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_ROBERTA: \
                PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
                    roberta_model_new,
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_GPT2: \
                PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
                    gpt2_model_new,
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_OPENAI: \
                PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
                    openai_model_new,
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_TRANSFO_XI: \
                PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
                    transfo_xl_model_new,
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_XLM: \
                PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
                    xlm_model_new,
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_XLNET: \
                PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
                    xlnet_model_new,
        }
        model_new = switcher.get(
            model_type)
        return model_new(config)

    @staticmethod
    def bert_pretrained_model_from_pretrained( \
        pretrained_model_name_or_path: str, \
        *argv, **kwargs):
        """
        Load a pretrained BERT model.
        """
        return BertPreTrainedModel.from_pretrained(
            pretrained_model_name_or_path,
            *argv, **kwargs)
    @staticmethod
    def roberta_pretrained_model_from_pretrained( \
        pretrained_model_name_or_path: str, \
        *argv, **kwargs):
        """
        Load a pretrained ROBERTA model.
        """
        return RobertaPreTrainedModel.from_pretrained(
            pretrained_model_name_or_path,
            *argv, **kwargs)
    @staticmethod
    def gpt2_pretrained_model_from_pretrained( \
        pretrained_model_name_or_path: str, \
        *argv, **kwargs):
        """
        Load a pretrained GPT2 model.
        """
        return GPT2PreTrainedModel.from_pretrained(
            pretrained_model_name_or_path,
            *argv, **kwargs)
    @staticmethod
    def openai_pretrained_model_from_pretrained( \
        pretrained_model_name_or_path: str, \
        *argv, **kwargs):
        """
        Load a pretrained OpenAI model.
        """
        return OpenAIGPTPreTrainedModel.from_pretrained(
            pretrained_model_name_or_path,
            *argv, **kwargs)
    @staticmethod
    def transfo_xl_pretrained_model_from_pretrained( \
        pretrained_model_name_or_path: str, \
        *argv, **kwargs):
        """
        Load a pretrained TransfoXL model.
        """
        return TransfoXLPreTrainedModel.from_pretrained(
            pretrained_model_name_or_path,
            *argv, **kwargs)
    @staticmethod
    def xlm_pretrained_model_from_pretrained( \
        pretrained_model_name_or_path: str, \
        *argv, **kwargs):
        """
        Load a pretrained XLM model.
        """
        return XLMPreTrainedModel.from_pretrained(
            pretrained_model_name_or_path,
            *argv, **kwargs)
    @staticmethod
    def xlnet_pretrained_model_from_pretrained( \
        pretrained_model_name_or_path: str, \
        *argv, **kwargs):
        """
        Load a pretrained XLNet model.
        """
        return XLNetPreTrainedModel.from_pretrained(
            pretrained_model_name_or_path,
            *argv, **kwargs)
    @staticmethod
    def pretrained_model_from_pretrained( \
        pretrained_model_name_or_path: str, \
        model_type: str = '', \
        *argv, **kwargs):
        """
        Load a model model.
        """
        if pretrained_model_name_or_path is None:
            DebuggingHelper.throw_exception(
                'input argument, pretrained_model_name_or_path, is None')
        model_key_type_map: Dict[str, str] = \
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_model_key_type_map()
        # ---- NOTE-PYLINT ---- E1135: Value '' doesn't support membership test
        #                              (unsupported-membership-test)
        # pylint: disable=E1135
        if pretrained_model_name_or_path in model_key_type_map:
            # ---- NOTE-PYLINT ---- E1136: Value '' is unsubscriptable
            #                              ((unsubscriptable-object)
            # pylint: disable=E1136
            model_type = model_key_type_map[pretrained_model_name_or_path]
        else:
            if DatatypeHelper.is_none_empty_whitespaces_or_nan(model_type):
                DebuggingHelper.throw_exception(
                    f'pretrained_model_name_or_path={pretrained_model_name_or_path} is not in '
                    f'model_key_type_map|{model_key_type_map}|'
                    f', model_type|{model_type}| is empty')
        switcher = {
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_BERT: \
                PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
                    bert_pretrained_model_from_pretrained,
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_ROBERTA: \
                PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
                    roberta_pretrained_model_from_pretrained,
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_GPT2: \
                PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
                    gpt2_pretrained_model_from_pretrained,
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_OPENAI: \
                PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
                    openai_pretrained_model_from_pretrained,
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_TRANSFO_XI: \
                PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
                    transfo_xl_pretrained_model_from_pretrained,
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_XLM: \
                PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
                    xlm_pretrained_model_from_pretrained,
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_XLNET: \
                PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
                    xlnet_pretrained_model_from_pretrained,
        }
        pretrained_model_from_pretrained = switcher.get(
            model_type)
        if pretrained_model_from_pretrained is None:
            DebuggingHelper.throw_exception(
                'pretrained_model_from_pretrained is None, '
                f'pretrained_model_name_or_path={pretrained_model_name_or_path}'
                f', model_type={model_type}'
                f', model_key_type_map|{model_key_type_map}|')
        return pretrained_model_from_pretrained(pretrained_model_name_or_path, *argv, **kwargs)

    @staticmethod
    def bert_model_from_pretrained( \
        pretrained_model_name_or_path: str, \
        *argv, **kwargs):
        """
        Load a pretrained BERT model.
        """
        return BertModel.from_pretrained(
            pretrained_model_name_or_path,
            *argv, **kwargs)
    @staticmethod
    def roberta_model_from_pretrained( \
        pretrained_model_name_or_path: str, \
        *argv, **kwargs):
        """
        Load a pretrained ROBERTA model.
        """
        return RobertaModel.from_pretrained(
            pretrained_model_name_or_path,
            *argv, **kwargs)
    @staticmethod
    def gpt2_model_from_pretrained( \
        pretrained_model_name_or_path: str, \
        *argv, **kwargs):
        """
        Load a pretrained GPT2 model.
        """
        return GPT2Model.from_pretrained(
            pretrained_model_name_or_path,
            *argv, **kwargs)
    @staticmethod
    def openai_model_from_pretrained( \
        pretrained_model_name_or_path: str, \
        *argv, **kwargs):
        """
        Load a pretrained OpenAI model.
        """
        return OpenAIGPTModel.from_pretrained(
            pretrained_model_name_or_path,
            *argv, **kwargs)
    @staticmethod
    def transfo_xl_model_from_pretrained( \
        pretrained_model_name_or_path: str, \
        *argv, **kwargs):
        """
        Load a pretrained TransfoXL model.
        """
        return TransfoXLModel.from_pretrained(
            pretrained_model_name_or_path,
            *argv, **kwargs)
    @staticmethod
    def xlm_model_from_pretrained( \
        pretrained_model_name_or_path: str, \
        *argv, **kwargs):
        """
        Load a pretrained XLM model.
        """
        return XLMModel.from_pretrained(
            pretrained_model_name_or_path,
            *argv, **kwargs)
    @staticmethod
    def xlnet_model_from_pretrained( \
        pretrained_model_name_or_path: str, \
        *argv, **kwargs):
        """
        Load a pretrained XLNet model.
        """
        return XLNetModel.from_pretrained(
            pretrained_model_name_or_path,
            *argv, **kwargs)
    @staticmethod
    def model_from_pretrained( \
        pretrained_model_name_or_path: str, \
        model_type: str = '', \
        *argv, **kwargs):
        """
        Load a model model.
        """
        if pretrained_model_name_or_path is None:
            DebuggingHelper.throw_exception(
                'input argument, pretrained_model_name_or_path, is None')
        model_key_type_map: Dict[str, str] = \
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_model_key_type_map()
        # ---- NOTE-PYLINT ---- E1135: Value '' doesn't support membership test
        #                              (unsupported-membership-test)
        # pylint: disable=E1135
        if pretrained_model_name_or_path in model_key_type_map:
            # ---- NOTE-PYLINT ---- E1136: Value '' is unsubscriptable
            #                              ((unsubscriptable-object)
            # pylint: disable=E1136
            model_type = model_key_type_map[pretrained_model_name_or_path]
        else:
            if DatatypeHelper.is_none_empty_whitespaces_or_nan(model_type):
                DebuggingHelper.throw_exception(
                    f'pretrained_model_name_or_path={pretrained_model_name_or_path} is not in '
                    f'model_key_type_map|{model_key_type_map}|'
                    f', model_type|{model_type}| is empty')
        switcher = {
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_BERT: \
                PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
                    bert_model_from_pretrained,
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_ROBERTA: \
                PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
                    roberta_model_from_pretrained,
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_GPT2: \
                PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
                    gpt2_model_from_pretrained,
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_OPENAI: \
                PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
                    openai_model_from_pretrained,
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_TRANSFO_XI: \
                PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
                    transfo_xl_model_from_pretrained,
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_XLM: \
                PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
                    xlm_model_from_pretrained,
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_XLNET: \
                PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
                    xlnet_model_from_pretrained,
        }
        model_from_pretrained = switcher.get(
            model_type)
        if model_from_pretrained is None:
            DebuggingHelper.throw_exception(
                'model_from_pretrained is None, '
                f'pretrained_model_name_or_path={pretrained_model_name_or_path}'
                f', model_type={model_type}'
                f', model_key_type_map|{model_key_type_map}|')
        return model_from_pretrained(pretrained_model_name_or_path, *argv, **kwargs)

    @staticmethod
    def gpt2_lm_head_model_from_pretrained( \
        pretrained_model_name_or_path: str, \
        *argv, **kwargs):
        """
        Load a pretrained GPT2 model.
        """
        return GPT2LMHeadModel.from_pretrained(
            pretrained_model_name_or_path,
            *argv, **kwargs)
    @staticmethod
    def openai_lm_head_model_from_pretrained( \
        pretrained_model_name_or_path: str, \
        *argv, **kwargs):
        """
        Load a pretrained OpenAI model.
        """
        return OpenAIGPTLMHeadModel.from_pretrained(
            pretrained_model_name_or_path,
            *argv, **kwargs)
    @staticmethod
    def transfo_xl_lm_head_model_from_pretrained( \
        pretrained_model_name_or_path: str, \
        *argv, **kwargs):
        """
        Load a pretrained TransfoXL model.
        """
        return TransfoXLLMHeadModel.from_pretrained(
            pretrained_model_name_or_path,
            *argv, **kwargs)
    @staticmethod
    def xlm_lm_head_model_from_pretrained( \
        pretrained_model_name_or_path: str, \
        *argv, **kwargs):
        """
        Load a pretrained XLM model.
        """
        return XLMWithLMHeadModel.from_pretrained(
            pretrained_model_name_or_path,
            *argv, **kwargs)
    @staticmethod
    def xlnet_lm_head_model_from_pretrained( \
        pretrained_model_name_or_path: str, \
        *argv, **kwargs):
        """
        Load a pretrained XLNet model.
        """
        return XLNetLMHeadModel.from_pretrained(
            pretrained_model_name_or_path,
            *argv, **kwargs)
    @staticmethod
    def lm_head_model_from_pretrained( \
        pretrained_model_name_or_path: str, \
        model_type: str = '', \
        *argv, **kwargs):
        """
        Load a LM head model.
        """
        if pretrained_model_name_or_path is None:
            DebuggingHelper.throw_exception(
                'input argument, pretrained_model_name_or_path, is None')
        model_key_type_map: Dict[str, str] = \
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_model_key_type_map()
        # ---- NOTE-PYLINT ---- E1135: Value '' doesn't support membership test
        #                              (unsupported-membership-test)
        # pylint: disable=E1135
        if pretrained_model_name_or_path in model_key_type_map:
            # ---- NOTE-PYLINT ---- E1136: Value '' is unsubscriptable
            #                              ((unsubscriptable-object)
            # pylint: disable=E1136
            model_type = model_key_type_map[pretrained_model_name_or_path]
        else:
            if DatatypeHelper.is_none_empty_whitespaces_or_nan(model_type):
                DebuggingHelper.throw_exception(
                    f'pretrained_model_name_or_path={pretrained_model_name_or_path} is not in '
                    f'model_key_type_map|{model_key_type_map}|'
                    f', model_type|{model_type}| is empty')
        switcher = {
            # PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_BERT: \
            #     PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
            #         bert_lm_head_model_from_pretrained,
            # PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_ROBERTA: \
            #     PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
            #         roberta_lm_head_model_from_pretrained,
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_GPT2: \
                PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
                    gpt2_lm_head_model_from_pretrained,
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_OPENAI: \
                PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
                    openai_lm_head_model_from_pretrained,
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_TRANSFO_XI: \
                PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
                    transfo_xl_lm_head_model_from_pretrained,
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_XLM: \
                PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
                    xlm_lm_head_model_from_pretrained,
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_XLNET: \
                PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
                    xlnet_lm_head_model_from_pretrained,
        }
        lm_head_model_from_pretrained = switcher.get(
            model_type)
        return lm_head_model_from_pretrained(pretrained_model_name_or_path, *argv, **kwargs)

    @staticmethod
    def gpt2_double_heads_model_from_pretrained( \
        pretrained_model_name_or_path: str, \
        *argv, **kwargs):
        """
        Load a pretrained GPT2 model.
        """
        return GPT2DoubleHeadsModel.from_pretrained(
            pretrained_model_name_or_path,
            *argv, **kwargs)
    @staticmethod
    def openai_double_heads_model_from_pretrained( \
        pretrained_model_name_or_path: str, \
        *argv, **kwargs):
        """
        Load a pretrained OpenAI model.
        """
        return OpenAIGPTDoubleHeadsModel.from_pretrained(
            pretrained_model_name_or_path,
            *argv, **kwargs)
    @staticmethod
    def double_heads_model_from_pretrained( \
        pretrained_model_name_or_path: str, \
        model_type: str = '', \
        *argv, **kwargs):
        """
        Load a LM head model.
        """
        if pretrained_model_name_or_path is None:
            DebuggingHelper.throw_exception(
                'input argument, pretrained_model_name_or_path, is None')
        model_key_type_map: Dict[str, str] = \
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_model_key_type_map()
        # ---- NOTE-PYLINT ---- E1135: Value '' doesn't support membership test
        #                              (unsupported-membership-test)
        # pylint: disable=E1135
        if pretrained_model_name_or_path in model_key_type_map:
            # ---- NOTE-PYLINT ---- E1136: Value '' is unsubscriptable
            #                              ((unsubscriptable-object)
            # pylint: disable=E1136
            model_type = model_key_type_map[pretrained_model_name_or_path]
        else:
            if DatatypeHelper.is_none_empty_whitespaces_or_nan(model_type):
                DebuggingHelper.throw_exception(
                    f'pretrained_model_name_or_path={pretrained_model_name_or_path} is not in '
                    f'model_key_type_map|{model_key_type_map}|'
                    f', model_type|{model_type}| is empty')
        switcher = {
            # PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_BERT: \
            #     PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
            #         bert_double_heads_model_from_pretrained,
            # PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_ROBERTA: \
            #     PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
            #         roberta_double_heads_model_from_pretrained,
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_GPT2: \
                PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
                    gpt2_double_heads_model_from_pretrained,
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_OPENAI: \
                PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
                    openai_double_heads_model_from_pretrained,
            # PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_TRANSFO_XI: \
            #     PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
            #         transfo_xl_double_heads_model_from_pretrained,
            # PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_XLM: \
            #     PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
            #         xlm_double_heads_model_from_pretrained,
            # PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_XLNET: \
            #     PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
            #         xlnet_double_heads_model_from_pretrained,
        }
        double_heads_model_from_pretrained = switcher.get(
            model_type)
        return double_heads_model_from_pretrained(pretrained_model_name_or_path, *argv, **kwargs)

    @staticmethod
    def bert_sequence_classification_model_from_pretrained( \
        pretrained_model_name_or_path: str, \
        *argv, **kwargs):
        """
        Load a pretrained BERT model.
        """
        return BertForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path,
            *argv, **kwargs)
    @staticmethod
    def roberta_sequence_classification_model_from_pretrained( \
        pretrained_model_name_or_path: str, \
        *argv, **kwargs):
        """
        Load a pretrained ROBERTA model.
        """
        return RobertaForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path,
            *argv, **kwargs)
    @staticmethod
    def xlm_sequence_classification_model_from_pretrained( \
        pretrained_model_name_or_path: str, \
        *argv, **kwargs):
        """
        Load a pretrained XLM model.
        """
        return XLMForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path,
            *argv, **kwargs)
    @staticmethod
    def xlnet_sequence_classification_model_from_pretrained( \
        pretrained_model_name_or_path: str, \
        *argv, **kwargs):
        """
        Load a pretrained XLNet model.
        """
        return XLNetForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path,
            *argv, **kwargs)
    @staticmethod
    def sequence_classification_model_from_pretrained( \
        pretrained_model_name_or_path: str, \
        model_type: str = '', \
        *argv, **kwargs):
        """
        Load a pretrained model.
        """
        if pretrained_model_name_or_path is None:
            DebuggingHelper.throw_exception(
                'input argument, pretrained_model_name_or_path, is None')
        model_key_type_map: Dict[str, str] = \
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_model_key_type_map()
        # ---- NOTE-PYLINT ---- E1135: Value '' doesn't support membership test
        #                              (unsupported-membership-test)
        # pylint: disable=E1135
        if pretrained_model_name_or_path in model_key_type_map:
            # ---- NOTE-PYLINT ---- E1136: Value '' is unsubscriptable
            #                              ((unsubscriptable-object)
            # pylint: disable=E1136
            model_type = model_key_type_map[pretrained_model_name_or_path]
        else:
            if DatatypeHelper.is_none_empty_whitespaces_or_nan(model_type):
                DebuggingHelper.throw_exception(
                    f'pretrained_model_name_or_path={pretrained_model_name_or_path} is not in '
                    f'model_key_type_map|{model_key_type_map}|'
                    f', model_type|{model_type}| is empty')
        switcher = {
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_BERT: \
                PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
                    bert_sequence_classification_model_from_pretrained,
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_ROBERTA: \
                PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
                    roberta_sequence_classification_model_from_pretrained,
            # PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_GPT2: \
            #     PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
            #         gpt2_sequence_classification_model_from_pretrained,
            # PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_OPENAI: \
            #     PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
            #         openai_sequence_classification_from_pretrained,
            # PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_TRANSFO_XI: \
            #     PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
            #         transfo_xl_sequence_classification_from_pretrained,
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_XLM: \
                PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
                    xlm_sequence_classification_model_from_pretrained,
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_XLNET: \
                PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
                    xlnet_sequence_classification_model_from_pretrained,
        }
        sequence_classification_model_from_pretrained = switcher.get(
            model_type)
        return sequence_classification_model_from_pretrained(pretrained_model_name_or_path, *argv, **kwargs)

    @staticmethod
    def bert_question_answering_model_from_pretrained( \
        pretrained_model_name_or_path: str, \
        *argv, **kwargs):
        """
        Load a pretrained BERT model.
        """
        return BertForQuestionAnswering.from_pretrained(
            pretrained_model_name_or_path,
            *argv, **kwargs)
    @staticmethod
    def roberta_question_answering_model_from_pretrained( \
        pretrained_model_name_or_path: str, \
        *argv, **kwargs):
        """
        Load a pretrained ROBERTA model.
        """
        return RobertaForQuestionAnswering.from_pretrained(
            pretrained_model_name_or_path,
            *argv, **kwargs)
    @staticmethod
    def xlm_question_answering_model_from_pretrained( \
        pretrained_model_name_or_path: str, \
        *argv, **kwargs):
        """
        Load a pretrained XLM model.
        """
        return XLMForQuestionAnswering.from_pretrained(
            pretrained_model_name_or_path,
            *argv, **kwargs)
    @staticmethod
    def xlnet_question_answering_model_from_pretrained( \
        pretrained_model_name_or_path: str, \
        *argv, **kwargs):
        """
        Load a pretrained XLNet model.
        """
        return XLNetForQuestionAnswering.from_pretrained(
            pretrained_model_name_or_path,
            *argv, **kwargs)
    @staticmethod
    def question_answering_model_from_pretrained( \
        pretrained_model_name_or_path: str, \
        model_type: str = '', \
        *argv, **kwargs):
        """
        Load a pretrained model.
        """
        if pretrained_model_name_or_path is None:
            DebuggingHelper.throw_exception(
                'input argument, pretrained_model_name_or_path, is None')
        model_key_type_map: Dict[str, str] = \
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_model_key_type_map()
        # ---- NOTE-PYLINT ---- E1135: Value '' doesn't support membership test
        #                              (unsupported-membership-test)
        # pylint: disable=E1135
        if pretrained_model_name_or_path in model_key_type_map:
            # ---- NOTE-PYLINT ---- E1136: Value '' is unsubscriptable
            #                              ((unsubscriptable-object)
            # pylint: disable=E1136
            model_type = model_key_type_map[pretrained_model_name_or_path]
        else:
            if DatatypeHelper.is_none_empty_whitespaces_or_nan(model_type):
                DebuggingHelper.throw_exception(
                    f'pretrained_model_name_or_path={pretrained_model_name_or_path} is not in '
                    f'model_key_type_map|{model_key_type_map}|'
                    f', model_type|{model_type}| is empty')
        switcher = {
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_BERT: \
                PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
                    bert_question_answering_model_from_pretrained,
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_ROBERTA: \
                PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
                    roberta_question_answering_model_from_pretrained,
            # PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_GPT2: \
            #     PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
            #         gpt2_question_answering_model_from_pretrained,
            # PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_OPENAI: \
            #     PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
            #         openai_question_answering_from_pretrained,
            # PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_TRANSFO_XI: \
            #     PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
            #         transfo_xl_question_answering_from_pretrained,
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_XLM: \
                PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
                    xlm_question_answering_model_from_pretrained,
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_XLNET: \
                PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
                    xlnet_question_answering_model_from_pretrained,
        }
        question_answering_model_from_pretrained = switcher.get(
            model_type)
        return question_answering_model_from_pretrained(pretrained_model_name_or_path, *argv, **kwargs)

    @staticmethod
    def bert_tokenizer_new( \
        vocab_file, \
        do_lower_case=True, \
        do_basic_tokenize=True, \
        never_split=None, \
        unk_token="[UNK]", \
        sep_token="[SEP]", \
        pad_token="[PAD]", \
        cls_token="[CLS]", \
        mask_token="[MASK]", \
        tokenize_chinese_chars=True, \
        **kwargs):
        # ---- NOTE-PYLINT ---- R0913: Too many arguments
        # pylint: disable=R0913
        """
        Load a pretrained BERT tokenizer.
        """
        return BertTokenizer(
            vocab_file=vocab_file,
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            never_split=never_split,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            **kwargs)
    @staticmethod
    def roberta_tokenizer_new( \
        vocab_file, \
        merges_file, \
        errors="replace", \
        bos_token="<s>", \
        eos_token="</s>", \
        sep_token="</s>", \
        cls_token="<s>", \
        unk_token="<unk>", \
        pad_token="<pad>", \
        mask_token="<mask>", \
        **kwargs):
        # ---- NOTE-PYLINT ---- R0913: Too many arguments
        # pylint: disable=R0913
        """
        Load a pretrained ROBERTA tokenizer.
        """
        return RobertaTokenizer(
            vocab_file=vocab_file,
            merges_file=merges_file,
            errors=errors,
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            **kwargs)
    @staticmethod
    def gpt2_tokenizer_new( \
        vocab_file, \
        merges_file, \
        errors="replace", \
        unk_token="<|endoftext|>", \
        bos_token="<|endoftext|>", \
        eos_token="<|endoftext|>", \
        **kwargs):
        # ---- NOTE-PYLINT ---- R0913: Too many arguments
        # pylint: disable=R0913
        """
        Load a pretrained BERT tokenizer.
        """
        return GPT2Tokenizer(
            vocab_file=vocab_file,
            merges_file=merges_file,
            errors=errors,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            **kwargs)
    @staticmethod
    def openai_tokenizer_new( \
        vocab_file, \
        merges_file, \
        unk_token="<unk>", \
        **kwargs):
        # ---- NOTE-PYLINT ---- R0913: Too many arguments
        # pylint: disable=R0913
        """
        Load a pretrained OpenAI tokenizer.
        """
        return OpenAIGPTTokenizer(
            vocab_file=vocab_file,
            merges_file=merges_file,
            unk_token=unk_token,
            **kwargs)
    @staticmethod
    def transfo_xl_tokenizer_new( \
        special=None, \
        min_freq=0, \
        max_size=None, \
        lower_case=False, \
        delimiter=None, \
        vocab_file=None, \
        pretrained_vocab_file=None, \
        never_split=None, \
        unk_token="<unk>", \
        eos_token="<eos>", \
        additional_special_tokens=["<formula>"], **kwargs):
        # ---- NOTE-PYLINT ---- R0913: Too many arguments
        # pylint: disable=R0913
        # ---- NOTE-PYLINT ---- W0102: Dangerous default value [] as argument (dangerous-default-value)
        # pylint: disable=W0102
        """
        Load a pretrained TransfoXL tokenizer.
        """
        return TransfoXLTokenizer(
            special=special,
            min_freq=min_freq,
            max_size=max_size,
            lower_case=lower_case,
            delimiter=delimiter,
            vocab_file=vocab_file,
            pretrained_vocab_file=pretrained_vocab_file,
            never_split=never_split,
            unk_token=unk_token,
            eos_token=eos_token,
            additional_special_tokens=additional_special_tokens,
            **kwargs)
    @staticmethod
    def xlm_tokenizer_new( \
        vocab_file, \
        merges_file, \
        unk_token="<unk>", \
        bos_token="<s>", \
        sep_token="</s>", \
        pad_token="<pad>", \
        cls_token="</s>", \
        mask_token="<special1>", \
        additional_special_tokens=[ \
            "<special0>", \
            "<special1>", \
            "<special2>", \
            "<special3>", \
            "<special4>", \
            "<special5>", \
            "<special6>", \
            "<special7>", \
            "<special8>", \
            "<special9>", \
        ], \
        lang2id=None, \
        id2lang=None, \
        do_lowercase_and_remove_accent=True, \
        **kwargs):
        # ---- NOTE-PYLINT ---- R0913: Too many arguments
        # pylint: disable=R0913
        # ---- NOTE-PYLINT ---- W0102: Dangerous default value [] as argument (dangerous-default-value)
        # pylint: disable=W0102
        """
        Load a pretrained XLM tokenizer.
        """
        return XLMTokenizer(
            vocab_file=vocab_file,
            merges_file=merges_file,
            unk_token=unk_token,
            bos_token=bos_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            additional_special_tokens=additional_special_tokens,
            lang2id=lang2id,
            id2lang=id2lang,
            do_lowercase_and_remove_accent=do_lowercase_and_remove_accent,
            **kwargs)
    @staticmethod
    def xlnet_tokenizer_new( \
        vocab_file, \
        do_lower_case=False, \
        remove_space=True, \
        keep_accents=False, \
        bos_token="<s>", \
        eos_token="</s>", \
        unk_token="<unk>", \
        sep_token="<sep>", \
        pad_token="<pad>", \
        cls_token="<cls>", \
        mask_token="<mask>", \
        additional_special_tokens=["<eop>", "<eod>"], \
        **kwargs):
        # ---- NOTE-PYLINT ---- R0913: Too many arguments
        # pylint: disable=R0913
        # ---- NOTE-PYLINT ---- W0102: Dangerous default value [] as argument (dangerous-default-value)
        # pylint: disable=W0102
        """
        Load a pretrained XLNet tokenizer.
        """
        return XLNetTokenizer(
            vocab_file=vocab_file,
            do_lower_case=do_lower_case,
            remove_space=remove_space,
            keep_accents=keep_accents,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            additional_special_tokens=additional_special_tokens,
            **kwargs)

    @staticmethod
    def bert_tokenizer_from_pretrained( \
        pretrained_model_name_or_path: str, \
        *argv, **kwargs):
        """
        Load a pretrained BERT tokenizer.
        """
        return BertTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            *argv, **kwargs)
    @staticmethod
    def roberta_tokenizer_from_pretrained( \
        pretrained_model_name_or_path: str, \
        *argv, **kwargs):
        """
        Load a pretrained ROBERTA tokenizer.
        """
        return RobertaTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            *argv, **kwargs)
    @staticmethod
    def gpt2_tokenizer_from_pretrained( \
        pretrained_model_name_or_path: str, \
        *argv, **kwargs):
        """
        Load a pretrained GPT2 tokenizer.
        """
        return GPT2Tokenizer.from_pretrained(
            pretrained_model_name_or_path,
            *argv, **kwargs)
    @staticmethod
    def openai_tokenizer_from_pretrained( \
        pretrained_model_name_or_path: str, \
        *argv, **kwargs):
        """
        Load a pretrained OpenAI tokenizer.
        """
        return OpenAIGPTTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            *argv, **kwargs)
    @staticmethod
    def transfo_xl_tokenizer_from_pretrained( \
        pretrained_model_name_or_path: str, \
        *argv, **kwargs):
        """
        Load a pretrained TransfoXL tokenizer.
        """
        return TransfoXLTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            *argv, **kwargs)
    @staticmethod
    def xlm_tokenizer_from_pretrained( \
        pretrained_model_name_or_path: str, \
        *argv, **kwargs):
        """
        Load a pretrained XLM tokenizer.
        """
        return XLMTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            *argv, **kwargs)
    @staticmethod
    def xlnet_tokenizer_from_pretrained( \
        pretrained_model_name_or_path: str, \
        *argv, **kwargs):
        """
        Load a pretrained XLNet tokenizer.
        """
        return XLNetTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            *argv, **kwargs)
    @staticmethod
    def tokenizer_from_pretrained( \
        pretrained_model_name_or_path: str, \
        model_type: str = '', \
        *argv, **kwargs):
        """
        Load a tokenizer tokenizer.
        """
        if pretrained_model_name_or_path is None:
            DebuggingHelper.throw_exception(
                'input argument, pretrained_model_name_or_path, is None')
        model_key_type_map: Dict[str, str] = \
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_model_key_type_map()
        # ---- NOTE-PYLINT ---- E1135: Value '' doesn't support membership test
        #                              (unsupported-membership-test)
        # pylint: disable=E1135
        if pretrained_model_name_or_path in model_key_type_map:
            # ---- NOTE-PYLINT ---- E1136: Value '' is unsubscriptable
            #                              ((unsubscriptable-object)
            # pylint: disable=E1136
            model_type = model_key_type_map[pretrained_model_name_or_path]
        else:
            if DatatypeHelper.is_none_empty_whitespaces_or_nan(model_type):
                DebuggingHelper.throw_exception(
                    f'pretrained_model_name_or_path={pretrained_model_name_or_path} is not in '
                    f'model_key_type_map|{model_key_type_map}|'
                    f', model_type|{model_type}| is empty')
        switcher = {
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_BERT: \
                PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
                    bert_tokenizer_from_pretrained,
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_ROBERTA: \
                PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
                    roberta_tokenizer_from_pretrained,
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_GPT2: \
                PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
                    gpt2_tokenizer_from_pretrained,
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_OPENAI: \
                PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
                    openai_tokenizer_from_pretrained,
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_TRANSFO_XI: \
                PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
                    transfo_xl_tokenizer_from_pretrained,
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_XLM: \
                PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
                    xlm_tokenizer_from_pretrained,
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_XLNET: \
                PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
                    xlnet_tokenizer_from_pretrained,
        }
        tokenizer_from_pretrained = switcher.get(
            model_type)
        return tokenizer_from_pretrained(pretrained_model_name_or_path, *argv, **kwargs)

    @staticmethod
    def bert_for_pretraining_model_from_pretrained( \
        pretrained_model_name_or_path: str, \
        *argv, **kwargs):
        """
        Load a pretrained BERT model.
        """
        return BertForPreTraining.from_pretrained(
            pretrained_model_name_or_path,
            *argv, **kwargs)
    # @staticmethod
    # def roberta_for_pretraining_model_from_pretrained( \
    #     pretrained_model_name_or_path: str, \
    #     *argv, **kwargs):
    #     """
    #     Load a pretrained ROBERTA model.
    #     """
    #     return RobertaForPreTraining.from_pretrained(
    #         pretrained_model_name_or_path,
    #         *argv, **kwargs)
    @staticmethod
    def pretraining_model_from_pretrained( \
        pretrained_model_name_or_path: str, \
        model_type: str = '', \
        *argv, **kwargs):
        """
        Load a model model.
        """
        if pretrained_model_name_or_path is None:
            DebuggingHelper.throw_exception(
                'input argument, pretrained_model_name_or_path, is None')
        model_key_type_map: Dict[str, str] = \
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_model_key_type_map()
        # ---- NOTE-PYLINT ---- E1135: Value '' doesn't support membership test
        #                              (unsupported-membership-test)
        # pylint: disable=E1135
        if pretrained_model_name_or_path in model_key_type_map:
            # ---- NOTE-PYLINT ---- E1136: Value '' is unsubscriptable
            #                              ((unsubscriptable-object)
            # pylint: disable=E1136
            model_type = model_key_type_map[pretrained_model_name_or_path]
        else:
            if DatatypeHelper.is_none_empty_whitespaces_or_nan(model_type):
                DebuggingHelper.throw_exception(
                    f'pretrained_model_name_or_path={pretrained_model_name_or_path} is not in '
                    f'model_key_type_map|{model_key_type_map}|'
                    f', model_type|{model_type}| is empty')
        switcher = {
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_BERT: \
                PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
                    bert_for_pretraining_model_from_pretrained,
            # PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_ROBERTA: \
            #     PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
            #         roberta_for_pretraining_model_from_pretrained,
            # PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_GPT2: \
            #     PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
            #         gpt2_for_pretraining_model_from_pretrained,
            # PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_OPENAI: \
            #     PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
            #         openai_for_pretraining_model_from_pretrained,
            # PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_TRANSFO_XI: \
            #     PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
            #         transfo_xl_for_pretraining_model_from_pretrained,
            # PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_XLM: \
            #     PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
            #         xlm_for_pretraining_model_from_pretrained,
            # PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_XLNET: \
            #     PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
            #         xlnet_for_pretraining_model_from_pretrained,
        }
        pretraining_model_from_pretrained = switcher.get(
            model_type)
        return pretraining_model_from_pretrained(pretrained_model_name_or_path, *argv, **kwargs)

    @staticmethod
    def bert_for_masked_lm_model_from_pretrained( \
        pretrained_model_name_or_path: str, \
        *argv, **kwargs):
        """
        Load a pretrained BERT model.
        """
        return BertForMaskedLM.from_pretrained(
            pretrained_model_name_or_path,
            *argv, **kwargs)
    @staticmethod
    def roberta_for_masked_lm_model_from_pretrained( \
        pretrained_model_name_or_path: str, \
        *argv, **kwargs):
        """
        Load a pretrained ROBERTA model.
        """
        return RobertaForMaskedLM.from_pretrained(
            pretrained_model_name_or_path,
            *argv, **kwargs)
    @staticmethod
    def masked_lm_model_from_pretrained( \
        pretrained_model_name_or_path: str, \
        model_type: str = '', \
        *argv, **kwargs):
        """
        Load a model model.
        """
        if pretrained_model_name_or_path is None:
            DebuggingHelper.throw_exception(
                'input argument, pretrained_model_name_or_path, is None')
        model_key_type_map: Dict[str, str] = \
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_model_key_type_map()
        # ---- NOTE-PYLINT ---- E1135: Value '' doesn't support membership test
        #                              (unsupported-membership-test)
        # pylint: disable=E1135
        if pretrained_model_name_or_path in model_key_type_map:
            # ---- NOTE-PYLINT ---- E1136: Value '' is unsubscriptable
            #                              ((unsubscriptable-object)
            # pylint: disable=E1136
            model_type = model_key_type_map[pretrained_model_name_or_path]
        else:
            if DatatypeHelper.is_none_empty_whitespaces_or_nan(model_type):
                DebuggingHelper.throw_exception(
                    f'pretrained_model_name_or_path={pretrained_model_name_or_path} is not in '
                    f'model_key_type_map|{model_key_type_map}|'
                    f', model_type|{model_type}| is empty')
        switcher = {
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_BERT: \
                PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
                    bert_for_masked_lm_model_from_pretrained,
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_ROBERTA: \
                PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
                    roberta_for_masked_lm_model_from_pretrained,
            # PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_GPT2: \
            #     PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
            #         gpt2_for_masked_lm_model_from_pretrained,
            # PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_OPENAI: \
            #     PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
            #         openai_for_masked_lm_model_from_pretrained,
            # PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_TRANSFO_XI: \
            #     PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
            #         transfo_xl_for_masked_lm_model_from_pretrained,
            # PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_XLM: \
            #     PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
            #         xlm_for_masked_lm_model_from_pretrained,
            # PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_XLNET: \
            #     PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
            #         xlnet_for_masked_lm_model_from_pretrained,
        }
        masked_lm_model_from_pretrained = switcher.get(
            model_type)
        return masked_lm_model_from_pretrained(pretrained_model_name_or_path, *argv, **kwargs)

    @staticmethod
    def bert_for_next_sentence_prediction_model_from_pretrained( \
        pretrained_model_name_or_path: str, \
        *argv, **kwargs):
        """
        Load a pretrained BERT model.
        """
        return BertForNextSentencePrediction.from_pretrained(
            pretrained_model_name_or_path,
            *argv, **kwargs)
    # @staticmethod
    # def roberta_for_next_sentence_prediction_model_from_pretrained( \
    #     pretrained_model_name_or_path: str, \
    #     *argv, **kwargs):
    #     """
    #     Load a pretrained ROBERTA model.
    #     """
    #     return RobertaForNextSentencePrediction.from_pretrained(
    #         pretrained_model_name_or_path,
    #         *argv, **kwargs)
    @staticmethod
    def next_sentence_prediction_model_from_pretrained( \
        pretrained_model_name_or_path: str, \
        model_type: str = '', \
        *argv, **kwargs):
        """
        Load a model model.
        """
        if pretrained_model_name_or_path is None:
            DebuggingHelper.throw_exception(
                'input argument, pretrained_model_name_or_path, is None')
        model_key_type_map: Dict[str, str] = \
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_model_key_type_map()
        # ---- NOTE-PYLINT ---- E1135: Value '' doesn't support membership test
        #                              (unsupported-membership-test)
        # pylint: disable=E1135
        if pretrained_model_name_or_path in model_key_type_map:
            # ---- NOTE-PYLINT ---- E1136: Value '' is unsubscriptable
            #                              ((unsubscriptable-object)
            # pylint: disable=E1136
            model_type = model_key_type_map[pretrained_model_name_or_path]
        else:
            if DatatypeHelper.is_none_empty_whitespaces_or_nan(model_type):
                DebuggingHelper.throw_exception(
                    f'pretrained_model_name_or_path={pretrained_model_name_or_path} is not in '
                    f'model_key_type_map|{model_key_type_map}|'
                    f', model_type|{model_type}| is empty')
        switcher = {
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_BERT: \
                PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
                    bert_for_next_sentence_prediction_model_from_pretrained,
            # PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_ROBERTA: \
            #     PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
            #         roberta_for_next_sentence_prediction_model_from_pretrained,
            # PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_GPT2: \
            #     PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
            #         gpt2_for_next_sentence_prediction_model_from_pretrained,
            # PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_OPENAI: \
            #     PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
            #         openai_for_next_sentence_prediction_model_from_pretrained,
            # PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_TRANSFO_XI: \
            #     PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
            #         transfo_xl_for_next_sentence_prediction_model_from_pretrained,
            # PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_XLM: \
            #     PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
            #         xlm_for_next_sentence_prediction_model_from_pretrained,
            # PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_XLNET: \
            #     PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
            #         xlnet_for_next_sentence_prediction_model_from_pretrained,
        }
        next_sentence_prediction_model_from_pretrained = switcher.get(
            model_type)
        return next_sentence_prediction_model_from_pretrained(pretrained_model_name_or_path, *argv, **kwargs)

    @staticmethod
    def bert_for_multiple_choice_model_from_pretrained( \
        pretrained_model_name_or_path: str, \
        *argv, **kwargs):
        """
        Load a pretrained BERT model.
        """
        return BertForMultipleChoice.from_pretrained(
            pretrained_model_name_or_path,
            *argv, **kwargs)
    @staticmethod
    def roberta_for_multiple_choice_model_from_pretrained( \
        pretrained_model_name_or_path: str, \
        *argv, **kwargs):
        """
        Load a pretrained ROBERTA model.
        """
        return RobertaForMultipleChoice.from_pretrained(
            pretrained_model_name_or_path,
            *argv, **kwargs)
    @staticmethod
    def multiple_choice_model_from_pretrained( \
        pretrained_model_name_or_path: str, \
        model_type: str = '', \
        *argv, **kwargs):
        """
        Load a model model.
        """
        if pretrained_model_name_or_path is None:
            DebuggingHelper.throw_exception(
                'input argument, pretrained_model_name_or_path, is None')
        model_key_type_map: Dict[str, str] = \
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_model_key_type_map()
        # ---- NOTE-PYLINT ---- E1135: Value '' doesn't support membership test
        #                              (unsupported-membership-test)
        # pylint: disable=E1135
        if pretrained_model_name_or_path in model_key_type_map:
            # ---- NOTE-PYLINT ---- E1136: Value '' is unsubscriptable
            #                              ((unsubscriptable-object)
            # pylint: disable=E1136
            model_type = model_key_type_map[pretrained_model_name_or_path]
        else:
            if DatatypeHelper.is_none_empty_whitespaces_or_nan(model_type):
                DebuggingHelper.throw_exception(
                    f'pretrained_model_name_or_path={pretrained_model_name_or_path} is not in '
                    f'model_key_type_map|{model_key_type_map}|'
                    f', model_type|{model_type}| is empty')
        switcher = {
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_BERT: \
                PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
                    bert_for_multiple_choice_model_from_pretrained,
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_ROBERTA: \
                PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
                    roberta_for_multiple_choice_model_from_pretrained,
            # PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_GPT2: \
            #     PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
            #         gpt2_for_multiple_choice_model_from_pretrained,
            # PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_OPENAI: \
            #     PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
            #         openai_for_multiple_choice_model_from_pretrained,
            # PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_TRANSFO_XI: \
            #     PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
            #         transfo_xl_for_multiple_choice_model_from_pretrained,
            # PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_XLM: \
            #     PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
            #         xlm_for_multiple_choice_model_from_pretrained,
            # PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_XLNET: \
            #     PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
            #         xlnet_for_multiple_choice_model_from_pretrained,
        }
        multiple_choice_model_from_pretrained = switcher.get(
            model_type)
        return multiple_choice_model_from_pretrained(pretrained_model_name_or_path, *argv, **kwargs)

    @staticmethod
    def bert_for_token_classification_model_from_pretrained( \
        pretrained_model_name_or_path: str, \
        *argv, **kwargs):
        """
        Load a pretrained BERT model.
        """
        return BertForTokenClassification.from_pretrained(
            pretrained_model_name_or_path,
            *argv, **kwargs)
    @staticmethod
    def roberta_for_token_classification_model_from_pretrained( \
        pretrained_model_name_or_path: str, \
        *argv, **kwargs):
        """
        Load a pretrained ROBERTA model.
        """
        return RobertaForTokenClassification.from_pretrained(
            pretrained_model_name_or_path,
            *argv, **kwargs)
    @staticmethod
    def token_classification_model_from_pretrained( \
        pretrained_model_name_or_path: str, \
        model_type: str = '', \
        *argv, **kwargs):
        """
        Load a model model.
        """
        if pretrained_model_name_or_path is None:
            DebuggingHelper.throw_exception(
                'input argument, pretrained_model_name_or_path, is None')
        model_key_type_map: Dict[str, str] = \
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_model_key_type_map()
        # ---- NOTE-PYLINT ---- E1135: Value '' doesn't support membership test
        #                              (unsupported-membership-test)
        # pylint: disable=E1135
        if pretrained_model_name_or_path in model_key_type_map:
            # ---- NOTE-PYLINT ---- E1136: Value '' is unsubscriptable
            #                              ((unsubscriptable-object)
            # pylint: disable=E1136
            model_type = model_key_type_map[pretrained_model_name_or_path]
        else:
            if DatatypeHelper.is_none_empty_whitespaces_or_nan(model_type):
                DebuggingHelper.throw_exception(
                    f'pretrained_model_name_or_path={pretrained_model_name_or_path} is not in '
                    f'model_key_type_map|{model_key_type_map}|'
                    f', model_type|{model_type}| is empty')
        switcher = {
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_BERT: \
                PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
                    bert_for_token_classification_model_from_pretrained,
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_ROBERTA: \
                PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
                    roberta_for_token_classification_model_from_pretrained,
            # PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_GPT2: \
            #     PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
            #         gpt2_for_token_classification_model_from_pretrained,
            # PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_OPENAI: \
            #     PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
            #         openai_for_token_classification_model_from_pretrained,
            # PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_TRANSFO_XI: \
            #     PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
            #         transfo_xl_for_token_classification_model_from_pretrained,
            # PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_XLM: \
            #     PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
            #         xlm_for_token_classification_model_from_pretrained,
            # PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_XLNET: \
            #     PytorchLanguageUnderstandingTransformersPretainedModelHelper.\
            #         xlnet_for_token_classification_model_from_pretrained,
        }
        token_classification_model_from_pretrained = switcher.get(
            model_type)
        return token_classification_model_from_pretrained(pretrained_model_name_or_path, *argv, **kwargs)

    @staticmethod
    def get_model_named_parameters( \
        model):
        """
        get_model_named_parameters()
        """
        if model is None:
            DebuggingHelper.throw_exception(
                'input argument, model, is None')
        return model.named_parameters()

    @staticmethod
    def get_optimizer_scheduer_configuration( \
        model, \
        optimizer_number_training_optimization_steps: int, \
        device_use_gpu_fp16: bool = False, \
        optimizer_learning_rate: float = 5e-5, \
        optimizer_warmup_proportion: float = 0.1, \
        optimizer_warmup_steps: int = 0, \
        optimizer_adam_epsilon: float = 1e-6, \
        optimizer_weight_decay: float = 0.0, \
        optimizer_no_decay_parameter_list: List[str] = None, \
        device_gpu_fp16_opt_level: str = None):
        # ---- NOTE-PYLINT ---- R0913: Too many arguments
        # pylint: disable=R0913
        # ---- NOTE-PYLINT ---- R0914: Too many local variables
        # pylint: disable=R0914
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        """
        Return an optimizer.
        REFERENCE: https://github.com/huggingface/pytorch-transformers/blob/master/examples/run_glue.py
        """
        if model is None:
            DebuggingHelper.throw_exception(
                'input argument, model, is None')
        if optimizer_no_decay_parameter_list is None:
            optimizer_no_decay_parameter_list = ['bias', 'LayerNorm.weight', 'LayerNorm.weight']
        model_named_parameters = \
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_model_named_parameters(model)
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model_named_parameters if not any(
                nd in n for nd in optimizer_no_decay_parameter_list)], 'weight_decay': optimizer_weight_decay},
            {'params': [p for n, p in model_named_parameters if any(
                nd in n for nd in optimizer_no_decay_parameter_list)], 'weight_decay': 0.0}]
        optimizer = AdamW(
            params=optimizer_grouped_parameters,
            lr=optimizer_learning_rate,
            eps=optimizer_adam_epsilon)
        if (optimizer_warmup_steps is None) or \
            (optimizer_warmup_steps < 0) or \
            (optimizer_warmup_steps >= optimizer_number_training_optimization_steps):
            optimizer_warmup_steps = int(optimizer_warmup_proportion * optimizer_number_training_optimization_steps)
        if (optimizer_warmup_steps is None) or \
            (optimizer_warmup_steps < 0) or \
            (optimizer_warmup_steps >= optimizer_number_training_optimization_steps):
            optimizer_warmup_steps = 0
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=optimizer_warmup_steps,
            num_training_steps=optimizer_number_training_optimization_steps)
        # ---- NOTE-TODO ---- scheduler = WarmupLinearSchedule(
        # ---- NOTE-TODO ----     optimizer=optimizer,
        # ---- NOTE-TODO ----     warmup_steps=optimizer_warmup_steps,
        # ---- NOTE-TODO ----     t_total=optimizer_number_training_optimization_steps)
        # ---- NOTE-conditional-for-FP16 ---- if device_use_gpu_fp16:
        # ---- NOTE-conditional-for-FP16 ----     try:
        # ---- NOTE-conditional-for-FP16 ----         # ---- NOTE-PYLINT ---- C0415: Import outside toplevel (apex) (import-outside-toplevel)
        # ---- NOTE-conditional-for-FP16 ----         # pylint: disable=C0415
        # ---- NOTE-conditional-for-FP16 ----         from apex import amp
        # ---- NOTE-conditional-for-FP16 ----     except ImportError:
        # ---- NOTE-conditional-for-FP16 ----         raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use FP16 training.")
        # ---- NOTE-conditional-for-FP16 ----     model, optimizer = amp.initialize(model, optimizer, device_gpu_fp16_opt_level)
        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
            'optimizer_number_training_optimization_steps': optimizer_number_training_optimization_steps,
            # ---- NOTE-TODO ---- 'device_use_gpu_fp16': device_use_gpu_fp16,
            'optimizer_learning_rate': optimizer_learning_rate,
            'optimizer_warmup_proportion': optimizer_warmup_proportion,
            'optimizer_warmup_steps': optimizer_warmup_steps,
            'optimizer_adam_epsilon': optimizer_adam_epsilon,
            'optimizer_weight_decay': optimizer_weight_decay,
            'device_gpu_fp16_opt_level': device_gpu_fp16_opt_level
        }
