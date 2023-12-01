# -*- coding: utf-8 -*-
import logging
import re
from typing import Any, Dict, Mapping

from kiara.exceptions import KiaraProcessingException
from kiara.models.values.value import ValueMap
from kiara.modules import KiaraModule, ValueSetSchema
from kiara_plugin.tabular.models.array import KiaraArray


class LDAModule(KiaraModule):
    """Perform Latent Dirichlet Allocation on a tokenized corpus.

    This module computes models for a range of number of topics provided by the user.
    """

    _module_type_name = "generate.LDA.for.tokens_array"

    KIARA_METADATA = {
        "tags": ["LDA", "tokens"],
    }

    def create_inputs_schema(
        self,
    ) -> ValueSetSchema:

        inputs: Dict[str, Dict[str, Any]] = {
            "tokens_array": {"type": "array", "doc": "The text corpus."},
            "num_topics_min": {
                "type": "integer",
                "doc": "The minimal number of topics.",
                "default": 7,
            },
            "num_topics_max": {
                "type": "integer",
                "doc": "The max number of topics.",
                "default": 7,
                "optional": True,
            },
            "compute_coherence": {
                "type": "boolean",
                "doc": "Whether to compute the coherence score for each model.",
                "default": False,
            },
            "words_per_topic": {
                "type": "integer",
                "doc": "How many words per topic to put in the result model.",
                "default": 10,
            },
        }
        return inputs

    def create_outputs_schema(
        self,
    ) -> ValueSetSchema:

        outputs: Mapping[str, Mapping[str, Any]] = {
            "topic_models": {
                "type": "dict",
                "doc": "A dictionary with one coherence model table for each number of topics.",
            },
            "coherence_table": {
                "type": "table",
                "doc": "Coherence details.",
                "optional": True,
            },
            "coherence_map": {
                "type": "dict",
                "doc": "A map with the coherence value for every number of topics.",
            },
        }
        return outputs

    def create_model(self, corpus, num_topics: int, id2word: Mapping[str, int]):
        from gensim.models import LdaModel

        model = LdaModel(
            corpus, id2word=id2word, num_topics=num_topics, eval_every=None
        )
        return model

    def compute_coherence(self, model, corpus_model, id2word: Mapping[str, int]):

        from gensim.models import CoherenceModel

        coherencemodel = CoherenceModel(
            model=model,
            texts=corpus_model,
            dictionary=id2word,
            coherence="c_v",
            processes=1,
        )
        coherence_value = coherencemodel.get_coherence()
        return coherence_value

    # def assemble_coherence(self, models_dict: Mapping[int, Any], words_per_topic: int):
    #
    #     import pandas as pd
    #     import pyarrow as pa
    #
    #     # Create list with topics and topic words for each number of topics
    #     num_topics_list = []
    #     topics_list = []
    #     for (
    #         num_topics,
    #         model,
    #     ) in models_dict.items():
    #
    #         num_topics_list.append(num_topics)
    #         topic_print = model.print_topics(num_words=words_per_topic)
    #         topics_list.append(topic_print)
    #
    #     df_coherence_table = pd.DataFrame(columns=["topic_id", "words", "num_topics"])
    #
    #     idx = 0
    #     for i in range(len(topics_list)):
    #         for j in range(len(topics_list[i])):
    #             df_coherence_table.loc[idx] = ""
    #             df_coherence_table["topic_id"].loc[idx] = j + 1
    #             df_coherence_table["words"].loc[idx] = ", ".join(
    #                 re.findall(r'"(\w+)"', topics_list[i][j][1])
    #             )
    #             df_coherence_table["num_topics"].loc[idx] = num_topics_list[i]
    #             idx += 1
    #
    #     coherence_table = pa.Table.from_pandas(df_coherence_table, preserve_index=False)
    #     return coherence_table

    def assemble_coherence(self, models_dict: Mapping[int, Any], words_per_topic: int):

        import pandas as pd
        import pyarrow as pa

        # Extract topics and their respective number of topics
        topics_data = [
            (num_topics, model.print_topics(num_words=words_per_topic))
            for num_topics, model in models_dict.items()
        ]

        # Prepare data for DataFrame
        df_data = [
            {
                "topic_id": topic_id + 1,
                "words": ", ".join(re.findall(r'"(\w+)"', topic_info[1])),
                "num_topics": num_topics,
            }
            for num_topics, topics in topics_data
            for topic_id, topic_info in enumerate(topics)
        ]

        # Create DataFrame from the prepared data
        df_coherence_table = pd.DataFrame(
            df_data, columns=["topic_id", "words", "num_topics"]
        )

        # Convert DataFrame to PyArrow Table
        coherence_table = pa.Table.from_pandas(df_coherence_table, preserve_index=False)
        return coherence_table

    def process(self, inputs: ValueMap, outputs: ValueMap) -> None:

        from gensim import corpora

        logging.getLogger("gensim").setLevel(logging.ERROR)
        tokens_array: KiaraArray = inputs.get_value_data("tokens_array")
        tokens = tokens_array.arrow_array.to_pylist()

        words_per_topic = inputs.get_value_data("words_per_topic")

        num_topics_min = inputs.get_value_data("num_topics_min")
        num_topics_max = inputs.get_value_data("num_topics_max")
        if not num_topics_max:
            num_topics_max = num_topics_min

        if num_topics_max < num_topics_min:
            raise KiaraProcessingException(
                "The max number of topics must be larger or equal to the min number of topics."
            )

        compute_coherence = inputs.get_value_data("compute_coherence")
        id2word = corpora.Dictionary(tokens)
        corpus = [id2word.doc2bow(text) for text in tokens]

        # model = gensim.models.ldamulticore.LdaMulticore(
        #     corpus, id2word=id2word, num_topics=num_topics, eval_every=None
        # )

        models = {}
        model_tables = {}
        coherence = {}

        # multi_threaded = False
        # if not multi_threaded:

        for nt in range(num_topics_min, num_topics_max + 1):
            model = self.create_model(corpus=corpus, num_topics=nt, id2word=id2word)
            models[nt] = model
            topic_print_model = model.print_topics(num_words=words_per_topic)
            # dbg(topic_print_model)
            # df = pd.DataFrame(topic_print_model, columns=["topic_id", "words"])
            # TODO: create table directly
            # result_table = Table.from_pandas(df)
            model_tables[nt] = topic_print_model

            if compute_coherence:
                coherence_result = self.compute_coherence(
                    model=model, corpus_model=tokens, id2word=id2word
                )
                coherence[nt] = coherence_result

        # else:
        #     def create_model(num_topics):
        #         model = self.create_model(corpus=corpus, num_topics=num_topics, id2word=id2word)
        #         topic_print_model = model.print_topics(num_words=30)
        #         df = pd.DataFrame(topic_print_model, columns=["topic_id", "words"])
        #         # TODO: create table directly
        #         result_table = Table.from_pandas(df)
        #         coherence_result = None
        #         if compute_coherence:
        #             coherence_result = self.compute_coherence(model=model, corpus_model=tokens, id2word=id2word)
        #         return (num_topics, model, result_table, coherence_result)
        #
        #     executor = ThreadPoolExecutor()
        #     results: typing.Any = executor.map(create_model, range(num_topics_min, num_topics_max+1))
        #     executor.shutdown(wait=True)
        #     for r in results:
        #         models[r[0]] = r[1]
        #         model_tables[r[0]] = r[2]
        #         if compute_coherence:
        #             coherence[r[0]] = r[3]

        # df_coherence = pd.DataFrame(coherence.keys(), columns=["Number of topics"])
        # df_coherence["Coherence"] = coherence.values()

        if compute_coherence:
            coherence_table = self.assemble_coherence(
                models_dict=models, words_per_topic=words_per_topic
            )
        else:
            coherence_table = None

        coherence_map = {k: v.item() for k, v in coherence.items()}

        outputs.set_values(
            topic_models=model_tables,
            coherence_table=coherence_table,
            coherence_map=coherence_map,
        )
