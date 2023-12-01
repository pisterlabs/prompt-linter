import glob
import os
import time

import torch
from decouple import config

from python.embeddings.openai_embedder import OpenAIEmbedder
from python.ranker.column_ranker_model import ColumnRankerModel
from python.ranker.dataset_generator.scores_to_numpy import (
    column_scores_to_numpy,
    table_scores_to_numpy,
    values_scores_to_numpy,
)
from python.ranker.dataset_generator.utils import ranker_models_path
from python.ranker.table_ranker_model import TableRankerModel
from python.ranker.training_ranker import TrainingRanker
from python.ranker.values_ranker_model import ValuesRankerModel
from python.schema.ranker import SCHEMA_RANKING_TYPE, ElementRank
from python.sql.types import DbElementIds, ElementIdsAndScores
from python.sql.utils.touch_points import convert_db_element_ids_to_db_element
from python.utils.batteries import log_execution_time
from python.utils.logging import log


class LearnedRanker:
    def __init__(self, datasource_id: int, match_limit: int = 10_000, value_hint_search_limit: int = 5000):
        self.datasource_id = datasource_id

        self.training_ranker = TrainingRanker(
            datasource_id, match_limit=match_limit, value_hint_search_limit=value_hint_search_limit
        )
        self.table_ranker_model = TableRankerModel.load_from_checkpoint(self.get_latest_checkpoint("table"))
        self.table_ranker_model.eval()

        self.column_ranker_model = ColumnRankerModel.load_from_checkpoint(self.get_latest_checkpoint("column"))
        self.column_ranker_model.eval()

        self.values_ranker_model = ValuesRankerModel.load_from_checkpoint(self.get_latest_checkpoint("values"))
        self.values_ranker_model.eval()

    def get_latest_checkpoint(self, model_name: str) -> str:
        logs_dir = f"{ranker_models_path()}/{model_name}_ranker/lightning_logs"
        checkpoints = glob.glob(os.path.join(logs_dir, "version_*", "checkpoints", "epoch=*.ckpt"))
        if not checkpoints:
            raise ValueError(f"No checkpoints found in {logs_dir}")
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        return latest_checkpoint

    def rank(
        self,
        query: str,
        embedder=OpenAIEmbedder,
        cache_results=True,
        table_weights=None,
        column_weights=None,
        value_weights=None,
    ) -> SCHEMA_RANKING_TYPE:

        with log_execution_time("Ranking fetch"):
            # Run the query through the training ranker, get back a merged list of table, column, and value matches
            table_scores, column_scores, value_scores = self.training_ranker.rank(
                query, embedder=embedder, cache_results=True
            )

            table_ranks = self.get_element_ranks_via_models(table_scores, "tables")
            column_ranks = self.get_element_ranks_via_models(column_scores, "columns")
            values_ranks = self.get_element_ranks_via_models(value_scores, "values")

            rankings: list[ElementRank] = [*table_ranks, *column_ranks, *values_ranks]

            # Sort rankings by score
            rankings.sort(key=lambda x: x.score, reverse=True)

            # Print with table and column names
            if os.getenv("DEBUG_RANKER"):
                for ranking in rankings:
                    db_element = convert_db_element_ids_to_db_element(
                        DbElementIds(ranking.table_id, ranking.column_id, ranking.value_hint)
                    )
                    log.debug("rank", score=ranking.score, element=db_element)

        return rankings

    def get_element_ranks_via_models(self, element_scores: ElementIdsAndScores, element_type: str) -> list[ElementRank]:
        # Since python 3.10, dicts are ordered
        elements = list(element_scores.keys())
        scores = list(element_scores.values())

        if element_type == "tables":
            x_np = table_scores_to_numpy(scores)
            x = torch.from_numpy(x_np)
            y_np = self.table_ranker_model(x).detach().numpy()
        elif element_type == "columns":
            x_np = column_scores_to_numpy(scores)
            x = torch.from_numpy(x_np)
            y_np = self.column_ranker_model(x).detach().numpy()
        else:
            x_np = values_scores_to_numpy(scores)
            x = torch.from_numpy(x_np)
            y_np = self.values_ranker_model(x).detach().numpy()
            y_np *= 0.9  # Manually scale

        element_ranks: list[ElementRank] = []

        for idx, y_score in enumerate(y_np):
            element_ranks.append(
                ElementRank(
                    table_id=elements[idx][0],
                    column_id=elements[idx][1],
                    value_hint=elements[idx][2],
                    score=y_score,
                )
            )

        return element_ranks
