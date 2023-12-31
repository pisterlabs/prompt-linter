"""This module provides functionality to parse and read in the knowledge bases."""

from typing import List, Tuple
import abc
import re
import collections
from bs4 import BeautifulSoup
from langchain import LLMChain
import pandas as pd


from dr_claude import datamodels

KG_CSV_COLUMNS = ("Disease Code", "Disease", "Symptom Code", "Symptom")


class KnowledgeBaseReader(abc.ABC):
    """Abstract class for reading in disease-symptom knowledge bases."""

    @abc.abstractmethod
    def load_knowledge_base(self) -> datamodels.DiseaseSymptomKnowledgeBase:
        ...


class CSVKnowledgeBaseReader(KnowledgeBaseReader):
    def __init__(self, csv_path: str) -> None:
        self._df = pd.read_csv(csv_path)

    def load_knowledge_base(self) -> datamodels.DiseaseSymptomKnowledgeBase:
        return make_knowledge_base_from_df(self._df)


class LLMWeightUpdateReader(KnowledgeBaseReader):
    """Update the weights of an existing KnowledgeBase using an LLMChain"""

    def __init__(
        self, kb_reader: KnowledgeBaseReader, weight_updater_chain: LLMChain
    ) -> None:
        self._kb_reader = kb_reader
        self._weight_updater_chain = weight_updater_chain

    def load_knowledge_base(self) -> datamodels.DiseaseSymptomKnowledgeBase:
        return super().load_knowledge_base()


class NYPHKnowldegeBaseReader(KnowledgeBaseReader):
    """
    knowledge database of disease-symptom associations generated by an
    automated method based on information in textual discharge summaries
    of patients at New York Presbyterian Hospital admitted during 2004.

    From: https://people.dbmi.columbia.edu/~friedma/Projects/DiseaseSymptomKB/index.html
    """

    def __init__(self, path_to_html: str) -> None:
        with open(path_to_html, "r") as f:
            self._html_content = f.read()

    def load_symptom_df(self) -> pd.DataFrame:
        soup = BeautifulSoup(self._html_content, "html.parser")
        table = soup.find("table", {"class": "MsoTableWeb3"})
        rows = table.find_all("tr")
        data: List = []

        for row in rows[1:]:
            cells = row.find_all("td")
            row_data = [cell.get_text(strip=True) for cell in cells]
            data.append(row_data)

        df = pd.DataFrame(data)
        df.columns = ("Disease", "Disease Occurrence", "Symptom")
        transformed_rows = []
        prev_diseases = None
        prev_symptoms = None
        for _, row in df.iterrows():
            diseases = parse_umls_string(row.Disease)
            if not diseases:
                diseases = prev_diseases
            prev_diseases = diseases
            symptoms = parse_umls_string(row.Symptom)
            if not symptoms:
                symptoms = prev_symptoms
            prev_symptoms = symptoms
            for _, d_code, d_name in get_disease_representation(diseases):
                for _, s_code, s_name in symptoms:
                    transformed_rows.append(
                        (
                            d_code,
                            " ".join(d_name.replace("\n", " ").split()),
                            s_code,
                            " ".join(s_name.replace("\n", " ").split()),
                        )
                    )
        return pd.DataFrame(
            transformed_rows,
            columns=KG_CSV_COLUMNS,
        )

    def load_knowledge_base(self) -> datamodels.DiseaseSymptomKnowledgeBase:
        df = self.load_symptom_df()
        return make_knowledge_base_from_df(df)


def make_df_from_knowledge_base(
    kb: datamodels.DiseaseSymptomKnowledgeBase,
) -> pd.DataFrame:
    rows: List[Tuple[str, str, str, str]] = []
    for condition, symptoms in kb.condition_symptoms.items():
        for s in symptoms:
            rows.append((condition.umls_code, condition.name, s.umls_code, s.name))
    return pd.DataFrame(rows, columns=KG_CSV_COLUMNS)


def make_knowledge_base_from_df(
    df: pd.DataFrame, default_weight: float = 0.5, default_noise=0.03
) -> datamodels.DiseaseSymptomKnowledgeBase:
    condition_to_symptoms = collections.defaultdict(list)
    curr_disease = None
    for _, row in df.iterrows():
        weight = row.get("Weight", default_weight)
        noise = row.get("Noise", default_noise)
        if row.Disease != curr_disease:
            curr_disease = row.Disease
        condition = datamodels.Condition(
            name=curr_disease, umls_code=row["Disease Code"]
        )
        condition_to_symptoms[condition].append(
            datamodels.WeightedSymptom(
                name=row.Symptom,
                umls_code=row["Symptom Code"],
                weight=weight,
                noise_rate=noise,
            )
        )
    return datamodels.DiseaseSymptomKnowledgeBase(
        condition_symptoms=condition_to_symptoms
    )


def parse_umls_string(umls_string):
    pattern = r"(UMLS):(C\d{7})_([\w\s]+)"
    matches = re.findall(pattern, umls_string)
    return matches


def get_disease_representation(diseases):
    code = diseases[0][1]
    name = " or ".join([" ".join(d[2].replace("\n", " ").split()) for d in diseases])
    return [("UMLS", code, name)]
