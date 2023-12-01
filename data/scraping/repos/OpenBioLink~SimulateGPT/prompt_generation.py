from pathlib import Path
from langchain.prompts import StringPromptTemplate

from pydantic import BaseModel, validator
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import pandas as pd


class CrcPromptTemplate(StringPromptTemplate, BaseModel):
    """A custom prompt template that takes in the function name as input, and formats the prompt template to provide the source code of the function."""

    @validator("input_variables")
    def validate_input_variables(cls, v):
        """Validate that the input variables are correct."""
        return v

    def format(self, **kwargs) -> str:
        # Load the prompt
        prompt = Path(
            "/home/moritz/Projects/Simulator/experiments/crc/prompt_template"
        ).read_text()

        return prompt.format(**kwargs)

    def _prompt_type(self):
        return "crc"


ATT_DIR = []
# Load patient data
df = pd.read_csv(
    "/home/moritz/Projects/Simulator/experiments/crc/crc_apc_impact_2020/data_clinical_patient.txt",
    sep="\t",
    index_col=0,
    comment="#",
)
# join first sample per patient
sample_df = pd.read_csv(
    "/home/moritz/Projects/Simulator/experiments/crc/crc_apc_impact_2020/data_clinical_sample.txt",
    sep="\t",
    index_col=0,
    comment="#",
)
df = df.join(sample_df.reset_index().groupby("PATIENT_ID").first())

# Group by clinical presentation
grouped = df.groupby(
    [
        "STAGE_AT_DIAGNOSIS",
        "CANCER_TYPE_DETAILED",
        "TUMOR_LOCATION",
        "DIFFERENTIATION",
        "CARCINOMATOSIS",
    ]
)
ages = grouped["AGE_AT_MET"].mean()
pfs_months = grouped["PFS_MONTHS"].median()
pfs_months = grouped["PFS_MONTHS"].median()
pfs_months_std = grouped["PFS_MONTHS"].std()
pfs_months_std.name = "PFS_MONTHS_STD"

group_counts = grouped["PFS_MONTHS"].count()
group_counts.name = "COUNT"

# Aggregate the series into a dataframe
grouped = (
    pd.concat([ages, pfs_months, pfs_months_std, group_counts], axis=1)
    .loc[group_counts >= 4]
    .reset_index()
)

example_indices = [21, 4, 7, 2]

crc_prompt = CrcPromptTemplate(
    input_variables="AGE_AT_DIAGNOSIS CARCINOMATOSIS STAGE_AT_DIAGNOSIS PRIMARY_SITE PATIENT_GRADE CANCER_TYPE_DETAILED".split(
        " "
    )
)


def _format_row(row):
    format_dict = row.to_dict()

    format_dict["AGE_AT_DIAGNOSIS"] = format_dict["AGE_AT_MET"]
    # format_dict["SEX"] = format_dict["SEX"].lower()
    format_dict["PRIMARY_SITE"] = (
        " at the rectum"
        if format_dict["TUMOR_LOCATION"] == "Rectum"
        else f" on the {format_dict['TUMOR_LOCATION'].lower()} side of the colon"
    )
    format_dict["CANCER_TYPE_DETAILED"] = format_dict["CANCER_TYPE_DETAILED"].lower()
    format_dict["CARCINOMATOSIS"] = (
        "no " if format_dict["CARCINOMATOSIS"] == "No" else ""
    )

    format_dict["PATIENT_GRADE"] = format_dict["DIFFERENTIATION"].lower()
    return format_dict


examples_dict = {}
for i, row in grouped.loc[example_indices].reset_index(drop=True).iterrows():
    examples_dict.update(
        {f"EXAMPLE{i+1}_{key}": value for key, value in _format_row(row).items()}
    )

# generate prompts
def get_prompt(row):
    # provide examples
    format_dict = examples_dict.copy()
    format_dict.update(_format_row(row))

    return crc_prompt.format(**format_dict)


Path("prompts").mkdir(exist_ok=True)
for i, row in grouped.drop(example_indices).iterrows():
    Path(f"prompts/crc_apc_impact_2020_{i}").write_text(get_prompt(row))

# store metadata of cases to be simulated
grouped.drop(example_indices).to_csv(
    "/home/moritz/Projects/Simulator/experiments/crc/crc_apc_impact_2020.csv"
)
