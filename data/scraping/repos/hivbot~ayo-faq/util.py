import pandas as pd
from langchain.docstore.document import Document
import re

SHEET_URL_X = "https://docs.google.com/spreadsheets/d/"
SHEET_URL_Y = "/edit#gid="
SHEET_URL_Y_EXPORT = "/export?gid="
SPLIT_PAGE_BREAKS = False
SYNONYMS = None


def get_id(sheet_url: str) -> str:
    x = sheet_url.find(SHEET_URL_X)
    y = sheet_url.find(SHEET_URL_Y)
    return sheet_url[x + len(SHEET_URL_X) : y] + "-" + sheet_url[y + len(SHEET_URL_Y) :]


def xlsx_url(get_id: str) -> str:
    y = get_id.rfind("-")
    return SHEET_URL_X + get_id[0:y] + SHEET_URL_Y_EXPORT + get_id[y + 1 :]


def read_df(xlsx_url: str, page_content_column: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx_url, header=0, keep_default_na=False)
    if SPLIT_PAGE_BREAKS:
        df = split_page_breaks(df, page_content_column)
    df = remove_empty_rows(df, page_content_column)
    if SYNONYMS is not None:
        df = duplicate_rows_with_synonyms(df, page_content_column, SYNONYMS)
    return df


def split_page_breaks(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    split_values = df[column_name].str.split("\n")

    new_df = pd.DataFrame({column_name: split_values.explode()})
    new_df.reset_index(drop=True, inplace=True)

    column_order = df.columns

    new_df = new_df.reindex(column_order, axis=1)

    other_columns = column_order.drop(column_name)
    for column in other_columns:
        new_df[column] = (
            df[column].repeat(split_values.str.len()).reset_index(drop=True)
        )

    return new_df


def transform_documents_to_dataframe(documents: Document) -> pd.DataFrame:
    keys = []
    values = {"document_score": [], "page_content": []}

    for doc, score in documents:
        for key, value in doc.metadata.items():
            if key not in keys:
                keys.append(key)
                values[key] = []
            values[key].append(value)
        values["document_score"].append(score)
        values["page_content"].append(doc.page_content)

    return pd.DataFrame(values)


def remove_duplicates_by_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    df.drop_duplicates(subset=column_name, inplace=True, ignore_index=True)

    return df


def dataframe_to_dict(df: pd.DataFrame) -> dict:
    df_records = df.to_dict(orient="records")

    return df_records


def duplicate_rows_with_synonyms(df: pd.DataFrame, column: str, synonyms: list[list[str]]) -> pd.DataFrame:
    new_rows = []
    for index, row in df.iterrows():
        new_rows.append(row)
        text = row[column]
        for synonym_list in synonyms:
            for synonym in synonym_list:
                pattern = r'(?i)\b({}(?:s|es|ed|ing)?)\b'.format(synonym)
                if re.search(pattern, text):
                    for replacement in synonym_list:
                        if replacement != synonym:
                            new_row = row.copy()
                            new_row[column] = re.sub(pattern, replacement, text)
                            new_rows.append(new_row)
    new_df = pd.DataFrame(new_rows, columns=df.columns)
    new_df = new_df.reset_index(drop=True)
    return new_df


def remove_empty_rows(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    df = df[df[column_name].str.strip().astype(bool)]
    df = df.reset_index(drop=True)
    return df
