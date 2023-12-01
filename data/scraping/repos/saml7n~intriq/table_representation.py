from dataclasses import dataclass
from bs4 import BeautifulSoup
from langchain.docstore.document import Document
from loguru import logger


# @dataclass
# class table_entity:
#     name: str
#     columns: list
#     rows: list
#     metadata: dict
#     table_type: str
#     table_id: str

_DOC_NAME = 'doc_name'
_TABLE_NAME = 'table_name'


def table_to_docs(html_content, doc_name, table_name):
    docs = []
    table_rows = _extract_table_data(html_content)
    if not table_rows:
        logger.warning('No table rows found in html content')
        return docs
    columns_to_embed = list(table_rows[0].keys())
    columns_to_metadata = []
    sheet_metadata = {_DOC_NAME: doc_name, _TABLE_NAME: table_name}
    for row in table_rows:
        to_metadata = {
            col: row[col]for col in columns_to_metadata if col in row
        }
        values_to_embed = {k: row[k] for k in columns_to_embed if k in row}
        to_embed = "\n".join(
            f"{k.strip()}: {v.strip()}" for k, v in values_to_embed.items()
        )
        newDoc = Document(
            page_content=to_embed,
            metadata={**sheet_metadata, **to_metadata}
        )
        docs.append(newDoc)
    return docs


def _extract_table_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    table = soup.find('table')

    if not table or not table.find_all('tr'):
        logger.warning('No table or table rows found in html content')
        return []

    headers = [th.get_text(strip=True) for th in table.find_all('th')]
    if not headers:
        logger.warning('No table headers found in html content. Transposing..')
        rows = [
            [td for td in tr.find_all('td')]
            for tr in table.find_all('tr')
        ]
        transposed = _transpose_rows_as_headers(rows)
        headers = transposed[0]
        data_rows = transposed[1:]
    else:
        data_rows = [
            [td.get_text(strip=True) for td in tr.find_all('td')] for tr in table.find_all('tr')[1:]
        ]
    logger.info(f'Found headers {headers}')

    table_data = []
    for row in data_rows:
        if len(row) == len(headers):
            row_data = {headers[i]: row[i] for i in range(len(row))}
            table_data.append(row_data)
        else:
            logger.warning(
                f'''Row has different number of cells than header. headers is of length {len(headers)} and row is of length {len(row)}'''
            )
            pass

    return table_data


def _transpose_rows_as_headers(table_rows):
    if not table_rows or not table_rows[0]:
        return []

    transposed = []
    for i in range(len(table_rows[0])):
        transposed.append([row[i].get_text(strip=True) for row in table_rows])
    return transposed
