from typing import Dict, List, Optional, Union

import sqlalchemy as sa

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class SQLAlchemyLoader(BaseLoader):
    """
    Load documents by querying database tables supported by SQLAlchemy.
    Each document represents one row of the result.
    """

    def __init__(
        self,
        query: Union[str, sa.Select],
        url: str,
        page_content_columns: Optional[List[str]] = None,
        metadata_columns: Optional[List[str]] = None,
        source_columns: Optional[List[str]] = None,
        include_rownum_into_metadata: bool = False,
        include_query_into_metadata: bool = False,
        sqlalchemy_kwargs: Optional[Dict] = None,
    ):
        """

        Args:
            query: The query to execute.
            url: The SQLAlchemy connection string of the database to connect to.
            page_content_columns: The columns to write into the `page_content`
              of the document. Optional.
            metadata_columns: The columns to write into the `metadata` of the document.
              Optional.
            source_columns: The names of the columns to use as the `source` within the
              metadata dictionary. Optional.
            include_rownum_into_metadata: Whether to include the row number into the
              metadata dictionary. Optional. Default: False.
            include_query_into_metadata: Whether to include the query expression into
              the metadata dictionary. Optional. Default: False.
            sqlalchemy_kwargs: More keyword arguments for SQLAlchemy's `create_engine`.
        """
        self.query = query
        self.url = url
        self.page_content_columns = page_content_columns
        self.metadata_columns = metadata_columns
        self.source_columns = source_columns
        self.include_rownum_into_metadata = include_rownum_into_metadata
        self.include_query_into_metadata = include_query_into_metadata
        self.sqlalchemy_kwargs = sqlalchemy_kwargs or {}

    def load(self) -> List[Document]:
        try:
            import sqlalchemy as sa
        except ImportError:
            raise ImportError(
                "Could not import sqlalchemy python package. "
                "Please install it with `pip install sqlalchemy`."
            )

        engine = sa.create_engine(self.url, **self.sqlalchemy_kwargs)

        docs = []
        with engine.connect() as conn:
            if isinstance(self.query, sa.Select):
                result = conn.execute(self.query)
                query_sql = str(self.query.compile(bind=engine))
            elif isinstance(self.query, str):
                result = conn.execute(sa.text(self.query))
                query_sql = self.query
            else:
                raise TypeError(
                    f"Unable to process query of unknown type: {self.query}"
                )
            field_names = list(result.mappings().keys())

            if self.page_content_columns is None:
                page_content_columns = field_names
            else:
                page_content_columns = self.page_content_columns

            if self.metadata_columns is None:
                metadata_columns = []
            else:
                metadata_columns = self.metadata_columns

            for i, row in enumerate(result.mappings()):
                page_content = "\n".join(
                    f"{column}: {value}"
                    for column, value in row.items()
                    if column in page_content_columns
                )

                metadata: Dict[str, Union[str, int]] = {}
                if self.include_rownum_into_metadata:
                    metadata["row"] = i
                if self.include_query_into_metadata:
                    metadata["query"] = query_sql

                source_values = []
                for column, value in row.items():
                    if column in metadata_columns:
                        metadata[column] = value
                    if self.source_columns and column in self.source_columns:
                        source_values.append(value)
                if source_values:
                    metadata["source"] = ",".join(source_values)

                doc = Document(page_content=page_content, metadata=metadata)
                docs.append(doc)

        return docs
