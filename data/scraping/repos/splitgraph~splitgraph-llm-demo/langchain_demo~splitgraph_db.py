# based on: https://python.langchain.com/en/latest/modules/chains/examples/sqlite.html
from typing import Any, List
from langchain import SQLDatabase
from unittest import mock

import sqlalchemy
from sqlalchemy.engine import Connection, Engine

from langchain_demo.repo_info import RepositoryInfo
from langchain_demo.repo_to_md import table_info_to_markdown


def get_repository_tables(repo_info: RepositoryInfo) -> List[str]:
    return [table["name"] for table in repo_info["tables"]]


def get_repository_list_tables(repo_list: List[RepositoryInfo]) -> List[str]:
    return [
        table_name for repo in repo_list for table_name in get_repository_tables(repo)
    ]


def get_repository_list_table_info(repo_list: List[RepositoryInfo]) -> str:
    return "\n".join(
        [
            table_info_to_markdown(repo["namespace"], repo["repository"], table)
            for repo in repo_list
            for table in repo["tables"]
        ]
    )


class SplitgraphInspector:
    pg_conn_str: str
    repo_list: List[RepositoryInfo]

    def __init__(self, pg_conn_str: str, repo_list: List[RepositoryInfo]):
        self.pg_conn_str = pg_conn_str
        self.repo_list = repo_list

    def get_table_names(self, schema: str) -> List[str]:
        # TODO: maybe return the "{repository}.{table}"
        return get_repository_list_tables(self.repo_list)

    def get_view_names(self, schema: str) -> List[str]:
        return []

    def get_indexes(
        self, table_name: str
    ) -> List[sqlalchemy.engine.interfaces.ReflectedIndex]:
        # TODO: no indexes returned at the moment.
        # returns a list of ReflecedIndex TypeDicts with the following fields set:
        # name: Optional[str]
        # unique: bool
        # column_names: List[Optional[str]]
        return []


class SplitgraphSQLDatabase(SQLDatabase):
    repo_list: List[RepositoryInfo]

    def __init__(self, *args, repo_list: List[RepositoryInfo] = [], **kwargs):
        super(SplitgraphSQLDatabase, self).__init__(*args, **kwargs)
        self.repo_list = repo_list

    def get_table_info(self, table_names: List[str] | None = None) -> str:
        return get_repository_list_table_info(self.repo_list)

    def run(self, command: str, fetch: str = "all") -> str:
        # TODO: replace unqualified table names with fully qualified table names
        # before executing query.
        return super().run(command, fetch)


class DummyMetadata(sqlalchemy.MetaData):
    """Dummy class which prevents SQLAlchemy from attempting to query
    introspection tables which don't exist on the DDN."""

    def reflect(
        self,
        bind: Engine | Connection,
        schema: str | None = None,
        views: bool = False,
        only: sqlalchemy.Sequence[str] | None = None,  # type: ignore
        extend_existing: bool = False,
        autoload_replace: bool = True,
        resolve_fks: bool = True,
        **dialect_kwargs: Any,
    ) -> None:
        pass


def get_splitgraph_db(
    ddn_pg_conn_str: str, repo_list: List[RepositoryInfo]
) -> SQLDatabase:
    """ddn_pg_conn_str is the postgresql connection string, repo_list is the list
    of repositories the chain should be aware of."""
    # The constructor of the SQLDatabase base class calls SQLAlchemy's inspect()
    # which fails on the DDN since some of the required introspection tables are
    # not available.
    # In itself, class inheritence doesn't solve the issue, since the base class
    # __init__() method is still called.
    # Inheritence is useful for typing, we need an instance of SQLDatabase to
    # use SQLDatabaseChain.
    #
    # The current solution is to mock out the inspect() function and replace it
    # with a SplitgraphInspector instance.
    with mock.patch("langchain.sql_database.inspect") as mock_inspect:
        mock_inspect.return_value = SplitgraphInspector(ddn_pg_conn_str, repo_list)
        return SplitgraphSQLDatabase.from_uri(
            ddn_pg_conn_str,
            metadata=DummyMetadata(),
            include_tables=get_repository_list_tables(repo_list),
            repo_list=repo_list,
        )
