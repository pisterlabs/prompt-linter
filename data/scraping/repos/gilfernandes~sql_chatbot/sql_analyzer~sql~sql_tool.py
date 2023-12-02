from langchain.tools.sql_database.tool import BaseSQLDatabaseTool
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.tools.base import BaseTool
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

from typing import Optional, List, Any
from json import dumps


class ListViewSQLDatabaseTool(BaseSQLDatabaseTool, BaseTool):
    """Tool for getting view names."""

    name = "sql_db_list_views"
    description = "Input is an empty string, output is a comma separated list of views in the database."

    def _run(
        self,
        tool_input: str = "",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Get the schema for a specific view."""
        return ", ".join(self.db._inspector.get_view_names())

    async def _arun(
        self,
        tool_input: str = "",
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        raise NotImplementedError("ListTablesSqlDbTool does not support async")


class ListIndicesSQLDatabaseTool(BaseSQLDatabaseTool, BaseTool):
    """Tool for getting view names."""

    name = "sql_db_list_indices"
    description = """Input is an a list of tables, output is a JSON string with the names of the indices, column names and wether the index is unique.

    Example Input: "table1, table2, table3, table4"
    """

    def _run(
        self,
        table_names: str = "",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Get the indices for all tables."""
        tables: List[str] = table_names.split(", ")
        indices_list: List[List[Any]] = []
        try:
            for table in tables:
                indices: List[Any] = self.db._inspector.get_indexes(table)
                indices_list.extend(indices)
            return dumps(indices_list)
        except Exception as e:
            return f"Error: {e}"

    async def _arun(
        self,
        table_names: str = "",
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        raise NotImplementedError("ListTablesSqlDbTool does not support async")


class InfoViewSQLDatabaseTool(BaseSQLDatabaseTool, BaseTool):
    """Tool for getting metadata about a SQL database."""

    name = "sql_view_schema"
    description = """
    Input to this tool is a comma-separated list of views, output is the schema and sample rows for those views.    

    Example Input: "view1, view2, view3"
    """

    def _run(
        self,
        view_names: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Get the schema for views in a comma-separated list."""
        try:
            views = view_names.split(", ")
            view_info = ""
            meta_tables = [
                tbl
                for tbl in self.db._metadata.sorted_tables
                if tbl.name in set(views)
                and not (self.db.dialect == "sqlite" and tbl.name.startswith("sqlite_"))
            ]
            for i, view in enumerate(views):
                view_def = self.db._inspector.get_view_definition(view)
                view_info += view_def
                view_info += "\n\n/*"
                view_info += f"\n{self.db._get_sample_rows(meta_tables[i])}\n"
                view_info += "*/"
            # return view_info
            return self.db.get_table_info_no_throw()
        except Exception as e:
            """Format the error message"""
            return f"Error: {e}"

    async def _arun(
        self,
        table_name: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        raise NotImplementedError("SchemaSqlDbTool does not support async")


class ExtendedSQLDatabaseToolkit(SQLDatabaseToolkit):
    def get_tools(self) -> List[BaseTool]:
        base_tools = super(ExtendedSQLDatabaseToolkit, self).get_tools()
        db = base_tools[0].db
        base_tools.append(ListViewSQLDatabaseTool(db=db))
        base_tools.append(InfoViewSQLDatabaseTool(db=db))
        base_tools.append(ListIndicesSQLDatabaseTool(db=db))
        return base_tools
