from langchain.utilities import SQLDatabase

from util.logger import log

databases_connection_pool = dict()


class DatastepSqlDatabase:
    @log("Подключение к базе данных")
    def __init__(
        self,
        database_connection_string: str,
        include_tables: list[str],
        tenant_id: int
    ):
        self.database = SQLDatabase.from_uri(
            database_connection_string,
            include_tables=include_tables,
            view_support=True
        )
        # TODO: Использовать пулинг из Алхимии
        # if not databases_connection_pool.get(tenant_id, None):
        #     databases_connection_pool[tenant_id] = SQLDatabase.from_uri(
        #         database_connection_string,
        #         include_tables=include_tables,
        #         view_support=True
        #     )
        #
        # self.database = databases_connection_pool[tenant_id]

    @log("Исполнение SQL в базе данных")
    def run(self, sql_query):
        response = self.database._execute(sql_query)
        return response
