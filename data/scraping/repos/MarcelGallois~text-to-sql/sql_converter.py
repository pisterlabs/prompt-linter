import openai_client

class SQLConverter:
    def __init__(self):
        self.openai_client = openai_client.OpenAIClient()

    def convert_to_sql(self, text):
        response = self.openai_client.generate_text(text)

        sql = self._process_response(response)

        return sql
    
    def check_sql(self, sql, text):
        
        response = self.openai_client.validate_sql(sql, text)

        valid_sql = self._process_response(response)

        return valid_sql

    def _process_response(self, response):
        sql_statements = []
        for statement in response.split(';'):
            statement = statement.strip()
            if statement:
                sql_statements.append(statement)
        return '; '.join(sql_statements)


def convert_to_sql(text):
    converter = SQLConverter()
    return converter.convert_to_sql(text)
