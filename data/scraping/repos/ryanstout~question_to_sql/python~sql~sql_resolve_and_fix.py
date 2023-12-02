"""
Takes in generated SQL (from OpenAI for example) and a SimpleSchema and does the following:
1. Attempts to parse it using sqlglot, trying various dialects
2. It resolves all tables, columns, and aliases, and raises if one of them doesn't exist
3. Since we predict types
"""


from sqlglot import exp, parse_one

from python.sql.sql_inspector import SqlInspector
from python.sql.sql_parser import SqlParser
from python.sql.types import SimpleSchema
from python.sql.utils.snowflake_keywords import SNOWFLAKE_KEYWORDS


class SqlResolveAndFix:
    def run(self, sql: str, simple_schema: SimpleSchema):
        ast = SqlParser().run(sql)
        SqlInspector(ast, simple_schema, SqlParser.in_dialect)

        if SqlParser.out_dialect == "postgres":
            # Postgres needs the input to divides cast to float for "what percent" questions
            ast = ast.transform(self.cast_divides_to_float)

        ast = ast.transform(self.add_fully_qualified_name)

        sql = ast.sql(pretty=True, max_text_width=40)

        return sql

    def cast_divides_to_float(self, node):
        if not isinstance(node, exp.Div):
            return node

        # code.InteractiveConsole(locals=locals()).interact()
        # Need to cast one side on div's to float for postgres
        # Useful for "what percent" questions
        expr = node.args["this"]

        if isinstance(node, exp.TableAlias):
            # Transform the contents of the AS node
            # code.InteractiveConsole(locals=locals()).interact()
            return node

        node.args["this"] = parse_one(f"{str(expr)}::float")
        return node

    def quote_if_keyword(self, name: str):
        if name.upper() in SNOWFLAKE_KEYWORDS:
            return f'"{name}"'
        else:
            return name

    def add_fully_qualified_name(self, node):
        if isinstance(node, exp.Identifier):
            new_name = self.quote_if_keyword(node.args["this"])
            return parse_one(new_name, read=SqlParser.in_dialect)

        return node
