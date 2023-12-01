from langchain.tools import StructuredTool

from ecommerce_agent.sql_query_schema.write_html_report_args_schema import WriteHtmlReportArgsSchema


def write_html_report(file_name, html_report):
    print(f"Writing HTML report to {file_name}")
    with open(file_name, 'w') as file:
        file.write(html_report)


write_html_report_tool = StructuredTool.from_function(name="write_html_report",
                                                      description="Write an HTML file to disk. Use this tool whenever someone asks for a report",
                                                      func=write_html_report,
                                                      args_schema=WriteHtmlReportArgsSchema
                                                      )
