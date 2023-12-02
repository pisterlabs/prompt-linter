from contextqa.parsers.models import Summary
from langchain.output_parsers import PydanticOutputParser

summary_parser = PydanticOutputParser(pydantic_object=Summary)
