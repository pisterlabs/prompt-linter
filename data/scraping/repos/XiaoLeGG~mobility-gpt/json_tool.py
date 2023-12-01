from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional, Type
import json

class JsonSchema(BaseModel):
    class Config:
        arbitrary_types_allowed = True
    content: str = Field(description="The JSON format data.")
    output_file: str = Field(description="The file path to store json.")
    
class JsonTool(BaseTool):
    name = "json"
    description = str(
        "Store the JSON format data in the output file."
        "Notice that the content must be JSON formated data!!!"
    )
    args_schema: Type[JsonSchema] = JsonSchema
    def _run(
            self,
            content: str,
            output_file: str
    ) -> str:
        """Store json."""
        try:
            json_data = json.loads(content)
            json.dump(json_data, open(output_file, "w"))
            return "Successfully wrote json."
        except Exception as e:
            return "Error occurs:\n" + str(e)

            