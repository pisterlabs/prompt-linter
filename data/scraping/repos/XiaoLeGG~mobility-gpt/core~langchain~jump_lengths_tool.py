
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional, Type
from ..tools.measure import jump_lengths as jl

class JumpLengthsSchema(BaseModel):
    class Config:
        arbitrary_types_allowed = True
    input_file: str = Field(description="The data file path to be processed.")
    output_file: str = Field(description="The file path where the processed data stored.")


class JumpLengthsTool(BaseTool):
    name = "jump_lengths"
    description = "Compute the jump lengths (in kilometers) of a set of individuals. A jump length is defined as the geographic distance between two consecutive points. Warning: The input must be sorted in ascending order by datetime."
    args_schema: Type[JumpLengthsSchema] = JumpLengthsSchema
    def _run(
            self,
            input_file: str,
            output_file: str
    ):
        """Use the tool."""
        return jl.jump_lengths(input_file, output_file)
