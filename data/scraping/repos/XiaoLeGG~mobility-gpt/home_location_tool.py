
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional, Type
from ..tools.measure import home_location as hl

class HomeLocationSchema(BaseModel):
    class Config:
        arbitrary_types_allowed = True
    input_file: str = Field(description="The data file path to be processed.")
    output_file: str = Field(description="The file path where the processed data stored.")
    start_night_time: str = Field(description="The start time of the night.", default="22:00")
    end_night_time: str = Field(description="The end time of the night.", default="06:00")

class HomeLocationTool(BaseTool):
    name = "home_location"
    description = "This function compute the home location of a set of individuals. The home location is defined as the location `v` for every individual `u` visits most during nighttime."
    args_schema: Type[HomeLocationSchema] = HomeLocationSchema
    def _run(
            self,
            input_file: str,
            output_file: str,
            start_night_time: str='22:00',
            end_night_time: str='06:00'
    ):
        """Use the tool."""
        return hl.home_location(input_file, output_file, start_night_time, end_night_time)
