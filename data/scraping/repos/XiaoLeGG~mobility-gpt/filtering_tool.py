from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional, Type
from ..tools.preprocess import filtering


class FilteringSchema(BaseModel):
    class Config:
        arbitrary_types_allowed = True
    input_file: str = Field(description="The data file path to be processed.")
    output_file: str = Field(description="The file path where the processed data stored.")
    max_speed: float = Field(description="Indicate that the points with a speed(km/h) from previous point that beyond the max_speed will be deleted.", default=200)
    include_loop: bool = Field(description="Indicate whether to delete short and fast loops in the trajectories.", default=True)
    loop_intensity: float = Field(description="Indicate the intensity of deleting loops.", default=1.0)


class FilteringTool(BaseTool):
    name = "filtering"
    description = "Filter the useless or unreasonable points such as object suddenly moves too fast or object moves in a short and fast circles."
    args_schema: Type[FilteringSchema] = FilteringSchema
    def _run(
            self,
            input_file: str,
            output_file: str,
            max_speed: float=200,
            include_loop: bool=True,
            loop_intensity: float=1.0
    ) -> int:
        """Use the tool."""
        return filtering.noise_filtering(input_file, output_file, max_speed, include_loop, loop_intensity)