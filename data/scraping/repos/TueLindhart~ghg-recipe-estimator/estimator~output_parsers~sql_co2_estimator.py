from typing import List, Optional

from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


class CO2perKg(BaseModel):
    ingredient: str = Field(description="Name of ingredient")
    comment: str = Field(description="Comment about result. For instance what closest result is.")
    unit: str = Field(description="The unit which is kg CO2e per kg")
    co2_per_kg: Optional[float] = Field(description="kg CO2 per kg for ingredient", default=None)


class CO2Emissions(BaseModel):
    emissions: List[CO2perKg]


sql_co2_output_parser = PydanticOutputParser(pydantic_object=CO2Emissions)
