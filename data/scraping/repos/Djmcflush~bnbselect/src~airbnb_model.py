from langchain.output_parsers import PydanticOutputParser
from langchain.pydantic_v1 import BaseModel, Field  # validator
from typing import List

class AirbnbListing(BaseModel):
    pool: bool = Field(description="Is there a pool in the image"),
    wifi: bool = Field(description="Whether the listing has WiFi"),
    kitchen: bool = Field(description="Is their a kitchen in the image"),
    parking: bool = Field(description="Is there a parking lot in the image"),
    tv: bool = Field(description="Is there a tv in the image"),
    bathrooms: int = Field(description="Number of bathrooms in the image"),
    bedrooms: int = Field(description="Number of bedrooms in the image"),
    beds: int = Field(description="Number of beds in the image"),
    couches: int = Field(description="Number of couches in the image"),
    price: float = Field(description="Price per night of the listing"),
    is_apartment: bool = Field(description="Is this a image of an apartment"),
    is_house: bool = Field(description="Is this a image of a house"),
    is_penthouse: bool = Field(description="Is this a image of a penthouse"),
    trees: bool = Field(description="Are there trees in the image"),
    is_urban: bool = Field(description="Is this a image of an urban area"),
    is_rural: bool = Field(description="Is this a image of a rural area"),
    is_suburban: bool = Field(description="Is this a image of a suburban area"),
    has_stairs: bool = Field(description="Are there stairs in the image"),
    has_elevator: bool = Field(description="is there an elevator in the image"),
    has_garden: bool = Field(description="is there a garden in the image"),
    has_balcony: bool = Field(description="is there a balcony in the image"),
    has_gym: bool = Field(description="is there a gym in the image"),
    has_pool: bool = Field(description="is there a pool in the image"),
    has_air_conditioning: bool = Field(description="Is there an ac unit in the image"),
    has_heating: bool = Field(description="is there a heater in the image"),
    has_washing_machine: bool = Field(
        description="is there a washing machine in the image"
    ),
    has_dryer: bool = Field(description="is there a dryer in the image"),
    has_dishwasher: bool = Field(description="is there a dishwasher in the image"),
    has_fireplace: bool = Field(description="is there a fireplace in the image"),
    has_bathtub: bool = Field(description="is there a bathtub in the image"),
    has_hot_tub: bool = Field(description="is there a hot tub in the image"),
    has_patio: bool = Field(description="is there a patio in the image"),
    has_bbq: bool = Field(description="is there a grill in the image"),
    has_pizza_oven: bool = Field(description="is there a pizza oven in the image"),
    has_garage: bool = Field(description="is there a garage in the image"),
    has_basement: bool = Field(description="is there a basement in the image"),
    has_attic: bool = Field(description="is there an attic in the image"),
    has_security_system: bool = Field(
        description="is there a security system in the image"
    ),
    has_security_cameras: bool = Field(description="Are there cameras in the image"),
    # You can add custom validation logic easily with Pydantic.


def airbnb_output_parser():
    return PydanticOutputParser(pydantic_object=AirbnbListing)




class FinalListingData(BaseModel):
    description: str = Field(description="The description of the listing"),
    caption: str = Field(description="The airbnb data of the listing"),
    url: str = Field(description="The url of the listing"),
    price: str = Field(description="The price of the listing"),
    images: str = Field(description="The image urls of the listing"),
    reviews: List[str] = Field(description="The reviews of the listing"),
    disambiguated_text: str = Field(description="The disambiguated text of the listing"),
def final_listing_parser():
    return PydanticOutputParser(pydantic_object=FinalListingData)
