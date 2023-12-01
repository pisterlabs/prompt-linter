from dataclasses import dataclass
from langchain.schema import Document

@dataclass
class Item:
    """Class representing a piece of clothing."""
    cod_modelo_color: str
    cod_color_code: str
    des_color_specification_esp: str
    des_agrup_color_eng: str
    des_sex: str
    des_age: str
    des_line: str
    des_fabric: str
    des_product_category: str
    des_product_aggregated_family: str
    des_product_family: str
    des_product_type: str
    des_filename: str
    
    img_description: str
    description: str
    
    def describe (self) -> str:
        return f"Sex: {self.des_sex}\nAge: {self.des_age}\nFabric: {self.des_fabric}\nProduct Type: {self.des_product_type}\nDescription: {self.description}"
    
    @classmethod
    def from_document (cls, document: Document) -> "Item":
        description = document.page_content
        meta = document.metadata
        
        return cls(description=description, **meta)