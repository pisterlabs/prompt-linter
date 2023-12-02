from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


class LoanIntel(BaseModel):
    loan_name: str = Field(description="Name of the loan")
    loan_interest: float = Field(description="Interest rate of the loan")
    message: str = Field(description="answer to the user")
    bank_name: str = Field(description="Name of the bank")

    def to_dict(self):
        return {
            "loan_name": self.loan_name,
            "loan_interest": self.loan_interest,
            "message": self.message,
            "bank_name": self.bank_name,
        }


loan_intel_parser: PydanticOutputParser = PydanticOutputParser(
    pydantic_object=LoanIntel
)
