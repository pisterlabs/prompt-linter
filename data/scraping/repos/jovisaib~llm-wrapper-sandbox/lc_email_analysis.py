from langchain.llms import OpenAI
from typing import List, Optional
from langchain.prompts import (
    PromptTemplate,
)



from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, ValidationError



# query = """
# Hola equipo de Cermondo,

# Queria reclamar sobre el iPad navideño que me habeis enviado, como puedo procecer a su devolución?

# Gracias,
# Jane Smith
# jane.smith@email.com
# """

query = """
Liebes Gizmo und Widget Support-Team,

Ich bin enttäuscht über die Gizmo 3000 und Widget Pro Geräte, die ich kürzlich gekauft habe, da ich einige Probleme hatte. Aufgrund der folgenden Probleme:

Der Akku des Gizmo 3000 hält nur 4 Stunden und die Touchscreen-Reaktion ist träge. Widget Pro hatte ständig Verbindungsprobleme und schaltete sich manchmal zufällig aus.

Ich habe mich an beide Support-Teams gewandt, aber es dauerte lange, und die Mitarbeiter boten keine Hilfe an. Ich fordere eine schnelle Lösung, wie z.B. Austausch, Rückerstattung oder effektive Fehlerbehebung.

Mit freundlichen Grüßen,

Jane Smith
jane.smith@email.com
"""



class Product(BaseModel):
    product_name: List[str]
    sender: Optional[str]
    sender_business_name: Optional[str]
    sender_sentiment_in_english: Optional[str]
    urgency_level_from_0_to_10: Optional[int]



model = OpenAI(temperature=0)

parser = PydanticOutputParser(pydantic_object=Product)
prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\nquery: {query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

try:
    _input = prompt.format_prompt(query=query)
    output = model(_input.to_string())
    print(parser.parse(output))
except ValidationError as exc:
    print(repr(exc.errors()[0]['type']))


response_language = "Spanish"

print(model(f"What language is the following text written in?\n{query}\n"))
print(model(f"As the company in charge called Cermondo, it attends to the claim of the following customer giving guidelines on how to proceed for its return. The answer must be written in Spanish.:\n{query}\n"))
