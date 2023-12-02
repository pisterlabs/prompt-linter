from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.schema import BaseOutputParser
fragment = "CQDs were synthesized by the usage of O. basilicum L. extract via a simple hydrothermal method (Fig. 1). In a typical one-step synthesizing procedure, 2.0 g of O. basilicum L. seed was added to 100 mL of distilled water and stirred at 50 °C for 2 h. Then, the obtained extract was filtered and transferred into a 100 mL Teflon-lined stainless-steel autoclave to be heated at 180 °C for 4 h. Once the autoclave was cooled naturally at room temperature and the solution was centrifuged (12,000 rpm) for 15 min, the prepared brown solution was filtered through a fine-grained 0.45 μm membrane to remove larger particles. Finally, the solution was freeze-dried to attain the dark brown powder of CQDs."
fragment_2 = "Initially, 0.15 g PVP-K30, 0.1 g FA, and 0.05 g Ce(Ac)3 were dissolved in 15 mL ultrapure water with ultrasonication for 10 min. Then the solution was poured into a 20 mL Teflon-lined stainless-steel reactor and treated at 160 °C for 24 h. After that, the solution was cooled to room temperature and centrifuged at 8000 rpm for 10 min to remove impurities and large particles. The supernatant was collected and kept in the refrigerator at 4 °C as the stock solution for further study. The final FA-modified cerium-doped carbon-dots (Ce-CDs-FA) powder was obtained by drying under vacuum at 60 °C or lyophilized in a freeze-dryer. For comparison, the nano Ce-CDs were synthesized by the same procedure without FA."
fragment_3 = 'Glucose (0.5 g) was poured into a polytetrafluoroethylene-lined stainless steel autoclave (25 mL) containing 5 mL ethanol and 5 mL deionized water. Then it was placed into a constant temperature drying oven and heated to 180 °C for 8 h. The brown liquid obtained was filtered with a 0.22 μm microporous membrane to remove impurities and then dialyzed with a dialysis membrane (MW = 500 Da) for 48 h. At the beginning of dialysis, the water in the dialysis bag should be changed frequently and then changed every 4 h until it is clear and transparent. Then the brown B-CD powders were obtained by freeze-drying for 48 h.'
fragment_4 = 'S-doped C-dots were synthesized using a hydrothermal method. Briefly, 25 mL sodium citrate solution (0.1 M) and sodium thiosulfate were added into a 50 mL Teflon-lined stainless steel autoclave. After that, the autoclave was kept at a fixed temperature (160, 180, 200, 220 or 240 °C) for 6 h. The product could be used after filtration with a cylinder filtration membrane filter (0.22 μm).'

class CommaSeparatedListOutputParser(BaseOutputParser):
    """Parse the output of an LLM call to a comma-separated list."""


    def parse(self, text: str):
        """Parse the output of an LLM call."""
        return text.strip().split(", ")


    
template = """You are materials chemist text mining fragments of science articles for data.
A user will pass in a fragment of text and you will return the {parameter} and units of the hydrothermal reaction described to be carried out in a autoclave in the text in a comma separated list.
Each value with should be separated by a comma. ONLY return the {parameter} and units, in a comma seprated list and nothing more."""
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
# system_message_prompt.format(parameter = 'temperature')
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
# chat_prompt.format_messages(parameter = 'temperature')
chain = LLMChain(
    llm = ChatOpenAI(),
    prompt = chat_prompt,
    output_parser = CommaSeparatedListOutputParser()
)

# fragment = "CQDs were synthesized by the usage of O. basilicum L. extract via a simple hydrothermal method (Fig. 1). In a typical one-step synthesizing procedure, 2.0 g of O. basilicum L. seed was added to 100 mL of distilled water and stirred at 50 °C for 2 h. Then, the obtained extract was filtered and transferred into a 100 mL Teflon-lined stainless-steel autoclave to be heated at 180 °C for 4 h. Once the autoclave was cooled naturally at room temperature and the solution was centrifuged (12,000 rpm) for 15 min, the prepared brown solution was filtered through a fine-grained 0.45 μm membrane to remove larger particles. Finally, the solution was freeze-dried to attain the dark brown powder of CQDs."
result = chain.run({'parameter': 'temperature', 'text':fragment})
print(result)


