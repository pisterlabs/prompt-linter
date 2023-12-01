import os
import docx
from dotenv import load_dotenv, find_dotenv
from pprint import pprint
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain

load_dotenv(find_dotenv())


def read_docx(filename: str) -> str:
    """Reads a DOCX file and returns its content as a string."""
    doc = docx.Document(filename)
    return '\n'.join(para.text for para in doc.paragraphs)


def get_files_from_dir(dir_path: str) -> list:
    """Returns a list of files in a given directory."""
    return [os.path.join(dir_path, file_path) for file_path in os.listdir(dir_path) if
            os.path.isfile(os.path.join(dir_path, file_path))]


def main():
    DOCX_PATH = 'data/'
    list_of_files = get_files_from_dir(DOCX_PATH)

    chat = ChatOpenAI(temperature=0, model_name="gpt-4", max_tokens=700)

    text = read_docx(list_of_files[0])
    num_graphics = text.count("graphic-number")

    template = """
    In a document you will find {num_graphics} codes in a format 
    graphic-number-xxx where xxx are three integers.
    For example graphic-number-003.
    Your aim is to make a brief summary of the text around the codes, 
    especially in a paragraph just before the text.
    You provide a reply in a format:
        ("graphic-number-001": "description to the graphic")

    Document: {document}
    """

    prompt = PromptTemplate(
        input_variables=["num_graphics", "document"],
        template=template
    )

    chain = LLMChain(llm=chat, prompt=prompt)
    captions = chain.run(document=text, num_graphics=num_graphics)
    pprint(captions, width=150)


if __name__ == "__main__":
    main()

'''
example output: 
('("graphic-number-001": "This graphic likely illustrates the tools needed for the installation process, as it is mentioned after the list of required tools.")\n'
 '("graphic-number-002": "This graphic likely shows how to position the bike for the installation process, as it is mentioned in the context of positioning the bike with wheels upwards.")\n'
 '("graphic-number-003": "This graphic likely demonstrates the process of removing front cranks from the bike, as it is mentioned in the context of crank and front derailleur removal.")\n'
 '("graphic-number-004": "This graphic likely shows the bottom bracket that comes with the GTRO, as it is mentioned in the context of bottom bracket replacement.")\n'
 '("graphic-number-005": "This graphic likely shows the positioning of the wave-spring in the bottom bracket’s niche, as it is mentioned in the context of gearbox installation.")\n'
 '("graphic-number-006": "This graphic likely shows the positioning of the wave-spring in the bottom bracket’s niche, as it is mentioned in the context of gearbox installation.")\n'
 '("graphic-number-007": "This graphic likely shows the positioning of the reaction lever on the chainstay, as it is mentioned in the context of gearbox installation.")\n'
 '("graphic-number-008": "This graphic likely shows the correct positioning of the lever against the straight area of the chainstay or kickstand plate, as it is mentioned in the context of gearbox installation.")\n'
 '("graphic-number-009": "This graphic likely shows the incorrect positioning of the lever against the edge of the kickstand’s plate, as it is mentioned in the context of gearbox installation.")\n'
 '("graphic-number-010": "This graphic likely shows the incorrect positioning of the lever against the edge of the kickstand’s plate, as it is mentioned in the context of gearbox installation.")\n'
 '("graphic-number-011": "This graphic likely shows the correct positioning of the shifting cable against the down tube, as it is mentioned in the context of gearbox installation.")\n'
 '("graphic-number-012": "This graphic likely shows the correct positioning of the reaction lever against the chainstay, as it is mentioned in the context of gearbox installation.")\n'
 '("graphic-number-013": "This graphic likely shows the installation of the cable-slider, as it is mentioned in the context of cable-slider installation.")\n'
 '("graphic-number-014": "This graphic likely shows the installation of the GTRO’s shifter on the handlebar, as it is mentioned in the context of shifter installation.")\n'
 '("graphic-number-015": "This graphic likely shows the installation of the GTRO’s shifter on the handlebar, as it is mentioned in the context of shifter installation.")\n'
 '("graphic-number-016": "This graphic likely shows the installation of the GTRO’s shifter on the handlebar, as it is mentioned in the context of shifter installation.")\n'
 '("graphic-number-017": "This graphic likely shows the installation of the lever to the pedal boom for bicycles with no chainstay, as it is mentioned in the context of no-chainstay bicycles.")\n'
 '("graphic-number-018": "This graphic likely shows the application of grease under the plastic cap of the crank bolt for usage in wet/humid areas, as it is mentioned in the context of GTRO usage in highly wet/humid area.")\n'
 '("graphic-number-019": "This graphic likely shows the installation of bolts for the adapter, as it is mentioned in the context of adapter for beltring or larger chainring.")')
'''
