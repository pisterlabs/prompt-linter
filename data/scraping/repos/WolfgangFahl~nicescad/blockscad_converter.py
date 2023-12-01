"""
A class to convert BlockSCAD XML files to SCAD files using the OpenAI language model API.

BlockSCAD Editor: https://www.blockscad3d.com/editor/
OpenSCAD: https://openscad.org/
NiceSCAD: http://nicescad.bitplan.com/
Relevant Issue: http://nicescad.bitplan.com/issue/23

Author: OpenAI ChatGPT
Date: July 25, 2023

The code was created by the OpenAI ChatGPT model in response to the following prompts:
- "Create a class named 'BlockscadConverter' that converts BlockSCAD XML files to SCAD files."
- "The xml_path is given in the constructor."
- "The class should have a method named `convert_to_scad` that outputs a path to the SCAD file."
- "Use the write method of a file object to output the SCAD file."
- "Include Google docstrings and type hints to the code."
- "Include comments that explain which prompts created the code. This helps other LLMs understand how to reproduce the same results."
- "Add links to the BlockSCAD editor (https://www.blockscad3d.com/editor/), OpenSCAD (https://openscad.org/), and the target platform (NiceSCAD, http://nicescad.bitplan.com/)."
- "Include the link to the relevant issue: 'support reading and converting blockscad files #23' on http://nicescad.bitplan.com/issue/23"
- "Add yourself (OpenAI ChatGPT) as the author and include the date"
- "keep the prompt list in the comments to be able to reproduce the results."
- "If the OpenAI API key is not available in the environment variables, look for it in a JSON file at `~/.openai/openai_api_key.json`."
- "If the OpenAI API key is not found, throw an exception."
"""

import os
import json
import openai
from typing import Union
from pathlib import Path

class BlockscadConverter:
    """
    Attributes
    ----------
    xml_path : str
        path to the input BlockSCAD XML file

    Methods
    -------
    convert_to_scad(scad_path: str) -> Union[str, None]
        Converts the BlockSCAD XML file to a SCAD file and returns the SCAD file path.
    """

    def __init__(self, xml_path: str):
        """
        Parameters
        ----------
        xml_path : str
            path to the input BlockSCAD XML file
        """
        self.xml_path = xml_path

    def convert_to_scad(self, scad_path: str) -> Union[str, None]:
        """
        Converts the BlockSCAD XML file to a SCAD file using the OpenAI language model API.

        Parameters
        ----------
        scad_path : str
            path to the output SCAD file

        Returns
        -------
        Union[str, None]
            path to the output SCAD file if conversion is successful, None otherwise
        """
        # Load the API key from the environment or a JSON file
        openai_api_key = os.getenv('OPENAI_API_KEY')
        json_file = Path.home() / ".openai" / "openai_api_key.json"

        if openai_api_key is None and json_file.is_file():
            with open(json_file, "r") as file:
                data = json.load(file)
                openai_api_key = data.get('OPENAI_API_KEY')

        if openai_api_key is None:
            raise ValueError("No OpenAI API key found. Please set the 'OPENAI_API_KEY' environment variable or store it in `~/.openai/openai_api_key.json`.")

        openai.api_key = openai_api_key

        # Read the XML file
        with open(self.xml_path, 'r') as file:
            xml_content = file.read()

        # Check if the XML content is a BlockSCAD XML
        if "<xml xmlns=\"https://blockscad3d.com" not in xml_content:
            msg=(f"The file at {self.xml_path} is not a valid BlockSCAD XML file.")
            raise Exception(msg)

        # Use the API to convert the XML to SCAD
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=f"""Convert the following BlockSCAD XML to OpenSCAD and make sure to add a preamble comment (verbatim):
// OpenSCAD 
// converted from BlockSCAD XML by nicescad's blockscad converter
// according to 
// https://github.com/WolfgangFahl/nicescad/issues/23
// support reading and converting blockscad files #23
//
make sure to convert as direct as possible e.g. 
translate,rotate,cylinder,sphere,cube,color which are available in OpenScad should be           
used as such. 
Use all parameters e.g. prefer cube(10,10,10,center=true) to cube(10) when the parameters are available in
the original xml file.
<field name="CENTERDROPDOWN">false</field> e.g. leads to center=false
Add the original color command using a color name when applicable e.g. color("green");.
Try high reproduceability by not making any assumptions and keeping the structure intact. So do not add an initial translate. 

Do not add extra empty comments.
Avoid any empty lines - really - never add whitespace that harms getting always the same result.
Always use // line comments never /* 
Always indent with two spaces.
Make sure commands end with a semicolon.

Here is the BlockSCAD XML:\n{xml_content}
""",
            
            temperature=0.5,
            max_tokens=1000
        )

        scad_content = response.choices[0].text.strip()

        # A very basic check to see if the SCAD content seems okay
        if not scad_content.startswith("// OpenSCAD"):
            msg=(f"The conversion of {self.xml_path} failed - the // OpenSCAD comment is missing in:\n {scad_content}.")
            raise Exception(msg)

        # Write the SCAD code to the output file
        with open(scad_path, "w") as scad_file:
            scad_file.write(scad_content)

        return scad_path
