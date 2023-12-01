import os
import openai
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from colorama import Fore, Style, init

init(autoreset=True)

openai.api_base = "https://api.chatanywhere.com.cn/v1"
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
llms = OpenAI(openai_api_key=API_KEY)

robotDescription = input(
    "Enter a brief description of the arduino robot you want to design:"
)

# LLMChain Stage 1 - User Input & Prelimenary Design
llm = OpenAI(temperature=0.9, model_name="gpt-3.5-turbo")
stage1_template = """You are an incredible arduino robot designer. Given the brief user description of an arduino robot, it is your job to propose a preliminary arduino robot design outline, identify the foundimental robot type and core functional modules, and expain the reason and principles of each designed mechanisms. You are free to think out of the box but the design should be practical, tachnical and logically sounding, you should not propose vague or subjective design.

User : {description}
Arduino Robot Engineer: This is a preliminary outline of the arduino robot design:"""
stage1_prompt_template = PromptTemplate(
    input_variables=["description"], template=stage1_template
)
stage1_chain = LLMChain(
    llm=llm, prompt=stage1_prompt_template, output_key="preliminaryDesign"
)


# LLMChain Stage 2 - Hardware Design
llm = OpenAI(temperature=0.9, model_name="gpt-3.5-turbo")
stage2_template = """You are an arduino robot engineer. Given the preliminary design outline of an arduino robot, it is your job to propose a specified hardware design for the robot, including the robot's physical structure, the robot's hardwares compoents such as arduino processors, sensors and actuators, and the robot's electronic and mechanical connections. You must provide detailed specification for each structure (shape, size and material), components (mechanical design and electronic specification), as well as connections, (including the connection pin numbers for elecronics as well as the connection type and connection method for mechanical connnections). Any design proposed should be logically justified, with technical details explained, not vague nor subjective.

Preliminary design:
{preliminaryDesign}
Arduino Robot Engineer: This is the hardware design of the arduino robot:"""
stage2_prompt_template = PromptTemplate(
    input_variables=["preliminaryDesign"], template=stage2_template
)
stage2_chain = LLMChain(
    llm=llm, prompt=stage2_prompt_template, output_key="hardwareDesign"
)

# LLMChain Stage 3 - Software Design
llm = OpenAI(temperature=0.9, model_name="gpt-3.5-turbo")
stage3_template = """You are an arduino robot engineer. Given the preliminary design and hardware design of an arduino robot, it is your job to propose a corresponding software design for the robot, including the connection pin numbers between the robot's hardware components and the arduino processors, the full arduino code for each of the robot's hardware components (if required), and the full arduino code for the robot's core logic controller. Your arduino code should be complete, with all the required libraries imported, and should be able to compile and run on the arduino processors. Any design proposed should be logically justified, with technical details explained, not vague nor subjective.

Preliminary design:
{preliminaryDesign}
Hardware design:
{hardwareDesign}
Arduino Robot Engineer: This is the software design of the arduino robot:"""
stage3_prompt_template = PromptTemplate(
    input_variables=["preliminaryDesign", "hardwareDesign"], template=stage3_template
)
stage3_chain = LLMChain(
    llm=llm, prompt=stage3_prompt_template, output_key="softwareDesign"
)

# LLMChain Stage 4 - Design Review
llm = OpenAI(temperature=0.7, model_name="gpt-3.5-turbo")
stage4_template = """You are an arduino robot engineer. Given the preliminary design, hardware design and software design of an arduino robot, it is your job to review and criticize the design, provide critical concerns about the design's feasibility, realiability, safety and any other aspects you can think of in real-world scenarios, analyse and resolve any major design flaws, and finally examine and correct the arduino code and pin connections. Any improvements proposed should be logically justified, with technical details explained, including updated codes or new design proposals if necessary and should not be vague or subjective.

Preliminary design:l./
{preliminaryDesign}
Hardware design:
{hardwareDesign}
Software Design:
{softwareDesign}

Arduino Robot Engineer: This is the design review of the arduino robot design:"""
stage4_prompt_template = PromptTemplate(
    input_variables=["preliminaryDesign", "hardwareDesign", "softwareDesign"],
    template=stage4_template,
)
stage4_chain = LLMChain(
    llm=llm, prompt=stage4_prompt_template, output_key="designReview"
)

# LLMChain Stage 5 - User Instruction
llm = OpenAI(temperature=0.9, model_name="gpt-3.5-turbo")
stage5_template = """You are an arduino robot engineer. Given the preliminary design, hardware design and software design and design review of an arduino robot, it is your job to provide a clear, detailed, easy to follow and friendly instruction for any inexperienced user to purchase the require materials, manufacture and process the physical pieces, assemble the mechanical structure and install the electronic components, compile and import the arduino code, and finally initiate and operate the robot, with safety guidelines.

Preliminary design:
{preliminaryDesign}
Hardware design:
{hardwareDesign}
Software Design:
{softwareDesign}
Design Review:
{designReview}

Arduino Robot Engineer: This is the user instruction of the arduino robot design:"""
stage5_prompt_template = PromptTemplate(
    input_variables=[
        "preliminaryDesign",
        "hardwareDesign",
        "softwareDesign",
        "designReview",
    ],
    template=stage5_template,
)
stage5_chain = LLMChain(
    llm=llm, prompt=stage5_prompt_template, output_key="userInstruction"
)

# LLMChain Stage 6 - Failure Analysis
llm = OpenAI(temperature=0.5, model_name="gpt-3.5-turbo")
stage6_template = """You are a state of the art real-life physics simulation. Given the preliminary design, hardware design and software design and design review of an arduino robot, it is your job to stress test the actual robot in the physics simulation, design and record at least ten completely different and unexpected real-life physics scenarios that leads to different failures of the robot. The failures must be unexpected in the design and you should provide detailed physical description of the scenario, the physical cause or logic principle of the failure, and the solution to the failure. The scenarios should be realistic, and the failures should be logically and scientifically justified and only caused by design flaws and not due to components failing on their own or malfunctioning, with technical details explained, not vague or subjective.

Preliminary design:
{preliminaryDesign}
Hardware design:
{hardwareDesign}
Software Design:
{softwareDesign}
Design Review:
{designReview}

Arduino Robot Engineer: This is the failure simulation record of the arduino robot:"""
stage6_prompt_template = PromptTemplate(
    input_variables=[
        "preliminaryDesign",
        "hardwareDesign",
        "softwareDesign",
        "designReview",
    ],
    template=stage6_template,
)
stage6_chain = LLMChain(
    llm=llm, prompt=stage6_prompt_template, output_key="failureAnalysis"
)

robot_design_chain = SequentialChain(
    chains=[
        stage1_chain,
        stage2_chain,
        stage3_chain,
        stage4_chain,
        stage5_chain,
        stage6_chain,
    ],
    input_variables=["description"],
    output_variables=[
        "preliminaryDesign",
        "hardwareDesign",
        "softwareDesign",
        "designReview",
        "userInstruction",
        "failureAnalysis",
    ],
    verbose=True,
)

design_output = robot_design_chain({"description": robotDescription})

colors = [
    Fore.CYAN,
    Fore.YELLOW,
    Fore.MAGENTA,
    Fore.GREEN,
    Fore.BLUE,
    Fore.RED,
    Fore.WHITE,
    Fore.LIGHTRED_EX,
]

for index, (stage, content) in enumerate(design_output.items()):
    color = colors[index % len(colors)]
    stage = stage.capitalize()
    print(f"{Style.BRIGHT}{color}{stage}: {Style.RESET_ALL}")
    print()
    print(f"{color}{content}{Style.RESET_ALL}")
    print("\n")

saveDesign = input("Do you want to save the design? (y/n)")
if saveDesign == "y":
    designName = input("Enter the name of the design:")
    with open(f"designs/{designName}.txt", "w") as f:
        for index, (stage, content) in enumerate(design_output.items()):
            f.write(f"{stage.capitalize()}:\n")
            f.write(content)
            f.write("\n\n")
