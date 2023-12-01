import os
from dotenv import load_dotenv
import json

# Retrieval
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper

# Prompting
from langchain.prompts import PromptTemplate

# LLM response
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

# For API setup
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

template = """You are a helpful research assistant and teacher. 
I would like you to help me understand how topics relate to each other and the strcuture of knowledge.
I will provide you with the name of a topic and some information on the topic from wikipedia. 
Based on the wikipedia description of the topic, your goal is to write a list of related topics / subtopics / necessary techniques or methods to be learned in order to fully grasp and understand a complex topic.
The list should be formatted with the following steps as below:

1/ Wrap each topic like this: [[TOPIC NAME]]
2/ Return a nested dictionary which contains the heirarchy of the topics.

Here is an example of a correctly formatted list:
{{"MAIN TOPIC": {{"MAIN TOPIC - SUB TOPIC 1": ["SUB 1 SUB TOPIC 1", "SUB 1 SUB TOPIC 2"], "MAIN TOPIC - SUB TOPIC 2": ["SUB 2 SUB TOPIC 1", "SUB 2 SUB TOPIC 2", "SUB 2 SUB TOPIC 3"], "MAIN TOPIC - SUB TOPIC 3": ["SUB 3 SUB TOPIC 1"]}}}}

Here is an example response on "General Relativity" following above rules:
{{"General Relativity": {{"General Relativity - Background Knowledge": {{"Special Relativity": ["Lorentz Transformations", "Time Dilation", "Length Contraction", "Minkowski Space-Time"], "Newton's Law of Universal Gravitation": ["Gravity Force", "Gravitational Constant", "Inverse Square Law"], "Differential Geometry": ["Manifolds", "Metric Tensor", "Christoffel Symbols", "Geodesics", "Riemann Curvature Tensor"], "Tensor Calculus": ["Tensor Basics", "Contravariantand Covariant Tensors", "Tensor Operations", "Metric Tensor", "Ricci Curvature"]}}, "General Relativity - Basic Concepts": {{"Spacetime": ["Event", "Worldline", "Light Cone"], "Einstein Field Equations": ["Energy-Momentum Tensor", "Einstein's Field Equations Solutions"], "Curvature of Spacetime": ["Geodesic Deviation", "Geodetic Effect", "Frame-Dragging"], "Energy-Momentum Tensor": ["Stress-Energy Tensor", "Perfect Fluid"]}}, "General Relativity - Predictions and Experimental Tests": {{"GravitationalTimeDilation": ["Hafele–Keating Experiment"], "Gravitational Lensing": ["Einstein Cross", "Einstein Ring"], "Gravitational Redshift of Light": ["Pound–Rebka Experiment"], "Shapiro Time Delay": ["Radar Echo Delay Tests"], "Singularities and Black Holes": ["Event Horizon", "Schwarzschild Radius", "Hawking Radiation", "Interiors of Black Holes"]}}, "General Relativity - Applications in Cosmology": {{"Big Bang Theory": ["Cosmic Inflation", "Timeline of the Big Bang"], "Cosmic Microwave Background Radiation": ["Discovery of CMB", "CMB Spectrum", "CMB Anisotropies"], "Expansion of the Universe": ["Hubble's Law", "Dark Energy", "Fate of the Universe"]}}, "General Relativity - Astrophysical Implications": {{"Black Holes": ["Schwarzschild Black Holes", "Kerr Black Holes", "Black Hole Thermodynamics"], "Microquasars and Active Galactic Nuclei": ["Accretion Disk", "Astrophysical Jets"], "Gravitational Waves": ["Detection by LIGO", "Gravitational Wave Astronomy", "Sources of Gravitational Waves"], "Gravitational Lensing": ["Strong Lensing", "Weak Lensing", "Microlensing"], "Expanding Universe Models": ["Friedmann-Lemaître-Robertson-Walker metric", "Observational Evidence"]}}, "General Relativity - Unresolved Issues": {{"Quantum Gravity": ["String Theory", "Loop Quantum Gravity"], "Unification with Other Fundamental Forces": ["Grand Unified Theory", "Theory of Everything"]}}, "General Relativity - Philosophical and Theoretical Implications": {{"Concept of Time": ["Arrow of Time", "Block Universe"], "Concept of Space": ["Absolute vs Relative Space"], "The Nature of Gravity": ["Problem of Time", "Quantum Gravity"], "Interpretations of General Relativity": ["Hole Argument", "Einstein's Equivalence Principle"]}}}}}}

Here is your topic name: {topic}
Wikipedia exerpt on {topic}:
{wiki_info}

Begin!
"""

prompt = PromptTemplate(
    input_variables=["topic", "wiki_info"], 
    template=template
)

class Query(BaseModel):
    topic: str
    #openai_api_key: str

def generate_response(query: Query):
    """Generates JSON valid dictionary of topic tree structure by querying wikipedia"""
    llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key, model="gpt-3.5-turbo-16k-0613")

    # Query chatGPT
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True 
    )

    wiki_info = wikipedia.run(query.topic)
    raw_response = chain.run(topic=query.topic, wiki_info=wiki_info)

    try:
        response = json.loads(raw_response.strip())
    except ValueError as e:
        print("Error loading response as JSON. Error:", str(e))
        print("The reponse from the LLM couldn't be parsed properly: Raw Response from LLM:\n\n", raw_response)
        return
    
    return response

app = FastAPI()

origins = [
    "*" # Allow everyting for now
    #"app://obsidian.md",
    #"http://localhost"
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/graph")
async def api_generate_response(query: Query):
    return generate_response(query)

