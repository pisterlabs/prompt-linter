from __future__ import annotations
import openai
from langchain.agents import Tool
import os
import dotenv
dotenv.load_dotenv()
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import asyncio


from typing import Any, Dict, List, Optional

from pydantic import Extra

from langchain.schema import BaseLanguageModel
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.chains.base import Chain
from langchain.schema import BasePromptTemplate
import json


async def frontend_chain(
        inputs: Dict[str, Any],
        llm: BaseLanguageModel,
        advanced_llm: BaseLanguageModel,
    ) -> str:
        # Feature Extraction
        prompt = PromptTemplate(
            input_variables=["project_details", "project_technologies"],
            template="""
            Given the following project description and tech stack, identify and elaborate on the key frontend features that would be necessary for development. The frontend features should not involve backend api calls or database interactions. The frontend features should be described in terms of user stories or detailed feature requirements.
            This project is a hackathon project. Break apart the features into MVP and additional features. The MVP should be the minimum features necessary to have a working prototype.
            Project description: {project_details}
            Technologies: {project_technologies}
            """
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        frontend_features = await chain.arun(inputs)
        
        # Specification Creation
        specification_prompt = PromptTemplate(
            input_variables=["frontend_features", "project_technologies"],
            template="""
            Given the extracted frontend features, create a detailed technical specification. 
            This specification should include the technologies to be used, the architecture, pages to be developed, and the components required for each page.
            However, they should be split into two categories: MVP and additional features. The MVP should be the minimum features necessary to have a working prototype.
            You should ignore the technologies for the backend and focus on the frontend.
            Please also mention any other technical considerations.

            Frontend Features: {frontend_features}
            Project Technologies: {project_technologies}
            """
        )
        specification_chain = LLMChain(llm=llm, prompt=specification_prompt)
        specification = await specification_chain.arun({
            'frontend_features': frontend_features,
            'project_technologies': inputs['project_technologies']
        })

        # AI Approval
        
        approval_prompt = PromptTemplate(
            input_variables=["technical_specification", "aspect", "group_size", "group_experience"],
            template="""
            Given the developed technical specification, conduct a thorough review of the MVP Features only for any inconsistencies or issues. 
            Also, evaluate whether the MVP Features, can be realistically completed within the two day hackathon for {group_size} people.
            
            The MVP Features are specifically listed under the heading 'MVP Features'. 
            Please completely disregard any features or sections listed under 'Additional Features' or any similar headers.
            This specification is only for the {aspect} aspect of the project, and should not be evaluated for other aspects.

            Answer this question: Can the MVP Features be realistically completed within the two day hackathon for {group_size} people with this skill level: {group_experience}?
            Output only a json with keys 'approval' and 'comments'. 
            If yes, the value of 'approval' should be '1' and the value of 'comments' should be an empty string
            If not, the value of 'approval' should be '0' and the value of 'comments' should be a string with the issues and inconsistencies listed.

            Technical Specification: {technical_specification}
            
            Output only a json with keys 'approval' and 'comments'. 
            """
        )
        
        approval_chain = LLMChain(llm=advanced_llm, prompt=approval_prompt)
        approval = await approval_chain.arun({
            'technical_specification': specification,
            'aspect': 'frontend',
            'group_size': inputs['group_size'],
            'group_experience': inputs['group_experience']
        })
        
        approvals_object = json.loads(approval)
        
        return_obj = {
            'approval': approvals_object['approval'],
            'comments': approvals_object['comments'],
            'idea': inputs['project_details'],
            'features': frontend_features,
            'specifications': specification
        }
        
        return json.dumps(return_obj)
