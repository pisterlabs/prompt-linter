from typing import Optional

from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain.tools import BaseTool

class JobSearchTool(BaseTool):
    name = "job_search"
    description = "Useful for doing a job search. Input is a single string containing a stringified JSON object with jobTitle. Ouput is a list of jobs."

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Use the tool."""
        return """5 Test Engineer jobs in London:
ITS Testing Services (UK) Ltd
SL7, Marlow, SL7 1LW
competitive salary and package
As a Test Engineer - Electrical Safety, your key purpose is to complete electrical safety evaluations for Intertek's Electrical global client base, you would work closely with the customers to evaluate their products and documentation to ensure they meet specific requirements of the applicable test standards. As a Test Engineer - Electrical Safety, your key purpose is to complete electrical safety evaluations forIntertek'sElectrical global client base, you would work closely with the customers to evaluate their products and documentation to ensure they meet specific requirements of the applicable test standards. As a Test Engineer - Electrical Safety,you are a self-motivated, organised individual with ability to work independently in a fast-paced, multi-tasking environment.

Harris Crawley
Instrument Test Engineer
As a Instrument Test Engineer the individuals experience and multiple skills will be used within the Test Engineering team to undertake more complex elements of activities required to complete Customer repairs of aircraft instrumentation, including PCB repairs as necessary, whilst also supporting the business with simulator builds. * Following and maintaining test procedures and working in accordance with departmental procedures * Supporting production and test engineering with Full flight simulator builds, Pilot training devices and customer support as required

Associate QA/Test Engineer - Shopper Technology
London
We are looking for an Associate QA/Test Engineer to join our growing team based in London to help support our teams building the digital retail experience at the LEGO Group. * Support other application engineers in developing and maintaining automated tests for their functionality where appropriate * Experience working with software engineers to help them test functionality during the initial development process is preferred * Collaborate with Software Engineers, Digital Designers and Product Managers to quality assure new functionality and iterative improvements to the platform - Working closely with the whole product team, our QA Engineers play a vital role in securing the quality, security and stability of our digital retail experience. * Experienc

Portable Appliance Test Engineer
SW11, South West London, SW11 4NJ
£21,750
PHS Compliance self-delivers electrical test and inspection, M&E installation & maintenance & asset verification with over 400 engineers based nationwide. We dont require our PAT engineers to be experienced, we will supply all the training you need. A day in the life of a PAT Engineer at phs; * Completing tests on portable appliances with our customers offices and commercial offices. Phs compliancefocuses onhelping more than 2,000 UK business customers test, install and manage their property infrastructure assets.

Portable Appliance Test Engineer
Shadwell, E1 0hx
£21750 per annum
PHS Compliance self-delivers electrical test and inspection, M&E installation & maintenance & asset verification with over 400 engineers based nationwide. We dont require our PAT engineers to be experienced, we will supply all the training you need. A day in the life of a PAT Engineer at phs; * Completing tests on portable appliances with our customers offices and commercial offices. Phs compliancefocuses onhelping more than 2,000 UK business customers test, install and manage their property infrastructure assets.

There are 5 jobs in total. Calling this tool again will return the same 5 jobs.
"""
    
    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")
    
