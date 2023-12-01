import os
import openai
import dotenv
from typing import List

from langchain.chains import LLMMathChain
from langchain.utilities import SerpAPIWrapper
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool, StructuredTool, Tool, tool

from scripts.rag import get_tool, create_query_tool

GPT_VERSION = "gpt-3.5-turbo"
TEMPERATURE = 0.3

dotenv.load_dotenv()

openai.api_key = os.environ.get("OPENAI_API_KEY")
print(openai.api_key)

VERBOSE=True

class EmailGenerator:
    def __init__(
            self,
            num_iterations,
            initial_prompt,
            user_name,
            company_name,
            company_kb_id,
            customer_company_name,
            customer_name,
            customer_url: List[str]
        ):
        # company_points: list[str] = get_company_info(company)
        # customer_points: list[str] = get_customer_info(customer)
        self.user_name = user_name
        self.customer_name = customer_name
        
        company_tool = create_query_tool(f"{company_name} website", f"Get info about {company_name} by asking ACTUAL QUESTIONS, e.g. 'What is ChatGPT for Enterprise' instead of 'ChatGPT for enterprise'", company_kb_id)
        
        
        # company_tool = get_tool(
        #     title=f"{company_name} website",
        #     description=f"Get info about {company_name} by asking ACTUAL QUESTIONS, e.g. 'What is ChatGPT for Enterprise' instead of 'ChatGPT for enterprise'",
        #     url_list=company_url_list
        # )
        company_llm = initialize_agent(
            [company_tool], ChatOpenAI(model_name=GPT_VERSION, temperature=TEMPERATURE, max_tokens=1000), agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose = VERBOSE, handle_parsing_errors=True
        )
        
        customer_tool = get_tool(f"{customer_company_name} website", f"Get info about {customer_company_name} by asking ACTUAL QUESTIONS, e.g. 'What is ChatGPT for Enterprise' instead of 'ChatGPT for enterprise'", customer_url)
        customer_llm = initialize_agent(
            [customer_tool], ChatOpenAI(model_name=GPT_VERSION, temperature=TEMPERATURE, max_tokens=1000), agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose = VERBOSE, handle_parsing_errors=True
        )
        
        
        company_points = company_llm.run(f"Explain who {company_name} is in detail. Then, generate a list of points about the solutions that {company_name} provides.")
        customer_points = customer_llm.run(f"Explain who {customer_company_name} is in detail. Then, generate a list of points about the problems that {customer_company_name} has.")
        print("INIT: CUSTOMER POINTS", customer_points)
        print("INIT: COMPANY POINTS", company_points)
        # company_points = """
        # 1. OpenAI makes state-of-the-art language models
        # 2. OpenAI provides easy-to-use APIs to call out
        # """

        # customer_points = """
        
        # """
        self.customerAgent = CustomerAgent(customer_points, customer_llm)
        self.salesAgent = SalesAgent(company_points, company_llm)

        self.num_iterations = num_iterations
        self.initial_prompt = initial_prompt

        self.email = self._generate_initial(company_points, customer_points)

        print("INIT: finished init")

    def generate(self):

        for _ in range(self.num_iterations):
            print("EMAIL: iteration")
            customer_feedback = self.customerAgent.critique(self.email)
            sales_feedback = self.salesAgent.critique(self.email)

            self.email = self._refine(customer_feedback, sales_feedback)
            print("EMAIL: ", self.email)

        return self.email
    
    def _generate_initial(self, company_points, customer_points):
        message=[{"role": "user", "content": f"""
            Your name is {self.user_name}. Your are reaching out to a potential customer, {self.customer_name}.
            Information about your company: {company_points}
            Information about your client: {customer_points}
        """}, {"role": "system", "content": self.initial_prompt}]
        response = openai.ChatCompletion.create(
            model=GPT_VERSION,
            messages = message,
            temperature=TEMPERATURE,
            max_tokens=1000
        )
        print("EMAIL: generated initial email", response)
        return response

    def _refine(self, customer_feedback, sales_feedback) -> str:
        print("refine")
        prompt = f"""
        Here is your original email:
        {self.email}
        
        The customer has provided the following feedback:
        {customer_feedback}

        Your sales team has provided the following feedback:
        {sales_feedback}

        Your name is {self.user_name}. The client's name is {self.customer_name}.
        Please rewrite the email, addressed to {self.customer_name}, according to these suggestions.
        Please aim to make the email useful to both the customer and the sales team.
        """
        message=[{"role": "user", "content": prompt}, {"role": "system", "content": self.initial_prompt}]
        response = openai.ChatCompletion.create(
            model=GPT_VERSION,
            messages = message,
            temperature=TEMPERATURE,
            max_tokens=1000
        )
        content = response["choices"][0]["message"]["content"]
        return content


class CustomerAgent:
    def __init__(self, initial_prompt, customer_llm):
        self.initial_prompt = initial_prompt # what the customer is
        self.customer_llm = customer_llm
    
    def critique(self, draft):
        """
        OpenAI API Call
        """
        prompt = f" Here is an email you receive: \n\n{draft}.\n\nPlease give a thorough critique on the above email. Explain exactly why the solutions the company provides are useless to you. Please number your critiques and classify each as MAJOR or MINOR."
        message=[{"role": "user", "content": prompt}, {"role": "system", "content": self.initial_prompt}]
        
        response = self.customer_llm.run(message)
        print("CRITIQUE: CUSTOMER", response)

        return response


class SalesAgent:
    def __init__(self, initial_prompt, company_llm):
        self.initial_prompt = initial_prompt
        self.company_llm = company_llm
    
    def critique(self, draft):
        """
        OpenAI API Call
        """
        prompt = f" Here is an email your subordinate has sent: \n\n{draft}.\n\nPlease give a thorough critique on the above email. Explain exactly why the email does or does not represent the solutions the company can actually provide. Please number your critiques and classify each as MAJOR or MINOR."
        message=[{"role": "user", "content": prompt}, {"role": "system", "content": self.initial_prompt}]
        
        response = self.company_llm.run(message)
        print("CRITIQUE: SALES", response)

        return response


if __name__ == "__main__":
    
    # company_url_list = [
    #     "https://glean.com/"
    #     "https://glean.com/product/overview/",
    #     "https://glean.com/product/workplace-search-ai",
    #     "https://glean.com/product/assistant",
    #     "https://glean.com/product/platform",
    #     "https://glean.com/product/knowledge-management"
    # ]
    company_kb_id = "f1970b8b-408b-4aa2-aefa-e0a1efa3bdbc"
    company_name = "Glean"
    
    email_gen = EmailGenerator(
        num_iterations=3,
        company_name=company_name,
        initial_prompt=f"You are an email-writing assistant for {company_name} that writes first-contact emails to potential clients.",
        user_name="Vignav",
        company_kb_id=company_kb_id,
        customer_name="Ron",
        customer_company_name="Jane Street",
        customer_url=["https://www.janestreet.com/"]
    )


    msg = email_gen.generate()

    print(msg)
