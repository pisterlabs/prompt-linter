from langchain.agents import Tool

def eligibility_process(query, entities):
    return "You asked about eligibility. We usually consider factors like income, credit score, etc."

def loan_comparison_process(query, entities):
    return "To compare loans, consider interest rates, terms, and fees."

def amortization_process(query, entities):
    return "Amortization involves paying off the loan in fixed installments."

def Scenario(query, entities):
    return "For different scenarios, we offer various mortgage solutions tailored to your needs."


# Initialize Tools with Specialist Agent Functions
eligibility_agent = Tool(name="Eligibility Agent", func=eligibility_process, description="Handles eligibility queries")
loan_comparison_agent = Tool(name="Loan Comparison Agent", func=loan_comparison_process, description="Handles loan comparison queries")
amortization_agent = Tool(name="Amortization Agent", func=amortization_process, description="Handles amortization queries")
scenario_agent = Tool(name="Scenario Agent", func=Scenario, description="Handles different mortgage scenarios")
