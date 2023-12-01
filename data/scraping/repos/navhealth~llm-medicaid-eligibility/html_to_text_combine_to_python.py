if __name__ == "__main__":
    from utils import create_llm

    llm = create_llm()

    category_urls = [
        "https://www.hca.wa.gov/free-or-low-cost-health-care/i-need-medical-dental-or-vision-care/individual-adults",
        "https://www.hca.wa.gov/free-or-low-cost-health-care/i-need-medical-dental-or-vision-care/parents-and-caretakers",
        "https://www.hca.wa.gov/free-or-low-cost-health-care/i-need-medical-dental-or-vision-care/pregnant-individuals",
        # "https://www.hca.wa.gov/free-or-low-cost-health-care/i-need-medical-dental-or-vision-care/children",
        # "https://www.hca.wa.gov/free-or-low-cost-health-care/i-need-medical-dental-or-vision-care/noncitizens",
        "https://www.hca.wa.gov/free-or-low-cost-health-care/i-need-medical-dental-or-vision-care/aged-blind-or-disabled",
        # "https://www.hca.wa.gov/free-or-low-cost-health-care/i-need-medical-dental-or-vision-care/age-65-and-older-or-medicare-eligible",
        # "https://www.hca.wa.gov/free-or-low-cost-health-care/i-need-medical-dental-or-vision-care/foster-care",
        # "https://www.hca.wa.gov/free-or-low-cost-health-care/i-need-medical-dental-or-vision-care/long-term-care-and-hospice",
        # "https://www.hca.wa.gov/free-or-low-cost-health-care/i-need-medical-dental-or-vision-care/medicare-savings-program",
    ]

    from html_to_text_to_python import url_to_rules

    rules_joined = "\n\n".join(url_to_rules(url, llm=llm) for url in category_urls)

    from langchain.prompts import PromptTemplate
    from langchain.chains import ConversationChain

    conversational_llm_prompt = "The following is a conversation between a human and an AI. The AI is an expert on Medicaid eligibility and is able to write quality Python code. If the AI does not know the answer to a question, it truthfully says it does not know.\n\nCurrent conversation:\n{history}\nHuman: {input}\nAI:"
    conversational_llm = ConversationChain(
        llm=llm,
        prompt=PromptTemplate(
            input_variables=["history", "input"], template=conversational_llm_prompt
        ),
    )

    prompts = [
        f"""Below are descriptions of eligibility requirements for {len(category_urls)} different categories of Medicaid beneficiaries. Using all these of eligibility requirements, write a single Python program that determines whether the user is eligible for any Medicaid program. The Python program should prompt the user with questions and use their responses to determine if they are eligible: 

{rules_joined}""",
        "Ensure that the code incorporates all information from the income tables, and not just a single value. Re-write the code: ",
    ]

    for prompt in prompts:
        output = conversational_llm({"input": prompt})["response"]
        print(conversational_llm.memory.chat_memory.messages[-2].content)
        print("=====================================================")
        print(output)
        print("=====================================================")
