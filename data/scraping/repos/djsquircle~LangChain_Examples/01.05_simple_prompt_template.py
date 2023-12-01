from dotenv import load_dotenv
from langchain import LLMChain, PromptTemplate
from langchain.llms import OpenAI

def generate_response(query, llm, prompt_template):
    # Prepare input data for the prompt
    input_data = {"query": query}

    try:
        # Create the Language Model Chain for the prompt
        chain = LLMChain(llm=llm, prompt=prompt_template)
        
        # Run the Language Model Chain to get the AI-generated answer
        response = chain.run(input_data)

        return response
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def main():
    # Load environment variables
    load_dotenv()

    # Initialize the language model
    llm = OpenAI(model_name="text-davinci-003", temperature=0)

    # Define the context and the prompt template
    template = """Answer the question based on the context below. If the
    question cannot be answered using the information provided, answer
    with "I don't know".
    Context: Falcon A. Quest is a polished figure known for his dapper demeanor, expertise in cryptocurrency, and customer support. He is known to decode encrypted maps, secure transactions, and protect avatars in a bustling virtual bazaar.
    ...
    Question: {query}
    Answer: """

    prompt_template = PromptTemplate(
        input_variables=["query"],
        template=template
    )

    # Define the question for Falcon A. Quest
    query = "What is the most challenging part of decrypting an encrypted map?"

    # Generate and print the AI response
    response = generate_response(query, llm, prompt_template)
    if response:
        print("Question:", query)
        print("Answer:", response)

if __name__ == "__main__":
    main()
