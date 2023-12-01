from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from dotenv import load_dotenv
import sys

#config
modelname = "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5"




# load the Environment Variables. 
load_dotenv()


def main():


    # get user input
    def get_text():
        input_text = input("Chat: ")
        return input_text


    def chain_setup():


        template = """<|prompter|>{question}<|endoftext|>
        <|assistant|>"""
        
        prompt = PromptTemplate(template=template, input_variables=["question"])

        llm=HuggingFaceHub(repo_id=modelname, model_kwargs={"max_new_tokens":1200})

        llm_chain=LLMChain(
            llm=llm,
            prompt=prompt
        )
        return llm_chain
    user_input = ""
    chatenabled = False
    while True:
        if not chatenabled:
            try:
                user_input = sys.argv[1]
            except:
                raise Exception("Error! You dint't write a prompt!")
        else:
            user_input = get_text()
        if user_input == "exit":
            break
        # generate response
        def generate_response(question, llm_chain):
            response = llm_chain.run(question)
            return response

        ## load LLM
        llm_chain = chain_setup()
        #generate response
        response = generate_response(user_input, llm_chain)
        print(response)
        try:
            if sys.argv[2] == "-chat":
                chatenabled = True
                continue
            else:
                break
        except:
            break

if __name__ == '__main__':
    main()
