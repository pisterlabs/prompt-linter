from langchain import PromptTemplate, LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
import textwrap
import os, time


def summarise (text): 
    llm = ChatOpenAI(openai_api_key="sk-xxxxxxxxxxx",temperature=0,model_name="gpt-3.5-turbo")
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Concisly summarise the following text and start your summary with ""---"": {text}",
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    summary = chain.run(text)
    return summary    

# open the file at the given filepath and return its content
def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


if __name__ == '__main__':

    #set parameters
    input_directory = 'Texts'
    output_directory = 'Summaries'

    # loop through all files in folder
    for filename in os.listdir(input_directory):
        filepath = os.path.join(input_directory, filename)
        text = open_file(filepath)

        # break them down into chunks, 2000 characters each
        chunks = textwrap.wrap(text, 1400,break_long_words=False)
        count = 0
        index = 0

        print ("Summarising: "+ filename +"...\n")
        
        # loop over the chunks
        for chunk in chunks:

            # try generate a summary through all available bot instances
            count = count + 1
            success = False
            error_not_yet_shown = True
            retry_count = 0
            while not success:
                try:
                    summary = summarise(chunk)
                    summary = summary.replace("---", "\n- ")
                    success = True
                    retry_count = 0 
                except Exception as not_summarised:
                    print(not_summarised)
                    if error_not_yet_shown: 
                        print ("Hold tight, retrying until it works...\n")
                        error_not_yet_shown = False 
                    success = False
                    with open ('error_log.txt', 'w', encoding = 'utf-8') as f:
                        f.write ("Summarisation failed for: "+filename+" for the following chunk of text:\n\n\""+chunk+"\"\n\n")
                    f.close()
                    retry_count += 1
                if retry_count >= 5:
                    time.sleep(30)
                    retry_count = 0

            print('\n\n\n', count, 'of', len(chunks), ' - ', summary)

            # append to and save the summary file in the Summaries/ folder
            with open(os.path.join(output_directory, filename), 'a', encoding='utf-8') as f:
                f.write(summary + '\n\n')
            f.close()
