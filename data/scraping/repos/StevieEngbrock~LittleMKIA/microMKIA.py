from silence_tensorflow import silence_tensorflow # import and call silence_tensor_flow to make tensorflow shutup about files it thinks I need but don't
silence_tensorflow()
import logging
logging.disable(logging.WARNING) # disable logging warnings to get rid of warnings about things that aren't really errors
from langchain import HuggingFacePipeline,PromptTemplate # import the stuff for setting up langchain with huggingface
from langchain.memory import ConversationBufferMemory # import stuff langchain uses to remember stuff
from langchain.chains import ConversationChain # import stuff it uses in relation to conversations
from transformers import pipeline # import the main pipeline from transformers
import readline # import readline for a slightly nicer and slightly easier to us interface
from transformers import GenerationConfig # import stuff to configure for text generation
import re # import re to match regular expression

modelPath = "LittleMKIA" # local path to the language model
mode1 = "text2text-generation" # task we want the transformers pipeline to perform

config = GenerationConfig.from_pretrained(modelPath) # set up the configuration object

pipe = pipeline(task= mode1, model=modelPath,min_length = 20,max_new_tokens = 200,temperature = 0.7,early_stopping = True,
no_repeat_ngram_size=3,do_sample = True,top_k = 150,generation_config=config) # set up the pipeline

llm = HuggingFacePipeline(pipeline=pipe) # make transformers pipeline usable by langchain

# create a template for the prompt
template  = '''
You are MKIA an intelligent companion and assistent.
{history}
User: {input}'''

# create the prompt from the template
prompt = PromptTemplate(
    input_variables=[ "input","history"],
    template=template)

# set up a memory object
mem = ConversationBufferMemory(k = 1000,memory_key = "history",return_messase = False,ai_prefix = "MKIA")

# make a conversation chain and pass all the necessary parameters to it, tell it we don't want verbose so we only get regular output
chat_chain = ConversationChain(
    llm=llm,
    prompt = prompt,
    memory= mem,
    verbose=False
)
#create a function that will act as the program's main loop
def loop():
    while 1: # python is optimize for while 1: not while True: so I will use while 1:
        In = input('User > ') # ask for input
        if re.match('think[:] (.*)|think[:](.*)|Think[:] (.*)|Think[:](.*)',In) != None:
            # if the input text matches the pattern then we will bypass langchain

            In2 = re.sub('think[:]|Think[:]','',In).strip()
            # remove the prefix at the begining

            out= pipe(In2)[0]['generated_text']
            # get the output directly from the language model

            print(out)

        elif In == 'quit':
            break

        else:
           out = chat_chain.run(input=In)      # we feed the input to langchain and get the result
           print(f'MKIA > {out}') # let the user know what MKIA said and that she said it
           print('\n\n') # print 2 newlines to help output be prettier

loop()
