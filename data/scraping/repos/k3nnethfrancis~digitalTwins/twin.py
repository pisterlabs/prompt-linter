import langchain
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory, ConversationBufferMemory
from src.config import username, personality, rules, initPlan, twinInstructions
from src.chains import initialize_chain, initialize_meta_chain, initialize_revise_chain, get_chat_history, get_formatted_chat_history

from dotenv import load_dotenv
import os

# Get the current project directory
project_dir = os.path.dirname(os.path.realpath(__file__))

# Load .env file from the project's root directory
load_dotenv(os.path.join(project_dir, '../bot.env'))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")



def main(user_input, inner_loop_iters=1, max_chat_iters=5, verbose=False, debug_mode=False):
    
    langchain.debug = debug_mode
    twin = username
    instructions = twinInstructions

    twin_memory = ConversationBufferWindowMemory(
        memory_key="chat_history", 
        return_messages=True, 
        ai_prefix=twin
    )
    twin_propose_memory = ConversationBufferWindowMemory(
        memory_key="chat_history", 
        return_messages=True, 
        ai_prefix=twin
    )
    full_history = ConversationBufferWindowMemory(
        memory_key="full_history", 
        return_messages=True, 
        ai_prefix=twin
        )
    # initialize the initial conversation chain
    chain = initialize_chain(
        instructions, 
        memory=twin_memory
    )
    propose_chain = initialize_chain(
        instructions,
        memory=twin_propose_memory
    ) # initialize the initial conversation chain
    output = chain.predict(human_input=user_input) # assign the output to a var and include memory for the convo
    
    full_history.save_context({"Human": user_input}, {twin: output})
    twin_propose_memory.save_context({"Human": user_input}, {twin: output})

    if verbose:
        #print the twins output
        print(f'{twin}: {output} [END TWIN 1]') # print the first twin response
        print()
        print(
            f'''---\nMEM STATE 0:\n{get_formatted_chat_history(chain.memory)}\n---'''
        )
        print()
        print('...INITIALIZING CONVERATION LOOP...')
        
        ## this kicks off the first query to the twin that it will self reflect about before answering
        for i in range(max_chat_iters):
            print(f'[Iter {i+1}/{max_chat_iters}]')
            
            human_input = input() # get input from the human user
            print()
            print(
                f'''---\nMEM STATE {i}:\n{get_formatted_chat_history(chain.memory)}\n---'''
            )   
            print()
            print('...INITIALIZING INNER SELF-REFLECTION LOOP...')

            for j in range(inner_loop_iters):
                print(f'(Step {j+1}/{inner_loop_iters})')
                print()
                proposed_output = propose_chain.predict(
                                        human_input=human_input,
                                        chat_history=get_formatted_chat_history(propose_chain.memory)
                )

                full_history.chat_memory.add_user_message(human_input)
                twin_memory.chat_memory.add_user_message(human_input)

                print(
                    f'''MEM STATE {j+2}:\n{get_formatted_chat_history(chain.memory)}\n---'''
                )
                
                print(f'{twin} [proposed response]: {proposed_output} [END TWIN 3]')
                print()
                print(
                    f'''HISTORY STATE {j}: {full_history.chat_memory}'''
                )
                print()
                # The AI reflects on its performance using the meta chain
                meta_chain = initialize_meta_chain(personality=personality, rules=rules, memory=full_history) # inject the twins personality and rules for the simulation
                meta_output = meta_chain.predict(chat_history=get_formatted_chat_history(meta_chain.memory)) # assign the output to a var with memory
                print(f'{twin} [self-reflection]: {meta_output} [END REFLECTION 1]')
                print(
                    f'''MEMORY STATE {j+3}:\n{get_formatted_chat_history(chain.memory)}\n---'''
                ) 
                print()
                
                # initialize the revise chain
                revise_chain = initialize_revise_chain()
                #revision = revise_chain.predict(chat_history=get_chat_history(chain.memory), meta_reflection=meta_output, proposed_response=proposed_output) # include history and the meta reflection output
                revision = revise_chain.predict(chat_history=get_formatted_chat_history(chain.memory), meta_reflection=meta_output, proposed_response=proposed_output) # include history and the meta reflection output
                # print(f'{twin} [revised response]: {revision} [END REVISION 1]')
                print(f'{twin}: {revision} [END REVISION 1]')
                print()
                # human_input = input()
                # print(f'Human: {human_input} [END6]')

                #save the revised exchange to memory to continue the loop
                twin_memory.chat_memory.add_ai_message(revision)
                #memory.save_context({"Human": human_input}, {twin: revision})
                print(
                    f'''MEMORY STATE {j+4}:\n{get_formatted_chat_history(chain.memory)}\n---'''
                ) 
                print()
                #mem.append(revision)
                print('...ENDING INNER SELF-REFLECTION LOOP..')
                print()
    else:
                #print the twins output
        print(f'{twin}: {output}') # print the first twin response
        
        ## this kicks off the first query to the twin that it will self reflect about before answering
        for i in range(max_chat_iters):
            
            human_input = input() # get input from the human user

            for j in range(inner_loop_iters):
                proposed_output = propose_chain.predict(
                                        human_input=human_input,
                                        chat_history=get_formatted_chat_history(propose_chain.memory)
                )

                full_history.chat_memory.add_user_message(human_input)
                twin_memory.chat_memory.add_user_message(human_input)
                print('\n---')
                print(f'{twin} [...thinking]: {proposed_output}')
                # The AI reflects on its performance using the meta chain
                meta_chain = initialize_meta_chain(personality=personality, rules=rules, memory=full_history) # inject the twins personality and rules for the simulation
                meta_output = meta_chain.predict(chat_history=get_formatted_chat_history(meta_chain.memory)) # assign the output to a var with memory
                print(f'{twin} [..reflecting]: {meta_output}')
                print('---\n')
                # initialize the revise chain
                revise_chain = initialize_revise_chain()
                revision = revise_chain.predict(chat_history=get_formatted_chat_history(chain.memory), meta_reflection=meta_output, proposed_response=proposed_output) # include history and the meta reflection output
                
                print(f'{twin}: {revision}')

                #save the revised exchange to memory to continue the loop
                twin_memory.chat_memory.add_ai_message(revision)
            
        
        print('\n'+'#'*80+'\n')

    print(f'End of conversation! Thanks for Chatting!')


print('...send a message to start the conversation...')
# memory is not working exactly how I would like but it 'works'
init_msg = input()

main(
    user_input=init_msg,
    max_chat_iters=10,
    verbose=False,
    debug_mode=False
)