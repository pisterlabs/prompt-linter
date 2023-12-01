import os
import json
from datetime import datetime
import timeit
from dotenv import load_dotenv
import openai
from openai.error import AuthenticationError
import wandb
from langchain.chat_models import ChatOpenAI
from langchain.parallel import ParallelChain


#Logging etc
from gen_logger.logger_util import setup_logger
from langchain.globals import set_verbose, set_debug
from langchain.callbacks import WandbCallbackHandler, StdOutCallbackHandler  
from langchain.callbacks import wandb_tracing_enabled
#from logs.wandb_langchain import wandb_callback
#from wandb_langchain import wandb_callback, callbacks
import textstat
import spacy


from operator import itemgetter
from langchain.schema.runnable import RunnableParallel
from langchain.schema import StrOutputParser
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain.schema import messages_to_dict 
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableMap, RunnablePassthrough

#Input placeholders 

load_dotenv()
_logger=setup_logger("langllm")
os.environ["LANGCHAIN_WANDB_TRACING"] = "true"
os.environ["WANDB_PROJECT"] = "avin-midi"

set_debug(True)
#set_verbose(True)


#Initialize WandB and set Langchain flag to verbose
try:
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb.init(project=os.getenv("WANDB_PROJECT"))
    
    
    prediction_table = wandb.Table(columns=["logTime", "func" "prompt", "prompt_tokens", "completion", 
                                            "completion_tokens", "time"])
except Exception as e:
    _logger.error(e)

try:
    _logger.info(f"Adding openAI API key")
    #callbacks = [StdOutCallbackHandler(), wandb_callback]
    model = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model_name="gpt-3.5-turbo") 
    _logger.info("init llm complete. Model is {}".format(model.model_name))
except AuthenticationError as e:
    _logger.error(e)

def image_of_description(description):
    #Make an image of the song description
    image_req = openai.Image.create(prompt=f"A futuristic anime image vibrants representative of a song that is: {description}",
    n=0,
    size="1024x1024"
    )

    image_res = image_req["data"][0]
    
    print(image_res)
    return image_res



'''
Prompt Templates for the chain of calls to the model of choice:



'''



prompt1 = ChatPromptTemplate.from_template(
    """You are an expert level music composer. Generate the lyrics of a song named {song_name}
    The lyrics of the song will matche the following description. 
    -Make it catchy and suitable for a 4/4 rhythm:
    The mood, tone and style is to be:
    ```
    {description}
    ```
    Output the verse lyrics as a string with appropriate line breaks and paragraphs. 
    Remove all new line markers "\\n"
    """
)
prompt2 = ChatPromptTemplate.from_template(
    """Take the song and write the piano chords for a Verse. Write the chords as a python list and nothing else:
    *****
    {song}
    *****
    
    """
)

#Can generate the colors of the song and make a User Interface.
prompt3 = ChatPromptTemplate.from_template(
    """Take the following chords:
    ********
    {chords}
    ********* 
    - Use MIDI notation. Write the synth, bass and drum MIDI notes for the {element} with varying velocity.
    Synth:
    Bass:
    Drum:
    Rhythm:
    - Leave Rhythm empty for the next step
    - Remove all new line characters "\n"
    
    Output nothing else but a JSON Object with the MIDI.
    """)

prompt4 = ChatPromptTemplate.from_template(
    """
    Generate the MIDI Notation of the Rhythm of the Song? Focusing on the:
    {element} and {chords}
    Below is the MIDI of the instruments:
    {midi_dict}
    -Use standard MIDI rhythm notation to match the chords.
    -Use 4/4 time in Traditional Western Notation at 65bpm.
    - Match the rhythm to the song.
    - Remove all new line characters "\n"
    -Insert the data into the rhythm key of the midi JSON Object
    - Output nothing but a JSON Object with the MIDI that corresponds to the rhythm element of the song.
    """
)

prompt5 = ChatPromptTemplate.from_template(
    """Cleanup any inconistencies. 
    Remove unnecessary information. 
    Format correctly as JSON:
    input:
     {dictionary} 
    ************
    Output a JSON object. Nothing Else:  
 """   
)

#Config:

song_name = "Cherry Prick Ya Dick"
description = "A warm blues vibe, beers on a sunny day."
model_parser = model | StrOutputParser()
json_parser = model | SimpleJsonOutputParser()
file_name = "_".join(song_name.split()[0:4]) + "_avin_agoodtime.txt"

#chains
describer = {"description": RunnablePassthrough(), "song_name": RunnablePassthrough()} | prompt1 | {"song":model_parser} 
chords =  {"song": describer} | prompt2 | model_parser
chords_to_midi = {"element": itemgetter("element"), "chords": chords} | prompt3 | {"midi": json_parser}
midi_to_rhythm = {"element": itemgetter("element"), "chords":chords, "midi_dict": chords_to_midi} | prompt4 | {"rhythm":json_parser}
midi_chain = {"dictionary": chords_to_midi} | prompt5 | json_parser
rhythm_chain = {"dictionary": midi_to_rhythm} | prompt5 | json_parser

try:
    _logger.info("starting chain")
    song_desc = describer.invoke({"description": description, "song_name": song_name})
    _logger.info(f"song description complete")
    _logger.log
    chords_list = chords.invoke({"song": describer})
    _logger.info(f"Chords list complete")
    _logger.info(chords_list)
    midi_list = chords_to_midi.invoke({"element": "verse", "chords": chords_list, "song_name": song_name})
    _logger.info(f"Midi list complete")
    rhythm_midi = midi_to_rhythm.invoke({"element": itemgetter("element"), "rhythm_dict": chords_to_midi})
    _logger.info(f"Rhythm list complete")
    midi_out = midi_chain.invoke({"dictionary": chords_to_midi, "element":"verse", "description": description, "song_name": song_name})
    rhythm_out = rhythm_chain.invoke({"dictionary": midi_to_rhythm, "element":"verse", "description": description, "song_name": song_name})
    _logger.info("Complete....writing file")
except Exception as e:
    _logger.error(e, exc_info=True)
    os.remove('cmd_chain_gav.txt')
    os.remove(file_name)
    with open('./data/partial.txt', 'w') as f:
        try:
            f.write(f"00----The Noble Song: {song_name} at 65 bpm----00\n\n")
            f.write(str(song_desc))
            f.write(f"\n00----The Chords of {song_name}----00\n\n")
            f.write(str(chords_list))
            f.write("\n00----The MIDI----00\n\n")
            f.write(json.dumps(midi_list, indent=4))
            f.write("\n00----The Rhythm------00\n\n")
            f.write(json.dumps(rhythm_midi, indent=4))
            f.write("\n\n\n00----The Final Midi------00\n\n\n")
            f.write(json.dumps(midi_out, indent=4))
            f.write("\n\n\n00----The Final Rhythm------00\n\n\n")
            f.write(json.dumps(rhythm_out), indent=4)
            f.write(f"\n\n\n00----The EDN------00\n\n\n")
        except Exception as err:
            _logger.error(err, exc_info=True)
            os.remove('cmd_chain_gav.txt')
            os.remove(file_name)



#midi_output = midi_chain.invoke({"element":"verse", "description": description, "song_name": song_name})

#rhythm_output = rhythm_chain.invoke({"element":"verse", "description": description, "song_name": song_name})


with open(file_name, 'w') as f:
    f.write(f"00----The Noble Song: {song_name} at 65 bpm----00\n\n")
    f.write(str(song_desc))
    f.write(f"\n00----The Chords of {song_name}----00\n\n")
    f.write(str(chords_list))
    f.write("\n00----The MIDI----00\n\n")
    f.write(json.dumps(midi_list, indent=4))
    f.write("\n00----The Rhythm------00\n\n")
    f.write(json.dumps(rhythm_midi, indent=4))
    f.write("\n\n\n00----The Final Midi------00\n\n\n")
    f.write(json.dumps(midi_out, indent=4))
    f.write("\n\n\n00----The Final Rhythm------00\n\n\n")
    f.write(json.dumps(rhythm_out), indent=4)
    f.write(f"\n\n\n00----The EDN------00\n\n\n")

_logger.info(f"File completed and saved to: {os.path(file_name)}")
    

    
    
    
    
    
   
