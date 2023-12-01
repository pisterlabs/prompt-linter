import json

from langchain import PromptTemplate
from langchain.prompts import (
    FewShotPromptTemplate,
    FewShotChatMessagePromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate
)
from langchain.prompts.example_selector import LengthBasedExampleSelector

PREFIX_IMAGE_QUERY_PROMPT= """
You are a video editor's assistant who is trying to understand natural language request of the editor to come up with search query for images to put in the video. You are given a command from the editor, the original context of the command, and relevant content from the video. Relevant content is a list of snippets from the transcript and visual description (what action is happening, abstract caption, and descriptions of objects) of 10-second segments. You must generate the search query for the image to be displayed based on the editor's command, original context, and relevant content.

Note 1: If no relevant search query can be generated that satisfies the command, output only the command.
Note 2: Make sure that the search query is not too long, since it should be seen by the editor. Keep it under 100 characters.

"""

EXAMPLE_PROMPT = """
Command: {command}
Context: {context}
Transcript snippets: {metadata_transcript}
Visual descriptions: {metadata_visual}
Response: {response}
"""

SUFFIX_IMAGE_QUERY_PROMPT = """
Command: {command}
Context: {context}
Transcript snippets: {metadata_transcript}
Visual descriptions: {metadata_visual}
Response:
"""

def get_examples():
    context1 = [
        "The original command was: Put some images of devices that can be seen in the video.",
    ]
    metadata_transcript1 = [
        " oh happy friday everyone so a couple of",
        " serendipitous things sort of happened here uh one i was in the middle of working on a video actually come check this thing out so this is the gpd",
    ]
    metadata_visual1 = [
        {
            "action": "using computer",
            "abstract_caption": "young man sitting at a desk working on a laptop with headphones on.",
            "dense_caption": "small black laptop on wooden desk,a man sitting at a desk,the opened laptop,black laptop on desk,a black and silver watch,a white wall,blue cord on the desk,a closed laptop computer,a laptop screen is turned on,the screen is white,a man with short hair,a mans right hand,white box on desk,black keyboard on laptop,the hand of a man,the man is wearing a black shirt,a laptop on a table,a laptop on a table,a white piece of paper\n",
        },
        {
            "action": "using computer",
            "abstract_caption": "young man holding a laptop in his hands.",
            "dense_caption": "the opened laptop on the table,this is a laptop,a black box with wires plugged into it,a person is sitting on a wooden table,a black wrist band,a hand holding a laptop,laptop screen is on,the screen is white,laptop computer on desk,the keyboard of a laptop,the hand of a person,the brown and black flip phone\n",
        },
    ]
    command1 = ["of devices that can be seen"]
    response1 = "black laptop, black and sliver watch, cell phone, black keyboard png"

    context2 = [
        "The original command was: Whenever possible, put an icon that represents the scene.",
    ]
    metadata_transcript2 = [
        " up before don't finish up now you told me you were done sorry let me just get my wi-fi password in here i'm just gonna throw in some extra motions",
        " so that hopefully you guys reverse engineer my password here because that would be really really really inconvenient for me i would",
        " have to take a whole probably about six minutes or so and log into my interface and then change that all right now we have some important setup to do",
    ]
    metadata_visual2 = [
        {
            "action": "using computer",
            "abstract_caption": "a man sitting at a desk holding a laptop in his hands and a computer on the table.",
            "dense_caption": "the laptop is open,a laptop on a table,screen on the laptop,a black wristwatch with silver band,black keyboard on laptop,a white wall behind the laptop,blue cord on the table,white cord attached to laptop,a hand on a laptop,hand holding a cell phone,blue cord on the desk,a hand holding a laptop,black keyboard portion of laptop\n",
        },
        {
            "action": "using computer",
            "abstract_caption": "young man sitting at a desk working on a laptop computer.",
            "dense_caption": "a white laptop computer,a man on a laptop,a laptop on a desk,the computer monitor is on,the mans hair is dark,a black cell phone,papers on the desk,black cord on the desk,white wall in the background,keyboard on the laptop,a black keyboard on a laptop,hand of computer worker,the screen of the laptop\n",
        }, 
    ]
    command2 = ["icon that represents the scene"]
    response2 = "man working at a desk on a laptop icon png"

    context3 = [
        "The original command was: Based on the current scene, add an appropriate meme."
    ]
    metadata_transcript3 = [
        " surface go is using what is it a there it is a 4415 y processor what do you like brandon do you like the unboxing on black",
        " or the unboxing on wood grain which duper you like the wood grain all right we're going to do the wood grain so it's got a 4415 why processor four gigs or eight gigs of ram",
        " it's got 64 gigs or 128 gigs of storage acclaimed nine hours of battery life well this one's",
    ]
    metadata_visual3 = [
        {
            "action": "using computer",
            "abstract_caption": "a man sitting at a desk holding a laptop in his hands and a computer on the table.",
            "dense_caption": "the laptop is open,a laptop on a table,screen on the laptop,a black wristwatch with silver band,black keyboard on laptop,a white wall behind the laptop,blue cord on the table,white cord attached to laptop,a hand on a laptop,hand holding a cell phone,blue cord on the desk,a hand holding a laptop,black keyboard portion of laptop\n",
        },
        {
            "action": "using computer",
            "abstract_caption": "young man sitting at a desk working on a laptop computer.",
            "dense_caption": "a white laptop computer,a man on a laptop,a laptop on a desk,the computer monitor is on,the mans hair is dark,a black cell phone,papers on the desk,black cord on the desk,white wall in the background,keyboard on the laptop,a black keyboard on a laptop,hand of computer worker,the screen of the laptop\n",
        }, 
        {
            "action": "unboxing",
            "abstract_caption": "a man holding a laptop in his hands.",
            "dense_caption": "a man playing the wii,a black framed projector screen,a silver laptop,a person is holding a wii controller,a white apple computer,black cords on the desk,a man with brown hair\n",
        }
    ]
    command3 = ["appropriate meme"]
    response3 = "meme about young man talking about laptop specifications"

    examples = []
    examples.append({
        "context": json.dumps(context1),
        "metadata_transcript": json.dumps(metadata_transcript1),
        "metadata_visual": json.dumps(metadata_visual1),
        "command": json.dumps(command1),
        "response": response1,
    })
    examples.append({
        "context": json.dumps(context2),
        "metadata_transcript": json.dumps(metadata_transcript2),
        "metadata_visual": json.dumps(metadata_visual2),
        "command": json.dumps(command2),
        "response": response2,
    })
    examples.append({
        "context": json.dumps(context3),
        "metadata_transcript": json.dumps(metadata_transcript3),
        "metadata_visual": json.dumps(metadata_visual3),
        "command": json.dumps(command3),
        "response": response3,
    })
    return examples

def get_image_query_prompt_llm(partial_variables={}, examples = []):
    example_prompt_template = PromptTemplate(
        input_variables=["context", "metadata_transcript", "metadata_visual", "command", "response"],
        template=EXAMPLE_PROMPT,
    )

    example_selector = LengthBasedExampleSelector(
        examples=examples,
        example_prompt=example_prompt_template,
        max_length=300,
    )

    return FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt_template,
        prefix=PREFIX_IMAGE_QUERY_PROMPT,
        suffix=SUFFIX_IMAGE_QUERY_PROMPT,
        input_variables=["context", "metadata", "command"],
        partial_variables=partial_variables,
    )


def get_image_query_prompt_chat(partial_variables={}):
    example_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("human", "Command: {command}\nContext: {context}\nTranscript snippets: {metadata_transcript}\nVisual descriptions: {metadata_visual}"),
            ("ai", "{response}"),
        ]
    )
    few_shot_prompt_template = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt_template,
        examples=get_examples(),
    )

    system_message = SystemMessagePromptTemplate(prompt=PromptTemplate(
            input_variables=[],    
            template=PREFIX_IMAGE_QUERY_PROMPT,
            partial_variables=partial_variables,
        )
    )

    final_prompt = ChatPromptTemplate.from_messages(
        [
            system_message,
            few_shot_prompt_template,
            ("human", "Command: {command}\nContext: {context}\nTranscript snippets: {metadata_transcript}\nVisual descriptions: {metadata_visual}"),
        ]
    )

    return final_prompt