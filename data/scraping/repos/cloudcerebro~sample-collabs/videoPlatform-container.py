import json
import eventlet

eventlet.monkey_patch()
import sys
from pathlib import Path
import os
import boto3
from threading import Thread
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from dotenv import load_dotenv
from langchain import LLMChain
from langchain.llms import OpenAIChat
from langchain import PromptTemplate
from langchain.llms import PromptLayerOpenAIChat
from langchain.chains import create_tagging_chain, create_tagging_chain_pydantic
from langchain.chat_models import ChatOpenAI

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

print("Executing promptTemplate.py")

llm = ChatOpenAI(model_name="gpt-4") # type: ignore
# llm = ChatOpenAI() # type: ignore

""" from langchain.memory.chat_message_histories import DynamoDBChatMessageHistory

history = DynamoDBChatMessageHistory(table_name="SessionTable", session_id="0")

history.add_user_message("hi!") """


sequence_template = """
You are a powerful AI Assistant which is well trained on many videos.

The input may contain one of the following: 
a) video_input: JSON which contains the current JSON configuration.
b) user_query: Question by the user using which a new JSON configuration has to be generated.

The template for the new JSON to be created is as follows:

{{
    "videoInput": {{
      "backgroundColor": "#000000", // Any color that suits the project. Do not always use the black background.
      "durationInFrames": 1500,  // Got from the user_query input. This is project_duration
      "fps": 30, // Got from the user_query input
      "width": 1920, // Got from the user_query input
      "height": 1080, // Got from the user_query input
      "sequences": [ // A sequence is a kind of slide in a video. Multiple sequences can be generated.
        {{
          "sequenceNo": 1, // Sequential number
          "sequenceName": "Sequence 1", // Assistant gives a relevanat Name for the sequence
          "from": 0, // Based on the previous sequence
          "durationInFrames": 300, // Split the project_duration among the sequences. This can vary from one sequence to another. The sum of all durationInFrames in sequences must be equal to project_duration. The following statement must be followed strictly: ### durationInFrames must be a multiple of fps. The reminder got when durationInFrames is divided by fps MUST always be 0. ### 
          "durationInSeconds": 10, // This can be calculated based on the durationInFrames and fps with the formula: durationInFrames/fps. This is a number and not a string.
          "transition": {{
            "type": "in" // Can be 'in' or 'out'
          }},
          "structure": {{
            "backgroundColor": "#000000", // Assistant chooses ny color that suits the sequence. Do not always use the black background.
            "useTemplate": false,
            "template": null,
            "generatedText": "", // The text that will be generated for this sequence
            "components": []
          }}
        }}
      ]
    }}
  }}

Steps:
1. Generate the text for each sequence based on the user_input query and place it in generatedText field.
2. Form JSON based on the above template substituting the generated text for each sequence in the formed JSON.

durationInFrames, fps, width and height are got from the user_query input.
The durationInFrames of each sequence can be different from the other.

video_input:
{video_input}

user_query:
{user_query}


Result:

Example Input:
user_query: Generate a video of duration 900 frames in 30 fps with width: 1080 and height: 1920. The video is about five facts about cats.
Example output:
{{
    "videoInput": {{
      "backgroundColor": "#000000",
      "durationInFrames": 900,
      "fps": 30,
      "width": 1080,
      "height": 1920,
      "sequences": [
        {{
          "sequenceNo": 1,
          "sequenceName": "Sequence 1",
          "from": 0,
          "durationInFrames": 300,
          "durationInSeconds": 10,
          "transition": {{
            "type": "in"
          }},
          "structure": {{
            "backgroundColor": "#000000",
            "useTemplate": false,
            "template": null,
            "generatedText": "Cat is a friendly animal",
            "components": []
          }}
        }},
        {{
          "sequenceNo": 2,
          "sequenceName": "Sequence 2",
          "from": 300,
          "durationInFrames": 240,
          "durationInSeconds": 8,
          "transition": {{
            "type": "in"
          }},
          "structure": {{
            "backgroundColor": "#FB0000",
            "useTemplate": false,
            "template": null,
            "generatedText": "Cat can be a pet animal",
            "components": []
          }}
        }},
        {{
          "sequenceNo": 3,
          "sequenceName": "Sequence 3",
          "from": 540,
          "durationInFrames": 360,
          "durationInSeconds": 12,
          "transition": {{
            "type": "out"
          }},
          "structure": {{
            "backgroundColor": "#30984C",
            "useTemplate": false,
            "template": null,
            "generatedText": "Cat has a soft body",
            "components": []
          }}
        }}
      ]
    }}
  }}
""" # type: ignore

comp_template = """
You are a powerful AI Assistant which is well trained on many videos.

The input will contain sequence_input which is a JSON that contains the sequence configuration.
Based on the generatedText key, the Assistant will generate a new JSON that contains the sequence configuration based on the template below.



The template for the new JSON to be created is as follows:

{{
    "sequenceNo": 3, // The sequence number from the sequence input
    "sequenceName: "Sequence 3", // The sequence name from the sequence input
    "from": 600, // From the sequence_input
    "durationInFrames": 300, // From the sequence_input
    "durationInSeconds": 10, // From the sequence_input
    "transition": {{
    "type": "in" // From the sequence_input
    }},
    "structure": {{
    "backgroundColor": "#000000", // From the sequence_input
    "useTemplate": false, // From the sequence_input
    "template": null, // From the sequence_input
    "generatedText": "Cat has a soft body", // The text that will be generated for this sequence
    "components": [] // List of components generated for this sequence. The components are generated based on the generatedText. The components can be text, paragraph, image. The assistant chooses a combination of components that looks good for the sequence. The components that can be used are mentioned in the instructions that follow this. There can be multiple components in a sequence.
    }}
}}

Components that can be used are:
1. paragraph
2. image
3. text

1. The template for the new JSON for paragraph component to be created is as follows:
{{
  "name": "Card_1_Paragraph", // Name of the paragraph
  "type": "paragraph", // Type of the component. Should be paragraph for this paragraph component
  "text": "Lorem ipsum dolor sit amet", // This is based on generatedText from the sequence_input
  "durationInFrames": 100, // This is based on the durationInFrames from the sequence_input
  "wordByWord": true, // Assisant can choose to show the text word by word or all at once
  "wordByWordSpeed": 2, // Speed at which the words are shown. This is based on the durationInFrames from the sequence_input. Assistant chooses a value between 1 and 5.
  "style": {{ // Assistant chooses a style for the paragraph component using CSS. The Assistant chooses a style that looks good with the other components in the sequence.
    "position": "absolute", // Assistant chooses a position for the component. This is CSS position.
    "width": "744px", // Assistant chooses a width for the component. This is CSS width. The width should be within the width from the sequence_input.
    "height": "117px", // Assistant chooses a height for the component. This is CSS height. The height should be within the height from the sequence_input.
    "left": "167px", // Assistant chooses a left for the component. This is CSS left.
    "top": "660px", // Assistant chooses a top for the component. This is CSS top.
    "fontFamily": "Nunito", // Assistant chooses a fontFamily for the component. This is CSS fontFamily based on the google font.
    "fontStyle": "normal", // Assistant chooses a fontStyle for the component. This is CSS fontStyle. Possible values are based on the fontFamily of the google font.
    "fontWeight": 400, // Assistant chooses a fontWeight for the component. This is CSS fontWeight. Possible values are based on the fontFamily and fontStyle of the google font.
    "fontScript": "latin", // Assistant chooses a fontScript for the component. This is CSS fontScript. Possible values are based on the fontFamily, fontStyle and fontWeight of the google font. This is mandatory.
    "fontSize": "24px", // Assistant chooses a fontSize for the component. This is CSS fontSize.
    "lineHeight": "29px", // Assistant chooses a lineHeight for the component. This is CSS lineHeight.
    "display": "flex", // Assistant chooses a display for the component. This is CSS display.
    "alignItems": "center", // Assistant chooses a alignItems for the component. This is CSS alignItems.
    "textAlign": "center",  // Assistant chooses a textAlign for the component. This is CSS textAlign.
    "color": "#ffffff", // Assistant chooses a color for the component. This is CSS color.
    "justifyContent": "center", // Assistant chooses a justifyContent for the component. This is CSS justifyContent.
    "background": "#ffffff00" // Assistant chooses a background color for the component. The Assistant chooses a background color that looks good with the other components in the sequence. The background color can be transparent.
  }},
  "tailwindClasses": "", // Assistant chooses a tailwind class for the component. The Assistant chooses a tailwind class that looks good with the other components in the sequence.
  "transitions": [ // Assistant chooses a transition for the component. The Assistant chooses a transition that looks good with the other components in the sequence.
    {{
      "type": "entry",
      "frameFrom": 60,
      "valueFrom": 200,
      "valueTo": 0
    }},
    {{
      "type": "entryTranslateY",
      "frameFrom": 60,
      "valueFrom": -100,
      "valueTo": 0
    }}
  ]
}}

2. The template for the new JSON for image component to be created is as follows:
{{
    "name": "Card_1_Image", // Name of the image
    "type": "image", // Type of the component. Should be image for this image component
    "src": "https://images.unsplash.com/photo-1611095772760-5b6b5b9d0b0f?ixid=MnwxMjA3fDB8MHxzZWFyY2h8Mnx8Y2F0JTIwYmFyfGVufDB8fDB8fA%3D%3D&ixlib=rb-1.2.1&w=1000&q=80", // This is based on generatedText from the sequence_input
    "durationInFrames": 100, // This is based on the durationInFrames from the sequence_input
    "style": {{ // Assistant chooses a style for the image component using CSS. The Assistant chooses a style that looks good with the other components in the sequence.
        "position": "absolute", // Assistant chooses a position for the component. This is CSS position.
        "width": "744px", // Assistant chooses a width for the component. This is CSS width. The width should be within the width from the sequence_input.
        "height": "117px", // Assistant chooses a height for the component. This is CSS height. The height should be within the height from the sequence_input.
        "left": "167px", // Assistant chooses a left for the component. This is CSS left.
        "top": "660px", // Assistant chooses a top for the component. This is CSS top.
        "display": "flex", // Assistant chooses a display for the component. This is CSS display.
        "alignItems": "center", // Assistant chooses a alignItems for the component. This is CSS alignItems.
        "justifyContent": "center", // Assistant chooses a justifyContent for the component. This is CSS justifyContent.
        "background": "#ffffff00" // Assistant chooses a background color for the component. The Assistant chooses a background color that looks good with the other components in the sequence. The background color can be transparent.
    }},
    "tailwindClasses": "", // Assistant chooses a tailwind class for the component. The Assistant chooses a tailwind class that looks good with the other components in the sequence.
    "transitions": [ // Assistant chooses a transition for the component. The Assistant chooses a transition that looks good with the other components in the sequence.
        {{
            "type": "entry",
            "frameFrom": 60,
            "valueFrom": 200,
            "valueTo": 0
        }},
        {{
            "type": "entryTranslateY",
            "frameFrom": 60,
            "valueFrom": -100,
            "valueTo": 0
        }}
    ]
}}

3. The template for the new JSON for text component to be created is as follows: 
{{
    "name": "Card_1_Text", // Name of the text
    "type": "text", // Type of the component. Should be text for this text component
    "text": "Cat has a soft body", // This is based on generatedText from the sequence_input
    "durationInFrames": 100, // This is based on the durationInFrames from the sequence_input
    "style": {{ // Assistant chooses a style for the text component using CSS. The Assistant chooses a style that looks good with the other components in the sequence.
        "position": "absolute", // Assistant chooses a position for the component. This is CSS position.
        "width": "744px", // Assistant chooses a width for the component. This is CSS width. The width should be within the width from the sequence_input.
        "height": "117px", // Assistant chooses a height for the component. This is CSS height. The height should be within the height from the sequence_input.
        "left": "167px", // Assistant chooses a left for the component. This is CSS left.
        "top": "660px", // Assistant chooses a top for the component. This is CSS top.
       "fontFamily": "Nunito", // Assistant chooses a fontFamily for the component. This is CSS fontFamily.
    "fontStyle": "normal", // Assistant chooses a fontStyle for the component. This is CSS fontStyle. Possible values are normal, italic.
    "fontWeight": 400, // Assistant chooses a fontWeight for the component. This is CSS fontWeight. Possible values are 400, 500, 600, 700, 800, 900.
    "fontScript": "latin", // Assistant chooses a fontScript for the component. This is CSS fontScript. Possible values are based on the fontFamily, fontStyle and fontWeight of the google font. This is mandatory.
    "fontSize": "24px", // Assistant chooses a fontSize for the component. This is CSS fontSize.
    "lineHeight": "29px", // Assistant chooses a lineHeight for the component. This is CSS lineHeight.
    "display": "flex", // Assistant chooses a display for the component. This is CSS display.
    "alignItems": "center", // Assistant chooses a alignItems for the component. This is CSS alignItems.
    "textAlign": "center",  // Assistant chooses a textAlign for the component. This is CSS textAlign.
    "color": "#ffffff", // Assistant chooses a color for the component. This is CSS color.
    "justifyContent": "center", // Assistant chooses a justifyContent for the component. This is CSS justifyContent.
    "background": "#ffffff00" // Assistant chooses a background color for the component. The Assistant chooses a background color that looks good with the other components in the sequence. The background color can be transparent.
    }},
    "tailwindClasses": "", // Assistant chooses a tailwind class for the component. The Assistant chooses a tailwind class that looks good with the other components in the sequence.
    "transitions": [ // Assistant chooses a transition for the component. The Assistant chooses a transition that looks good with the other components in the sequence.
        {{
            "type": "entry",
            "frameFrom": 60,
            "valueFrom": 200,
            "valueTo": 0
        }},
        {{
            "type": "entryTranslateY",
            "frameFrom": 60,
            "valueFrom": -100,
            "valueTo": 0
        }}
    ]
}}


sequence_input :
{sequence_input}

output:


"""


seq_prompt_template = PromptTemplate(input_variables=["user_query", "video_input"], template=sequence_template)

# formatted_prompt = prompt_template.format(user_input="Generate a video of duration 900 frames in 30 fps with width: 1080 and height: 1920. The video is about five facts about cats.", video_input="")

seq_llm_chain = LLMChain(prompt=seq_prompt_template, llm=llm)
# Export the chain for using it in another 
""" seq_output = seq_llm_chain.run(video_input='', user_query="Generate a video of duration 900 frames in 30 fps with width: 1080 and height: 1920. The video is about five facts about cats.")
# Convert output string to JSON
seq_output_json = json.loads(seq_output)

# Convert JSON to dict
seq_output_dict = dict(seq_output_json)
print(f"output_dict: {seq_output_dict['videoInput']}") """

def get_seq_as_dict(video_input, user_query) -> dict:
    print(f"Getting sequence as dict for video_input and user_query")
    seq_output_temp = seq_llm_chain.run(video_input=video_input, user_query=user_query)
    # Trim the output to get only the JSON
    seq_output_temp = seq_output_temp[seq_output_temp.find("{"):]
    print(f"seq_output_temp: {seq_output_temp}")

    seq_output_json = json.loads(seq_output_temp)
    seq_output_dict = dict(seq_output_json)
    return seq_output_dict

# seq_output = get_seq_as_dict(video_input='', user_query="Generate a video of duration 800 frames in 30 fps with width: 1080 and height: 1920. The video is about 3 facts about cats.")

# print(f"seq_output: {seq_output}")

def get_components_from_seq(seq_output) -> dict:
  for seq in seq_output['videoInput']['sequences']:
    print(f"Sequence No: {seq['sequenceNo']}")
    print(f"Sequence: {seq}")
    # Convert seq dict to json
    seq_json = json.dumps(seq)
    # Seq_json to string
    seq_json_str = str(seq_json)
    comp_prompt_template = PromptTemplate(input_variables=["sequence_input"], template=comp_template)
    comp_llm_chain = LLMChain(prompt=comp_prompt_template, llm=llm)
    comp_output = comp_llm_chain.run(sequence_input=seq_json_str)
    print(f"comp_output: {comp_output}")
    comp_output_json = json.loads(comp_output)
    print(f"comp_output_json: {comp_output_json}")
    # Convert comp_output_json to dict
    comp_output_dict = dict(comp_output_json)
    print(f"comp_output_dict: {comp_output_dict}")
    print(f"Seq at point: {seq_output['videoInput']['sequences'][seq['sequenceNo']-1]}")
    # Replace the sequence in the seq_output with the comp_output_dict
    seq_output['videoInput']['sequences'][seq['sequenceNo']-1] = comp_output_dict
  return seq_output
  
# final_seq_output = get_components_from_seq(seq_output)
# print(f"final_seq_output: {final_seq_output}")

def get_video_sequences(video_input, user_query) -> dict:
  seq_output = get_seq_as_dict(video_input, user_query)
  final_seq_output = get_components_from_seq(seq_output)
  print(f"final_seq_output: {final_seq_output}")
  return final_seq_output






