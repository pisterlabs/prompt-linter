import base64
from langchain import PromptTemplate, LLMChain
from langchain import HuggingFacePipeline
from transformers import logging
from dotenv import dotenv_values
from utils.init_models import loadModelAndTokenizer
import json
import os
from utils.templating import setTemplate
from utils.edgeTTS import run_tts
from moe.main import synthesize
from exllama.model import ExLlamaCache
from exllama.generator import ExLlamaGenerator
import os
import glob


from typing import Union
from fastapi import APIRouter, FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from schemas import ExllamaCfg, UpdateLlm, SystemSchema, ChatModel, UpdateTtsModel

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

current_path = os.path.dirname(os.path.realpath(__file__))
project_path = os.path.abspath(os.getcwd())

def loadConfigs():
    global memories, character, llm_settings, llm_loader_settings, tts_settings, memories
    global MODEL_TYPE, MODEL_LOADER, LANGUAGE, SPEED, SPEAKER_ID, VOICE, PITCH, RATE, VOLUME

    with open(os.path.join(project_path, "configs/llm_loader_settings.json"), "r") as f:
        f.seek(0)  # Move to the beginning of the file
        llm_loader_settings = json.loads(f.read())

    with open(os.path.join(project_path, "configs/character.json"), "r") as f:
        f.seek(0)  # Move to the beginning of the file
        character = json.loads(f.read())

    with open(os.path.join(project_path, "configs/memories.json"), "r") as f:
        f.seek(0)  # Move to the beginning of the file
        memories = json.loads(f.read())

    with open(os.path.join(project_path, "configs/llm_settings.json"), "r") as f:
        f.seek(0)  # Move to the beginning of the file
        llm_settings = json.loads(f.read())

    with open(os.path.join(project_path, "configs/tts_settings.json"), "r") as f:
        f.seek(0)  # Move to the beginning of the file
        tts_settings = json.loads(f.read())

    MODEL_TYPE = llm_loader_settings['model_type']
    MODEL_LOADER = llm_loader_settings['model_loader']

    # MoeTTS Settings
    LANGUAGE = tts_settings['language']
    SPEED = tts_settings['speed']
    SPEAKER_ID = tts_settings['speaker_id']

    # edgeTTS Settings
    VOICE = tts_settings['voice']
    PITCH = tts_settings['pitch']
    RATE = tts_settings['rate']
    VOLUME = tts_settings['volume']

def saveReply(question, bot_response):
    
    replace_name_reply = bot_response.replace('<USER>', memories['MC_name'])
    print(f"{character['char_name']}:{replace_name_reply}")

    # Insert the chat history
    memories['history'].append(f"You: {question}")
    memories['history'].append(f"{character['char_name']}:{replace_name_reply}")

    # Save the chat history to a JSON file
    with open(os.path.join(project_path, "configs/memories.json"), "w", encoding='utf-8') as outfile:
        json.dump(memories, outfile, ensure_ascii=False, indent=2)

    synthesize(text=LANGUAGE+replace_name_reply+LANGUAGE, speed=float(SPEED), out_path="reply.wav", speaker_id=int(SPEAKER_ID))


##### LLM INIT #####
llm_init = APIRouter(
  prefix="/init",
  tags=["Initialize LLM models and configs"],
  responses={404: {"description": "Not found"}},
)
@llm_init.get("/configs")
def load_configs():
    # init configs
    loadConfigs()
    return {
        "llm_settings": llm_settings,
        "llm_loader_settings": llm_loader_settings,
        "character": character,
        "memories": memories,
        "tts_settings": tts_settings
    }

@llm_init.get("/model")
def init_models():

    global init_model, model, tokenizer
    global MODEL_NAME_OR_PATH

    # load ENV
    env = dotenv_values(".env")
    MODEL_NAME_OR_PATH = env['MODEL_NAME_OR_PATH']

    if "/" not in MODEL_NAME_OR_PATH:
        MODEL_NAME_OR_PATH = os.path.abspath(os.path.join("models/LLM", MODEL_NAME_OR_PATH))
        st_pattern = os.path.join(MODEL_NAME_OR_PATH, "*.safetensors")
        try:
            MODEL_BASENAME = glob.glob(st_pattern)[0] # find all files in the directory that match the * pattern
        except:
            MODEL_BASENAME=None

        init_model = loadModelAndTokenizer(model_name_or_path=MODEL_NAME_OR_PATH, model_basename=MODEL_BASENAME)
        model = init_model["model"]
        tokenizer = init_model["tokenizer"]
        return {
            "success": True,
            "message": "Model loaded successfully",
        }
    else:
        raise HTTPException(
            status_code=404, 
            detail="The models can only load inside the models/LLM folder, please remove any slashes '/' in MODEL_NAME_OR_PATH in .env"
        )


##### LLM ROUTER #####
llm_router = APIRouter(
  prefix="/llm",
  tags=["Chat with me (:>_<:)"],
  responses={404: {"description": "Not found"}},
)

@llm_router.post("/chat")
def chat(ChatModel: ChatModel):
    try:
        if MODEL_TYPE == "GPTQ":
            if MODEL_LOADER == "AutoGPTQ":
                model.seqlen = 4096
                # Prevent printing spurious transformers error when using pipeline with AutoGPTQ
                logging.set_verbosity(logging.CRITICAL)

                pipeline = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_length=int(llm_settings['max_length']),
                    max_new_tokens=int(llm_settings['max_new_tokens']),
                    temperature=float(llm_settings['temperature']),
                    top_p=float(llm_settings['top_p']),
                    typical_p=float(llm_settings['typical_p']),
                    repetition_penalty=float(llm_settings['repetition_penalty']),
                    penalty_alpha=float(llm_settings['penalty_alpha']),
                    do_sample=float(llm_settings['do_sample'])
                )
                llm = HuggingFacePipeline(pipeline=pipeline)

                question = ChatModel.questions
                
                template = setTemplate() # set and execute the right template of the models

                # prompt = PromptTemplate(template=template, input_variables=["question"]) # generate the prompt
                prompt = PromptTemplate.from_template(template)
                prompt.format(question=question)

                # using pipeline from Langchain
                llm_chain = LLMChain(prompt=prompt, llm=llm) # create a chain
                bot_reply = llm_chain.run(question) # run the chain

                # saveReply(question, bot_reply)
                replace_name_reply = str(bot_reply).replace('<USER>', memories['MC_name'])

                print(f"{character['char_name']}:{replace_name_reply}")

                # Insert the chat history
                memories['history'].append(f"You: {question}")
                memories['history'].append(f"{character['char_name']}:{replace_name_reply}")

                # Save the chat history to a JSON file
                with open(os.path.join(project_path, "configs/memories.json"), "w", encoding='utf-8') as outfile:
                    json.dump(memories, outfile, ensure_ascii=False, indent=2)

                if tts_settings['tts_type'] == "MoeTTS":
                    # MoeTTS
                    synthesize(text=LANGUAGE+replace_name_reply+LANGUAGE, speed=float(SPEED), out_path="reply.wav", speaker_id=int(SPEAKER_ID))

                elif tts_settings['tts_type'] == "EdgeTTS":
                    # "voice": "en-US-AnaNeural",
                    # "pitch": "+0Hz",
                    # "rate":"+0%",
                    # "volume": "+0%"
                    # EdgeTTS
                    run_tts(
                        replace_name_reply,
                        VOICE,
                        RATE,
                        VOLUME,
                        PITCH,
                        output_file="reply.wav"
                    )

                file_path = os.path.join(project_path, "reply.wav")

                try:
                    with open(file_path, "rb") as audio_file:
                        audio_content = base64.b64encode(audio_file.read()).decode("utf-8")

                    response_data = {
                        "question": question,
                        "reply_text": replace_name_reply,
                        "reply_audio": audio_content
                    }
                    return JSONResponse(content=response_data)
                except FileNotFoundError:
                    raise HTTPException(status_code=404, detail="File not found")
                
            elif MODEL_LOADER == "ExLlama":
                # create cache for inference
                cache = ExLlamaCache(model)
                generator = ExLlamaGenerator(model, tokenizer, cache)   # create generator

                # Configure generator
                # generator.disallow_tokens([tokenizer.eos_token_id])
                generator.settings.token_repetition_penalty_max = float(llm_settings['token_repetition_penalty_max'])
                generator.settings.temperature = float(llm_settings['temperature'])
                generator.settings.top_p = float(llm_settings['top_p'])
                generator.settings.top_k = int(llm_settings['top_k'])
                generator.settings.typical = float(llm_settings['typical'])
                generator.settings.beams = int(llm_settings['beams'])
                generator.settings.beam_length = int(llm_settings['beam_length'])
                generator.settings.token_repetition_penalty_sustain = int(llm_settings['token_repetition_penalty_sustain'])
                generator.settings.token_repetition_penalty_decay = int(llm_settings['token_repetition_penalty_decay'])

                question = ChatModel.questions
                template = setTemplate() # set and execute the right template of the models
                prompt = template.format(question=question)

                print("max_new_tokens:", llm_settings['max_new_tokens'])

                output = generator.generate_simple(prompt, max_new_tokens=int(llm_settings['max_new_tokens']))

                replace_name_reply = str(output[len(prompt):]).replace('<USER>', memories['MC_name'])

                print(f"{character['char_name']}:{replace_name_reply}")

                # Insert the chat history
                memories['history'].append(f"You: {question}")
                memories['history'].append(f"{character['char_name']}:{replace_name_reply}")

                # Save the chat history to a JSON file
                with open(os.path.join(project_path, "configs/memories.json"), "w", encoding='utf-8') as outfile:
                    json.dump(memories, outfile, ensure_ascii=False, indent=2)

                if tts_settings['tts_type'] == "MoeTTS":
                    # MoeTTS
                    synthesize(text=LANGUAGE+replace_name_reply+LANGUAGE, speed=float(SPEED), out_path="reply.wav", speaker_id=int(SPEAKER_ID))

                elif tts_settings['tts_type'] == "EdgeTTS":
                    # "voice": "en-US-AnaNeural",
                    # "pitch": "+0Hz",
                    # "rate":"+0%",
                    # "volume": "+0%"
                    # EdgeTTS
                    run_tts(
                        replace_name_reply,
                        VOICE,
                        RATE,
                        VOLUME,
                        PITCH,
                        output_file="reply.wav"
                    )

                file_path = os.path.join(project_path, "reply.wav")

                try:
                    with open(file_path, "rb") as audio_file:
                        audio_content = base64.b64encode(audio_file.read()).decode("utf-8")

                    response_data = {
                        "question": question,
                        "reply_text": replace_name_reply,
                        "reply_audio": audio_content
                    }

                    return JSONResponse(content=response_data)
                except FileNotFoundError:
                    raise HTTPException(status_code=404, detail="File not found")
        else:
            raise HTTPException(status_code=404, detail=f"Model Type Not Found: {MODEL_TYPE}")
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"{str(e)}, Please Initialize config before chatting")

@llm_router.get("/character")
def get_character():
    try:
        return character
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"{str(e)}, please Initialize the config first")

@llm_router.post("/upload")
def upload_character(file: UploadFile):
    if file.content_type != 'application/json':
        raise HTTPException(status_code=400, detail="Only JSON files are allowed")
    
    key_check = ["char_name", "char_persona", "example_dialogue", "world_scenario"]
    file_json = json.loads(file.file.read().decode("utf-8"))
    
    # Check if all keys are present
    if all(key in file_json for key in key_check):
        with open(os.path.join(project_path, "configs/character.json"), "w", encoding='utf-8') as outfile:
            json.dump(file_json, outfile, ensure_ascii=False, indent=2)
        loadConfigs()
        return FileResponse("configs/character.json", filename="character.json", media_type="application/json")
    else:
        raise HTTPException(status_code=400, detail="Invalid JSON file")

@llm_router.get("/memories")
def get_memories():
    try:
        return memories
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"{str(e)}, please Initialize the config first")

@llm_router.delete("/memories")
def delete_memories():
    try:
        memories['history'] = []
        with open(os.path.join(project_path, "configs/memories.json"), "w", encoding='utf-8') as outfile:
            json.dump(memories, outfile, ensure_ascii=False, indent=2)

        return memories
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"{str(e)}, please Initialize the config first")


##### SETTINGS ROUTER #####
setings_router = APIRouter(
  prefix="/settings",
  tags=["Settings and Configurations"],
  responses={404: {"description": "Not found"}},
)
@setings_router.get("/llm_settings")
def get_llm_settings():
    try:
        return llm_settings
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"{str(e)}, please Initialize the config first")

@setings_router.put("/llm_settings", response_model=UpdateLlm)
def update_llm_settings(llm: UpdateLlm):
    try:
        with open(os.path.join(project_path, "configs/llm_settings.json"), "w", encoding='utf-8') as outfile:
            json.dump(json.loads(llm.json()), outfile, ensure_ascii=False, indent=2)

        # reload configs
        loadConfigs()
        return llm
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"{str(e)}, please Initialize the config first")

@setings_router.get("/llm_loader_settings")
def llm_loader_settings():
    try:
        return llm_loader_settings
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"{str(e)}, please Initialize the config first")

@setings_router.put("/llm_loader_settings", response_model=SystemSchema)
def llm_loader_settings(system: SystemSchema):
    """
    template_type: # for now is 'pygmalion' and 'prompt'
    model_type: # GPTQ
    model_loader: # AutoGPTQ, HuggingFaceBig, ExLlama
    """
    try:
        with open(os.path.join(project_path, "configs/llm_loader_settings.json"), "w", encoding='utf-8') as outfile:
            json.dump(json.loads(system.json()), outfile, ensure_ascii=False, indent=2)

        # reload configs
        loadConfigs()

        return system
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"{str(e)}, please Initialize the config first")

@setings_router.get("/tts_settings")
def get_tts_settings():
    try:
        return tts_settings
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"{str(e)}, please Initialize the config first")

@setings_router.put("/tts_settings", response_model=UpdateTtsModel)
def update_tts_settings(tts: UpdateTtsModel):
    
    try:
        with open(os.path.join(project_path, "configs/tts_settings.json"), "w", encoding='utf-8') as outfile:
            json.dump(json.loads(tts.json()), outfile, ensure_ascii=False, indent=2)

        # reload configs
        loadConfigs()

        return tts
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"{str(e)}, please Initialize the config first")


app.include_router(llm_init)
app.include_router(llm_router)
app.include_router(setings_router)