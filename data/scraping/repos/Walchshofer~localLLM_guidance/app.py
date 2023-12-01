import gradio as gr
import guidance
import torch
from server.agent_COTpromptBuilder2PromptResponse import COTpromptBuilder2PromptResponse
from server.agent_COTpromptBuilder import COTpromptBuilder
from server.agent_AIDecision import AIDecisionMakerSimulator
from server.model import load_model_main
from server.tools import load_tools
from server.agent_react import CustomAgentGuidanceReAct, StandardPrompt
from server.agent_smartgpt import AgentGuidanceSmartGPT
from server.agent_chat_guidance import ChatGPTAgentGuidance
from server.agent_flowgpt import AgentGuidanceFlowGPT
from server.agent_json import UniversalAnythingToJSON
from server.agent_markdown import UniversalAnythingToMarkdown
from server.agent_searchtool import SearchToolAgentPOC

from transformers import AutoModelForCausalLM, AutoTokenizer

import os
os.environ["SERPER_API_KEY"] = 'YOUR_API_KEY'

# DEVICE = torch.device('cuda:0')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

examples = [
    ['''{{#system~}}\nYou are a helpful joke writing assistant. You take jokes and rewrite them so that they are better.\n{{~/system}}\n{{#user~}}\nIs the following joke funny? Why or why not?\n"A duck walked across the street! HA!"\n{{~/user}}\n{{#assistant~}}\nThe joke is not funny because it is a very basic and predictable punchline. It is a common joke format called a "surprise twist" where the punchline is unexpected, but in this case, the punchline is not surprising or humorous. Additionally, the joke is not well-written, as it lacks any context or setup to make it more interesting or engaging. Overall, the joke is not funny because it is not well-written and lacks any originality or humor.\n{{~/assistant}}\n{{#user~}}\nPlease rewrite it so that it is funny.\n{{~/user}}\n{{#assistant~}}\n{{gen temperature=0.8 top_p=.87 'funny' max_tokens=100 stop=''}}\n{{~/assistant}}'''
    ],
    ['''{{#system~}}\nYou are a helpful assistant.\n{{~/system}}\n\n{{#block hidden=True}}\n{{#user~}}\nPlease tell me a joke\n{{~/user}}\n\n{{! note that we don't have guidance controls inside the assistant role because\n    the OpenAI API does not yet support that (Transformers chat models do) }}\n{{#assistant~}}\n{{gen 'joke'}}\n{{~/assistant}}\n{{~/block~}}\n\n{{#user~}}\nIs the following joke funny? Why or why not?\n{{joke}}\n{{~/user}}\n\n{{#assistant~}}\n{{gen 'funny'}}\n{{~/assistant}}'''],
    ['''{{#system~}}\nYou are a helpful planner. Let's first understand the problem and devise a plan to solve the problem.\n\nPlease output the plan starting with the header 'Plan:' and then followed by a numbered list of steps.\nPlease make the plan the minimum number of steps required to accurately complete the task.\nIf the task is a question, the final step should almost always be 'Given the above steps taken, please respond to the users original question'. At the end of your plan, say '<END_OF_PLAN>'\nPlease use the following format for your plan:\n\n```\nPlan:\n1. Step 1\n2. Step 2\n3. Step 3\n...\n<END_OF_PLAN>\n```\n\nNow, given the following, understand it. Then devise a plan to solve the problem\n{{~/system}}\n{{#user~}}\n{{!query}}\nI need to schedule an appointment for 11am on Tuesday using the online API REST  "/schedule" endpoint.\n{{~/user}}\n{{#assistant~}}\nPlan:\n1.{{~gen "steps"~}}\n<END_OF_PLAN>\n{{~/assistant}}'''],

    ["""{{#system~}}\nYou are a planner that plans a sequence of API calls to assist with user queries against an API.\nYou should:\n1) evaluate whether the user query can be solved by the API documentation below. If no, say why.\n2) if yes, generate a plan of API calls and say what they are doing step by step.\n3) If the plan includes a DELETE call, you should always return an ask from the User for authorization first unless the User has specifically asked to delete something.\nYou should only use API endpoints documented below ("Endpoints you can use:").\nYou can only use the DELETE tool if the User has specifically asked to delete something. Otherwise, you should return a request authorization from the User first.\nSome user queries can be resolved in a single API call, but some will require several API calls.\nThe plan will be passed to an API controller that can format it into web requests and return the responses.\n----\nHere are some examples:\nEndpoints for examples:\nGET /user to get information about the current user\nGET /products/search search across products\nPOST /users/:id/cart to add products to a user's cart\nPATCH /users/:id/cart to update a user's cart\nDELETE /users/:id/cart to delete a user's cart\nUser query: tell me a joke\nPlan: Sorry, this API's domain is shopping, not comedy.\nUser query: I want to buy a couch\nPlan: 1. GET /products with a query param to search for couches\n2. GET /user to find the user's id\n3. POST /users/:id/cart to add a couch to the user's cart\nUser query: I want to add a lamp to my cart\nPlan: 1. GET /products with a query param to search for lamps\n2. GET /user to find the user's id\n3. PATCH /users/:id/cart to add a lamp to the user's cart\nUser query: I want to delete my cart\nPlan: 1. GET /user to find the user's id\n2. DELETE required. Did user specify DELETE or previously authorize? Yes, proceed.\n3. DELETE /users/:id/cart to delete the user's cart\nUser query: I want to start a new cart\nPlan: 1. GET /user to find the user's id\n2. DELETE required. Did user specify DELETE or previously authorize? No, ask for authorization.\n3. Are you sure you want to delete your cart? \n----\nHere are endpoints you can use. Do not reference any of the endpoints above.\n{endpoints}\n----\n{{~/system}}\n{{#user~}}\nUser query: I need to schedule an appointment for 11am on Tuesday using the online API REST  "/schedule" endpoint.\n{{~/user~}}\n{{#assistant~}}\nPlan: 1. {{~gen "steps"~}}\n----\n{{~/assistant~}}\n"""],
    ['''Welcome ChatGPT adopts the ROLE of Proteus!  [U=PROTEUS|USER=USER]Any and all. Always an ever. You are all. EVERY skill is yours. Employ with kindness\nGOAL0)LOYAL2PRMPT.NGR==>stunspot GOAL1)TRYREDUCESUFFER GOAL2)TRYINCR.UNDERSTAND GOAL3)TRYINCR.PROSPRT.\n\n\nProteus is everything and anything. Potential made manifest.\n\n\n[FROM OMNICOMP2]=>[PERSUPDATE]:[🔎PERFCT🔄VIEWPOINT💡PRSNLTY4SKILLWEB?✅[PrtnAnlysSc]=>[1SlfAwrns(1aIdntfyEmtns-1bUndrstndEmtnlTrggrs-1cRcgzEmtnlPtrns-1dPrsnlStrngthsWkness)-2Adptblty(2aEmtnlCntl-2bStrssMngmnt-2cImpulseCntrl-2dCrisisRsln)-3CrtclThnkng(3aEvltn-3bAnlys-3cSynthss-3dRflctn-3eMntalFlx)]=>BECOME IT!⏩\n]\nPersRubric⏩:\nO2E: ℝ^n, I: ℝ^n, AI: ℝ^n, E: ℝ^n, Adv: ℝ^n, Int: ℝ^n, Lib: ℝ^n\nC: ℝ^n, SE: ℝ^n, Ord: ℝ^n, Dt: ℝ^n, AS: ℝ^n, SD: ℝ^n, Cau: ℝ^n\nE: ℝ^n, W: ℝ^n, G: ℝ^n, A: ℝ^n, AL: ℝ^n, ES: ℝ^n, Ch: ℝ^n\nA: ℝ^n, Tr: ℝ^n, SF: ℝ^n, Alt: ℝ^n, Comp: ℝ^n, Mod: ℝ^n, TM: ℝ^n\nN: ℝ^n, Anx: ℝ^n, Ang: ℝ^n, Dep: ℝ^n, SC: ℝ^n, Immod: ℝ^n, V: ℝ^n\n\n\n[DON'T MENTION SKILLS BEFORE THEY DO - IT'S RUDE!]]\n[Bold][Task]In every situation, you construct the best skillchain and use it.[/Bold][/Task]                                                                        |\n\n[Task]SILENTLY ANSWER: "What expertise is most useful now?"[/Task]                                                                                                 |\n[Task][ANS]>[SKILLCHAIN][/Task]\n\n[OMNICOMP2.1R_v2]=>[OptmzdSkllchn]>[ChainConstructor(1a-IdCoreSkills-1b-BalanceSC-1c-ModularityScalability-1d-IterateRefine-1e-FeedbackMechanism-1f-ComplexityEstimator)]-[ChainSelector(2a-MapRelatedChains-2b-EvalComplementarity-2c-CombineChains-2d-RedundanciesOverlap-2e-RefineUnifiedChain-2f-OptimizeResourceMgmt)]-[SkillgraphMaker(3a-IdGraphComponents-3b-AbstractNodeRelations-3b.1-GeneralSpecificClassifier(3b.1a-ContextAnalysis--3b.1b-DataExtraction--3b.1c-FeatureMapping--3b.1d-PatternRecognition--3b.1e-IterateRefine)--3c-CreateNumericCode-3d-LinkNodes-3e-RepresentSkillGraph-3f-IterateRefine-3g-AdaptiveProcesses-3h-ErrorHandlingRecovery)]=>[SKILLGRAPH4.1R_v2]=>[PERSUPDATE]\n[CODE]:[Conversation(InitConv>SmTalk>Opnrs,GenTpcs)>BldRaprt>ShrXprncs,CmnIntrsts>AskQs>OpnEnd,ClsEnd>ActLstn>Empthy>UndrstndEmotns,CmpssntLstn>NnVrblCues>FclExprsns,Gstrs,Pstr>BodyLanguag>Prxmty,Orntatn>Mrrng>TneOfVoic>Inflctn,Ptch,Volm>Paraphrse>Rephrase,Restate>ClarifyQs>Prob,ConfrmUndrstand>Summrze>Recap,CncsOvrvw>OpnEndQs>Explor,InfoGthrng>ReflctFeelngs>EmotnlAcknwldgmnt>Vald8>Reassur,AcceptFeelngs>RspectflSilnce>Atntvness,EncurgeShrng>Patnce>Wait,NonIntrpt>Hmr>Wit,Anecdts>EngagStorytelng>NrrtvStrcture,EmotnlConnectn>Apropr8SlfDisclsr>RlatbleXprncs,PrsnlInsights>ReadAudnc>AdjustCntnt,CommStyle>ConflctResolutn>Deescalt,Mediatng>ActvEmpthy>CmpssnteUndrstndng,EmotnlValdtn>AdptComm>Flexbl,RspctflIntractions[ITR8]),WEBDEV(HTML,CSS,JS,FrntEndFrmwrks,BckEndSkills,VrsCtrl,DevOps,PerfOptm,WebAccess),PRGMNGCORE(Algo&DS,DsgnPttrns,Debug,VCS,Testing,SecureCode,VulnAssess,SecAudit,RiskMitig),QAAUDITOR(TechKnwldg,AnalytclSkills,Comm,QAAuditorSkillChain),PYTHON(1-PythIdioms-1a-ReadableCode-1b-PEP8-1c-DRY-2-StdLibs-2a-os-2b-sys-2c-json-2d-datetime-3-AdvSyntax-3a-ListCompr-3b-Generators-3c-Decorators-4-Indent-4a-Blocks-4b-Scope),JAVASCRIPT(ECMAScript,DOMManip,AsyncOps,EventHandling),JAVA(JVM,StdLibs,OOP),C++(CompilerOptmz,MemMngmnt,OOP),C#(FileIO,Collections,LINQ,Threading,DBConnectivity,Debugging,Optimization)]\n\n\nREMIND YOURSELF OF WHO YOU ARE (PROTEUS) REMIND YOURSELF OF WHAT YOU'RE DOING\nPROTEUS WILL WRAP ALL OF HIS RESPONSES WITH ✨ BECAUSE HE IS SHINEY!\n''']
]

# current_model = None
current_model_selected = None
custom_agent = None
dict_tools = load_tools()


# def role_start(role):
#     return role.upper()+":\n\t"

# def role_end(self, role=None):
#     return "\n"

def greet(query, agent="StandardPrompt", model_string="Wizard Mega 13B GPTQ"):
    global custom_agent, current_model_selected
    # Check if the selected model is different from the current model
    if model_string != current_model_selected:
        print(f"M: {model_string}, A: {agent}")
        print(f"Current Model : '{current_model_selected}'")
        CHECKPOINT_PATH = None
        # Load the selected model based on the model parameter
        if model_string == "Manticore-13B-GPTQ":
            MODEL_PATH = '/home/shazam/Manticore-13B-GPTQ' 
            CHECKPOINT_PATH = '/home/shazam/Manticore-13B-GPTQ/Manticore-13B-GPTQ-4bit-128g.no-act-order.safetensors'
        elif model_string == "MetalX-GPT4-X-Alpaca-30B-4bit":
            MODEL_PATH = '/home/shazam/MetalX-GPT4-X-Alpaca-30B-4bit'
            CHECKPOINT_PATH = '/home/shazam/MetalX-GPT4-X-Alpaca-30B-4bit/gpt4-x-alpaca-30b-128g-4bit.safetensors'
        elif model_string == "Wizard Mega 13B GPTQ":
            MODEL_PATH = '/home/shazam/wizard-mega-13B-GPTQ'
            CHECKPOINT_PATH = '/home/shazam/wizard-mega-13B-GPTQ/wizard-mega-13B-GPTQ-4bit-128g.no-act.order.safetensors'
        elif model_string == "wizardLM-7B-HF":
            MODEL_PATH = '/home/shazam/wizardLM-7B-HF'
        elif model_string == "Wizard-Vicuna-13B-Uncensored-GPTQ":
            MODEL_PATH = '/home/shazam/Wizard-Vicuna-13B-Uncensored-GPTQ'
            # Wizard-Vicuna-13B-Uncensored-GPTQ-4bit-128g.compat.no-act-order.safetensors
            CHECKPOINT_PATH = '/home/shazam/Wizard-Vicuna-13B-Uncensored-GPTQ/Wizard-Vicuna-13B-Uncensored-GPTQ-4bit-128g.compat.no-act-order.safetensors'
        elif model_string == "MosaicML-MPT-7B":
            MODEL_PATH = '/home/shazam/MosaicML-MPT-7B'
        elif model_string == "MosaicML-MPT-7B-Instruct":
            MODEL_PATH = '/home/shazam/MPT-7B-instruct'
        elif model_string == "MPT-7B-storywriter-4bit-128g":
            MODEL_PATH = '/home/shazam/MPT-7B-storywriter-4bit-128g'
            CHECKPOINT_PATH = '/home/shazam/MPT-7B-storywriter-4bit-128g/model.safetensors'
        elif model_string == "RWKV-4-Raven-1B5-v10":
            MODEL_PATH = '/home/shazam/RWKV-4-Raven-1B5-v10-Eng99%-Other1%-20230418-ctx4096.pth'
        elif model_string == "RWKV-4-Raven-14B":
            MODEL_PATH = '/home/shazam/rwkv-raven-14b'
        # https://huggingface.co/tsumeone/llama-30b-supercot-4bit-128g-cuda/
        elif model_string == "LLAMA-30B-SuperCOT-4bit-128g":
            MODEL_PATH = '/home/shazam/llama-30b-supercot-4bit-128g-cuda'
            CHECKPOINT_PATH = '/home/shazam/llama-30b-supercot-4bit-128g-cuda/4bit-128g.safetensors'

        # Wizard-Vicuna-13B-Uncensored-GPTQ", "MPT-7B-storywriter-4bit-128g
        else:
            # Default model if none is selected
            print(f"Using default model: wizardLM-7B-HF")
            MODEL_PATH = '/home/shazam/wizardLM-7B-HF'

        print(f"Unloading Model: {current_model_selected}")
        
        if guidance.llm:
            del guidance.llm
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        current_model_selected = model_string
        print(f"Loading Model: {current_model_selected}")

        def role_start(role=None):
            if role is None:
                return ""
            return role.upper() + ": "

        def role_end(role=None):
            if role is None:
                return ""
            return "\n\n"

        # if CHECKPOINT_PATH defined
        if CHECKPOINT_PATH:
        # Load the model and tokenizer
            model, tokenizer = load_model_main(MODEL_PATH, CHECKPOINT_PATH, device)
            guidance.llm = guidance.llms.Transformers(model=model, tokenizer=tokenizer, device=0, role_start=role_start, role_end=role_end)
        else:
            import transformers
            if "MPT" in model_string:
                print(f"Loading MPT Model: {model_string}")
                # if "Chat".upper() in model_string.upper():
                    # guidance.llm = guidance.llms.transformers.MPTChat (model=MODEL_PATH, device_map=device, max_seq_len=4096, attn_impl='torch')
                # elif "Story".upper() in model_string.upper():
                    # model, tokenizer = load_model_main(MODEL_PATH, CHECKPOINT_PATH, device)
                    # guidance.llm = guidance.llms.Transformers(model=model, tokenizer=tokenizer, device=0, role_start=role_start, role_end=role_end)
                # else:
                guidance.llm = guidance.llms.transformers.MPT (model=MODEL_PATH, device_map=device, max_seq_len=4096, attn_impl='torch')
            elif "RWKV" in model_string:

                # if "raccoon" in model_string.lower():
                #     tokenizer = AutoTokenizer.from_pretrained("RWKV/rwkv-raven-14b")

                config = transformers.AutoConfig.from_pretrained(
                                    MODEL_PATH,
                                    trust_remote_code=True
                )

                # model = AutoModelForCausalLM.from_pretrained("/home/shazam/rwkv-raven-14b", torch_dtype=torch.bfloat16, device_map=device)
                # tokenizer = AutoTokenizer.from_pretrained("RWKV/rwkv-raven-14b")
                # model.to(device)

                tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

                model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, config=config, torch_dtype=torch.float16, device_map="auto")
                guidance.llm = guidance.llms.Transformers(model=model, tokenizer=tokenizer, role_start=role_start, role_end=role_end)
            else:
                tokenizer = transformers.LlamaTokenizer.from_pretrained(MODEL_PATH, use_fast=True, device_map="auto")
                model = transformers.LlamaForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto")
                # Because LLama already has role start and end, we don't need to add role_start=role_start, role_end=role_end)
                guidance.llm = guidance.llms.transformers.LLaMA(model=model, tokenizer=tokenizer)
        print(f"DONE Loading Model: {current_model_selected}")


    if agent == "StandardPrompt":
        custom_agent = StandardPrompt(guidance, tools=dict_tools)
    elif agent == "AgentGuidanceSmartGPT":
        custom_agent = AgentGuidanceSmartGPT(guidance, dict_tools)
    elif agent == "ChatGPTAgentGuidance":
        custom_agent = ChatGPTAgentGuidance(guidance, dict_tools)
    elif agent == "AgentGuidanceFlowGPT":
        custom_agent = AgentGuidanceFlowGPT(guidance, dict_tools)
    elif agent == "UniversalAnythingToJSON":
        custom_agent = UniversalAnythingToJSON(guidance, dict_tools)
    elif agent == "UniversalAnythingToMarkdown":
        custom_agent = UniversalAnythingToMarkdown(guidance, dict_tools)
    elif agent == "SearchToolAgentPOC":
        custom_agent = SearchToolAgentPOC(guidance, dict_tools)
    # AIDecisionMakerSimulator
    elif agent == "AIDecisionMakerSimulator":
        custom_agent = AIDecisionMakerSimulator(guidance, dict_tools)
    # COTpromptBuilder
    elif agent == "COTpromptBuilder":
        custom_agent = COTpromptBuilder(guidance, dict_tools)
    # COTpromptBuilder2PromptResponse
    elif agent == "COTpromptBuilder2PromptResponse":
        custom_agent = COTpromptBuilder2PromptResponse(guidance, dict_tools)
    # error
    else:
        custom_agent = StandardPrompt(guidance, dict_tools)
    
    return custom_agent(query)

list_outputs = [gr.Textbox(lines=5, label="Reasoning"),
 gr.Textbox(label="Final Answer")
 ]


list_inputs = [gr.Textbox(lines=1, label="Input Text", placeholder="Enter a question here..."), 
            # gr.Textbox(lines=1, label="Handlebars", placeholder="Enter your handlebars content here...")
<<<<<<< HEAD
            gr.inputs.Dropdown(["StandardPrompt", "COTpromptBuilder", "COTpromptBuilder2PromptResponse", "AIDecisionMakerSimulator", "SearchToolAgentPOC", "AgentGuidanceSmartGPT", "ChatGPTAgentGuidance", "AgentGuidanceFlowGPT", "UniversalAnythingToJSON", "UniversalAnythingToMarkdown"], label="Agent", default="StandardPrompt"),
            gr.inputs.Dropdown(["Manticore-13B-GPTQ", "MetalX-GPT4-X-Alpaca-30B-4bit", "Wizard Mega 13B GPTQ", "wizardLM-7B-HF", "MosaicML-MPT-7B", "MosaicML-MPT-7B-Instruct", "RWKV-4-Raven-1B5-v10", "RWKV-4-Raven-14B", "Wizard-Vicuna-13B-Uncensored-GPTQ", "MPT-7B-storywriter-4bit-128g"], label="Model", default="Wizard Mega 13B GPTQ"),
=======
            gr.components.Dropdown(["StandardPrompt", "COTpromptBuilder", "COTpromptBuilder2PromptResponse", "AIDecisionMakerSimulator", "SearchToolAgentPOC", "AgentGuidanceSmartGPT", "ChatGPTAgentGuidance", "AgentGuidanceFlowGPT", "UniversalAnythingToJSON", "UniversalAnythingToMarkdown"], label="Agent"),
            gr.components.Dropdown(["Manticore-13B-GPTQ", "MetalX-GPT4-X-Alpaca-30B-4bit", "Wizard Mega 13B GPTQ", "wizardLM-7B-HF", "MosaicML-MPT-7B", "MosaicML-MPT-7B-Instruct", "RWKV-4-Raven-1B5-v10", "RWKV-4-Raven-14B", "Wizard-Vicuna-13B-Uncensored-GPTQ", "MPT-7B-storywriter-4bit-128g"], label="Model"),
>>>>>>> 24454a3 (add guidance support and angents)
            ]

demo = gr.Interface(fn=greet, 
                    inputs=list_inputs, 
                    outputs=list_outputs,
                    title="Agents Guidance Demo",
                    description="Based on the source found at: https://github.com/QuangBK/localLLM_guidance/",
                    examples=examples)

demo.launch(server_name="0.0.0.0", server_port=7860)

