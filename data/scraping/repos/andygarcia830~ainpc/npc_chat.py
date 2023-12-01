# Copyright (c) 2023, Xurpas Inc. and contributors
# For license information, please see license.txt

import frappe,json,os,openai,re
import ainpc.ai_npc.doctype.world.world
from frappe.utils import get_site_name
from frappe.model.document import Document
from llama_index import StorageContext, load_index_from_storage,LLMPredictor,ServiceContext, PromptHelper
from langchain.llms import OpenAI
from langchain.callbacks.base import BaseCallbackHandler

class NPCChat(Document):
	pass

class NPCResponseHandler(BaseCallbackHandler):
	def __init__(self,session):
		self.session=session
		print(f'NPCResponseHandler INITIALIZED with session '+session)

	def on_llm_new_token(self, token: str, **kwargs) -> None:
		print(f"NPCResponseHandler token: {token}")
		frappe.publish_realtime(self.session,token)


def api_key():
	settings = frappe.get_doc('AI NPC OpenAI Settings')
	os.environ["OPENAI_API_KEY"] = settings.openai_api_key
			
	key=os.environ["OPENAI_API_KEY"]
    # print(f'OPENAI KEY={key}')
	openai.api_key=key
	return settings.model


@frappe.whitelist()
def ask_question(world,player,npc,mood,msg,jsonStr,gpt_session):
	llm = api_key()
	print(f'GPT SESSION={gpt_session}')
	jsonDict=json.loads(jsonStr)
	print(f'WORLD={world}')
    #only remember the last 10 questions.
	length = len(jsonDict)
	print(f'JSON LENGTH={length}')
	ctr = 0
	chatBuffer = []
	for item in jsonDict:
		if ctr >= length-10:
			chatBuffer.append(item)
		ctr+=1
	#fetch existing interaction
	playerdoc=frappe.get_doc('Player',player)
	child = frappe.new_doc("NPC Interaction")
	found = False
	
	npcdoc=frappe.get_doc('NPC',npc)
	index_path = npcdoc.data_path
	
	for item in playerdoc.npc_interaction:
		print(f'NPC INTERACTION={item}')
		if item.npc == npc:
			print(f'FOUND NPC Interaction {item.npc}')
			child = item
			found = True
			if item.indexed == 'TRUE':
				index_path=item.data_path

	
	if not found:
		playerdoc.append('npc_interaction',child)
		worlddoc=frappe.get_doc('World',world)
		worldname = re.sub('[^0-9a-zA-Z]+', '_',worlddoc.name)
		npcname = re.sub('[^0-9a-zA-Z]+', '_',npc)
		playername = re.sub('[^0-9a-zA-Z]+', '_',player)
		

		path=get_site_name(frappe.local.request.host)+'/private/files/'+worldname+'/players/'+playername+'/'+npcname
		try:
			os.makedirs(path)
		except:
			pass
		child.update({
		'npc': npc,
		'data_path': path,
		#'conversation_log': json.dumps(jsonDict)
		})

	maxWordCount=frappe.get_doc('AI NPC OpenAI Settings').max_response_word_count
	if maxWordCount == None or maxWordCount <= 0:
		maxWordCount=100

	

	print(f'PATH={index_path}')

	print(f'CHAT BUFFER LENGTH={len(chatBuffer)}')
	personality=npcdoc.personality
	add_inst=npcdoc.additional_instruction
	prompt='You are a '+npcdoc.description+' named '+npcdoc.name1+'.\n'
	
	prompt=prompt+f'Only give information from the documents provided.\nLimit your answers to less than {maxWordCount} words long.\n' 
	prompt=prompt+f'Speak in a way a {personality} does.\nSay your answers in a {mood} manner.\n'
	prompt=prompt+'Occasionally ask questions about the person talking to you.\n Do not introduce yourself unless asked.\n'
	if add_inst != None and add_inst != '':
		prompt=prompt+add_inst+'\n'
	if found:
		prompt=prompt+'The person you are speaking to is a '+playerdoc.description+' named '+playerdoc.name1+'.\n'
	prompt=prompt+'Chat History: '+json.dumps(chatBuffer)+'\n'
	prompt=prompt+'Answer in the same language the question was asked.\n'
	prompt=prompt+'Question: '+msg
	print(f'PROMPT={prompt}')
	
	storage_context = StorageContext.from_defaults(persist_dir=index_path)
	index = load_index_from_storage(storage_context)
	print(f'INDEX={index} LLM={llm}')

	# GPT-4 Service Context
	gpt_predictor = LLMPredictor(llm=OpenAI(temperature=0.7, model_name=llm, streaming=True, callbacks=[NPCResponseHandler(gpt_session)]))

	# define prompt helper
	# set maximum input size
	max_input_size = 8191
	# set number of output tokens
	num_output = 256
	# set maximum chunk overlap
	max_chunk_overlap = 0.5
	gpt_prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)
	
	service_context = ServiceContext.from_defaults(llm_predictor=gpt_predictor, prompt_helper=gpt_prompt_helper)

	response = index.as_query_engine(service_context=service_context).query(prompt)
	answer=response.response
	if answer.startswith('Answer:'):
		answer= answer[len('Answer:')::1]
	jsonDict.append((msg,answer.strip('\n')))
	
	
	playerdoc.save()
	return jsonDict


@frappe.whitelist()
def fetch_worlds():
	worlddoc = frappe.db.get_all('World')
	result = ''
	for item in worlddoc:
		result = result + item.name +'\n'
	result = result.strip('\n')
	return result


@frappe.whitelist()
def fetch_players(world):
	players = frappe.db.get_list('Player',filters={
        'world': world
    })
	print(f'PLAYERS={players}')
	result = ''
	for item in players:
		print(f'PLAYER={item.name}')
		result = result + item.name +'\n'
	result = result.strip('\n')
	return result
			

@frappe.whitelist()
def fetch_npcs(world):
    worlddoc=frappe.get_doc('World',world)
    # npcs = frappe.db.get_all('NPC')
    # print(f'NPCS={npcs}')
    result = ''
    for item in worlddoc.npcs:
          print(f'NPC={item.name}')
          result = result + item.name +'\n'
    result = result.strip('\n')
    return result

@frappe.whitelist()
def fetch_npc_list(world):
	worlddoc=frappe.get_doc('World',world)
	# npcs = frappe.db.get_all('NPC')
	# print(f'NPCS={npcs}')
	npclist=[]
	for item in worlddoc.npcs:
		npc={}
		npc['name']=item.name
		npc['description']=item.description
		npc['personaliity']=item.personality
		npclist.append(npc)


	return npclist

@frappe.whitelist()
def fetch_personality(npc):
    npcdoc = frappe.get_doc('NPC',npc)
    return npcdoc.personality

@frappe.whitelist()
def fetch_description(npc):
    npcdoc = frappe.get_doc('NPC',npc)
    return npcdoc.description


@frappe.whitelist()
def fetch_additional_instruction(npc):
    npcdoc = frappe.get_doc('NPC',npc)
    return npcdoc.additional_instruction



@frappe.whitelist()
def save(world,player,npc,jsonStr):
	playerdoc=frappe.get_doc('Player',player)
	print(f'JSONSTR={jsonStr}')
	for item in playerdoc.npc_interaction:
		if item.npc == npc:
			if item.conversation_log == None or item.conversation_log == '':
				item.conversation_log='[]'
			jsonDict=json.loads(item.conversation_log)
			print(f'JSONDICT={jsonDict}')
			# if len(jsonDict) > 0:
			# 	jsonDict=jsonDict[0]
			newChats=json.loads(jsonStr)
			# if len(newChats) > 0:
			# 	newChats=newChats[0]
			for item3 in newChats:
				print(f'ITEM3={item3}')
				jsonDict.append(item3)
			item.update({
				'conversation_log': json.dumps(jsonDict)
			})
			playerdoc.save()
			worlddoc=frappe.get_doc('World',world)
			npcdoc=frappe.get_doc('NPC',item.npc)
			history=''
			ctr = 1
			for item2 in jsonDict:
				print(f'ITEM2={item2}')
				#history=history+player+':'+item2[0]+'\n'#+'You:'+item2[1]+'\n'
				history=history+'-"'+item2[0]+'".\n'
				
			# history='"'+history+'"'
			#traintext=worlddoc.lore+'\n\n\nThis is the story of '+npc+':'+npcdoc.story+'\n\n\n'+npc+' is friends with '+player+'. These are the things '+player+' has told '+npc+':\n'+history+'\n\n\n'
			story = npcdoc.story
			if story == None:
				story=''
			lore = worlddoc.lore
			if lore == None:
				lore=''
		
			traintext=lore+'\n\n\nThis is the story of '+npc+':\n'+story+'\n\n\nMy name is '+player+'. These are the things I have told you so far:\n'+history+'\n\n\n'
			#traintext=worlddoc.lore+'\n\n\nThis is the story of '+npc+':'+npcdoc.story+'\n\n\n'+history+'\n\n\n'
			traindir=item.data_path+'/train'
			try:
				os.makedirs(traindir)
			except:
				pass
			try:
				os.remove(item.data_path+'/docstore.json')
				os.remove(item.data_path+'/index_store.json')
				os.remove(item.data_path+'/vector_store.json')
			except:
				pass
			f = open(traindir+'/backstory.txt', "a")
			f.write(traintext)
			f.close()
			ainpc.ai_npc.doctype.world.world.construct_index(traindir,item.data_path)
			item.update({
				'indexed': 'TRUE'
			})
			playerdoc.save()

