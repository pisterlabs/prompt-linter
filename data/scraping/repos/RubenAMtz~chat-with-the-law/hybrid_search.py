import ast
import numpy as np
from typing import List, Union, Optional, Tuple, Dict
from langchain.docstore.document import Document
import urllib.request
import json
import os
import ssl
import logging
import pathlib
from langchain.llms import AzureOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
import banana_dev as banana
import cohere


# from sentence_transformers import CrossEncoder
from langchain.tools import BaseTool
from langchain.vectorstores import Chroma

from dotenv import load_dotenv

env_path = pathlib.Path(f"../.env.prod")
load_dotenv(dotenv_path=env_path)


class CustomHybridSearch():
	"""
	Gets the relevant documents for a given user input.
	Two step process: 
	- Uses the user input to query the database and retrieve based on similarity search.
	- Uses an LLM model to extract keywords from the user input that are used to query the database.
	Uses a reranker model to rerank the top k documents retrieved from the database.
	"""
	def __init__(self, llm, db, embeddings, logger, top_k_wider_search=100, top_k_reranked_search=10, verbose=False):
		self.llm = llm
		self.db = db
		self.embeddings = embeddings
		self.top_k_wider_search = top_k_wider_search
		self.top_k_reranked_search = top_k_reranked_search
		self.logger = logger
		self.verbose = verbose

		self.keywords = None
		self.top_documents_wider_search_by_keywords = None
		self.top_documents_reranked_search_by_keywords = None
		self.top_documents_wider_search_by_embeddings = None
		self.top_documents_reranked_search_by_embeddings = None
		
		# set up logging
		self.logger.debug("Initialized CustomHybridSearch")

	def augment_user_input(self, user_input):
		"""
		Augments the user input following the HyDE method.
		"""
		return self.llm.generate([f"Eres un abogado altamente calificado, un cliente te pregunta: {user_input}. Tu le respondes: "])\
			.generations[0][0].text.replace('\n', '')  
	
	def extract_keywords(self, user_input):
		"""
		Extract keywords from the user input using the LLM model.
		"""
		keywords = self.llm.generate([f"""Estoy buscando extraer las palabras clave más importantes del siguiente texto: 
		================
		{user_input}
		================
		Con estas palabras clave realizaré una búsqueda en la base de datos. Asegúrate que el orden de importancia refleje el contexto del texto.
		Contéstame con una lista de tuplas (palabra, score) ordenadas por score.

		Por ejemplo:
		================
		No hay una respuesta única a esta pregunta, ya que depende de la situación. Si el cliente está preguntando sobre la legalidad de conducir un vehículo sin llantas, entonces la respuesta es que depende de la jurisdicción. En algunos estados, conducir un vehículo sin llantas es ilegal, mientras que en otros no lo es. Si el cliente está preguntando sobre la seguridad de conducir un vehículo sin llantas, entonces la respuesta es que no es seguro. La falta de llantas afecta la estabilidad y el control del vehículo, lo que puede resultar en un accidente.
		================
		[('vehículo', 0.9), ('llantas', 0.8), ('conducir', 0.7), ('accidente', 0.6), ('estados', 0.5), ('jurisdicción', 0.4), ('seguridad', 0.3), ('estabilidad', 0.2), ('control', 0.1)]

		================
		Es importante que entienda que manejar bajo los efectos del alcohol es un delito grave en la mayoría de los estados. Esto significa que si es arrestado por manejar bajo la influencia del alcohol, puede enfrentar multas, prisión y la suspensión de su licencia de conducir. Por lo tanto, le recomiendo que no maneje bajo los efectos del alcohol.
		================
		[('alcohol', 0.9), ('manejar', 0.8), ('multas', 0.7), ('prisión', 0.6), ('suspensión', 0.5), ('licencia', 0.4), ('conducir', 0.3), ('influencia', 0.2), ('recomiendo', 0.1)]

		================
		{user_input}
		================

		"""]).generations[0][0].text.replace('\n', '')  # HyDE method
		self.keywords = ast.literal_eval(keywords)
		if self.verbose:
			self.logger.info(keywords)
	
	def top_k_documents_by_keyword(self, user_input):
		"""
		Searches the database for the top k documents that contain the keywords
		using the embeddings of the keywords.
		"""
		documents_by_keyword = self.db._collection.get(where_document={"$or": [{"$contains": self.keywords[0][0]}, {"$contains": self.keywords[1][0]}]}, include=["metadatas", "documents", "embeddings"])
	
		# do cosine similarity between the user input and the document embeddings
		def cosine_similarity(a, b):
			return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
		
		# get the embeddings of the user input
		user_input_embedding = self.embeddings.embed_query(user_input)

		# get the cosine similarity between the user input and the documents
		scores = []
		for idx, document in enumerate(documents_by_keyword["embeddings"]):
			scores.append(cosine_similarity(user_input_embedding, document))

		# convert to numpy array
		scores = np.array(scores)

		# get the top k scores
		top_k = min(self.top_k_wider_search, len(scores))
		# top_k_scores = torch.topk(torch.from_numpy(scores), top_k)
		top_k_scores = np.argsort(scores)[::-1][:top_k]
		top_k_scores = (scores[top_k_scores], top_k_scores)
		# print(top_k_scores)

		# get the top k documents
		top_documents_by_keyword = {"documents":[], "metadatas":[]}
		for score, idx in zip(top_k_scores[0], top_k_scores[1]):
			top_documents_by_keyword["documents"].append(documents_by_keyword["documents"][idx])
			top_documents_by_keyword["metadatas"].append(documents_by_keyword["metadatas"][idx])
		
		# create langchain Document objects
		self.top_documents_wider_search_by_keywords = [Document(page_content=document, metadata=metadata) for document, metadata in zip(top_documents_by_keyword["documents"], top_documents_by_keyword["metadatas"])]

		
	def top_k_documents_by_embedding(self, user_input):
		self.top_documents_wider_search_by_embeddings = self.db.similarity_search(query=user_input, k=self.top_k_wider_search)
		

	def top_reranked_documents(self, user_input: str, top_documents: List[Document]) -> List[Document]:
		"""
		Reranks the top k documents using the reranker model.
		"""
		pairs = []
		for idx, doc in enumerate(top_documents):
			pairs.append([user_input, doc.page_content])
		
		top_k = min(self.top_k_reranked_search, len(pairs))
		# scores = self.reranker_model.predict(pairs, show_progress_bar=True)
		scores = self.call_reranker(pairs)
		# reranked_scores = torch.topk(torch.from_numpy(scores), k=top_k)
		reranked_scores = np.argsort(scores)[::-1][:top_k]
		reranked_scores = (scores[reranked_scores], reranked_scores)

		top_reranked_documents_ = []
		for score, idx in zip(reranked_scores[0], reranked_scores[1]):
			top_reranked_documents_.append((top_documents[idx], score))

		parsed_response = ""
		if self.verbose:
			for r, score in top_reranked_documents_:
				parsed_response += r.page_content
				parsed_response += "\n"
				parsed_response += "Fuente: " + r.metadata['fuente']
				parsed_response += "\n"
				parsed_response += "Nivel de Justicia: " + r.metadata['nivel de justicia']
				parsed_response += "\n"
				parsed_response += "Score: " + str(round(score.item(), 2)) # item is float
				parsed_response += "\n"
			self.logger.info(f"Resultado de la busqueda: \n{parsed_response}")
		
		# return only Document objects
		return [r[0] for r in top_reranked_documents_]
	
	def run(self, user_input) -> List[Document]:
		"""
		Runs the system.
		"""
		self.logger.info(f"Pregunta del usuario: {user_input}")
		user_input = self.augment_user_input(user_input)
		self.logger.info(f"Pregunta del usuario aumentada: {user_input}")
		self.extract_keywords(user_input)
		self.top_k_documents_by_keyword(user_input)
		self.top_k_documents_by_embedding(user_input)
		res = self.top_reranked_documents(user_input, self.top_documents_wider_search_by_keywords)
		res += self.top_reranked_documents(user_input, self.top_documents_wider_search_by_embeddings)

		return res
	
	def call_reranker(self, pairs: List[List[Union[str, Document]]]) -> np.ndarray[float]:

		def allowSelfSignedHttps(allowed):
			# bypass the server certificate verification on client side
			if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
				ssl._create_default_https_context = ssl._create_unverified_context

		allowSelfSignedHttps(True) # this line is needed if you use self-signed certificate in your scoring service.

		# Request data goes here
		# https://docs.microsoft.com/azure/machine-learning/how-to-deploy-advanced-entry-script
		data = {
			"data": {
				"inputs": {
					"source_sentence": pairs[0][0],
					"sentences": [pair[1] for pair in pairs],
					"wait_for_model": True,
				}
			}
		}

		body = str.encode(json.dumps(data))

		# API_URL = 'https://cross-encoder-v1.eastus2.inference.ml.azure.com/score'
		API_URL = os.environ.get("CROSSENCODER_URL", "")
		# Replace this with the primary/secondary key or AMLToken for the endpoint
		# api_key = f'{cross_encoder_azureml_key}'
		api_key = os.environ.get("CROSSENCODER_API_KEY", "")
		if not api_key:
			raise Exception("A key should be provided to invoke the endpoint")
		# The azureml-model-deployment header will force the request to go to a specific deployment.
		# Remove this header to have the request observe the endpoint traffic rules
		headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key), 'azureml-model-deployment': 'blue' }
		req = urllib.request.Request(API_URL, body, headers)
		try:
			response = urllib.request.urlopen(req, timeout=30)
			result = response.read()
			# print(result)
		except urllib.error.HTTPError as error:
			print("The request failed with status code: " + str(error.code))
			# Print the headers - they include the request ID and the timestamp, which are useful for debugging the failure
			print(error.info())
			print(error.read().decode("utf8", 'ignore'))
		
		# logging.info("Output from AzureML (bytes): {}".format(result))
		result = ast.literal_eval(result.decode('utf-8'))
		# logging.info("Output from AzureML: {}".format(result))
		return np.array(result)
	




class CustomHybridSearchTool(BaseTool):
	"""
	Gets the relevant documents for a given user input.
	Two step process: 
	- Uses the user input to query the database and retrieve based on similarity search.
	- Uses an LLM model to extract keywords from the user input that are used to query the database.
	Uses a reranker model to rerank the top k documents retrieved from the database.
	"""
	name: str = "La ley"
	description: str = "Esta herramienta contiene informacion relacionada a todas las leyes del pais, usala para contestar preguntas que esten relacionadas a temas que tengan que ver con la ley, justicia, etc."
	llm: AzureOpenAI = None
	db: Chroma = None
	embeddings: OpenAIEmbeddings = None
	top_k_wider_search: int = None
	top_k_reranked_search: int = None
	logger: logging.Logger = None
	verbose: bool = None
	keywords: List[str] = None
	top_documents_wider_search_by_keywords: List[Document] = None
	top_documents_reranked_search_by_keywords: List[Document] = None
	top_documents_wider_search_by_embeddings: List[Document] = None
	top_documents_reranked_search_by_embeddings: List[Document] = None
	references: str = None
	
	def __init__(self, llm, db, embeddings, logger, top_k_wider_search=100, top_k_reranked_search=10, verbose=False):
		super().__init__(return_direct=True, verbose=True)
		self.llm = llm
		self.db = db
		self.embeddings = embeddings
		self.top_k_wider_search = top_k_wider_search
		self.top_k_reranked_search = top_k_reranked_search
		self.logger = logger
		self.verbose = verbose

		self.keywords = None
		self.top_documents_wider_search_by_keywords = None
		self.top_documents_reranked_search_by_keywords = None
		self.top_documents_wider_search_by_embeddings = None
		self.top_documents_reranked_search_by_embeddings = None
		
		# set up logging
		self.logger.debug("Initialized CustomHybridSearch")

	def augment_user_input(self, user_input):
		"""
		Augments the user input following the HyDE method.
		"""
		return self.llm.generate([f"Eres un abogado altamente calificado, un cliente te pregunta: {user_input}. Tu le respondes: "])\
			.generations[0][0].text.replace('\n', '')  
	
	def extract_keywords(self, user_input):
		"""
		Extract keywords from the user input using the LLM model.
		"""
		keywords = self.llm.generate([f"""Estoy buscando extraer las palabras clave más importantes del siguiente texto: 
		================
		{user_input}
		================
		Con estas palabras clave realizaré una búsqueda en la base de datos. Asegúrate que el orden de importancia refleje el contexto del texto.
		Contéstame con una lista de tuplas (palabra, score) ordenadas por score.

		Por ejemplo:
		================
		No hay una respuesta única a esta pregunta, ya que depende de la situación. Si el cliente está preguntando sobre la legalidad de conducir un vehículo sin llantas, entonces la respuesta es que depende de la jurisdicción. En algunos estados, conducir un vehículo sin llantas es ilegal, mientras que en otros no lo es. Si el cliente está preguntando sobre la seguridad de conducir un vehículo sin llantas, entonces la respuesta es que no es seguro. La falta de llantas afecta la estabilidad y el control del vehículo, lo que puede resultar en un accidente.
		================
		[('vehículo', 0.9), ('llantas', 0.8), ('conducir', 0.7), ('accidente', 0.6), ('estados', 0.5), ('jurisdicción', 0.4), ('seguridad', 0.3), ('estabilidad', 0.2), ('control', 0.1)]

		================
		Es importante que entienda que manejar bajo los efectos del alcohol es un delito grave en la mayoría de los estados. Esto significa que si es arrestado por manejar bajo la influencia del alcohol, puede enfrentar multas, prisión y la suspensión de su licencia de conducir. Por lo tanto, le recomiendo que no maneje bajo los efectos del alcohol.
		================
		[('alcohol', 0.9), ('manejar', 0.8), ('multas', 0.7), ('prisión', 0.6), ('suspensión', 0.5), ('licencia', 0.4), ('conducir', 0.3), ('influencia', 0.2), ('recomiendo', 0.1)]

		================
		{user_input}
		================

		"""]).generations[0][0].text.replace('\n', '')  # HyDE method
		self.keywords = ast.literal_eval(keywords)
		if self.verbose:
			self.logger.info(keywords)

	def extract_relevant_info(self, response, query):
		"""
		Extracts relevant passages from a response using the LLM model.
		"""
		return self.llm.generate([f"""Estoy buscando extraer la información relevante del siguiente texto. Cada parrafo representa un extracto de la ley, los extractos fueron recuperados por una búsqueda, la pregúnta que produjo los extractos fue "{query}", entonces, quiero que respondas con los pedazos de texto que contengan la información necesaria para contestar la pregunta. El texto que me respondas debe ser el texto tal cual aparece en los extractos, no debe ser un resumen. Si se menciona el número de artículo, inclúyelo, asi como "Fuente" y "Nivel de Justicia", ignora "Score", es irrelevante. 

		Por ejemplo:
		================
		Pregunta: Puedo manejar en sentido contrario?
		================
		Extractos:
		vía.  ARTICULO 151.- Queda prohibido invadir un carril de sentido opuesto a la circulación, para rebasar una hilera de vehículos. ARTICULO 152.- El conductor de un vehículo que circule en el mismo sentido que otro, por una vía de 2 carriles y circulación en ambos sentidos, podrán rebasarlo por la izquierda sujetándose a las reglas siguientes: I.- Deberá anunciar su intención con luz direccional y además, con señal audible durante el día y cambio de luces durante la noche; lo pasará por la izquierda a una distancia segura y tratará de volver al carril de la derecha tan pronto como le sea posible, pero hasta alcanzar una distancia razonable y sin obstruir la marcha del vehículo rebasado; II.- Sin perjuicio de lo dispuesto en la fracción anterior, todo conductor debe, antes de efectuar un rebase, cerciorarse de que ningún conductor que le siga ha iniciado la misma maniobra. III.- El conductor de un vehículo que vaya a ser rebasado por la izquierda, deberá
		Fuente: LeydeTránsitodelEstadodeSonora
		Nivel de Justicia: estatal
		Score: 0.08
		el cambio de carril se hará de uno a la vez, transitando por cada uno una distancia considerable antes de pasar al siguiente. IV Hacerlo solamente en lugares donde haya suficiente visibilidad hacia atrás, de tal forma que se pueda observar el tránsito en el carril hacia donde se realiza el cambio. V En calles o avenidas que tengan más de tres carriles de tránsito en un solo sentido, si ocurriera el caso de que dos conductores pretendan cambiar de carril circulando ambos en carriles separados por uno o más carriles, el derecho de acceso al carril que se pretende ocupar será de quien entra de derecha a izquierda. ARTÍCULO 10.- En los lugares donde existan carriles diseñados o señalados para realizar vueltas exclusivamente, queda prohibido el tránsito en sentido contrario al diseñado o señalado. ARTÍCULO 11.- Quedan prohibidas las vueltas en “U”, exceptuándose los casos en que existan carriles de retorno para esa maniobra. ARTÍCULO 12.- Se permite transitar en reversa,
		Fuente: REGLAMENTO_DE_TRANSITO_MUNICIPAL
		Nivel de Justicia: municipal
		Score: -0.0
		================
		Extractos relevantes:
		ARTICULO 151.- Queda prohibido invadir un carril de sentido opuesto a la circulación, para rebasar una hilera de vehículos. ARTICULO 152.- El conductor de un vehículo que circule en el mismo sentido que otro, por una vía de 2 carriles y circulación en ambos sentidos, podrán rebasarlo por la izquierda sujetándose a las reglas siguientes: I.- Deberá anunciar su intención con luz direccional y además, con señal audible durante el día y cambio de luces durante la noche; lo pasará por la izquierda a una distancia segura y tratará de volver al carril de la derecha tan pronto como le sea posible, pero hasta alcanzar una distancia razonable y sin obstruir la marcha del vehículo rebasado; II.- Sin perjuicio de lo dispuesto en la fracción anterior, todo conductor debe, antes de efectuar un rebase, cerciorarse de que ningún conductor que le siga ha iniciado la misma maniobra.
		Fuente: LeydeTránsitodelEstadodeSonora
		Nivel de Justicia: estatal
		Hacerlo solamente en lugares donde haya suficiente visibilidad hacia atrás, de tal forma que se pueda observar el tránsito en el carril hacia donde se realiza el cambio. ARTÍCULO 10.- En los lugares donde existan carriles diseñados o señalados para realizar vueltas exclusivamente, queda prohibido el tránsito en sentido contrario al diseñado o señalado.
		Fuente: REGLAMENTO_DE_TRANSITO_MUNICIPAL
		Nivel de Justicia: municipal
		================
		Pregunta: {query}
		================
		{response}
		================
		Extractos relevantes:
		"""]).generations[0][0].text.replace('\n', '')
	
	def summarize(self, text, user_input):
		"""
		Summarize text using the LLM model.
		"""
		return self.llm.generate([f"""Estoy buscando resumir el siguiente texto. Cada parrafo representa un extracto de la ley, los extractos fueron recuperados por una búsqueda, la pregúnta que produjo los extractos fue "{user_input}", entonces, quiero que el resumen sea relevante a la pregunta.

		Por ejemplo:
		================
		Pregunta: Puedo manejar en sentido contrario?
		================
		Extractos:
		ARTICULO 151.- Queda prohibido invadir un carril de sentido opuesto a la circulación, para rebasar una hilera de vehículos. ARTICULO 152.- El conductor de un vehículo que circule en el mismo sentido que otro, por una vía de 2 carriles y circulación en ambos sentidos, podrán rebasarlo por la izquierda sujetándose a las reglas siguientes: I.- Deberá anunciar su intención con luz direccional y además, con señal audible durante el día y cambio de luces durante la noche; lo pasará por la izquierda a una distancia segura y tratará de volver al carril de la derecha tan pronto como le sea posible, pero hasta alcanzar una distancia razonable y sin obstruir la marcha del vehículo rebasado; II.- Sin perjuicio de lo dispuesto en la fracción anterior, todo conductor debe, antes de efectuar un rebase, cerciorarse de que ningún conductor que le siga ha iniciado la misma maniobra.
		Fuente: LeydeTránsitodelEstadodeSonora
		Nivel de Justicia: estatal
		Hacerlo solamente en lugares donde haya suficiente visibilidad hacia atrás, de tal forma que se pueda observar el tránsito en el carril hacia donde se realiza el cambio. ARTÍCULO 10.- En los lugares donde existan carriles diseñados o señalados para realizar vueltas exclusivamente, queda prohibido el tránsito en sentido contrario al diseñado o señalado.
		Fuente: REGLAMENTO_DE_TRANSITO_MUNICIPAL
		Nivel de Justicia: municipal
		================
		Resumen:
		El artículo 152 que puedes hacerlo sujetándose a las siguientes reglas: 1. Anunciando la intención con luces direccionales y además, con una señal audible durante el día y cambio de luces durante la noche; lo hará siempre por la izquierda y a evaluando una distancia segura y tratará de regresar al carril de la derecha tan pronto como pueda, pero hasta alcanzar una distancia razonable sin darle molestias al vehículo rebasado. 2. Antes de de realizar el rebase, asegurarse que el conductor de detrás, no haya iniciado un rebase.[1] Además, solo se podrá realizar en lugares donde exista suficiente visibilidad[2]. Sin embargo, el artículo 151 indica que queda prohíbido manejar en sentido contrario cuando se intente rebasar a una hilera de vehículos [1], además también se prohíbe hacerlo en lugares diseñados o señalados para realizar vueltas exclusivamente, indicado por el artículo 10[2].
[1] LeydeTránsitodelEstadodeSonora
[2] REGLAMENTO_DE_TRANSITO_MUNICIPAL
		================
		Pregunta: {user_input}
		================
		Extractos:
		{text}
		================
		Resumen:

		"""]).generations[0][0].text.replace('\n', '')
	
	
	def top_k_documents_by_keyword(self, user_input):
		"""
		Searches the database for the top k documents that contain the keywords
		using the embeddings of the keywords.
		"""
		documents_by_keyword = self.db._collection.get(where_document={"$or": [{"$contains": self.keywords[0][0]}, {"$contains": self.keywords[1][0]}]}, include=["metadatas", "documents", "embeddings"])
	
		# do cosine similarity between the user input and the document embeddings
		def cosine_similarity(a, b):
			return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
		
		# get the embeddings of the user input
		user_input_embedding = self.embeddings.embed_query(user_input)

		# get the cosine similarity between the user input and the documents
		scores = []
		for idx, document in enumerate(documents_by_keyword["embeddings"]):
			scores.append(cosine_similarity(user_input_embedding, document))

		# convert to numpy array
		scores = np.array(scores)

		# get the top k scores
		top_k = min(self.top_k_wider_search, len(scores))
		# top_k_scores = torch.topk(torch.from_numpy(scores), top_k)
		top_k_scores = np.argsort(scores)[::-1][:top_k]
		top_k_scores = (scores[top_k_scores], top_k_scores)
		# print(top_k_scores)

		# get the top k documents
		top_documents_by_keyword = {"documents":[], "metadatas":[]}
		for score, idx in zip(top_k_scores[0], top_k_scores[1]):
			top_documents_by_keyword["documents"].append(documents_by_keyword["documents"][idx])
			top_documents_by_keyword["metadatas"].append(documents_by_keyword["metadatas"][idx])
		
		# create langchain Document objects
		self.top_documents_wider_search_by_keywords = [Document(page_content=document, metadata=metadata) for document, metadata in zip(top_documents_by_keyword["documents"], top_documents_by_keyword["metadatas"])]

		
	def top_k_documents_by_embedding(self, user_input):
		self.top_documents_wider_search_by_embeddings = self.db.similarity_search(query=user_input, k=self.top_k_wider_search)
		

	def top_reranked_documents(self, user_input: str, top_documents: List[Document]) -> str:#List[Document]:
		"""
		Reranks the top k documents using the reranker model.
		"""
		pairs = []
		for idx, doc in enumerate(top_documents):
			pairs.append([user_input, doc.page_content])
		
		top_k = min(self.top_k_reranked_search, len(pairs))
		# scores = self.reranker_model.predict(pairs, show_progress_bar=True)
		# scores = self.call_reranker(pairs)
		# scores = self.call_reranker_cohere(pairs)
		scores = self.call_reranker_cohere(pairs)
		# coheres returns ordered list of documents by score
		if isinstance(scores, np.ndarray):
			reranked_scores = np.argsort(scores)[::-1][:top_k]
			reranked_scores = (scores[reranked_scores], reranked_scores)

			top_reranked_documents_ = []
			for score, idx in zip(reranked_scores[0], reranked_scores[1]):
				top_reranked_documents_.append((top_documents[idx], score))
		else:
			top_reranked_documents_ = []
			for index, (score, idx) in enumerate(scores):
				top_reranked_documents_.append((top_documents[idx], score))
				if (index + 1) == top_k:
					break
				

		parsed_response = ""
		if self.verbose:
			for r, score in top_reranked_documents_:
				parsed_response += r.page_content
				parsed_response += "\n"
				parsed_response += "Fuente: " + r.metadata['fuente']
				parsed_response += "\n"
				parsed_response += "Nivel de Justicia: " + r.metadata['nivel de justicia']
				parsed_response += "\n"
				parsed_response += "Score: " + str(round(score, 2)) # item is float
				parsed_response += "\n"
			self.logger.info(f"Resultado de la busqueda: \n{parsed_response}")
		
		# return only Document objects
		# return [r[0] for r in top_reranked_documents_]
		return parsed_response
	
	def _run(self, user_input) -> List[Document]:
		"""
		Runs the system.
		"""
		self.logger.info(f"Pregunta del usuario: {user_input}")
		user_input = self.augment_user_input(user_input)
		self.logger.info(f"Pregunta del usuario aumentada: {user_input}")
		# self.extract_keywords(user_input)
		# self.top_k_documents_by_keyword(user_input)
		self.top_k_documents_by_embedding(user_input)
		# res = self.top_reranked_documents(user_input, self.top_documents_wider_search_by_keywords)
		# relevant_res_keywords = self.extract_relevant_info(res, user_input)
		# res = self.top_reranked_documents(user_input, self.top_documents_wider_search_by_embeddings)
		# relevant_res_similarity = self.extract_relevant_info(res, user_input)
		# res = relevant_res_keywords + relevant_res_similarity
		# relevant_res = self.extract_relevant_info(res, user_input)
		
		# we now mix documents from both searches
		# mixed_documents = self.top_documents_wider_search_by_keywords + self.top_documents_wider_search_by_embeddings
		mixed_documents = self.top_documents_wider_search_by_embeddings
		res = self.top_reranked_documents(user_input, mixed_documents)
		self.references = res
		relevant_res = self.extract_relevant_info(res, user_input)
		res = self.summarize(relevant_res, user_input)
		return res
	
	def call_reranker(self, pairs: List[List[Union[str, str]]]) -> np.ndarray[float]:
		"""
		Reranker using Azure ML.
		"""

		def allowSelfSignedHttps(allowed):
			# bypass the server certificate verification on client side
			if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
				ssl._create_default_https_context = ssl._create_unverified_context

		allowSelfSignedHttps(True) # this line is needed if you use self-signed certificate in your scoring service.

		# Request data goes here
		# https://docs.microsoft.com/azure/machine-learning/how-to-deploy-advanced-entry-script
		data = {
			"data": {
				"inputs": {
					"source_sentence": pairs[0][0],
					"sentences": [pair[1] for pair in pairs],
					"wait_for_model": True,
				}
			}
		}

		body = str.encode(json.dumps(data))

		# API_URL = 'https://cross-encoder-v1.eastus2.inference.ml.azure.com/score'
		API_URL = os.environ.get("CROSSENCODER_URL", "")
		# Replace this with the primary/secondary key or AMLToken for the endpoint
		# api_key = f'{cross_encoder_azureml_key}'
		api_key = os.environ.get("CROSSENCODER_API_KEY", "")
		if not api_key:
			raise Exception("A key should be provided to invoke the endpoint")
		# The azureml-model-deployment header will force the request to go to a specific deployment.
		# Remove this header to have the request observe the endpoint traffic rules
		headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key), 'azureml-model-deployment': 'blue' }
		req = urllib.request.Request(API_URL, body, headers)
		try:
			response = urllib.request.urlopen(req, timeout=20)
			result = response.read()
			# print(result)
		except urllib.error.HTTPError as error:
			print("The request failed with status code: " + str(error.code))
			# Print the headers - they include the request ID and the timestamp, which are useful for debugging the failure
			print(error.info())
			print(error.read().decode("utf8", 'ignore'))
		
		# logging.info("Output from AzureML (bytes): {}".format(result))
		result = ast.literal_eval(result.decode('utf-8'))
		# logging.info("Output from AzureML: {}".format(result))
		return np.array(result)
	
	def call_reranker_banana(self, pairs: List[List[Union[str, str]]]) -> np.ndarray[float]:
		"""
		Re-ranking using banana deployment.
		"""
		#  create object out of pairs
		logging.info("Pairs: {}".format(pairs))
		body = {
			"inputs": {
				"source_sentence": pairs[0][0],
				"sentences": [pair[1] for pair in pairs]
			}
		}
		logging.info("Body: {}".format(body))

		api_key = os.environ.get("BANANA_API_KEY", "")
		model_key = os.environ.get("BANANA_MODEL_KEY", "")
		out = banana.run(api_key, model_key, body)
		return np.array(out["modelOutputs"][0]["outputs"])

	def call_reranker_cohere(self, pairs: List[List[Union[str, str]]]) -> List[Tuple[float, int]]:
		"""
		Re-ranking using coheres re-ranker

		Args:
			pairs (List[List[Union[str, str]]]): list of list, where each list contains the query and the document to be re-ranked
		"""
		# documents should be a list of strings
		# query should be a string
		co = cohere.Client(os.environ.get("COHERE_API_KEY", ""))

		query = pairs[0][0]
		documents = [pair[1] for pair in pairs]
		logging.info("Query: {}".format(query))
		logging.info("Documents: {}".format(documents))
		results = co.rerank(query=query, documents=documents, top_n=10, model='rerank-multilingual-v2.0') # Change top_n to change the number of results returned. If top_n is not passed, all results will be returned.
		for idx, r in enumerate(results):
			print(f"Document Rank: {idx + 1}, Document Index: {r.index}")
			print(f"Document: {r.document['text']}")
			print(f"Relevance Score: {r.relevance_score:.2f}")
			print("\n")
		# return score and index
		return [(r.relevance_score, r.index) for r in results]
	
	async def _arun(self, query: str) -> str:
		"""Use the tool asynchronously."""
		raise NotImplementedError("This tool does not support async")
	
	

