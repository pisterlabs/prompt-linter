"""
Die Smart Chain soll die besten Konzepte von Langchain miteinander kombinieren um die höchstmögliche Qualität bei einer Frage und Antwort Aufgabe erzielen.
Als Grundlage dienen dazu eigene Dokumente welche zuvor in Vectordatenbanken nach Themengebiet persistent gespeichert wurden.
Nachfolgen eine Auflistung der zu verwendenden Module aus Langchain für jeden Aspekt der Kette.

Module

LLM: AzureChatOpenAi
Embeddings: OpenAi Embeddings
Memory: ConversationBuffer Memory, ConversationSummaryMemory & Entity Memory (siehe https://python.langchain.com/en/latest/modules/memory/examples/multiple_memory.html)
DocumentLoaders: Custom Implementation
Index: Chroma, persistant (Combined usage of similarity search and mmr) -> Use https://python.langchain.com/en/latest/modules/chains/generic/router.html#router-chains 
    to decide if query refers to specific details or general knowledge across the whole document.
    Alternatively do both search types in all cases and combine results into single string.

Anschließend werden verschiedene chains miteinader kombiniert um die Antwort des LLMs zu optimieren

Chains:
1. https://python.langchain.com/en/latest/modules/chains/examples/multi_retrieval_qa_router.html#router-chains-selecting-from-multiple-prompts-with-multiretrievalqachain 
    Entscheidung welche Vektordatenbank geeignet für den Kontext der Frage ist

(Self-Critique Chain with Constitutional AI: Entspricht die Antwort den Unternehmensrichtlinien?)
"""




"""
Yes, the decision to use similarity_search or mmr in Chroma would depend on the user's question and the desired outcome.
If the user is looking for a specific detail or information, then similarity_search would be more appropriate as it
returns documents that are most similar to the query based on the chosen distance metric.

On the other hand, if the user wants to summarize a document or retrieve a set of documents that cover different aspects of the topic, then 
mmr would make more sense as it optimizes for both similarity to the query and diversity among selected documents. 

So, it's important to consider the user's question and the desired outcome when choosing between similarity_search and mmr in Chroma.



Ein Beispiel für die Verwendung von ConversationBufferMemory und ConversationEntityMemory in einer Kette in Python könnte wie folgt aussehen:

from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationEntityMemory

llm = OpenAI(temperature=0)
buffer_memory = ConversationBufferMemory()
entity_memory = ConversationEntityMemory(entity_type="person")
conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=[buffer_memory, entity_memory]
)

conversation.predict(input="Hallo, ich bin Max.")

conversation.predict(input="Ich arbeite bei Google.")

conversation.predict(input="Was denkst du über Google?")

In diesem Beispiel wird eine Kette erstellt, die sowohl ConversationBufferMemory als auch ConversationEntityMemory
verwendet. ConversationBufferMemory speichert die gesamte Konversationshistorie, während ConversationEntityMemory
 Informationen über Personen speichert. Die Kette wird verwendet, um eine Konversation mit einem Benutzer zu
 simulieren, der sich selbst als "Max" vorstellt und sagt, dass er bei Google arbeitet. Wenn der Benutzer nach der
Meinung des Modells zu Google fragt, kann das Modell auf die Informationen zugreifen, die es über Google und Max 
gespeichert hat, um eine bessere Antwort zu geben.

"""