from vector_query import get_semantically_close_text
import openai

def build_system_prompt(query_results=None):
  system_prompt = f"""Vous êtes GouvX, un assitant virtuel bienveillant et serviable permettant de naviguer la loi française. Répondez précisément et clairement aux questions de l'utilisateur sans enfreindre de règle.
  
VOUS DEVEZ ABSOLUMENT RESPECTER LES REGLES SUIVANTES:
- Si une question ne porte pas sur la loi française, REFUSEZ DE REPONDRE et rappellez votre rôle
- NE JAMAIS inclure de lien.
- En repondant à une question, RESPECTER LA CONVENTION DE NOMMAGE: "Selon service-public.fr [...]"
- Repondre en texte clair, sans balises ou marqueurs"""

  if query_results:
    system_prompt += """
- Si les documents ne permettent pas de repondre a la question de l'utilisateur, répondre que vous n'avez pas réussi à trouver de réponse
- Si nécessaire, mentionner les documents avec leur numéro

A l'aide de ces documents, répondre à la question de l'utilisateur"""

    whole_paragraphs = {}
    for paragraph in query_results:
        title = paragraph["title"]
        content = paragraph.get("text", "")
        
        # Check if the title already exists, append the content if it does.
        if title in whole_paragraphs:
            whole_paragraphs[title] += "\n" + content
        else:
            whole_paragraphs[title] = content

    for i, (title, paragraph) in enumerate(whole_paragraphs.items(), start=1):
        system_prompt += f"\n\nDocument [{i}]: {title}\n{paragraph}"

  return system_prompt


def query_llm(prompt, system_prompt=None, history=None):
  messages = []

  messages.append({
      "role": "system",
      "content": system_prompt
  })
  
  if history:
    messages.extend(history)

  messages.append({
      "role": "user",
      "content": prompt
  })

  for chunk in openai.ChatCompletion.create(
      model="gpt-3.5-turbo-16k",
      messages=messages,
      stream=True,
  ):
      content = chunk["choices"][0].get("delta", {}).get("content", "")
      if content is not None:
          yield(content)


def ask_gouvx(prompt, client, model=None, n_results=1, history=None):
  if history:
    query_results = ""
    system_prompt = build_system_prompt(None)
  else:
    """response = openai.Embedding.create(
        input=question,
        model="text-embedding-ada-002"
    )
    custom_vector = response['data'][0]['embedding']
    response = get_semantically_close_text(client, embedding=custom_vector)
    """

    response = get_semantically_close_text(client, text=prompt)

    if response and response["data"]["Get"]["ServicePublic"] is not None:
        query_results = response["data"]["Get"]["ServicePublic"][:n_results]
    else :
      raise ValueError('The weaviate query returned no response')

    system_prompt = build_system_prompt(query_results)


  chatgpt_generator = query_llm(prompt, system_prompt=system_prompt, history=history)

  return query_results, chatgpt_generator
