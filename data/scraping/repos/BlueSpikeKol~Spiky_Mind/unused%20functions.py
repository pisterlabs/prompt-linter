from gpt_api_old import AI_trainer
from memory_stream_old import MemoryStreamAccess
from memory_stream_old import MemoryObject
import re
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity
import openai.error
import tiktoken
import time
import logging
import openai
import AI_entities as AI
from utils import config_retrieval

openai.api_key = config_retrieval.OpenAIConfig.api_key
memory_stream = MemoryStreamAccess.MemoryStreamAccess()
encoder = tiktoken.encoding_for_model("gpt-3.5-turbo-16k")
def retry_on_openai_error(max_retries=7):
    """
    A decorator to automatically retry a function if it encounters specific OpenAI API errors.
    It uses exponential backoff to increase the wait time between retries.

    Parameters:
    - max_retries (int, optional): The maximum number of times to retry the function. Default is max_retries.

    Usage:
    Decorate any function that makes OpenAI API calls which you want to be retried upon encountering errors.

    Behavior:
    1. If the decorated function raises either openai.error.APIError, openai.error.Timeout, or openai.error.InvalidRequestError,
       the decorator will log the error and attempt to retry the function.
    2. The decorator uses an initial wait time of 5 seconds between retries. After each retry,
       this wait time is doubled (exponential backoff).
    3. If the error contains rate limit headers, the wait time is adjusted to respect the rate limit reset time.
    4. If the function fails after the specified max_retries, the program will pause and prompt
       the user to press Enter to retry or Ctrl+C to exit.

    Returns:
    The result of the decorated function if it succeeds within the allowed retries.

    Raises:
    - openai.error.APIError: If there's an error in the OpenAI API request.
    - openai.error.Timeout: If the OpenAI API request times out.
    - openai.error.InvalidRequestError: If the request to the OpenAI API is invalid.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            wait_time = 5  # Initial wait time of 5 seconds

            while retries <= max_retries:
                try:
                    response = func(*args, **kwargs)
                    return response
                except (openai.error.APIError, openai.error.Timeout, openai.error.InvalidRequestError) as e:
                    logging.error(f"Error calling OpenAI API in function {func.__name__}: {str(e)}")
                    retries += 1
                    print("Number of tries:", retries)
                    print(f"Error: {e}")

                    # Check if the error is due to token limit
                    if isinstance(e, openai.error.InvalidRequestError) and "maximum context length" in str(e):
                        user_input = input("The text is too long for the model. Do you want to ignore and return an empty string (I) or retry (R)? ").lower()

                        if user_input == 'i':
                            return ""

                    # Ask the user for action for other errors
                    else:
                        user_input = input("Do you want to retry (R) or ignore and return an empty string (I)? ").lower()
                        if user_input == 'i':
                            return ""

                    # Check for rate limit headers and adjust wait time if present
                    if hasattr(e, 'response') and e.response:
                        rate_limit_reset = e.response.headers.get('X-RateLimit-Reset')
                        if rate_limit_reset:
                            wait_time = max(int(wait_time), int(rate_limit_reset) - int(time.time()))

                    time.sleep(wait_time)
                    wait_time *= 2  # Double the wait time for exponential backoff

            # If max retries are reached, freeze the program
            print(f"Function {func.__name__} failed after {max_retries} retries.")
            input("Press Enter to retry or Ctrl+C to exit.")
            return wrapper(*args, **kwargs)  # Retry the function

        return wrapper
    return decorator



@retry_on_openai_error() # Generates questions and explanations to clarify ambiguities in a user's strategy using a GPT model.
def create_next_question_or_answer(context, model="text-davinci-003"):
    """
    Generates a list of questions and explanations aimed at clarifying ambiguities or gaps in a user's strategy.

    This function uses the OpenAI GPT model to identify unclear or ambiguous points in the user's strategy and generates questions that, if answered, would clarify those points. Each question is accompanied by an explanation of what the AI hopes to achieve by asking that question.

    Parameters:
    - context (str): The user's strategy or plan that needs clarification.
    - model (str, optional): The GPT model to use for generating questions and explanations. Defaults to "text-davinci-003".

    Returns:
    - list: A list containing two sub-lists. The first sub-list contains the questions generated by the GPT model, and the second sub-list contains the explanations for each question.
    """

    def separate_content_for_AI(text,
                                separator):  # this function separates everyting that is in between the separator variable into individual objects and puts it in a list and does the same for the content around the main content
        # Step 1: Find the indices of the first and second characters of the separator
        split_indices_first_char = [i for i, char in enumerate(text) if char == separator[0]]
        split_indices_second_char = [i for i, char in enumerate(text) if char == separator[1]]

        # Combine and sort the indices
        split_indices = sorted(split_indices_first_char + split_indices_second_char)

        # Step 2: Determine which parts were inside the separators
        parts = []
        prev_idx = 0
        for idx in split_indices:
            parts.append(text[prev_idx: idx])
            prev_idx = idx + 1
        parts.append(text[prev_idx:])

        # Step 3: Create lists of inside and outside parts, replacing empty strings with "unavailable"
        inside_separators = [part if part else 'unavailable text' for part in parts[1::2]]
        outside_separators = [part if part else 'unavailable text' for part in parts[::2]]

        # Step 4: Return the final list
        return [inside_separators, outside_separators]
    create_questions_template = """Your job is to point out holes in strategies of the user in order to make them more apparent and eventually help him. the way you will do this is by, first, making a list of the points (point 1, point 2, etc) that you either don't think are usefull or that you lack context for. second, for each item in that list you will create a question(question 1, question 2, etc) in between <> that, if answered, should clarify that point. third after each question you will do a small explanation as to what you hope comes out of the question above. here is an example and, at the same time, a model you must follow no matter what:
    "-point 1: it is unclear what is the opportunity the user is speaking of.
    -point 2: it is unclear how the user wants to use new technologies to achieve higher yields.
    -point 3: it is unclear where the user wants to build his product
    -point 4: etc

    <Question 1: Sorry, I am not sure to understand what the opportunity is and i would need some more context to establlish a better strategy. Could you provide more context?>
    With this question I should be able to establish a better context for the strategy by knowing the type of industry the user is in and how the opportunity is gonna change the business trends
    <Question 2: Sorry if I missunderstood, but I cannot see how your new technology can bring about higher crop yields. Although it does seems very effective in other areas, I would need some clarification as to how it increases yield.>
    With this question I hope to better understand the new technology by getting a better picture of all its capabilities.
    etc."

    Ok, you are now ready.
    User:
    """
    response = openai.Completion.create(
        model=model,
        temperature=0.2,
        prompt=create_questions_template + context,
        max_tokens=500
    ).choices[0].text.strip()
    print("texte brut" + response + "fin text brute")
    separator = "<>"
    questions_and_explanations = separate_content_for_AI(response, separator)
    AI_trainer.training_dataset_creation(context, response, "create_next_question_spiky_AI")

    return questions_and_explanations # Returns a list containing two sub-lists: one with questions and another with explanations for each question.

@retry_on_openai_error() # Generates a unifying label or process name for a collection of factual memories using a GPT model.
def generate_reflection_name (text,model="text-davinci-003"):
    """
    Generates a unifying label or process name that captures the essence of a collection of factual memories.

    This function uses the OpenAI GPT model to generate a unifying label or process name based on a collection of factual memories. These memories could be derived from a book and may detail specific events, techniques, concepts, or themes.

    Parameters:
    - text (str): The text containing a collection of factual memories.
    - model (str, optional): The GPT model to use for generating the unifying label or process name. Defaults to "text-davinci-003".

    Returns:
    - None: This function currently does not return any value but generates a response from the GPT model. Consider updating the function to return the generated response.
    """

    template = "Given a collection of factual memories derived from a book, each detailing specific events, techniques,concepts,  or themes, can you identify a unifying label or process that captures the common essence shared among these memories? here are the memories"

    response = openai.Completion.create(
        model=model,
        prompt=template + text,
        temperature=0.3,
        max_tokens= 100
    ).choices[0].text.strip()
    # Currently does not return any value; consider updating the function to return the generated unifying label or process name.

@retry_on_openai_error() # Decomposes a complex question into a list of simpler questions using a GPT model.
def create_retrieval_query(text, model="text-davinci-003"):
    """
    Generates a list of simplified and decomposed questions based on a complex input question.

    This function uses the OpenAI GPT model to break down a complex question into a set of simpler questions that, if answered, would collectively address the main question. The function also filters the generated questions to ensure they are valid and non-empty.

    Parameters:
    - text (str): The complex question to decompose.
    - model (str, optional): The GPT model to use for the decomposition. Defaults to "text-davinci-003".

    Returns:
    - list: A list of simplified and decomposed questions that aim to address the main question.
    """

    def create_string_list(reformulated_elements):
        # Ensure each question has a ? followed by \n
        reformulated_elements = reformulated_elements.replace("?", "?\n")

        # Separate the questions using \n
        string_list = reformulated_elements.strip().split("\n")

        return string_list

    template = f"""You are a prompt engineer and you need to simplify and decompose the input of the user into smaller questions that, if answered, would help to answer the main question, it would look a little bit like this:
    Complex Question: How does climate change impact global biodiversity and ecosystems?

    Reformulated Elements:
    -How do greenhouse gas emissions and global warming affect biodiversity and ecosystems?
    -What role does deforestation and habitat loss play in influencing biodiversity decline?
    -How does ocean acidification impact marine life and ecosystem health?
    -What are the consequences of rising species extinction rates on ecosystems?

    Complex Question: What are the key factors contributing to the rise of income inequality in developed countries?

    Reformulated Elements:    
    -How do technological advancements and automation impact income inequality in the job market?
    -What changes in labor policies and workers' rights influence income distribution?
    -How do taxation policies affect wealth distribution among different income groups?
    -What is the relationship between access to education and skill development opportunities and income disparity?

    Complex Question: How do cultural and social norms influence individual behavior and decision-making?

    Reformulated Elements:    
    -How does socialization, particularly family and peer interactions, shape individual behavior?
    -What cultural values influence moral reasoning and decision-making processes?
    -How does media and advertising impact perceptions and influence choices?
    -What role do societal expectations play in promoting conformity or deviance?  

    Complex Question: {text}

    Reformulated Elements:
    """

    response = openai.Completion.create(
        model=model,
        temperature=0.5,
        prompt=template,
        max_tokens=200
    ).choices[0].text.strip()
    print("interpretation of input : " + response)
    list_response = create_string_list(response)
    filtered_list = [s for s in list_response if s.strip() and any(c.isalpha() for c in s)]
    AI_trainer.training_dataset_creation(text, response, "create_retrieval_query")

    return filtered_list # Returns a list of simplified and decomposed questions generated by the GPT model.

@retry_on_openai_error() # Generates a list of key concepts in a subject text based on provided notes using a GPT model.
def create_subject_AI(text,notes,model="text-davinci-003"):
    """
    Generates a list of key concepts related to a subject text based on provided notes.

    This function uses the OpenAI GPT model to identify key concepts in the subject text that are also found in the provided notes.
    It can also identify new concepts introduced in the subject text that are not covered in the notes.

    Parameters:
    - text (str): The subject text to analyze.
    - notes (str): The notes that contain key concepts related to the subject text.
    - model (str, optional): The GPT model to use for the analysis. Defaults to "text-davinci-003".

    Returns:
    - list: A list of key concepts found in the subject text and notes.
    """

    create_subject_template = """
    In the text above, there are key concepts found in the provided notes. Additionally, the subject text might introduce new concepts or ideas. Your goal as a programmer is to:
    - Identify key concepts from the notes that are relevant to the subject text.
    - If you find that the subject text introduces new ideas not covered in the notes, list them as new concepts.

    Please ensure you follow this format for your answer:
    List of key concepts:
    <Concept A from notes>
    <Concept B from notes>
    <New Concept introduced in subject text>

    Consider all aspects of the subject text to provide a comprehensive list.

    Subject text:"""

    response = openai.Completion.create(
        model=model,
        temperature=0.2,
        prompt= "notes" + notes + create_subject_template + text,
        max_tokens=50
    ).choices[0].text.strip()
    concepts = re.findall(r'<(.*?)>', response)
    memory_stream.mydb.stream_close()
    return concepts # Returns a list of key concepts extracted from the GPT model's response.

@retry_on_openai_error() # Evaluates the importance of a text passage in relation to provided notes using either a GPT model or text embeddings.
def importance_memory_AI(text, notes="", model="text-davinci-003"):  # get importance score

    """
    Evaluates the importance of a given text in relation to provided notes using either a GPT model or text embeddings.

    This function uses one of two methods to evaluate the importance of a text passage in relation to the provided notes. The first method uses the OpenAI GPT model to generate an importance score based on a template. The second method uses text embeddings to compute the cosine similarity between the text and the notes.

    Parameters:
    - text (str): The text passage whose importance needs to be evaluated.
    - notes (str or list): The notes related to the subject of the text passage. Can be an empty string or a list of strings.
    - model (str, optional): The GPT model to use for generating the importance score. Defaults to "text-davinci-003".

    Returns:
    - int or float: An importance score. If the GPT model is used, the score is an integer between -10 and 10. If text embeddings are used, the score is a float scaled to be between 0 and 10.
    """

    templateImportanceMemory = "Above are your notes on the subject,there might be none. On the scale of -10 to 10, where -10 is completely unrelated and off-topic and 10 is completely on topic an essential to the subject, give a rating for the likely importance of the following text related to your notes. Place your rating in between square parentheses[] and only one short sentence to explain your rating:\n"
    templateImportanceWithNotes = "The text above are your own notes of the chapter you are in. The text you will receive is a passage of that chapter. From that passage, you will rate the importance of the passage related to your notes. On the scale of -10 to 10, where -10 is completely unrelated and off-topic and 10 is completely on topic an essential to the subject, give a rating for the likely importance of the following text related to your notes. Place your rating in between square parentheses[] and only one short sentence to explain your rating:\n"

    if isinstance(notes, list):
        notes = ' '.join(notes)

    if notes == "":
        template = templateImportanceMemory
        label = "importance_memory_AI"
    else:
        template = notes + templateImportanceWithNotes
        label = "importance_memory_with_notes_AI"

    # Randomly choose one of two processes
    if random.choice([True, False]):
        # First process
        response = openai.Completion.create(
            model=model,
            temperature=0.2,
            prompt=template + text,
            max_tokens = 200
        )
        # Extract the score from the response using a regular expression
        score = re.search(r'\[(.*?)\]', response.choices[0].text.strip())

        answer = response.choices[0].text.strip()  # store answers to train the AI
        AI_trainer.training_dataset_creation(notes + "\n" + text, answer, label)

        return int(score.group(1)) if score else 0
    else:
        # Second process
        text_embedding = get_embedding(text)
        notes_embedding = get_embedding(notes)

        # Compute cosine similarity
        score = np.dot(text_embedding, notes_embedding) / (
                np.linalg.norm(text_embedding) * np.linalg.norm(notes_embedding))
        score = round(score * 10, 2)  # Scale the score to be between 0 and 10

        AI_trainer.training_dataset_creation(text, str(score), "importance_by_embedding")

        return score # Returns an importance score, either as an integer (GPT model) or as a float (text embeddings).

@retry_on_openai_error() # Identifies or creates relevant concepts for a given text based on existing notes and updates the memory stream.
def subject_choice_and_creation_AI(text, notes, level_of_abstraction, memory_id, model="text-davinci-003"):
    """
    Identifies or creates relevant concepts for a given text based on existing notes and a level of abstraction.

    This function uses the OpenAI GPT model to identify the closest concept(s) in the provided notes that relate to the given text. If no relevant concepts are found, it creates new memory objects for the identified concepts. It also uses text embeddings to find similar existing concepts and updates the memory stream accordingly.

    Parameters:
    - text (str): The text for which relevant concepts need to be identified or created.
    - notes (str): Existing notes that may contain relevant concepts.
    - level_of_abstraction (int): The current level of abstraction for the memory object.
    - memory_id (str or int): The ID of the parent memory object to which new memory objects may be linked.
    - model (str, optional): The GPT model to use for identifying or creating concepts. Defaults to "text-davinci-003".

    Returns:
    - list: A list of categories or concepts that are relevant to the given text. If no relevant concepts are found, new memory objects are created and the function returns an empty list.
    """

    approved_concepts = set()  # To keep track of already approved concepts
    categories = []

    # Define a regex pattern to capture the concepts section
    pattern = re.compile(r"(?i)start\s+of\s+concepts?\s*([\s\S]*?)\s*end\s+of\s+concepts?", re.IGNORECASE)

    # Search for the pattern in the notes
    match = pattern.search(notes)

    # Extract concepts if the pattern is found
    if match:
        concepts_section = match.group(1).strip()
        concepts_string = "Concepts:\n" + concepts_section
    else:
        concepts_string = "no concepts"

    print(concepts_string)
    choose_subject_template = ". Write what you think is the closest (one or multiple) concept in the notes compared to this text:"

    try:
        response = openai.Completion.create(
            model=model,
            temperature=0.2,
            prompt="Notes with Concepts:" + concepts_string + choose_subject_template + text,
            max_tokens=100
        ).choices[0].text.strip()
    except Exception as e:
        return f"Error with OpenAI Completion: {e}"

    response_embedding = get_embedding(response)
    AI_trainer.training_dataset_creation(concepts_string + "." + text, response, "subject_choice_AI")

    cursor = memory_stream.mycursor
    cursor.execute(f"SELECT id FROM spiky_memory WHERE level_of_abstraction = {level_of_abstraction + 1} AND LENGTH(text) <= 20")
    ids = [row[0] for row in cursor.fetchall() if row[0] not in approved_concepts]

    vectors = memory_stream.index.query(ids)
    id_vector_dict = {id: vectors[id] for id in ids}
    similarities = {id: cosine_similarity(np.array(response_embedding).reshape(1, -1), np.array(vector).reshape(1, -1))[0][0] for id, vector in id_vector_dict.items()}
    top_concepts = sorted(similarities, key=similarities.get, reverse=True)[:3]

    approved_concepts_temp = []
    for concept in top_concepts:
        concept_str = concept
        approval = AI.similarity_approval_AI(concept_str, text)
        if approval is True:
            approved_concepts_temp.append(concept)

    if len(approved_concepts_temp) == len(top_concepts):
        categories.extend([concept for concept in approved_concepts_temp])
        approved_concepts.update(approved_concepts_temp)
    else:
        create_concept_memory(text, notes, level_of_abstraction,memory_id)

    memory_stream.mydb.stream_close()
    return categories # Returns a list of relevant concepts or categories; creates new memory objects if no relevant concepts are found.

@retry_on_openai_error() # Creates and adds memory objects for key concepts identified in a given text and notes to a memory stream.
def create_concept_memory(text, notes, level_of_abstraction,memory_id):
    """
    Creates memory objects for key concepts identified in a given text and notes.

    This function uses the `create_subject_AI` function to identify key concepts in the given text and notes. It then creates memory objects for each of these concepts and adds them to a memory stream. Each memory object is created with a new level of abstraction, which is one level higher than the provided `level_of_abstraction`.

    Parameters:
    - text (str): The text containing the subject matter.
    - notes (str): Notes related to the subject matter in the text.
    - level_of_abstraction (int): The current level of abstraction for the memory object.
    - memory_id (str or int): The ID of the parent memory object to which the new memory objects will be linked.

    Returns:
    - None: This function does not return any value but modifies the memory stream by adding new memory objects.
    """

    memory_subjects = create_subject_AI(text, notes)
    new_level_of_abstraction = level_of_abstraction + 1
    for memory_subject in memory_subjects:
        memory_subject = "Concept label : " + memory_subject
        memory_object = MemoryObject.MemoryObject(content=memory_subject,child_list_ID=[memory_id],level_of_abstraction=new_level_of_abstraction)
        memory_stream.add_memory(memory_object, get_embedding(memory_subject))
        # Modifies the memory stream by adding new memory objects, does not return any value.

@retry_on_openai_error() # Updates existing notes with new information and generates a question or answer to clarify ambiguities.
def spiky_notes_AI(context, previous_notes = "", model="text-davinci-003"):
    """
    Updates existing notes with new information and generates a question or answer to clarify ambiguities.

    This function uses the OpenAI GPT model to update existing notes with new information provided in the `context`. It then uses the `create_next_question_or_answer` function to generate a question or answer aimed at clarifying any ambiguities or gaps in the updated notes.

    Parameters:
    - context (str): The new information that needs to be incorporated into the existing notes.
    - previous_notes (str, optional): The existing notes that need to be updated. Defaults to an empty string.
    - model (str, optional): The GPT model to use for updating the notes and generating the question or answer. Defaults to "text-davinci-003".

    Returns:
    - list: A list containing two elements. The first element is a string containing the generated question or answer. The second element is a string containing the updated notes.
    """

    notes_template = ""
    context = "Notes: " + previous_notes + ". Update the notes with this information" + context
    new_conversation_notes = openai.Completion.create(
        model=model,
        prompt=notes_template + context,
        max_tokens=500
    ).choices[0].text.strip()

    AI_trainer.training_dataset_creation(context, new_conversation_notes, "spiky_AI_notes")

    response_string = create_next_question_or_answer()

    response = [response_string,new_conversation_notes]
    return response # Returns a list with two elements: the generated question or answer, and the updated notes.
@retry_on_openai_error()
def get_embedding(text, model="text-embedding-ada-002"):
    placeholder_text = "placeholder"
    try:
        if isinstance(text, list):
            text = ' '.join(text)
        text = text.replace("\n", " ")
        return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']
    except Exception as e:
        logging.error(f"get_embedding: an error occurred - {str(e)}")
        return openai.Embedding.create(input = [placeholder_text], model=model)['data'][0]['embedding']
