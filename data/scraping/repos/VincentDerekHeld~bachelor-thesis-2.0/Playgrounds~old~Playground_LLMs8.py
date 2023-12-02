import os

from openai import OpenAI
import spacy

# openai.api_key = os.environ["OPENAI_API_KEY"]
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def preprocess_text_with_LLM(doc):
    debug_mode = True
    text_contains_listings = False
    for token in doc:
        if token.tag_ == "LS":
            text_contains_listings = True
            break



    intro = """
    #### Intro: ###
    You are a system analyst who strictly and carefully follows the instructions. 
    Make sure that the previously given instructions are still followed.
    Keep as many words from the original text as possible, and never add information from the examples to the output.\n"""
    #  You return carefully only the current sentence as a complete sentence without adding any extra information, interpretation, explanation, numerations, listings, or "->".
    outro = """\n
    ### TEXT ### \n"""
    # Return only the transformed input without any additional information or annotations.
    answer_outro = "\n ### Answer / Response: ###"

    prompts = []
    inital_prompt = ("""
    ### Instruction: ###
    Read the text carefully and try to understand the content. 
    Return on this message an empty message (just a spaces without any other characters).
    ### Full Text ### \n""")

    # Relevance of Sentence
    if True: prompts.append("""
            Filter information from the text (initial query), that is not relevant for the process. The only information we are interested in are real process steps.
            ### Background Information: ###
            
            1) Introduction that describes the what the company is doing in general are not relevant and must be filtered.
                 Example: The Sentence "A small company manufactures customized bicycles." is not relevant. You return an empty message.
            
            2) Introductions that describes the goal of the process are not relevant and must be filtered.
                 Example: "to ensure valid results. "
                      -> "The organization shall determine the methods for monitoring, measurement, analysis and evaluation, to ensure valid results" -> "The organization shall determine the methods for monitoring, measurement, analysis and evaluation"
                 Example: "as evidence of the implementation of the audit programme(s)"
                       -> "Documented information shall be available as evidence of the implementation of the audit programme(s) and the audit results." -> "... shall document the results."
            
            4) Information that clarifies that something is not universally applicable are not relevant and must be filtered.
                Example: "as applicable"
                        "The organization shall determine the methods for monitoring, measurement, analysis and evaluation, as applicable," -> The organization shall determine the methods for monitoring, measurement, analysis and evaluation"
            
            5) Examples in the sentences are not relevant and must be filtered.
            
            6) Including parts are not relevant.
                Example: "The organization shall determine what needs to be monitored and measured, including information security processes and controls" -> "The organization shall determine what needs to be monitored and measured"
            
            7) References to other Articles or Paragraphs are not relevant for the process.
                Example: "referred to in Article 22(1) and (4)" must be filtered
                Example: "in accordance with Article 55"
                
            8) Information on how to solve actions or the goals of actions are not relevant and must be filtered.
                Example: "The selected methods should produce comparable and reproducible results to be considered valid." must be filtered from the sentence.
            
            9) Case descriptions are relevant:
                Example: "In the former case,..." and "If it is not...." are relevant.

            ### Instruction: ####
            Read the text carefully and decide information for information based on the eight provided background information from the sentences if the information in the sentence is relevant for the process or not.
            2)  If a sentence is relevant for the process, return the input sentence without any changes and interpretations.
            3)  If a sentence is in general not relevant for the process, return an empty message  (just a spaces without any other characters)
                else filter the not relevant information based on the seven background information from the sentences return the filtered sentence.""")

    # 3) Information that just addresses that the process starts or is finished are not relevant and must be filtered.
    # Example: "The process instance is then finished."
    print("text_contains_listings: " + text_contains_listings.__str__())
    if text_contains_listings:
        prompts.append(
            """
            ### Instruction: ####
            1) Decide if the text contains any listings. If the text contains listings, go to 2), else go to 3).
            2) Transform listings based on the structure of the text into a continuous text and filter the bullet points. Do not add any additional information or annotations.
            3) Return the sentence without any changes and without any additional information or annotations.
            """)
    if False:
        # Active Voice
        prompts.append("""
        ### Background Information: ###
        Example 1:
            Active Voice: "Whenever they receive an order, ..."
            Passive Voice: "... a new process instance is created."
            -> New Sentence: "Whenever they receive an order, they create a new process instance."
        Example 2:
            "If it is not available, it is back-ordered."
            Active Voice: "If the product is not available"
            Passive Voice: "it is back-ordered."
            -> To determine the active voice, the actor must be identified in the text from the inital request. New Sentence: "If the product is not available, they department back-orders the product."
        
        Somtimes for the transformation the actor is needed, which is not given in the sentence. Therefrom the actor must be identified in the text from the inital request.
        ### Instruction: ####
        If a part of a sentence is in passive voice, return this part of the sentence transformed into an active voice sentence (inlcuding the other parts), else return the sentence without any changes.
        If the actor is not given in the sentence, identify the actor in the text from the inital request and do not use information or verbatim from the examples.
        """)
    if False:
        # Implicit Actions
        prompts.append("""
        Some sentences contain implicit actions. Implicit or implied actions are actions that are actions that are not explicitly mentioned in the sentence, but can be inferred from the sentence.
            #Example 1: „Documented information shall be available as evidence of the implementation of the audit programme(s) and the audit results.“ -> „Documented information shall be available as evidence of the implementation of the audit programme(s) and the audit results.“
            #Example 2: "The actor then submits an order ticket to the kitchen to begin preparing the food." -> „The actor submits an order ticket to the kitchen. The kitchen prepares the food.“
        
        Most sentences do not contain implicit actions.
            #Example: "Whenever they receives an order, a new process instance is created." -> "Whenever they receive an order, they creates a new process instance."
        
        ### Instruction: ####
        1) Analyze the following sentence to determine if it contains any implicit actions. If it contains implicit actions go to 2), else go to 3).
        2) If implicit actions are identified, these must be converted into explicit actions following the following conditions, else return the sentence without any changes. 
            1. Condition: The original structure and order of the sentence must be retained as far as possible.
            2. Condition: The original wording of the sentence must be retained as far as possible.
        3) Return the input sentence without any changes and interpretations.
        """)

    # TODO
    # Reference Resolution
    if False: prompts.append("""
    Using the text provided, replace all pronouns with the specific names of the people or objects they refer to. 
    Do not add, infer, or assume any information beyond what is explicitly mentioned in the text. 
    If a pronoun's antecedent is not clear from the text, leave the pronoun as is and return the sentence without any changes and interpretations.
    Please focus solely on direct replacements for clarity and accuracy.
    """)
    if False: prompts.append("""
    ### Instruction: ####
    1) Remember the whole text from the first request and resolve references in the sentence such as "she", "he", "it", "they" with the name of the ACTOR(s) from the text, if given.
    2) Remember the whole text from the first request and resolve references in the sentence such as "this", "that", "these", "those" with the name of the OBJECT(s) from the text, if given.
    3) Remember the whole text from the first request and resolve references in the sentence such as "another" with the name of the ACTOR(s) or OBJECT(s) from the text, if given.
    If no reference is given or the reference cannot be resolved, return the sentence without any changes and interpretations.
    """)
    if False: prompts.append("""
    Please restructure the provided sentences to fit the specified format: '[ACTOR] [MODAL VERB if present] [VERB in active voice] [OBJECT].' 
    Remember that an ACTOR can be a person, a group (like a company or department), or a representative entity (like 'Kitchen' for kitchen staff). 
    Do not confuse objects with actors; for instance, 'auditors' in 'Select the auditors' are objects, not actors. K
    eep modal verbs such as 'must', 'shall', 'should', 'can' in their original form within the sentence. 
    Do not introduce new information; use only the actors and objects as they are presented in the text. 
    Your focus is on reformatting sentences without altering their original meaning or adding content.
    """)
    if False: prompts.append("""
    ### Background Information ###
    1. Actors are the subject of a sentence, or the person or thing that performs the action of the verb
        For the identification of the ACTOR, keep in mind, a actor can  for e.g. be a natural person, a organization, such as a company or a department, but sometimes also a place, a device or a system can be an valide Actor.
            Example: The "Kitchen" represents the "kitchen staff", and therefore the "Kitchen" can be an actor.
        Make sure not to use Objects as Actors:
            Example:  In the sentence "Select the auditors" the "auditors" are the object, not the actor.
            
    2. Modal verbs have to stay with the original format  in the sentence.
            -> Modal verbs, are verbs that express the strength of an expression. 
            Example modal verbs are: "must", "shall", "should", "can".
                Example 1: In the sentence "The organization shall determine..." the modal verb "shall" has to stay in the sentence.
    ### Instruction: ###
    Restructure every sentence to achieve the following structure of the sentence: "[ACTOR] [MODAL VERB if it exists in the sentence] [VERB in active] [OBJECT]."
    """)
    if False: prompts.append("""
    Transform the given sentences by following specific structural rules. First, identify placeholders such as '[ACTOR]', '[CONDITION]', '[OBJECT]', and '[ACTION]'. Then, replace these placeholders with appropriate words extracted from the context. Once you have the necessary components, restructure the sentences as per the following guidelines:
    If there is an action that does not depend on a condition, structure the sentence using the active voice in this format: '[ACTOR] [MODAL VERB if present] [VERB in active voice] [OBJECT]'.
    If an action is conditional, reformat the sentence to: 'If [CONDITION], [ACTOR] [ACTION], ELSE [alternate ACTION if provided in the text]'.
    When two actions are contingent on the same condition, combine them into a single sentence: 'If [CONDITION], [ACTOR] [ACTION], ELSE [alternate ACTION if mentioned]'.
    For example, take the sentences 'If the controller becomes aware of a personal data breach, the controller shall notify the supervisory authority within 72 hours.' and 'If the notification isn't made within 72 hours, the controller shall provide reasons for the delay.' Merge them into 'If the controller becomes aware of a personal data breach, the controller shall notify the supervisory authority within 72 hours, else the controller shall provide reasons for the delay.', as they follow an IF-ELSE structure.
    Please apply these rules to restructure sentences accurately and clearly, maintaining the original meaning and intent of the text.
    """)
    if False:
        if use_all_prompts: prompts.append("""
                         Carefully replace all placeholders, such as "[ACTOR], [CONDITION], [OBJECT], [ACTION]" with the right words from the text and restrcuture the sentence in the follwing way:
                         
                         1) If the action is not based on a condition restructure the sentence with the following structure: "[ACTOR] [MODAL VERB if exists in the sentence] [VERB in active] [OBJECT]."
                         
                         2) If the action is based on a condition restructure the sentence with the following structure: „If [CONDITION], [ACTOR] [ACTION], ELSE [other ACTION, if in the Text]“. 
                        
                         3) If two ACTIONs are based on the same CONDITION merge the sentences to the following structure: „If [CONDITION], [ACTOR] [ACTION], ELSE [other ACTION, if in the Text]“. 
                                Example: 
                                The Sentence „If the controller becomes aware of a personal data breach, the controller shall notify the supervisory authority within 72 hours.“ and
                                the Sentence „If the notification isn't made within 72 hours, the controller shall provide reasons for the delay.“ can be merged into one Sentence „If the controller becomes aware of a personal data breach, the controller shall notify the supervisory authority within 72 hours, else the controller shall provide reasons for the delay.“, as they are based on a IF - ELSE structure.
                """)

        if False: prompts.append("""
                Ensure carefully that all placeholders, such as "[ACTOR], [CONDITION], [OBJECT], [ACTION]" have been replaced with the correct actors, condtions, objects and actions from the text and the text does not contain any [].
                   """)

        if False: prompts.append("""
                        
                        If the CONDITION part is at the end of the sentence, move it to the front of the sentence. 
                        Example: The Sentence "The data subject has the right to have their personal data transmitted directly from one controller to another, if technically feasible." can be restructured into "If technically feasible, the data subject has the right to have their personal data transmitted directly from one controller to another.".
                        ###### TEXT ###### \n
                """)

    def generate_response(prompt) -> str:
        model_engine = "gpt-3.5-turbo-instruct"
        if debug_mode: print(f"*** Prompt: *** len: {len(prompt).__str__()} \n {prompt} \n")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": intro},
                {"role": "assistant", "content": ""},
                {"role": "user", "content": prompt},
            ]
        )
        response_text = response.choices[0].message.content
        response_text = response_text.strip()
        if debug_mode:
            print(f"*** Response: ***\n {response_text}")
            print("*" * 50)
        return response_text

    # Initital Instructions and
    result = ""
    generate_response(inital_prompt + doc.text)
    for number, sent in enumerate(doc.sents):
        # print(f"**** Sent. {number}: {sent.text}")
        current_sent: str = sent.text
        if current_sent.isspace() or current_sent.__len__() == 0:
            print(f"Skipped on Sent No. {number} because only whitespace")
            next(doc.sents)
        for prompt in prompts:
            #intro
            query = prompt + outro + current_sent + answer_outro
            current_sent = generate_response(query)
            print(f"current_sent: {current_sent}")
            print(f"current_sent.isspace(): {current_sent.isspace().__str__()}")
            print(f"current_sent.__len__(): {current_sent.__len__()}")
            if current_sent.isspace() or current_sent.__len__() == 0:
                print(f"Sent No. {number} has been returned as empty message.")
                next(doc.sents)
                break
        result = result + "" + current_sent + "\n"

    print("**** Full description: **** \n" + result.strip().replace("\n", " ").replace(".", ". ").replace("!",
                                                                                                          "! ").replace(
        "?", "? "))
    return result.strip()


def write_to_file(number: int, nlp):
    input_path = f"/Users/vincentderekheld/PycharmProjects/bachelor-thesis/project/Text/text_input_vh/Text{number.__str__()}.txt"
    text_input = open(input_path, 'r').read().replace("\n", " ")
    doc = nlp(text_input)
    content = preprocess_text_with_LLM(doc)
    output_path = f"/Users/vincentderekheld/PycharmProjects/bachelor-thesis/Evaluation/GPT-Text/Text{number.__str__()}-3.txt"
    with open(output_path, 'w') as file:
        file.write(content)
    print(f"Created Text: {number.__str__()}")


nlp = spacy.load('en_core_web_trf')
write_to_file(5, nlp)
