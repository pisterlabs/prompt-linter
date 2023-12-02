from pathlib import Path
import re
import time
from modal import Image, Secret, Stub, web_endpoint

image = Image.debian_slim().pip_install(
    "lxml~=4.9.2",
    "openai~=0.27.4",
    "tiktoken==0.3.0",
    # "guidance~=0.0.60",
    "langchain~=0.0.216",
)
stub = Stub(
    name="syntax-agent",
    image=image,
    secrets=[Secret.from_name("openai-qa")],
)


from lxml import etree
import requests

# Get the plain treedown representation for a token's sentence

# example endpoint: "https://labs.clear.bible/symphony-dev/api/GNT/Nestle1904/lowfat?usfm-ref=JHN%2014:1" - JHN 14:1


def process_element(element, usfm_ref, indent=0, brackets=False):
    if brackets:
        indent = 0
    treedown_str = ""
    open_bracket = "[" if brackets else ""
    close_bracket = "] " if brackets else ""

    if element.get("class") == "cl":
        treedown_str += "\n" + open_bracket + ("  " * indent)

    if element.get("role"):
        role = element.attrib["role"]
        if role == "adv":
            role = "+"
        if not brackets:
            treedown_str += "\n"
        treedown_str += open_bracket + ("  " * indent) + role + ": "

    # # bold the matching token using usfm ref # NOTE: not applicable, since I think you have to use a USFM ref without the word on the endpoint
    # if element.tag == "w" and element.get("ref") == usfm_ref:
    #     treedown_str += "**" + element.text + "**"
    #     treedown_str += element.attrib.get("after", "") + close_bracket

    if element.tag == "w" and element.text:
        treedown_str += (
            element.attrib.get("gloss", "")
            + element.attrib.get("after", "")
            + f"({element.text})"
        )
        treedown_str += close_bracket

    for child in element:
        treedown_str += process_element(child, usfm_ref, indent + 1, brackets)

    return treedown_str


def get_treedown_by_ref(usfm_ref, brackets=True):
    usfm_passage = usfm_ref.split("!")[0]
    endpoint = (
        "https://labs.clear.bible/symphony-dev/api/GNT/Nestle1904/lowfat?usfm-ref="
        # "http://localhost:8984/symphony-dev/api/GNT/Nestle1904/lowfat?usfm-ref="
        + usfm_passage
    )

    # uri encode endpoint
    endpoint = requests.utils.requote_uri(endpoint)

    # print(endpoint)

    text_response = requests.get(endpoint).text
    xml = etree.fromstring(text_response.encode("utf-8"))

    treedown = process_element(xml, usfm_passage, brackets=brackets)
    return treedown


# Long book names to USFM (3 uppercase letters) format
book_name_mapping = {
    "Genesis": "GEN",
    "Exodus": "EXO",
    "Leviticus": "LEV",
    "Numbers": "NUM",
    "Deuteronomy": "DEU",
    "Joshua": "JOS",
    "Judges": "JDG",
    "Ruth": "RUT",
    "1 Samuel": "1SA",
    "2 Samuel": "2SA",
    "1 Kings": "1KI",
    "2 Kings": "2KI",
    "1 Chronicles": "1CH",
    "2 Chronicles": "2CH",
    "Ezra": "EZR",
    "Nehemiah": "NEH",
    "Esther": "EST",
    "Job": "JOB",
    "Psalms": "PSA",
    "Psalm": "PSA",
    "Proverbs": "PRO",
    "Ecclesiastes": "ECC",
    "Song of Solomon": "SNG",
    "Isaiah": "ISA",
    "Jeremiah": "JER",
    "Lamentations": "LAM",
    "Ezekiel": "EZK",
    "Daniel": "DAN",
    "Hosea": "HOS",
    "Joel": "JOL",
    "Amos": "AMO",
    "Obadiah": "OBA",
    "Jonah": "JON",
    "Micah": "MIC",
    "Nahum": "NAM",
    "Habakkuk": "HAB",
    "Zephaniah": "ZEP",
    "Haggai": "HAG",
    "Zechariah": "ZEC",
    "Malachi": "MAL",
    "Matthew": "MAT",
    "Mark": "MRK",
    "Luke": "LUK",
    "John": "JHN",
    "Acts": "ACT",
    "Romans": "ROM",
    "1 Corinthians": "1CO",
    "2 Corinthians": "2CO",
    "Galatians": "GAL",
    "Ephesians": "EPH",
    "Philippians": "PHP",
    "Colossians": "COL",
    "1 Thessalonians": "1TH",
    "2 Thessalonians": "2TH",
    "1 Timothy": "1TI",
    "2 Timothy": "2TI",
    "Titus": "TIT",
    "Philemon": "PHM",
    "Hebrews": "HEB",
    "James": "JAS",
    "1 Peter": "1PE",
    "2 Peter": "2PE",
    "1 John": "1JN",
    "2 John": "2JN",
    "3 John": "3JN",
    "Jude": "JUD",
    "Revelation": "REV",
}
reverse_book_name_mapping = {v: k for k, v in book_name_mapping.items()}


@stub.function()
@web_endpoint("GET")
def get_syntax_for_query(query):
    """Need to extract the USFM ref from the query using regular expression
    Basically, search the query string for all of book_name_mapping.keys()
    beginning with the longest, and moving to the shortest. We are looking
    for exact matches, because '1 John' contains 'John', etc.

    then, return get_treedown_by_ref(usfm_ref)
    """

    book_names = sorted(
        list(book_name_mapping.keys()), key=lambda x: len(x), reverse=True
    )

    for name in book_names:
        if name in query:
            ref = re.search(f"{name} \d+[:\.]\d+", query)
            usfm_book = book_name_mapping[name]
            ref = re.sub(f"{name}", f"{usfm_book}", ref.group(0))
            return get_treedown_by_ref(ref).strip()

    return None


# import guidance


# def clean_string(s):
#     return s.encode("utf-8", "ignore").decode("utf-8")


# @stub.function()
# @web_endpoint("GET")
# def syntax_qa_chain_guidance(query):
#     """Use guidance to complete QA chain for syntax question"""
#     guidance.llm = guidance.llms.OpenAI("text-davinci-003")

#     example_qas = [
#         {
#             "question": "What is the subject of the main verb in Mark 1:15?",
#             "context": "And (Καὶ)] \n[[+: after (μετὰ)] the (τὸ)] \n[[v: delivering up (παραδοθῆναι)] [s: - (τὸν)] of John (Ἰωάνην)] [v: came (ἦλθεν)] [s: - (ὁ)] Jesus (Ἰησοῦς)] [+: into (εἰς)] - (τὴν)] Galilee (Γαλιλαίαν)] [+: \n[[v: proclaiming (κηρύσσων)] [o: the (τὸ)] gospel (εὐαγγέλιον)] - (τοῦ)] of God (Θεοῦ)] and (καὶ)] \n[[v: saying (λέγων)] [+: - (ὅτι)] \n[[v: Has been fulfilled (Πεπλήρωται)] [s: the (ὁ)] time (καιρὸς)] and (καὶ)] \n[[v: has drawn near (ἤγγικεν)] [s: the (ἡ)] kingdom (βασιλεία)] - (τοῦ)] of God·(Θεοῦ)] \n[[v: repent (μετανοεῖτε)] and (καὶ)] \n[[v: believe (πιστεύετε)] [+: in (ἐν)] the (τῷ)] gospel.(εὐαγγελίῳ)]",
#             "answer": "The subject of the main verb is Jesus ([s: - (ὁ)] Jesus (Ἰησοῦς)])",
#         },
#         {
#             "question": "Who is the object of Jesus' command in Matthew 28:19?",
#             "context": "therefore (οὖν)] \n[\n[[+: [v: Having gone (πορευθέντες)] [v: disciple (μαθητεύσατε)] [o: all (πάντα)] the (τὰ)] nations,(ἔθνη)] \n[[+: [v: baptizing (βαπτίζοντες)] [o: them (αὐτοὺς)] [+: in (εἰς)] the (τὸ)] name (ὄνομα)] of the (τοῦ)] Father (Πατρὸς)] and (καὶ)] of the (τοῦ)] Son (Υἱοῦ)] and (καὶ)] of the (τοῦ)] Holy (Ἁγίου)] Spirit,(Πνεύματος)] \n[[+: [v: teaching (διδάσκοντες)] \n[[o: [s: them (αὐτοὺς)] [v: to observe (τηρεῖν)] [o: all things (πάντα)] \n[[apposition: [o: whatever (ὅσα)] [v: I commanded (ἐνετειλάμην)] [io: you·(ὑμῖν)]",
#             "answer": "In the verse, he commanded 'you' ([io: you·(ὑμῖν)])",
#         },
#         {
#             "question": "In 1 Corinthians 15:33, what is the direct object of the verb 'be deceived'?",
#             "context": "[[+: Not (μὴ)] [v: be misled·(πλανᾶσθε)] \n[[v: Do corrupt (φθείρουσιν)] [o: morals (ἤθη)] good (χρηστὰ)] [s: companionships (ὁμιλίαι)] bad.(κακαί)]",
#             "answer": "The verb 'be deceived' ([v: be misled·(πλανᾶσθε)]) has no direct object in this verse.",
#         },
#         {
#             "question": "What are the circumstances of the main clause in Luke 15:20?",
#             "context": "And (καὶ)] \n[\n[[+: [v: having risen up (ἀναστὰς)] [v: he went (ἦλθεν)] [+: to (πρὸς)] the (τὸν)] father (πατέρα)] of himself.(ἑαυτοῦ)] now (δὲ)] \n[[+: Still (ἔτι)] [s: he (αὐτοῦ)] [+: far (μακρὰν)] [v: being distant (ἀπέχοντος)] \n[[v: saw (εἶδεν)] [o: him (αὐτὸν)] [s: the (ὁ)] father (πατὴρ)] of him (αὐτοῦ)] and (καὶ)] \n[[v: was moved with compassion,(ἐσπλαγχνίσθη)] and (καὶ)] \n[\n[[+: [v: having run (δραμὼν)] [v: fell (ἐπέπεσεν)] [+: upon (ἐπὶ)] the (τὸν)] neck (τράχηλον)] of him (αὐτοῦ)] and (καὶ)] \n[[v: kissed (κατεφίλησεν)] [o: him.(αὐτόν)]",
#             "answer": "The implied subject goes 'to his own father' ([+: to (πρὸς)] the (τὸν)] father (πατέρα)] of himself.(ἑαυτοῦ)])",
#         },
#         {
#             "question": "What does Jesus tell his disciples to do in Matthew 5:44 regarding their enemies, and what is the reason he gives for this command?",
#             "context": "however (δὲ)] \n[[s: I (ἐγὼ)] [v: say (λέγω)] [io: to you,(ὑμῖν)] [o: \n[[v: love (ἀγαπᾶτε)] [o: the (τοὺς)] enemies (ἐχθροὺς)] of you (ὑμῶν)]",
#             "answer": "Jesus tells his disciples to love their enemies ([[v: love (ἀγαπᾶτε)] [o: the (τοὺς)] enemies (ἐχθροὺς)] of you (ὑμῶν)])",
#         },
#     ]

#     program = guidance(
#         """
#         {{~#each example_qas}}
#         Q: {{this.question}}
#         Context: {{this.context}}
#         A: {{this.answer}}
#                 {{~/each}}
#         Q: {{query}}
#         Context: {{get_syntax_for_query query~}}
#         A: {{gen 'answer'}}"""
#     )  # Optionally, add token_healing=False to this gen to disable token healing for bracket notation.

#     # result = program(
#     #     example_qas=example_qas,
#     #     query=query,
#     #     get_syntax_for_query=get_syntax_for_query,
#     #     # await_missing=True,
#     #     # stream=False,
#     # )

#     # I need to wait for the program to execute
#     return program(
#         example_qas=example_qas,
#         query=query,
#         get_syntax_for_query=get_syntax_for_query,
#     )


from langchain import PromptTemplate, OpenAI, LLMChain


@stub.function()
@web_endpoint("GET")
def syntax_qa_chain(query):
    """Use langchain to complete QA chain for syntax question"""
    examples = """"""

    prompt_template = """The contexts provided below follow a simple syntax markup, where 
    s=subject
    v=verb
    o=object
    io=indirect object
    +=adverbial modifier
    p=non-verbal predicate
    
    Answer each question by extracting the relevant syntax information from the provided context:
    Q: What is the subject of the main verb in Mark 1:15?
    Context: And (Καὶ)] 
[[+: after (μετὰ)] the (τὸ)] 
[[v: delivering up (παραδοθῆναι)] [s: - (τὸν)] of John (Ἰωάνην)] [v: came (ἦλθεν)] [s: - (ὁ)] Jesus (Ἰησοῦς)] [+: into (εἰς)] - (τὴν)] Galilee (Γαλιλαίαν)] [+: 
[[v: proclaiming (κηρύσσων)] [o: the (τὸ)] gospel (εὐαγγέλιον)] - (τοῦ)] of God (Θεοῦ)] and (καὶ)] 
[[v: saying (λέγων)] [+: - (ὅτι)] 
[[v: Has been fulfilled (Πεπλήρωται)] [s: the (ὁ)] time (καιρὸς)] and (καὶ)] 
[[v: has drawn near (ἤγγικεν)] [s: the (ἡ)] kingdom (βασιλεία)] - (τοῦ)] of God·(Θεοῦ)] 
[[v: repent (μετανοεῖτε)] and (καὶ)] 
[[v: believe (πιστεύετε)] [+: in (ἐν)] the (τῷ)] gospel.(εὐαγγελίῳ)]
    A: The subject of the main verb is Jesus ([s: - (ὁ)] Jesus (Ἰησοῦς)])
    
    Q: Who is the object of Jesus' command in Matthew 28:19?
    Context: therefore (οὖν)] 
[
[[+: [v: Having gone (πορευθέντες)] [v: disciple (μαθητεύσατε)] [o: all (πάντα)] the (τὰ)] nations,(ἔθνη)] 
[[+: [v: baptizing (βαπτίζοντες)] [o: them (αὐτοὺς)] [+: in (εἰς)] the (τὸ)] name (ὄνομα)] of the (τοῦ)] Father (Πατρὸς)] and (καὶ)] of the (τοῦ)] Son (Υἱοῦ)] and (καὶ)] of the (τοῦ)] Holy (Ἁγίου)] Spirit,(Πνεύματος)] 
[[+: [v: teaching (διδάσκοντες)] 
[[o: [s: them (αὐτοὺς)] [v: to observe (τηρεῖν)] [o: all things (πάντα)] 
[[apposition: [o: whatever (ὅσα)] [v: I commanded (ἐνετειλάμην)] [io: you·(ὑμῖν)]
    A: In the verse, he commanded 'you' ([io: you·(ὑμῖν)])
    
    Q: What are the circumstances of the main clause in Luke 15:20?
    Context: And (καὶ)] 
[
[[+: [v: having risen up (ἀναστὰς)] [v: he went (ἦλθεν)] [+: to (πρὸς)] the (τὸν)] father (πατέρα)] of himself.(ἑαυτοῦ)] now (δὲ)] 
[[+: Still (ἔτι)] [s: he (αὐτοῦ)] [+: far (μακρὰν)] [v: being distant (ἀπέχοντος)] 
[[v: saw (εἶδεν)] [o: him (αὐτὸν)] [s: the (ὁ)] father (πατὴρ)] of him (αὐτοῦ)] and (καὶ)] 
[[v: was moved with compassion,(ἐσπλαγχνίσθη)] and (καὶ)] 
[
[[+: [v: having run (δραμὼν)] [v: fell (ἐπέπεσεν)] [+: upon (ἐπὶ)] the (τὸν)] neck (τράχηλον)] of him (αὐτοῦ)] and (καὶ)] 
[[v: kissed (κατεφίλησεν)] [o: him.(αὐτόν)]
    A: The implied subject goes 'to his own father' ([+: to (πρὸς)] the (τὸν)] father (πατέρα)] of himself.(ἑαυτοῦ)])
    
    Q: What does Jesus tell his disciples to do in Matthew 5:44 regarding their enemies, and what is the reason he gives for this command?
    Context: however (δὲ)] 
[[s: I (ἐγὼ)] [v: say (λέγω)] [io: to you,(ὑμῖν)] [o: 
[[v: love (ἀγαπᾶτε)] [o: the (τοὺς)] enemies (ἐχθροὺς)] of you (ὑμῶν)]
    A: Jesus tells his disciples to love their enemies ([[v: love (ἀγαπᾶτε)] [o: the (τοὺς)] enemies (ἐχθροὺς)] of you (ὑμῶν)])
    
    Q: {question}
    Context: {context}
    A: """

    llm = OpenAI(temperature=0)
    llm_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate(
            template=prompt_template,
            input_variables=["question", "context"],
        ),
    )

    context = get_syntax_for_query(query)

    return {
        "answer": llm_chain.predict(context=context, question=query),
        "context": context,
    }
