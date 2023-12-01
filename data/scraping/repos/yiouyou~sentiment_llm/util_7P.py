##### v0
# import pprint as pp
# key = "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"
# N_batch = 3

# def call_openai(chain, _content, _example):
#     from langchain.callbacks import get_openai_callback
#     _re = ""
#     _tokens = 0
#     _cost = 0
#     _log = ""
#     with get_openai_callback() as cb:
#         _re = chain.run(_content=_content, _example=_example)
#         _tokens = cb.total_tokens
#         _cost = cb.total_cost
#         _log += f"\nTokens: {cb.total_tokens} = (Prompt {cb.prompt_tokens} + Completion {cb.completion_tokens})\n"
#         _cost_str = format(cb.total_cost, ".5f")
#         _log += f"Cost: ${_cost_str}\n\n"
#     # print(_re)
#     return [_re, _tokens, _cost, _log]

# def split_note_to_sentences(txt_lines):
#     ##### split note to sentences
#     _sentences = []
#     for i in txt_lines:
#         i_li = i.strip()
#         if i_li:
#             for j in i_li.split(". "):
#                 jj = ""
#                 if j[-1] == '.':
#                     jj = j
#                 else:
#                     jj = j+"."
#                 _sentences.append(jj)
#     return _sentences

# def P7_openai(key, txt_lines, N_batch):
#     import os
#     import re
#     from langchain import OpenAI, PromptTemplate, LLMChain
#     _log = ""
#     _7P_str = ""
#     _total_cost = 0
#     ##### set OpenAI API Key and prompt
#     os.environ["OPENAI_API_KEY"] = key
#     llm = OpenAI(temperature=0)
#     template = """
# Ignore previous instructions. As a marketing strategy analyst, your task is to identify and extract the 7Ps from each customer comment using nouns, according to the 7Ps Marketing Mix.

# Below are some examples of 7Ps analysis for some customer comments in csv format, where the customer's comments are enclosed in double quotes, and after the comma is the 7Ps analysis results:
# {_example}

# The customer comments that require marketing strategy analysis are as follows:
# {_content}

# For each comment, identify and extract relative nouns of 7Ps from and only from the comment. Output the analysis result of each comment in JSON format in one line, with the 7Ps as the main key and the corresponding nouns as the values. The order of the main key is: Product, Price, Place, Promotion, People, Process, Physical evidence.

# Please output the analysis results in English lowercase:
# """
#     prompt = PromptTemplate(
#         input_variables=["_content", "_example"],
#         template=template,
#     )
#     ##### 随机取10个example
#     import random
#     with open("examples_7P.txt", "r", encoding="utf-8") as ef:
#         _example = "".join(random.sample(ef.readlines(), 30))
#     ##### LLMChain
#     chain = LLMChain(llm=llm, prompt=prompt)
#     ##### split note to sentences
#     _sentences = split_note_to_sentences(txt_lines)
#     ##### call OpenAI API with _content
#     all_re = ""
#     for i in range(0, len(_sentences)):
#         if i % N_batch == 0:
#             batch = _sentences[i:i+N_batch]
#             # print(batch)
#             _content = ""
#             n = int(i / N_batch)
#             for j in range(0, len(batch)):
#                 _content = _content + f"{n*N_batch +j +1}) {batch[j]}\n"
#             _log += _content
#             # print(prompt.format(_content=_content, _example=_example))
#             [b_re, b_tokens, b_cost, b_log] = call_openai(chain, _content, _example)
#             _log += b_log
#             _total_cost += b_cost
#             all_re += b_re + "\n"
#             # print(b_re)
#     _total_cost_str = format(_total_cost, ".5f")
#     _7P_str = all_re.strip()
#     return [_log, _7P_str, _total_cost_str, _sentences]

# def parse_7P_str(_str, _sentences):
#     import re
#     import json
#     _re = []
#     _li = _str.split("\n")
#     for i in _li:
#         if i:
#             # print(i)
#             _1 = i.split(" {")
#             _i = "{" + _1[1]
#             # print(f"_i: {_i}")
#             _re.append(_i)
#     for i in range(len(_sentences)):
#         # print(f"\n{i}, {_sentences[i]}")
#         _i_json = json.loads(_re[i])
#         # print(type(_i_json), _i_json)
#         for j in _i_json:
#             if _i_json[j].lower() not in _sentences[i].lower():
#                 # print(">>>", _i_json[j])
#                 _i_json[j] = ''
#             if _i_json[j].lower() == 'none':
#                 _i_json[j] = ''
#         _re[i] = json.dumps(_i_json, ensure_ascii=False)
#         # print(type(_re[i]), _re[i], "\n")
#     _re_str = '[' + ', '.join(_re) + ']'
#     return _re_str

# def P7_llm(_txt):
#     global key
#     _log = ""
#     _7P_str = ""
#     _total_cost = 0
#     txt_lines = _txt.split("\n")
#     [_log, _7P_str, _total_cost_str, _sentences] = P7_openai(key, txt_lines, N_batch)
#     # print(_log)
#     # print(_7P_str)
#     import ast
#     _7P_str = parse_7P_str(_7P_str, _sentences)
#     # print(_7P_str)
#     _7P = ast.literal_eval(_7P_str)
#     # print(type(_7P), _7P)
#     return [_7P, _total_cost_str]



##### v1
def call_openai_tagging(chain, _sentence):
    from langchain.callbacks import get_openai_callback
    _re = ""
    _tokens = 0
    _cost = 0
    _log = ""
    with get_openai_callback() as cb:
        _re = chain.run(_sentence)
        _tokens = cb.total_tokens
        _cost = cb.total_cost
        _log += f"\nTokens: {cb.total_tokens} = (Prompt {cb.prompt_tokens} + Completion {cb.completion_tokens})\n"
        _cost_str = format(cb.total_cost, ".5f")
        _log += f"Cost: ${_cost_str}\n\n"
    print(_sentence, _re)
    print(_log)
    return [_re, _tokens, _cost, _log]

def P7_openai_tagging(txt_lines):
    from langchain.chains import create_tagging_chain
    from dotenv import load_dotenv
    load_dotenv()
    from langchain.chat_models import ChatOpenAI
    import os
    llm = ChatOpenAI(temperature=0, model_name=os.getenv('OPENAI_MODEL'))
    #####
    schema = {
        "properties": {
            "Product": {
                "type": "string",
                "description": """
According to the 7Ps Marketing Mix, identify and extract the 'Product' of 7Ps using nouns. Remeber it's the item or service that we offer to our customers.

Below are some examples of 7Ps analysis for customer comments in csv format, where the customer's comments are enclosed in double quotes, and after the comma is the 7Ps analysis results:
"han sidder i VVS og har ikke meget med de her sager at gøre", {"product":"VVS", "price":"", "place":"", "promotion":"", "people":"", "process":"", "physical evidence":""}
"Jan Milling skriver i mail 28 juni at han skal forhandle renovation i oktober og omtaler vores "fine materiale".", {"product":"renovation", "price":"", "place":"", "promotion":"omtaler", "people":"", "process":"", "physical evidence":""}
"Der har været Opslag på Linked In af Laust omkring affaldssortering og vi talte om at tage vores dialog videre.", {"product":"affaldssortering", "price":"", "place":"", "promotion":"", "people":"", "process":"", "physical evidence":""}
"Vi skal tale med Mogens Bang om den her.", {"product":"", "price":"", "place":"", "promotion":"", "people":"", "process":"", "physical evidence":""}
"Er ved at få udvidet butik og har mange kasser stående, efter uge 40 kunne det være interessant, da de åbner en del flere kvadratmeter.", {"product":"kasser", "price":"", "place":"butik", "promotion":"", "people":"", "processes":"", "physical evidence":""}
"hørkjoler med markup 4-6", {"product":"hørkjoler", "price":"markup", "place":"", "promotion":"", "people":"", "processes":"", "physical evidence":""}
"Ja, måske, men det kommer an på priser og hvilken kvalitet og hvilket mindste køb han kan lave.", {"product":"kvalitet", "price":"priser, mindste køb", "place":"", "promotion":"", "people":"", "processes":"", "physical evidence":""}
"hans brand er et mærke som består af klassiske designs, som alt sammen produceres i Italien, økologisk bomuld etc.", {"product":"designs, bomuld, økologisk", "price":"", "place":"Italien", "promotion":"brand", "people":"", "processes":"produceres", "physical evidence":""}
"Jeg bliver i branchen, han arbejder som konsulent for Pasform for Zizzi i Billund og Sandgaard i Ikast dels for at styre deres&nbsp; Har en online shop til store piger med, Bambus, polyamid, viskose.", {"product":"online shop, Bambus, polyamid, viskose", "price":"", "place":"", "promotion":"", "people":"konsulent, store piger", "processes":"", "physical evidence":""}
"Materiale på Tankanlæg modtaget sammen med materiale på skueglas (deal 8776)", {'product':'Tankanlæg, skueglas', 'price':'', 'place':'', 'promotion':'', 'people':'', 'processes':'', 'physical evidence':''}
""",
            },
            "Price": {
                "type": "string",
                "description": """
According to the 7Ps Marketing Mix, identify and extract the 'Price' of 7Ps using nouns. Remeber it's the amount of money that we charge for our product or service.

Below are some examples of 7Ps analysis for some customer comments in csv format, where the customer's comments are enclosed in double quotes, and after the comma is the 7Ps analysis results:
"Vi skal tale med Mogens Bang om den her.", {"product":"", "price":"", "place":"", "promotion":"", "people":"", "process":"", "physical evidence":""}
"hørkjoler med markup 4-6", {"product":"hørkjoler", "price":"markup", "place":"", "promotion":"", "people":"", "processes":"", "physical evidence":""}
"Helle, fordi vi skal ikke investere mere i noget med de salgstider de har.", {"product":"", "price":"investere", "place":"", "promotion":"", "people":"", "processes":"salgstider", "physical evidence":""}
"Ja, måske, men det kommer an på priser og hvilken kvalitet og hvilket mindste køb han kan lave.", {"product":"kvalitet", "price":"priser, mindste køb", "place":"", "promotion":"", "people":"", "processes":"", "physical evidence":""}
"Jeg ved ikke hvor lang tid der skal påregnes. Henrik, siger at Bendix afleverer de nye tanke og tager de færdige tanke med tilbage til NH STÅL", {'product':'', 'price':'tanke', 'place':'', 'promotion':'', 'people':'', 'processes':'', 'physical evidence':''}
"Planlæg hvem der der udfører basic engineering samt tilbud", {'product':'', 'price':'tilbud', 'place':'', 'promotion':'', 'people':'', 'processes':'basic engineering', 'physical evidence':''}
"send info priser kvaliteter var interesseret vil gerne have noget info og piser", {'product':'', 'price':'priser', 'place':'', 'promotion':'', 'people':'', 'processes':'kvaliteter', 'physical evidence':''}
""",
            },
            "Place": {
                "type": "string",
                "description": """
According to the 7Ps Marketing Mix, identify and extract the 'Place' of 7Ps using nouns. Remeber it's the channels and locations that we use to distribute and sell our product or service.

Below are some examples of 7Ps analysis for some customer comments in csv format, where the customer's comments are enclosed in double quotes, and after the comma is the 7Ps analysis results:
"er en del af en kæde så det er dem jeg skal have fat i", {"product":"", "price":"", "place":"kæde", "promotion":"", "people":"", "process":"", "physical evidence":""}
"vil gerne høre lidt mere om det, et højaktuelt emne og hvad der er på markedet, omvendt det bekymrer ham at give 3.", {"product":"", "price":"", "place":"markedet", "promotion":"", "people":"", "process":"", "physical evidence":""}
"Vi skal tale med Mogens Bang om den her.", {"product":"", "price":"", "place":"", "promotion":"", "people":"", "process":"", "physical evidence":""}
"Er ved at få udvidet butik og har mange kasser stående, efter uge 40 kunne det være interessant, da de åbner en del flere kvadratmeter.", {"product":"kasser", "price":"", "place":"butik", "promotion":"", "people":"", "processes":"", "physical evidence":""}
"De har 2 butikker og en webshop, det de tidligere de har undersøgt var at man skulle bestille 100 styk, og det skal man ikke her, det synes hun var godt.", {"product":"", "price":"", "place":"butikker", "promotion":"", "people":"", "processes":"styk", "physical evidence":""}
"Vi er en multibrand store, han tror ikke det er aktuelt, han tror ikke de kan sælge noget med deres Cadovius brand i, folk skal kunne kende det so mde har på.", {"product":"", "price":"", "place":"store", "promotion":"multibrand", "people":"", "processes":"", "physical evidence":""}
"hans brand er et mærke som består af klassiske designs, som alt sammen produceres i Italien, økologisk bomuld etc.", {"product":"designs, bomuld, økologisk", "price":"", "place":"Italien", "promotion":"brand", "people":"", "processes":"produceres", "physical evidence":""}
"Hvor får vi produceret henne? Send en mail, hvem har vi i Aalborg.", {"product":"", "price":"", "place":"Aalborg", "promotion":"", "people":"", "processes":"produceret", "physical evidence":""}
"Nejtak Er medlem af en indkøbsforening, mister og Min tøjmand, som han faktisk er bestyrelsesformand for.", {"product":"", "price":"", "place":"indkøbsforening", "promotion":"", "people":"bestyrelsesformand", "processes":"", "physical evidence":""}
"Opfølgning på Norge, Ny Dobbelthal samt forbrig, 30 nye stationer i dk. samt 4g dongle i Ikast", {'product':'', 'price':'', 'place':'Norge, ny dobbelthal, stationer, Ikast', 'promotion':'', 'people':'', 'processes':'', 'physical evidence':''}
"Dimensioner under transport.", {'product':'dimensioner', 'price':'', 'place':'transport', 'promotion':'', 'people':'', 'processes':'', 'physical evidence':''}
""",
            },
            "Promotion": {
                "type": "string",
                "description": """
According to the 7Ps Marketing Mix, identify and extract the 'Promotion' of 7Ps using nouns. Remeber it's the ways that we communicate and advertise our product or service to our target market.

Below are some examples of 7Ps analysis for some customer comments in csv format, where the customer's comments are enclosed in double quotes, and after the comma is the 7Ps analysis results:
"Christian har skrevet at det ikke var aktuelt lige nu med hjemmesiden til udlejning, da der var kø, så der behøvede ikke PR", {"product":"hjemmesiden", "price":"", "place":"", "promotion":"PR", "people":"", "process":"kø", "physical evidence":""}
"Jan Milling skriver i mail 28 juni at han skal forhandle renovation i oktober og omtaler vores "fine materiale".", {"product":"renovation", "price":"", "place":"", "promotion":"omtaler", "people":"", "process":"", "physical evidence":""}
"Vi skal tale med Mogens Bang om den her.", {"product":"", "price":"", "place":"", "promotion":"", "people":"", "process":"", "physical evidence":""}
"Vi er en multibrand store, han tror ikke det er aktuelt, han tror ikke de kan sælge noget med deres Cadovius brand i, folk skal kunne kende det so mde har på.", {"product":"", "price":"", "place":"store", "promotion":"multibrand", "people":"", "processes":"", "physical evidence":""}
"hans brand er et mærke som består af klassiske designs, som alt sammen produceres i Italien, økologisk bomuld etc.", {"product":"designs, bomuld, økologisk", "price":"", "place":"Italien", "promotion":"brand", "people":"", "processes":"produceres", "physical evidence":""}
""",
            },
            "People": {
                "type": "string",
                "description": """
According to the 7Ps Marketing Mix, identify and extract the job titles or positions related to the 'People' component. Focus on the roles involved in creating, delivering, and supporting the product or service, and exclude the specific names of individuals.

Below are some examples of 7Ps analysis for some customer comments in csv format, where the customer's comments are enclosed in double quotes, and after the comma is the 7Ps analysis results:
"Ikke inde i Niras, han er facility mand", {"product":"", "price":"", "place":"", "promotion":"", "people":"facility mand", "process":"", "physical evidence":""}
"Har givet den videre til Jan, som er ejer.", {"product":"", "price":"", "place":"", "promotion":"", "people":"", "process":"", "physical evidence":""}
"Bad MM ringe tilbage eller sende tid for muligt møde på sms.", {"product":"", "price":"", "place":"", "promotion":"", "people":"", "process":"", "physical evidence":""}"Helle Pedersen gik direkte på tlfsvarer.", {"product":"", "price":"", "place":"", "promotion":"", "people":"", "process":"", "physical evidence":""}
"Vi skal tale med Mogens Bang om den her.", {"product":"", "price":"", "place":"", "promotion":"", "people":"", "process":"", "physical evidence":""}
"Hun står med en kunde.", {"product":"", "price":"", "place":"", "promotion":"", "people":"kunde", "processes":"", "physical evidence":""}
"chefen er på ferie, prøv om 14 dage.", {"product":"", "price":"", "place":"", "promotion":"", "people":"chefen", "processes":"", "physical evidence":""}
"Nejtak Er medlem af en indkøbsforening, mister og Min tøjmand, som han faktisk er bestyrelsesformand for.", {"product":"", "price":"", "place":"indkøbsforening", "promotion":"", "people":"bestyrelsesformand", "processes":"", "physical evidence":""}
"Jeg bliver i branchen, han arbejder som konsulent for Pasform for Zizzi i Billund og Sandgaard i Ikast dels for at styre deres&nbsp; Har en online shop til store piger med, Bambus, polyamid, viskose.", {"product":"online shop, Bambus, polyamid, viskose", "price":"", "place":"", "promotion":"", "people":"konsulent, store piger", "processes":"", "physical evidence":""}
"Skal sendes Direkte til kunden", {'product':'', 'price':'', 'place':'', 'promotion':'', 'people':'kunden', 'processes':'', 'physical evidence':''}
"Han kendte en ven i Alpaco Phillip som pressede på for at han skulle finde en i Pension Danmark. Brian Krogh måske, men Morten gider ikke bruge mere tid på det.", {'product':'', 'price':'', 'place':'', 'promotion':'', 'people':'ven', 'processes':'', 'physical evidence':''}
"Der er ingen Linda Segerfeldt ansat her Kathrine, tror ikke det er relevant vi er en del af H&amp;M, det kommer fra Stockholm deres produktion af nyhedsbreve. Vi skal igennem H&amp;M, . Content Koordinator Julie 28757751 kan jeg tale med", {'product':'', 'price':'', 'place':'Stockholm', 'promotion':'', 'people':'Content Koordinator', 'processes':'', 'physical evidence':''}
"kontakt kontoret ring til kæde chef er ved at skære ned på deres leverandører og vil være gode ved dem de har wunderwear samba se om jeg kan finde noget ellers ring igen så har hun et nummer", {'product':'', 'price':'', 'place':'', 'promotion':'', 'people':'kæde chef, leverandører', 'processes':'', 'physical evidence':''}

Remember, please DO NOT extract any individual human names and extract only the job titles or positions. If no job titles or positions are found, the value of the key "people" in result should be an empty string.
""",
            },
            "Process": {
                "type": "string",
                "description": """
According to the 7Ps Marketing Mix, identify and extract the 'Process' of 7Ps using nouns. Remeber it's the steps and procedures that we follow to ensure quality and efficiency in your product or service delivery.

Below are some examples of 7Ps analysis for some customer comments in csv format, where the customer's comments are enclosed in double quotes, and after the comma is the 7Ps analysis results:
"Har ringet ind til deres fysioterapi tidsbestilling, men det er outsourcet til Meyers køkkener, så det er nok dem jeg skal tale med", {"product":"", "price":"", "place":"", "promotion":"", "people":"", "process":"outsourcet", "physical evidence":""}
"Thorballe var positiv, mente tilbuddet var som det skulle være, og det hele er kun stoppet grundet outsourcing, som de ikke kendte til.", {"product":"", "price":"", "place":"", "promotion":"", "people":"", "process":"outsourcing", "physical evidence":""}
"Vi skal tale med Mogens Bang om den her.", {"product":"", "price":"", "place":"", "promotion":"", "people":"", "process":"", "physical evidence":""}
"De har 2 butikker og en webshop, det de tidligere de har undersøgt var at man skulle bestille 100 styk, og det skal man ikke her, det synes hun var godt.", {"product":"", "price":"", "place":"butikker", "promotion":"", "people":"", "processes":"styk", "physical evidence":""}
"Helle, fordi vi skal ikke investere mere i noget med de salgstider de har.", {"product":"", "price":"investere", "place":"", "promotion":"", "people":"", "processes":"salgstider", "physical evidence":""}
"Hvor får vi produceret henne? Send en mail, hvem har vi i Aalborg.", {"product":"", "price":"", "place":"", "promotion":"", "people":"", "processes":"produceret", "physical evidence":""}
"hans brand er et mærke som består af klassiske designs, som alt sammen produceres i Italien, økologisk bomuld etc.", {"product":"designs, bomuld, økologisk", "price":"", "place":"Italien", "promotion":"brand", "people":"", "processes":"produceres", "physical evidence":""}
"Hvor får vi produceret henne? Send en mail, hvem har vi i Aalborg.", {"product":"", "price":"", "place":"Aalborg", "promotion":"", "people":"", "processes":"produceret", "physical evidence":""}
"Henrik bad om en liste med alle sagerne på mail. Han kigger dem igen og sende kommentar tilbage. Lars Lynderup Hansen svarer ikke. Lagt besked på svarer Lars oplyser at den har vi leveret", {'product':'', 'price':'', 'place':'leveret', 'promotion':'', 'people':'', 'processes':'liste, sagerne', 'physical evidence':''}
"Mødet er booket indtil videre med Morten.Desuden; Vi kører nogle projekter på Arla, IT standarder og automationsstandarder, navne standarder bliver det første. De afholder 2 timers undervisning i Viby med automationsfolk fra alle mulige steder og de vil gerne invitere vores kontakt til Arla ind for at være med i den undervisning så vi bedre forstår hvad Arla skal bruge.", {'product':'', 'price':'', 'place':'', 'promotion':'', 'people':'automationsfolk', 'processes':'projekter, IT standarder, automationsstandarder, navne standarder, undervisning', 'physical evidence':''}
"Fjedre uden 3.1 certifikat Forsendelse er sendt fra Ekato d. 14/09", {'product':'fjedre', 'price':'', 'place':'forsendelse', 'promotion':'', 'people':'', 'processes':'certifikat', 'physical evidence':''}
"har hun ikke brug da det er en second hand, hun har", {'product':'', 'price':'', 'place':'', 'promotion':'', 'people':'', 'processes':'second hand', 'physical evidence':''}
"Rikke@libenodic.dk vil gerne have noget konkret på hvad vi har lavet og nogle pris eksempler hvilke systuer arbejder vi med. er det nogle ordenlige forhold de arbejder under, (det er vigtigt) er det nogle kontrolleret forhold de arbejder under?", {'product':'', 'price':'pris', 'place':'systuer', 'promotion':'', 'people':'', 'processes':'ordentlige forhold, kontrolleret forhold', 'physical evidence':''}
"Henrik bad om en liste med alle sagerne på mail. Han kigger dem igen og sende kommentar tilbage. Martin Laurberg oplyser at de ikke vandt opgaven. Den gik til hoved entreprenøren", {'product':'', 'price':'', 'place':'', 'promotion':'', 'people':'', 'processes':'liste', 'physical evidence':''}
"Udestående punkter: The torque for the fixing of the set screws will be suppliedWrong screw connections for the mounting of the gearbox on the lantern will be sent via tracked shipment to Yara'EKATO will send shipping notification when agitators leave our premisesData Sheet of the hardened set screw material or additional information will be sent to Yara 11-08 HPL Rykker Nico for update 14-08 HPL Rykker igen", {'product':'torque, set screws, screw connections, gearbox, lantern, agitators, Data sheet', 'price':'', 'place':'premises', 'promotion':'', 'people':'', 'processes':'fixing, supplied, mounting, shipment, tracked, shipping notification', 'physical evidence':''}
"Robert Voss har fået tilsendt OB da han bestilte O-ringene. Der er aldrig kommet en ordre. Gensender OB og beder om accept samt PO. Tilbud skal opdateres hvis Robert forsat ønsker O-ringene. 08-08-2023 - Reminder sendt til Robert 080-08-2023 - Robert vender tilbage med en lang liste over det som O-ringene skal efterleve. Eftersom den vedhæftede dokumentation kun nævner overensstemmelse til FDA, så mangler der en hel del ifht. vores krav angivet i V1.3.06 Specifikation of Food Contact Materials. Det gælder både EU lovgivning og egne krav. Vi skal have dokumentation for følgende:Declaration of compliance acc. to 1935/2004/EC on materials and articles intended to come into contact with foodDeclaration of compliance acc. to 2023/2006/EC Good Manufacturing PracticeDeclaration of compliance acc. to 1907/2006/EC concerning the Registration, Evaluation, Authorization and Restriction of Chemicals (REACH)Documentation for the suitability of the FCM to specified food category and conditions of useFree of Animal Derived Ingredients and produced without any culture steps (fermentation)Declaration stating that the materials are free of Bisphenol A.Declaration stating that the materials are free of phthalates.Declaration stating that the materials are free of latexHvis det er Nordic Engineering, der lagerfører og sælger O-ringene til os skal de også dokumentere:Declaration of compliance acc. to Bek. 681/2020 Food contact Material (Danish suppliers)Declaration of compliance acc. to Bek. 1352/2019 Authorization and registration (Danish suppliers)", {'product':'O-ringene', 'price':'Tilbud', 'place':'', 'promotion':'', 'people':'', 'processes':'ordre, dokumentation, EU lovgivning', 'physical evidence':''}
""",
            },
            "Physical evidence": {
                "type": "string",
                "description": """
According to the 7Ps Marketing Mix, identify and extract the 'Physical evidence' of 7Ps using nouns. Remeber it's the tangible and intangible aspects that show and prove our product or service quality and value.

Below are some examples of 7Ps analysis for some customer comments in csv format, where the customer's comments are enclosed in double quotes, and after the comma is the 7Ps analysis results:
"Vi skal tale med Mogens Bang om den her.", {"product":"", "price":"", "place":"", "promotion":"", "people":"", "process":"", "physical evidence":""}
""",
            },
        },
        "required": ["Product", "Price", "Place", "Promotion", "People", "Process", "Physical evidence"],
    }
    chain = create_tagging_chain(schema, llm)
    ##### split notes to sentences
    _sentences = []
    for i in txt_lines:
        i_li = i.strip()
        if i_li:
            for j in i_li.split(". "):
                jj = ""
                if j[-1] == '.':
                    jj = j
                else:
                    jj = j+"."
                _sentences.append(jj)
    # print(len(_sentences))
    ##### 
    _log = ""
    _total_cost = 0
    _7P = []
    ##### call OpenAI API with _content and _example
    _log += "-" * 40 + "\n"
    for i in range(0, len(_sentences)):
        [i_re, i_tokens, i_cost, i_log] = call_openai_tagging(chain, _sentences[i])
        _log += i_log
        _total_cost += i_cost
        _7P.append(i_re)
    _total_cost_str = format(_total_cost, ".5f")
    # print(len(_7P))
    # print(_7P)
    ##### parse response, generate _log and _7P_str
    _log += "-" * 40 + "\n"
    _log += str(_7P) + "\n"
    _log += "-" * 40 + "\n"
    _log += f"\nTotal Cost: ${_total_cost_str}\n"
    _7P_str = ""
    if len(_sentences) == len(_7P):
        _7P_str = str(_7P)
    else:
        _log += "Error: len(sentences) != len(7P)" + "\n"
    return [_log, _7P_str, _total_cost_str, _sentences]

def P7_llm_tagging(_txt):
    import re
    _log = ""
    _7P_str = ""
    _total_cost = 0
    txt_lines = _txt.split("\n")
    [_log, _7P_str, _total_cost_str, _sentences] = P7_openai_tagging(txt_lines)
    # print(_log)
    # print(_7P_str)
    import ast
    _7P = ast.literal_eval(_7P_str)
    # print(type(_7P), _7P)
    return [_7P, _total_cost_str]



if __name__ == "__main__":

    # _txt = "Ved ikke om de har noget organisk affald... på deres hovedkontor har de et køkken, men det er en ekstern operatør der driver det... det er Michael Kjær fra driften, et fælles køkken med andre virksomheder.. Ring til ham om det. NCC bestemmer desuden selv om de skal have vores projekt med i loopet på dgnb point i byggeriet... i deres koncept udvikling...; De er ved at definere det og vi kan vende retur til Martin i Januar, hvor han ved hvem vi skal have møde med om det."
    # [_re, _cost] = P7_llm(_txt)
    # print("\n>>>", type(_re), _re)
    # print(type(_cost), _cost)

    _txt = """Har ringet ind til deres fysioterapi tidsbestilling, men det er outsourcet til Meyers køkkener, så det er nok dem jeg skal tale med.
er en del af en kæde så det er dem jeg skal have fat i.
han sidder i VVS og har ikke meget med de her sager at gøre.
Christian har skrevet at det ikke var aktuelt lige nu med hjemmesiden til udlejning, da der var kø, så der behøvede ikke PR.
Ikke inde i Niras, han er facility mand.
Jan Milling skriver i mail 28 juni at han skal forhandle renovation i oktober og omtaler vores 'fine materiale'.
Thorballe var positiv, mente tilbuddet var som det skulle være, og det hele er kun stoppet grundet outsourcing, som de ikke kendte til.
Der har været Opslag på Linked In af Laust omkring affaldssortering og vi talte om at tage vores dialog videre.
vil gerne høre lidt mere om det, et højaktuelt emne og hvad der er på markedet, omvendt det bekymrer ham at give 3.
Har givet den videre til Jan, som er ejer.
Jeg foreslog møde sidste halvdel Juli eller i august.
De har fået at vide, at det ville være på plads om 2-3 uger.
Bad MM ringe tilbage eller sende tid for muligt møde på sms.
Helle Pedersen gik direkte på tlfsvarer.
Vil gerne have en mail sådan at han kan give en status og han vil også gerne have en intromail.
Rune vil ringe til kunden imorgen høre dem ad.
Talte med ham ifm.
bud til Bent Brandt.
Fullgte 12 maj op på, hvorvidt den vandt gehør.
Han har ikke nævnt nogen specifik dato men sagde ca om 14 dage hvilket Er denne dato.
Følge op på denne&nbsp;.
Skal ringe igen om 3 uger.
Opfølgning 29/6-2023.
Opfølgning på lejeaftale.
Vi skal tale med Mogens Bang om den her.
bud til Bent Brand.
opfølgning 16 Juni.
"""
    [_re, _cost] = P7_llm_tagging(_txt)
    print(type(_re))
    for i in _re:
        print(i)
    print(type(_cost), _cost)

