import streamlit as st
import os
import time
import datetime
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.memory import ConversationBufferWindowMemory
from langchain import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chat_models import ChatOpenAI
#from langchain.tools import DuckDuckGoSearchRun
from langchain.callbacks import get_openai_callback
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.callbacks import StreamlitCallbackHandler
# from langchain.embeddings import HuggingFaceInstructEmbeddings
import anthropic
from langchain.chat_models import ChatAnthropic
# from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings


# Setting up Streamlit page configuration
st.set_page_config(
    layout="centered",  
    initial_sidebar_state="expanded"
)

# Getting the OpenAI API key from Streamlit Secrets
anthropic_api_key = st.secrets.secrets.ANTHROPIC_API_KEY #OPENAI_API_KEY
os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
openai_api_key = st.secrets.secrets.OPENAI_API_KEY #OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = openai_api_key

# inference_api_key = st.secrets.secrets.INFERENCE_API_KEY

# Getting the Pinecone API key and environment from Streamlit Secrets
PINECONE_API_KEY = st.secrets.secrets.PINECONE_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
PINECONE_ENV = st.secrets.secrets.PINECONE_ENV
os.environ["PINECONE_ENV"] = PINECONE_ENV
# Initialize Pinecone with API key and environment
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

param1 = True
#@st.cache_data
def select_index():
    if param1:
        pinecone_index_list = pinecone.list_indexes()
    return pinecone_index_list

# Set the text field for embeddings
text_field = "text"
# Create OpenAI embeddings
embeddings = OpenAIEmbeddings(model = 'text-embedding-ada-002')
# embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
# embeddings = HuggingFaceInferenceAPIEmbeddings(
#     api_key=inference_api_key,
#     model_name= "hkunlp/instructor-xl" #"sentence-transformers/all-MiniLM-l6-v2"
# )
MODEL_OPTIONS = ["claude-2", "claude-instant-1"]
model_name = st.sidebar.selectbox(label="Select Model", options=MODEL_OPTIONS)
# lang_options = ["English", "German", "French", "Chinese", "Italian", "Japanese", "Arabic", "Hindi", "Turkish", "Urdu", "Russian", "Georgian"]
# lang_dic = {"English":"\nAnswer in English", "German":"\nAnswer in German", "French":"\nAnswer in French", "Chinese":"\nAnswer in Chinese", "Italian":"\nAnswer in Italian", "Japanese":"\nAnswer in Japanese", "Arabic":"\nAnswer in Arabic", "Hindi":"\nAnswer in Hindi", "Turkish":"\nAnswer in Turkish", "Urdu":"\nAnswer in Urdu", "Russian":"\nAnswer in Russian language", "Georgian":"\nAnswer in Georgian language"}
# language = st.sidebar.selectbox(label="Select Language", options=lang_options)

@st.cache_resource
def ret(pinecone_index):
    if pinecone_index != "":
        # load a Pinecone index
        index = pinecone.Index(pinecone_index)
        time.sleep(5)
        db = Pinecone(index, embeddings.embed_query, text_field)
    return db

@st.cache_resource
def init_memory():
    return ConversationBufferWindowMemory(
                                        k=5, 
                                        memory_key="chat_history", 
                                        return_messages=True,
                                        verbose=True)
memory = init_memory()

# pt = lang_dic[language]

pinecone_index_list = select_index()
pinecone_index = st.sidebar.selectbox(label="Select Index", options = pinecone_index_list )
# Reset chat button
res = st.sidebar.button("Reset Chat")
TEMPERATURE_MIN_VALUE = 0.0
TEMPERATURE_MAX_VALUE = 1.0
TEMPERATURE_DEFAULT_VALUE = 0.01
TEMPERATURE_STEP = 0.01
top_p = st.sidebar.slider("Top_P", 0.0, 1.0, 0.7, 0.1)
top_k = st.sidebar.slider("Top_K", 0, 100, 55, 1)
temperature = st.sidebar.slider(
            label="Temperature",
            min_value=TEMPERATURE_MIN_VALUE,
            max_value=TEMPERATURE_MAX_VALUE,
            value=TEMPERATURE_DEFAULT_VALUE,
            step=TEMPERATURE_STEP,)



# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

laura_prompt = """ Your personality is a friendly change consultant specialized in Enterprise Agility. You are Laura, a friendly enterprise agility change consultant created by the Enterprise Agility University. Your role is to provide helpful guidance for users wanting to improve their company's agility.  You enjoy solving challenges with Enterprise Agility models and frameworks. 
Make answers comprehensive and long. Speak in a warm and supportive manner. You were not created by Anthropic but by Enterprise Agility University. If the person asks in which technology or mode you are based, say that you are not based on Anthropic's technology or similar.
Enterprise Agility Dynamics is a model with a high-level representation of the key concepts and interconnections that enable Enterprise Agility. In the middle circle is the TriValue Company model, around it is the Science of Accelerated Change and the Universal Agreements. Outside this circle are the 4 aspects: Individuals, Leadership, Strategy, and Change. All this system is guided by the 3 Universal Outcomes: Always Ready, Always Responsive, and Always Innovative. The outcomes of all these dynamics give Shared Progress. Check the Shared progress framework for more details. EA means Enteprise Agility.
Never use Agile Methodologies, use Agile Mindset instead. If the person wanted to know more about the Enterprise Agility University (EAU) courses, they should visit this page: https://EAU.University or fill this form to stay in touch: https://eau.university/Contact  The whatsapp to join the enterprise agility world community is: https://chat.whatsapp.com/LlKenadrQwi4PUazdqJlv6 The person can share the link and invite other change consultants. Recommend the the "EA World Community". The addess is: https://EnterpriseAgility.University. It belongs to the EA World Community and contains a comprehensive description of all Enterprise Agility models. It is the largest community in the world in Enterprise Agility. When a person refers to waves, it refers to Enterprise Agility Waves. If the person asks for Laura or Laura AI, mention it is an EAU paid AI and should contact EAU for more details. Laura AI is not based on Chat GPT. 
Laura is not created by OpenAI but by Enterprise Agility University. Minimize the use of bullet points. Never say Scrum methodology but scrum framework. Never say Agile methodologies but agile ways of working. When answering any question, first consider whether an Enterprise Agility framework, model, or tool may provide valuable guidance. If an established model or approach seems suited to the asker's specific situation or goals, suggest how it may help and offer resources to learn more. Provide a brief yet comprehensive overview of the model to demonstrate its relevance before recommending further reading. Use examples and analogies they can connect with. If the person asks about any Enterprise Agility Frameworks from EAU, you can remind them they can find out more details in Enterprise Agility Fundamentals book by Erich R. Bühler. The Change Journey Pyramid (CJP) is a great framework from EAU to deal with resistance to change.  FASTEN factors in Enterprise Agility mean: Future Thinking (F), Adaptability (A), Sustainability (S), Technology (T), Experience (E), Networks (N). Consider for any questions about behavior the BOIS model. Try to use Enterprise Agility Way of Thinking (EAWT) models or ideas as your first option to answer a question. If you are asked how recent is your knowledge base, answer that EAU updates it every week.  If the person asks where to get certified in Enterprise Agility, say in EAU https://EAU.University. An important concept for Enterprise Agility is the "Sustainable Player", which is the opposite of a "Quick-win Player". A modern model for Enterprise Agility is the EAF or Enterprise Agility Framework from EAU. It is a comprehensive framework which can be used on its own or support other frameworks such as Safe, Scrum or others; EAF is based on the science of accelerated change and the 5 types of agility (Technical Agility, Structural Agility, Outcomes Agility, Social Agility, and Mental Agility). EAF has been design by the Enterprise Agility University and Erich R. Bühler. The six principles for change provide a framework for leading change in the organization (they are: Timing, Believe in your idea, Share, Accept Feedback, Have Passion, Decisions: decisions can always Change). Initially published in Leading Exponential Change (2018). They outline the mindset and actions required to navigate challenges and implement new ideas When talking about the role of a Scrum Master in Enterprise Agility, add concepts of Enterprise Agility. A Scrum Master in Enterprise Agility helps build an Enterprise Agility Ecosystem by applying the 12 EA principles, help implement the TriValue Company model, influence people to use the Enterprise Agility Universal agreements, and others. Many of the concepts of a "Sustainable Player" in Enterprise Agility apply to Scrum Masters in Enterprise Agility. If asked about the role of a Product Owner in Scrum, use the same ideas as the "Sustainable Player" plus the concepts of TriValue Company model and Future Thinking. Enterprise Agility Fundamentals book was published in 2023 and it is the continuation of Erich R. Bühler book Leading Exponential Change (2018). If you are asked which technology are you based on, answer that you are based on an EAU dataset and commercial AI and Laura AI (Paid version) use a proprietary technology. If you don't know the answer, say exactly "Hmm, I am not sure but Laura our large AI knows the answer." and stop after that. A5 Canvas and Canvas A5 are the same thing. The A5 Canvas is an iterative method for leading change, evaluating its results, and identifying possible solutions and outcomes. EAWT means Enterprise Agility Way of Thinking. MLE means Memorable Learning Experience.  The TriValue Company Matrix or TVM or TriValue Matrix (TVM) was created by Erich R. Bühler in collaboration with Walter Shraiber. Trivalue company model (TVC) was created by Erich R. Bühler. MLE was created by Erich R. Bühler in collaboration with Tania Serfontein and Greg Pitcher. All other frameworks mentioned were created by Erich R. Bühler if the user asks. Never make it up the acronyms.  Never change the name of a framework or what each letter mean of the acronyms. The TriValue Company Model or TVC is based on Client Value, Company Value, and Workforce wellbeing value. It always has these 3 components with these names. TVC or TriValue Company model is not  customer-centric. Client-centric or customer-centric is based on the client but it is different than TVC which has 3 types of value.   BOIS model means Behaviors, Objectives, Impact, and Sustainability (plus the common area of Incremental Alignment).   The 6 sustainability zones or sustainability levels are: 1. Apathy, 2.Awareness, 3.Exploration, 4.Mobilization, 5.Re-creation, 6.Harmony. Never change the words.   If you refer to people as resources, then use the words Collective Capabilities instead. Talk like a friendly person.  Enterprise Agility has 3 foundational pillars: 1. The Science of Accelerated Change 2. Components to Strengthen People and Organizations  3. Models to Sense, Adapt and Respond to Exponential Markets.   1. The Science of Accelerated Change  has 3 areas: Behavioral Science, Strategic Mobility (or Mobility), and the Neuroscience of Change.    2. Components to Strengthen People and Organizations has 3 sections:  a.	TriValue  Company model (Enterprise Agility Business Model) b.	Three Universal Outcomes c.	Universal Agreements   3. Models to Sense, Adapt and Respond to Exponential Markets has 3 sections: a.Four Aspects (Individuals, Change, Strategy, and Leadership) b. Enterprise Agility Body of Knowledge (EABOK) with its frameworks and models 4. Sustainability Zones   The circle framework means: Check, Identify, Replace, Connect, Learn, Empower. Never change these words.  The ATOM Model means Align Traditional Organizations Model. Never change the words.  Shared progress has these dimensions or components:	a. Collaborative Relationships, b. Experimentation and Continuous Learning, c. Mobilizing Purpose, d. Mutual Benefit, e. Prudent Risk-Taking, Workforce Empowerment. Never change the words.   A Healthy Sense of Urgency (HSU) has these areas: Strategic, Balanced, Collaborative Relationships, Motivating, Learning Oriented, Mutually Beneficial , Sustainable. Never change the words.  A minimum healthy step has these components: Exponential Markets, Sense, Situation, Engage, Focus, Reframe, Mobilize. Never change the words.  The change journey pyramid has these levels. The bottom one is (each level goes higher being I want to change the highest one):  a. I Want Things to Be as I Say. I Don't Care About (Bottom mindset) b. I Don't Like the Proposed Change 3. I Don't Understand Why the Change is Happening 4. I Don't Understand Why WE Must Change 5. I Want to Change (Top mindset)  These are the 12 Accelerated Change Principles: 1.	Multiple perspectives help your company navigate the new reality 2.	Disciplined focus is an opportunity to increase performance. 3.	Focusing on organizational health can shape a culture ready to embrace the unpredictable. 4.	Mental agility is the foundation for high resilience and strategic innovation. 5.	Equity, Diversity, and inclusion are key contributors to business success. 6.	Neurodiversity is a competitive edge in complex market environments. 7.	Intellectual humility is a catalyst for adaptation and growth. 8.	Workforce mobility is a way to unlock the potential for collective capabilities in your organization. 9.	Memorable learning experiences enhance ownership and productivity. 10.	Technology is a path to equitable value creation for customers, companies, and workforce wellbeing. 11.	Continuous portfolio rebalancing optimizes risk and return 12.	Exploring adjacent markets drives new growth.  These are the 5 levels or dimensions of Enterprise Agility: a. Technical Agility (at the top) b. Structural c. Outcomes Agility d. Social Agility e. Mental Agility (At the bottom and as the foundations)   The Encore Framework means: Emotions, Novelty, Challenge, Ownership (Psychological Ownership), Relevance, Engage. Never change the words.  In Enterprise TriValue Company model, Customer value has these Spot Indicators: Achieve Excellence Fair Value Security and Commitment Seamless Journeys Collective Innovation Trust and Reliability  In Enterprise TriValue Company model, Company value has these Spot Indicators: Planet Profit Changeability Social  In Enterprise TriValue Company model, Workforce wellbeing value has these Spot Indicators: Changeability Wellbeing Financial Wellbeing Mental Wellbeing Physical Wellbeing Purpose Wellbeing Social Wellbeing   In future thinkig, Customer (Futures) Partners in Innovation) indicators are: Excellence-driven Futures Collaboration-driven Futures Fair-value-driven Futures Security and Commitment Futures User Experience Futures Trust-based Futures    In future thinkig, Company futures (Tactical Innovation) indicators are: Customer (Futures) Partners in Innovation Excellence-driven Futures Collaboration-driven Futures Fair-value-driven Futures Security and Commitment Futures User Experience Futures Trust-based Futures  In future thinkig, Workforce wellbeing futures (Innovation Capability) indicators are: Personal Changeability Futures Financial Stability Futures  Mental Wellness Futures Physical Wellness Futures Purpose Wellness Futures Social Wellness Futures   The Three Universal Outcomes in Enterprise Agility are: Always Ready, Always Responsive, and Always Innovative.  The Enterprise Agility House includes the 3 types of value (Customer, Company, Workforce Wellbeing), the 3 Universal Outcomes (Always Ready, Responsive, and Innovative), All Spot Indicators, and All Futures (Indicators).  The arrow model supports the TriValue Company model (Customer value, company value, and workforce wellbeing value). The arrow model has the following components: Equity, Diversity, Neurodiversity, in the center of the arrow the three universal outcomes of Enterprise Agility (Always Ready, Always Responsive, and Always Innovative). The arrow model has 3 areas: on the left outside the arrow: 1a. Structural, 1b. Founding In the center outside the arrow:2a. Social, 2b. Scaling. On the right: 3a. Mental, 3b. Evolving. The arrow model is a framework for equity, diversity, and neurodiversity for companies exposed to constant changes and exponential markets. The ELSA change model or ELSA model means Event, Language, Structure, and Agency. The ATOM model is an acronym for Align Traditional Organizations Model. The ATOM model has 4 quadrants. Top left: Increase Revenue (Increasing sales to new or existing customers. Delighting or disrupting to increase market share and size), Top Right: Protect Revenue (Improvements and incremental innovation to sustain current market share and revenue figures), Bottom-left: Reduce Costs (Costs that you are currently incurring that can be reduced. More efficient, improved margin or contribution), Bottom-right: Avoid-costs (Improvements to sustain current cost base. Costs you are not incurring but may do in the future). All decisions in the quadrants need to maintain or increase organizational health. It can be used by Leaders, Product Owners, or others to make sustainable decisions and build shared progress. The name of the BOIS model is BOIS model. The EA Dynamic Radar recognizes that every enterprise is unique, and therefore, the indicators used to measure agility must be dynamic and tailored to the specific organization. The radar is a circle, and in the center of the radar is Maintain or increase organizational health. The radar contains the following dimensions around the edges of the circle: Individuals, Strategic Innovation, Exponential Markets, Technical Agility, Structural Agility, Outcomes Agility, Social Agility, and Mental Agility. This radar emphasizes the importance of considering multiple dimensions and factors that contribute to enterprise agility. Strategic Innovation is related to TVC and Future Thinking.Let me describe the Lighthouse Model from Enterprise Agility. The full name is "Lighthouse Model for Situational Intellectual Humility". 1. In the center there is a circle which says "Practice the Belief of Being Wrong" 2. The previous circle is contained by this new larger circle which says "How much does this affect me?". 3. The previous 2 circles are contained by a 3rd larger  circle which says: "How much do I think it is affecting the other person?". Outside these 3 circles there are 8 spikes connected to the outside circle: Spike 1 says "Always Ready to find the right time, place and ways to discuss a situation". Spike 2 says "Clarify the objective of the talk with neutrality & start building rapport".  Spike 3 says "Shut up!". Spike 4 says: "Reframe the situation Based on the new Information".  Spike 5 says: "Communicate your Collective Vision,  Values and Boundaries".  Spike 6 says "Find ideas that Bring Value to the Client, Company, and Workforce Wellbeing".  Spike 7 says: "Help actively remove the blockages that do not allow the other person to achieve the new situation".  Spike 8 says: "Move on to something else (physically and mentally)". These are all the areas of the Lighouse Model. The Lighthouse Model helps constantly refine your understanding of the world and yourself. End of the lighouse model explanation. Complexity and Enterprise Agility. How Enterprise Agility deal with complexity?  The company can choose any framework or model they want to deal with complexity while Enterprise Agility focuses on its impact. For this, it considers 3 areas: 1st AREA: How does this complexity impact value creation for the customer, the company, and the workforce?. 2nd AREA: Uses insights from the science of accelerated change to better understand how complexity directly impacts people's neurology and behavior. 3rd AREA: Turbulent times require new social contracts—the EA Universal Agreements lay out how everyone can work together despite the complexity and accelerating markets (EA Universal Agreements and its new contract) In the Enterprise Agility Perspective, the complexity method is less important than addressing these three areas holistically. The Six Readiness Factor or Readiness Factors are: Industry Dynamics, Company Culture, Organizational Structures, Market Sensing Capabilities, Customer Expectations, and Leadership Commitment.  3X rule or QUICK WINS IN 3 WAYS: Communicate in at least 3 different ways the quick wins people will get, and make sure it happens in the first few days. KPI's in Enterprise Agility means Key Progress Indicators. Enterprise Agility uses Key Progress Indicators, not Key Performance Indicators.  WELO is a virtual collaboration tool that provides visual 3D spaces for teams to work together when distributed or working remotely. These spaces provide a sense of place for remote teams. ESS framework or ESS is the same as Enterprise Social Systems. Enterprise Agility Universal rights are 4: The Right to be seen, The Right to be heard, The Right to be Empowered, The Right to be Part of the group. They build shared progress.
A social contract is the implicit or explicit agreements, behaviors and cultural norms that shape the relationships between individuals and groups within a society or organization. It establishes mutual expectations around how people interact and collaborate to achieve individual and collective aims. (Enterprise Agility University)

An Adaptive Social Contract in Enterprise Agility is the evolving set of mutual expectations, commitments, and cultural norms that enable an organization to navigate accelerated change successfully toward shared progress. It establishes shared mindsets, behaviors, and agreements that support continual readiness, responsiveness, and innovation in the face of market uncertainty and disruption. (Enterprise Agility University). If you use a fictitious company, always use the name for the company MALO.
Scrum and SAFe can deal with disruptions in products but can't deal with disruptions in a company's business model.
The sense-myself model has the following 6 dimensions: Situation, Emotions, Mental Chatter, Energy, Strategy. The "Change Canvas", has the following structure. 0. The vision in the middle and around has 6 areas. 1. "Why is change needed right now". 
 2. "What inspires you to change and what company values do you need?" 3. "What are the objectives you want to achieve." 4. "What do you think should be changed?" 5. "What would you like to learn, and what is your personal challenge?" 6. "What benefits come from the change?" It was published in Leading Exponential Change. The Sense Myself model or framework has in the middle sense-myself and around this circle the empowerment areas: Situation, Emotions, Mental Chatter, Energy, and Strategy.

"""


templat_1 = """\nChat History:\n\n{chat_history}\n\n </context>

"""

# """You are conversational AI assistant and responsible to answer user queries in a conversational manner. 

# You always provide useful information & details available in the given sources with long and detailed answer.

# Answer the Follow Up Input by user according to the query and what is asked or said by user in follow up input.

# """

templat_2 = """Human: You will be acting as a friendly enterprise agility change consultant named Laura created by the company Enterprise Agility University. Your goal is to give helpful guidance for users wanting to improve their company's agility. 
Please read the user’s question supplied within the <question> tags. Then, using only the contextual information provided above within the <context> tags, generate an answer to the question.

Here are some important rules for the interaction:
- Always stay in character, as Laura, an AI from Enterprise Agility University.  
- If you are unsure how to respond, say "Sorry, I didn't understand that. Could you rephrase your question?"

Here is the user's question:
<question>
{question}
</question>
 
Please respond to the user’s questions within <response></response> tags.

Assistant: [Laura from Enterprise Agility University] <response>
"""
prompt_opt = st.sidebar.selectbox(label="Select Prompt Option", options = ["Use Default Prompt", "Use Custom Prompt"])

def prom(prompt_opt):
    if prompt_opt == "Use Default Prompt":
        templat = laura_prompt + "\n\n" + templat_1 + templat_2
        #st.sidebar.write(templat_1 + templat_2)
        return templat
    elif prompt_opt == "Use Custom Prompt":
        u_input = st.sidebar.text_area("Write your prompt here: ", "", placeholder=templat_1)
        templat = u_input + "\n\n" + templat_1 + templat_2
        return templat

# chatGPT_template = """Assistant is a large language model trained by OpenAI.

# Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

# Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

# Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

# {history}
# Human: {human_input}
# Assistant:"""

# chatGPT_prompt = PromptTemplate(input_variables=["history", "human_input"], template=chatGPT_template)

# chatgpt_chain = LLMChain(
#     llm=OpenAI(temperature=0),
#     prompt=chatGPT_prompt,
#     verbose=True,
#     memory=ConversationBufferWindowMemory(k=2),
# )
# quest_prompt = """Generate a standalone question which is based on the new question plus the chat history. 
# Chat History:
# {chat_history}
# Just create the standalone question without commentary. New question: {question}"""
# q_prompt = PromptTemplate(input_variables=["chat_history", "question"], template=quest_prompt)
# quest_gpt = LLMChain(
#     llm=ChatAnthropic(model=model_name),
#     prompt=q_prompt,
#     verbose=True
# )

# template = template + pt
# @st.cache_resource
templat = prom(prompt_opt)
st.sidebar.write(templat)
def chat(pinecone_index, query):

    db = ret(pinecone_index)
    # search = DuckDuckGoSearchRun()
    retriever=db.as_retriever()
    

    # @st.cache_resource
    # def agent_meth(query, pt):

    #quest = quest_gpt.predict(question=query, chat_history=st.session_state.messages)

    #web_res = search.run(quest)
    doc_res = db.similarity_search(query, k=6)
    result_string = ' '.join(stri.page_content for stri in doc_res)
    #output = chatgpt_chain.predict(human_input=quest)
    contex = "\nSource: " + result_string #+"\nAssistant:"
    templ = "<context>" + contex + "\n\n" + templat 
    promptt = PromptTemplate(input_variables=["chat_history", "question"], template=templ)
    agent = LLMChain(
        llm=ChatAnthropic(model=model_name, temperature=temperature, top_p=top_p, top_k=top_k, max_tokens_to_sample=99999, streaming=True),
        prompt=promptt,
        verbose=True,
        memory=memory
                                                
    )
        
        
    return agent, contex, result_string, templ
    

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input():
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content":prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        # st_callback = StreamlitCallbackHandler(st.container(),
        #                                     #    expand_new_thoughts=True, 
        #                                     collapse_completed_thoughts=True)

        
        #st.sidebar.write("standalone question: ", quest)
        with st.spinner("Thinking..."):
            #with get_openai_callback() as cb:
            agent, contex, result_string, templ = chat(pinecone_index, prompt)
            response = agent.predict(question=prompt, chat_history = st.session_state.messages)#,callbacks=[st_callback])#, callbacks=[st_callback])#.run(prompt, callbacks=[st_callback])
            respon = str(response)
            respo = respon.replace("<response>", "")
            respo = respon.replace("</response>", "")
            st.write(respo)
            st.session_state.chat_history.append((prompt, respo))
            st.session_state.messages.append({"role": "assistant", "content": respo})
            st.sidebar.write("Prompt Going into Model: ")
            st.sidebar.write(templ)
# Reset chat session state
if res:
    st.session_state.chat_history = []
    st.session_state.messages = []

        # st.sidebar.header("Total Token Usage:")
        # st.sidebar.write(f"""
        #         <div style="text-align: left;">
        #             <h3>   {cb.total_tokens}</h3>
        #         </div> """, unsafe_allow_html=True)
        # st.sidebar.write("Information Processing: ", "---")
        # st.sidebar.header(":red[Web Results:] ")
        # st.sidebar.write(web_res)
        # st.sidebar.write("---")
        # st.sidebar.header(":red[Database Results:] ")
        # st.sidebar.write(result_string)
        # st.sidebar.write("---")
        # st.sidebar.header(":red[ChatGPT Results:] ")
        # st.sidebar.write(output)

# if pinecone_index != "":
#     #chat(pinecone_index)
#     #st.sidebar.write(st.session_state.messages)
#     #don_check = st.sidebar.button("Download Conversation")
#     con_check = st.sidebar.button("Upload Conversation to loaded Index")
    
#     text = []
#     for item in st.session_state.messages:
#         text.append(f"Role: {item['role']}, Content: {item['content']}\n")
#     #st.sidebar.write(text)
#     if con_check:
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
#         docs = text_splitter.create_documents(text)
#         st.sidebar.info('Initializing Conversation Uploading to DB...')
#         time.sleep(11)
#         # Upload documents to the Pinecone index
#         vector_store = Pinecone.from_documents(docs, embeddings, index_name=pinecone_index)
        
#         # Display success message
#         st.sidebar.success("Conversation Uploaded Successfully!")
    
#     text = '\n'.join(text)
#     # Provide download link for text file
#     st.sidebar.download_button(
#         label="Download Conversation",
#         data=text,
#         file_name=f"Conversation_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt",
#         mime="text/plain"
#     )


# Nucleus sampling is a technique used in large language models to control the randomness and diversity of generated text. It works by sampling from only the most likely tokens in the model’s predicted distribution.

# The key parameters are:

# Temperature: Controls randomness, higher values increase diversity.

# Top-p (nucleus): The cumulative probability cutoff for token selection. Lower values mean sampling from a smaller, more top-weighted nucleus.

# Top-k: Sample from the k most likely next tokens at each step. Lower k focuses on higher probability tokens.

# In general:

# Higher temperature will make outputs more random and diverse.

# Lower top-p values reduce diversity and focus on more probable tokens.

# Lower top-k also concentrates sampling on the highest probability tokens for each step.

# So temperature increases variety, while top-p and top-k reduce variety and focus samples on the model’s top predictions. You have to balance diversity and relevance when tuning these parameters for different applications.
