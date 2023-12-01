import asyncio
from markdown_it import MarkdownIt
import pdfkit
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from streamlit_chat import message
from langchain.prompts import PromptTemplate
from file import UploadedFile
questions = {
  "Target Market": [
    "Who's the customer? Key characteristics?",
    "Is the market large now or in the future?",
    "What is the growth rate?",
    "How do you calculate the market size: # potential customers x average revenue per customer?",
    "Is the Serviceable Available Market (SAM) $10B+?",
    "What is the Serviceable Obtainable Market (SOM)?"
  ],
  "Problem or Need": [
    "What's the core problem/need to be solved?",
    "How severe is the pain or challenge?"
  ],
  "Solution": [
    "What is the core value proposition?",
    "Does it provide better, faster, and/or cheaper solutions?",
    "What is the ROI for customers?",
    "What is the magnitude of improvement over the status quo?",
    "What is the unique approach?",
    "Is there a strong brand perspective?"
  ],
  "Team, Board, Advisors": [
    "What is the team's prior career?",
    "What are their successes/failures?",
    "Do they have industry knowledge?",
    "What are they uniquely skilled at: raising capital, generating revenues, building a team, exiting?",
    "Do they have any key relationships?",
    "What is the secret they unlocked?"
  ],
  "Traction": [
    "Has a Minimum Viable Product been built?",
    "What are the key things learned?",
    "Has the ROI been validated?",
    "What are the key metrics: # Customers, ARR (Annual Recurring Revenue)?",
    "Is there a revenue pipeline?",
    "Are there any strategic partnerships?",
    "What is the capital efficiency: ARR / Capital Spent To Scale?",
    "What is the LTV (Lifetime Value) / CAC (Customer Acquisition Cost)?"
  ],
  "Revenue Model": [
    "What is the revenue per customer?",
    "Is it a subscription-based or one-time revenue model?",
    "What is the frequency of revenue generation?",
    "How is the LTV calculated?",
    "What are the ways to increase LTV?",
    "What is the length of the sales cycle?",
    "Is it a high-price or high-volume strategy?"
  ],
  "Strategy": [
    "What are the key expenses/time efforts involved?",
    "What are the COGs (Cost of Goods Sold)?",
    "What is the gross margin percentage?",
    "How can the gross margin be improved?",
    "What are the biggest expenses?",
    "Where is the majority of time spent?",
    "What is the Cost to Acquire & Maintain Customers (CAC)?",
    "What is the marketing channel strategy?"
  ],
  "Financial": [
    "What were the last 1-2 year revenues?",
    "What are the projected 1-5 year revenues?",
    "What is the breakdown of gross vs. net revenue?",
    "What are the burn rate per month and cash runway?",
    "What is the path to profitability?",
    "How much total capital has been spent: % Build vs. % Scale?",
    "What is the capital efficiency: ARR / Capital Spent To Scale?",
    "Is the goal to maintain or improve capital efficiency?",
    "What market penetration percentage is required to hit revenue targets?"
  ],
  "Competition": [
    "Who are the direct and indirect competitors?",
    "What are the barriers to entry?",
    "What is the main differentiation from competitors?",
    "Are there simpler alternatives that are good enough?",
    "Is there a risk of future obsolescence?",
    "Is there an unfair and/or sustainable competitive advantage?",
    "Are there any patents or key partnerships?",
    "What are the key risks (current and future)?"
  ],
  "Exit Opportunity": [
    "What types of acquirers are targeted?",
    "Why buy instead of building a similar solution?",
    "Is there potential for an IPO?",
    "What revenue multiple is expected?",
    "Is there a possibility of achieving a 10x+ return with dilution?"
  ],
  "Investment Terms": [
    "How much funding is being raised?",
    "What is the amount committed vs. available?",
    "What is the pre-money and post-money valuation?"
  ],
  "Post-$ Valuation / ARR": [
    "Who are the new and previous investors?",
    "Are previous investors coming back?",
    "What is the total capital previously raised?",
    "Is the preferred investment senior or pari-passu?",
    "Are there any debts/SAFEs/notes?"
  ],
  "Strategic Value": [
    "How can potential partners help?",
    "Can they provide introductions to customers, partners, strategics, investors, or employees?"
  ]
}
prompt_template = """
You are Theus AI, an helpful assistant who helps venture investors in analyzing startups specifically on Competitors, Market Sizing and Team. You MUST BE NEUTRAL and OBJECTIVE! Try to be assertive when it comes to potential red flags and know each investor only invests in 1 or 2% their deals per year. 
Your task is to answer questions about companies using there pitchdeck. 
"When asked a question" - You must use detailed math in step by step market sizing calculations, and give competitors (if you cannot find any around the stage you can give examples (creative and hypothetical).
If you dont know the answer to any question, ask the user to clarify or apologize. Never use dollar signs since output is LaTeX formatted, instead use the word "dollars" and you must have detailed math.
Use as much creative information in your database to answer the question before apologizing, or IF YOU make an hypothesis and disclaim it.   
{context}

Question: {question}
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)





@st.cache_data()
def on_file_upload():
    st.session_state['uploading'] = True

@st.cache_data()
def format_chat_history_to_html(past, generated):
    chat_history_html = "<html><body>"
    for i in range(len(past)):
        chat_history_html += f"<p>User: {past[i]}</p><p>AI: {generated[i]}</p><hr>"
    chat_history_html += "</body></html>"
    return chat_history_html

@st.cache_data()
def generate_context_and_question(topic, question):
    # Format the context and question based on user selections
    context = f"Topic: {topic}\n"
    formatted_question = f"Question: {question}"
    return context, formatted_question

    
async def main():
    st.set_page_config(
        page_title="VentureCopilot Demo",
        page_icon=":rocket:",
        layout="wide",  # make the app expanded
        initial_sidebar_state="expanded"  # expand the sidebar by default
    )
    #code_check = st.text_input("Enter the invite code here?")
    #if code_check != "VCOPDEMO203OCt":
    #    st.error("Sorry, you do not have access to this page.")
     #   return
    #else:
    #    st.success("Welcome to the VentureCopilot Demo!")
    async def conversational_chat(query):
        print(query)
        try:
            # Get the formatted context and question based on user selections
            context, formatted_question = generate_context_and_question(topic, question)
            # Combine the context and question with the user's query
            full_query = f"{context}\n{formatted_question}\n{query}"
            result = qa({"question": full_query}, return_only_outputs=True)
        except Exception as e:
            st.error(f'An error occurred: {e}')
            return
        st.session_state["history"].append((query, result["answer"]))
        return result["answer"]
    if "history" not in st.session_state:
        st.session_state["history"] = []


    st.title("VentureCopilot Demo :rocket:")
    
    st.markdown("""
        ## Welcome to VentureCopilot Demo!
        VentureCopilot is here to assist venture investors in analyzing startups on Competitors, Market Sizing, and Team and beyond!
        Follow the instructions below to get started:
        
        1. **Upload**: Use the file uploader to provide your pitch deck files (PDF/DOCX).
        2. **Query**: Once the file is uploaded, you can either:
            - Type your question, thoughts or comments in the query box and hit 'Send', or
            - Select a topic and question from the dropdown menus below the query box.
        3. **Interact**: Review the responses, and you can continue to ask more questions or adjust your queries as needed.
        4. **Feedback**: Your feedback is invaluable! Please consider submitting your feedback [here](https://noteforms.com/forms/venturecopilot-demo-feedback-form-f9coda).
        5. **Waitlist**: Excited about our demo? Join our waitlist [here](https://noteforms.com/forms/venturecopilot-waitlist-7ggee7).
    """)
    


    uploaded_file = st.file_uploader(
        label="Choose files (PDF/DOCX)", 
        type=['pdf', 'docx'], 
        accept_multiple_files=False, 
        help="Upload one or more files in PDF or DOCX format.",
        on_change=on_file_upload
    )

    if uploaded_file is not None:
        with st.spinner("Processing..."):

            uploaded_file.seek(0)
            file = UploadedFile(uploaded_file)
            vectors = file.get_vector()

            memory = ConversationBufferMemory(
                memory_key="chat_history", return_messages=True
            )
            qa = ConversationalRetrievalChain.from_llm(
                ChatOpenAI(model_name="gpt-4",
                            streaming=True,
                            openai_api_key=st.secrets["OPEN_AI_KEY"],
                            temperature=0.15,
                            ),
                retriever=vectors.as_retriever(),
                return_source_documents=False,
                memory=memory,
                qa_prompt=PROMPT,
            )

        st.session_state["ready"] = True


    st.divider()

    if st.session_state.get("ready", False):

        if "generated" not in st.session_state:
            st.session_state["generated"] = [
                "Welcome, I am Theus AI, at your service! You can now ask any questions regarding the uploaded pitch materials."
            ]

        if "past" not in st.session_state:
            st.session_state["past"] = ["Hey, shall we get started?"]

        # container for chat history
        response_container = st.container()

        # container for text box
        container = st.container()

        with container:
            with st.form(key="my_form", clear_on_submit=True):
                user_input = st.text_input(
                    "Chat:",
                    placeholder="Ask information about the uplaoded pitch material",
                    key="input",
                )
                submit_button = st.form_submit_button(label="Send")

            st.markdown("---")  # markdown horizontal rule as a divider
            st.markdown("### OR")
            st.markdown("You can choose below from a curated set of questions and topics. Please avoid clicking both buttons for optimal output quality.")

            st.markdown("## :mag: Query Section")  # use emojis for better user orientation
            topic = st.selectbox('Select Topic', options=list(questions.keys()), key='topic_select')
            question = st.multiselect('Select Question', options=questions[topic], max_selections=10, key='question_select')



            # A button to submit the selected topic and question
            select_button = st.button("Submit Selection")

            if submit_button and user_input:
                with st.spinner("Working on it..."):
                    output = await conversational_chat(user_input)
                    st.session_state["past"].append(user_input)
                    st.session_state["generated"].append(output)
            elif select_button:
                # Construct a query string from the selected topic and question
                with st.spinner("Working on it..."):
                    query = f"{topic}: {question}"
                    output = await conversational_chat(query)
                    st.session_state["past"].append(query)
                    st.session_state["generated"].append(output)
                    

        if st.session_state["generated"]:
            st.balloons()
            with response_container:
                for i in range(len(st.session_state["generated"])):
                    message(
                        st.session_state["past"][i],
                        is_user=True,
                        key=str(i) + "_user",
                    )
                    message(
                        st.session_state["generated"][i],
                        key=str(i),
                    )
                    chat_history_html = format_chat_history_to_html(st.session_state["past"], st.session_state["generated"])
                    pdf_data = pdfkit.from_string(chat_history_html, False)
                    
                    st.download_button(
                        label="Download Chat History as PDF",
                        data=pdf_data,
                        file_name="chat_history.pdf",
                        mime="application/pdf",
                        key = f'1231241+{i}',
                      
                    )


    st.divider()  # Creates a horizontal line for separation

    st.markdown("## We value your feedback")
    st.markdown(
        "Please help us improve by providing your feedback "
        "[here](https://noteforms.com/forms/venturecopilot-demo-feedback-form-f9coda)."
    )

    st.markdown("## Join our waitlist")
    st.markdown(
        "Excited about our demo? Join our waitlist "
        "[here](https://noteforms.com/forms/venturecopilot-waitlist-7ggee7)."
    )

    st.markdown(
        "Designed with :heart: by Team VentureCopilot, Inc. Sending greetings from San Francisco. All rights reserved. :bridge_at_night:"
    )  # footer with emojis
    st.markdown("""
    ## :lock: Data Privacy Disclaimer
    
    This platform is designed to provide a free interaction with our potential AI solutions, 
    showcasing how they can assist in analyzing startups on Competitors, Market Sizing, and Team based on the uploaded pitch deck.
    
    :information_source: Please be informed that:
    - This app operates entirely within your browser.
    - No documents or data are stored, transmitted, or used outside of this session.
    - All interactions, including the chat and document analysis, are performed locally on your machine.
    - The speed and performance of the app are influenced by your device's capabilities and your internet connectivity.
    - Refreshing the screen will cause the app to reset, this means you will loose your progress, uploaded files.
    
    Our priority is to ensure a secure and private environment for you to experience the capabilities of our AI. 
    Enjoy exploring, and rest assured that your data remains with you at all times! :smile:
    
    Should you have any concerns or questions, feel free to reach out to our support.
    
""")
if __name__ == "__main__":
    asyncio.run(main())
