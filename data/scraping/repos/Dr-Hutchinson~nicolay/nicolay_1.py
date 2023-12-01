from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts import FewShotPromptTemplate
from langchain.chains import LLMChain
from langchain import OpenAI, SerpAPIWrapper, Wikipedia
from langchain.agents import initialize_agent, Tool
from langchain.agents.react.base import DocstoreExplorer
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
import csv
from datetime import datetime as dt
import pandas as pd
import numpy as np
import os
import re
import streamlit as st
import pygsheets
from google.oauth2 import service_account
import ssl
import base64


scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']

credentials = service_account.Credentials.from_service_account_info(
                    st.secrets["gcp_service_account"], scopes = scope)

gc = pygsheets.authorize(custom_credentials=credentials)

st.set_page_config(
    page_title="Nicolay: Exploring the Speeches of Abraham Lincoln with AI (version 0.1)",
    layout='wide',
    page_icon='🔍'
)

os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
openai.api_key = os.getenv("OPENAI_API_KEY")
#os.environ["SERPAPI_API_KEY"] = st.secrets["serpapi_api_key"]

st.title("Nicolay: Exploring the Speeches of Abraham Lincoln with AI")


st.write("This application uses OpenAI's GPT AI models to answer questions about the collected speeches of Abraham Lincoln. Choose one of the options below, and pose a question about Lincoln's speeches.")
semantic_search = "Semantic Search: Enter a question, and recieve sections of Lincoln's speeches that are the most closely related semantically."
#ask_a_paragraph = "Ask a Paragraph: Internet Search. Select a Section from the text, and then pose a question. GPT-3 will search the internet to answer your question."
#ask_wiki = "Ask a Paragraph: Wikipedia. Select a Section from the text, and then pose a question. GPT-3 will search Wikipedia to answer your question."
ask_a_source = "Ask Nicolay: Pose a question about Lincoln's speeches, and a GPT AI model will share answers drawn from the text. This process can take several minutes to complete."


search_method = st.radio("Choose a method:", (semantic_search, ask_a_source))
model_choice = st.selectbox("Choose an AI model for Ask Nicolay:", ('ChatGPT', 'GPT-4'), index=1)
#section_number = st.number_input('Select a section number if you have selected Ask a Paragraph. You can find the section numbers to the bottom left, or through a semantic search.', step=1)
submission_text = st.text_area("Enter your question below. ")
submit_button_1 = st.button(label='Click here to submit your question.')
if submit_button_1:

    st.subheader("Nicolay's analysis is underway. It can take several minutes for every step of the process to be completed. Thank you for your patience. Nicolay's progress will be documented below.")

    os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]

    if model_choice == 'GPT-3.5':
        model_select = 'gpt-3.5-turbo'
    else:
        model_select = 'gpt-4'

    # semantic search via text embeddings with OpenAI Ada embedding model

    datafile_path = "lincoln_index_embedded.csv"

    df = pd.read_csv(datafile_path)
    df["embedding"] = df.embedding.apply(eval).apply(np.array)

    def embeddings_search():
        # try this
        datafile_path = "lincoln_index_embedded.csv"
        df = pd.read_csv(datafile_path)
        df["embedding"] = df.embedding.apply(eval).apply(np.array)

        def search_text(df, product_description, n=3, pprint=True):
            product_embedding = get_embedding(
                product_description,
                engine="text-embedding-ada-002"
            )

            df["similarities"] = df.embedding.apply(lambda x: cosine_similarity(x, product_embedding))

                # Select the first three rows of the sorted DataFrame
            top_three = df.sort_values("similarities", ascending=False).head(3)

                # If `pprint` is True, print the output
                #if pprint:
                    #for _, row in top_three.iterrows():
                        #print(row["combined"])
                        #print()

                # Return the DataFrame with the added similarity values
            return top_three

            # Call the search_text() function and store the return value in a variable
        results_df = search_text(df, submission_text, n=3)

                # Reset the index and create a new column "index"
        results_df = results_df.reset_index()

                    # Access the values in the "similarities" and "combined" columns
        similarity1 = results_df.iloc[0]["similarities"]
        combined1 = str(results_df.iloc[0]["combined"])

        similarity2 = results_df.iloc[1]["similarities"]
        combined2 = str(results_df.iloc[1]["combined"])

        similarity3 = results_df.iloc[2]["similarities"]
        combined3 = str(results_df.iloc[2]["combined"])

        num_rows = results_df.shape[0]

            # Iterate through the rows of the dataframe
        for i in range(num_rows):
        # Get the current row
            row = results_df.iloc[i]

            # working code - don't DELETE

            with st.expander(label="Text Section  " + str(i+1) + ":", expanded=True):
                    # Display each cell in the row as a separate block of text
                st.markdown("**Question:**")
                st.write(submission_text)
                st.markdown("**Below is a section of the text along with its semantic similarity score. It is one of the three highest scoring sections in the text.**")
                st.write(row['similarities'])

                #combined_text = row['combined']
                #combined_text = combined_text.replace('\\n\\n', '\n\n')  # Convert plain string to actual newline characters
                #text_lines = combined_text.split('\n\n')

                #for line in text_lines:
                    #st.markdown(line.replace('\n', '<br>'))  # Replace '\n' with '<br>' for line breaks in markdown

            # end working code - don't DELETE
                combined_text = row['combined']
                combined_text = combined_text.replace('\\n\\n', '\n\n')  # Convert plain string to actual newline characters

                # Split the combined_text into sections
                text_num, source, summary, keywords, full_text = combined_text.split('\n\n', 4)

                # Remove the repeated labels from the values
                text_num = text_num.replace("Text #:", "").strip()
                source = source.replace("Source:", "").strip()
                summary = summary.replace("Summary:", "").strip()
                keywords = keywords.replace("Keywords:", "").strip()
                full_text = full_text.replace("Full Text:", "").strip()

                # Format each section with bold labels
                formatted_text_num = "**Text #:** {}".format(text_num)
                formatted_source = "**Source:** {}".format(source)
                formatted_summary = "**Summary:** {}".format(summary)
                formatted_keywords = "**Keywords:** {}".format(keywords)

                # Display the formatted sections
                st.markdown(formatted_text_num)
                st.markdown(formatted_source)
                st.markdown(formatted_summary)
                st.markdown(formatted_keywords)

                # Display the 'Full_Text' section with proper line breaks
                st.markdown("**Full Text:**")
                text_lines = full_text.split('\n')
                for line in text_lines:
                    st.markdown(line.replace('\n', '<br>'))


    def ask_nicolay():

        def search_text_2(df, product_description, n=3, pprint=True):
            product_embedding = get_embedding(
                product_description,
                engine="text-embedding-ada-002"
            )
            df["similarities"] = df.embedding.apply(lambda x: cosine_similarity(x, product_embedding))

            # Select the first three rows of the sorted DataFrame
            top_three = df.sort_values("similarities", ascending=False).head(3)

            # If `pprint` is True, print the output
            #if pprint:
                #for _, row in top_three.iterrows():
                    #print(row["combined"])
                    #print()

            # Return the DataFrame with the added similarity values
            return top_three

        # Q&A doc prompt with langchain with prompts for determining relevance and extracting quotations.

        results_df = search_text_2(df, submission_text, n=3)

        # Reset the index and create a new column "index"
        results_df = results_df.reset_index()

        # Access the values in the "similarities" and "combined" columns
        similarity1 = results_df.iloc[0]["similarities"]
        combined1 = str(results_df.iloc[0]["combined"])

        similarity2 = results_df.iloc[1]["similarities"]
        combined2 = str(results_df.iloc[1]["combined"])

        similarity3 = results_df.iloc[2]["similarities"]
        combined3 = str(results_df.iloc[2]["combined"])

        num_rows = results_df.shape[0]

        st.markdown("**Step 1 complete - identified the most semantically similar text sections.**")

        # Iterate through the rows of the dataframe
        for i in range(num_rows):
        # Get the current row
            row = results_df.iloc[i]

            # working code - don't DELETE

            with st.expander(label="Text Section  " + str(i+1) + ":", expanded=False):
                    # Display each cell in the row as a separate block of text
                st.markdown("**Question:**")
                st.write(submission_text)
                st.markdown("**Below is a section of the text along with its semantic similarity score. It is one of the three highest scoring sections in the text.**")
                st.write(row['similarities'])

                #combined_text = row['combined']
                #combined_text = combined_text.replace('\\n\\n', '\n\n')  # Convert plain string to actual newline characters
                #text_lines = combined_text.split('\n\n')

                #for line in text_lines:
                    #st.markdown(line.replace('\n', '<br>'))  # Replace '\n' with '<br>' for line breaks in markdown

            # end working code - don't DELETE
                combined_text = row['combined']
                combined_text = combined_text.replace('\\n\\n', '\n\n')  # Convert plain string to actual newline characters

                # Split the combined_text into sections
                text_num, source, summary, keywords, full_text = combined_text.split('\n\n', 4)

                # Remove the repeated labels from the values
                text_num = text_num.replace("Text #:", "").strip()
                source = source.replace("Source:", "").strip()
                summary = summary.replace("Summary:", "").strip()
                keywords = keywords.replace("Keywords:", "").strip()
                full_text = full_text.replace("Full Text:", "").strip()

                # Format each section with bold labels
                formatted_text_num = "**Text #:** {}".format(text_num)
                formatted_source = "**Source:** {}".format(source)
                formatted_summary = "**Summary:** {}".format(summary)
                formatted_keywords = "**Keywords:** {}".format(keywords)

                # Display the formatted sections
                st.markdown(formatted_text_num)
                st.markdown(formatted_source)
                st.markdown(formatted_summary)
                st.markdown(formatted_keywords)

                # Display the 'Full_Text' section with proper line breaks
                st.markdown("**Full Text:**")
                text_lines = full_text.split('\n')
                for line in text_lines:
                    st.markdown(line.replace('\n', '<br>'))


        #st.write("Step 1 complete - identified the most semantically similar text sections.")
        #st.dataframe(results_df)
        st.markdown("**Next step - relevancy check.**")

        ## k-shot prompts for relevance
        currency_question = """2. Section:\n\nText #: 58:\n\nSource: Second Annual Message. December 1, 1862\n\nSummary: In his Second Annual Message, Abraham Lincoln discusses the execution of the new commercial treaty between the United States and the Sultan of Turkey, as well as commercial and consular treaties with Liberia and Haiti. He describes the favorable relations maintained with European and other foreign states and the improved relations with neighboring countries in the Americas. Lincoln also addresses the financial situation, noting the suspension of specie payments and the introduction of United States notes as a temporary measure. He suggests the organization of banking associations under an act of Congress as a solution for providing public funds and a safe, uniform currency. Furthermore, he mentions the importance of developing the mineral resources in the Territories and the potential benefits of an Atlantic telegraph connecting the United States with Europe.\n\nKeywords: Abraham Lincoln, Second Annual Message, December 1 1862, commercial treaty, Sultan of Turkey, Liberia, Haiti, foreign relations, Americas, finances, suspension of specie payments, United States notes, banking associations, mineral resources, Territories, Atlantic telegraph.\n\nFull Text:\n\n"The new commercial treaty between the United States and the Sultan of Turkey has been carried into execution.\n\nA commercial and consular treaty has been negotiated, subject to the Senate's consent, with Liberia; and a similar negotiation is now pending with the republic of Hayti. A considerable improvement of the national commerce is expected to result from these measures.\n\nOur relations with Great Britain, France, Spain, Portugal, Russia, Prussia, Denmark, Sweden, Austria, the Netherlands, Italy, Rome, and the other European states, remain undisturbed. Very favorable relations also continue to be maintained with Turkey, Morocco, China and Japan.\n\nDuring the last year there has not only been no change of our previous relations with the independent states of our own continent, but, more friendly sentiments than have heretofore existed, are believed to be entertained by these neighbors, whose safety and progress, are so intimately connected with our own. This statement especially applies to Mexico, Nicaragua, Costa Rica, Honduras, Peru, and Chile.\n\nThe commission under the convention with the republic of New Granada closed its session, without having audited and passed upon, all the claims which were submitted to it. A proposition is pending to revive the convention, that it may be able to do more complete justice. The joint commission between the United States and the republic of Costa Rica has completed its labors and submitted its report.\n\nI have favored the project for connecting the United States with Europe by an Atlantic telegraph, and a similar project to extend the telegraph from San Francisco, to connect by a Pacific telegraph with the line which is being extended across the Russian empire.\n\nThe Territories of the United States, with unimportant exceptions, have remained undisturbed by the civil war, and they are exhibiting such evidence of prosperity as justifies an expectation that some of them will soon be in a condition to be organized as States, and be constitutionally admitted into the federal Union.\n\nThe immense mineral resources of some of those Territories ought to be developed as rapidly as possible. Every step in that direction would have a tendency to improve the revenues of the government, and diminish the burdens of the people. It is worthy of your serious consideration whether some extraordinary measures to promote that end cannot be adopted. The means which suggests itself as most likely to be effective, is a scientific exploration of the mineral regions in those Territories, with a view to the publication of its results at home and in foreign countries---results which cannot fail to be auspicious.\n\nThe condition of the finances will claim your most diligent consideration. The vast expenditures incident to the military and naval operations required for the suppression of the rebellion, have hitherto been met with a promptitude, and certainty, unusual in similar circumstances, and the public credit has been fully maintained. The continuance of the war, however, and the increased disbursements made necessary by the augmented forces now in the field, demand your best reflections as to the best modes of providing the necessary revenue, without injury to business and with the least possible burdens upon labor.\nThe suspension of specie payments by the banks, soon after the commencement of your last session, made large issues of United States notes unavoidable. In no other way could the payment of the troops, and the satisfaction of other just demands, be so economically, or so well provided for. The judicious legislation of Congress, securing the receivability of these notes for loans and internal duties, and making them a legal tender for other debts , has made them an universal currency; and has satisfied, partially, at least, and for the time, the long felt want of an uniform circulating medium, saving thereby to the people, immense sums in discounts and exchanges.\n\nA return to specie payments, however, at the earliest period compatible with due regard to all interests concerned, should ever be kept in view. Fluctuations in the value of currency are always injurious, and to reduce these fluctuations to the lowest possible point will always be a leading purpose in wise legislation. Convertibility, prompt and certain convertibility into coin, is generally acknowledged to be the best and surest safeguard against them; and it is extremely doubtful whether a circulation of United States notes, payable in coin, and sufficiently large for the wants of the people, can be permanently, usefully and safely maintained.\n\nIs there, then, any other mode in which the necessary provision for the public wants can be made, and the great advantages of a safe and uniform currency secured?\n\nI know of none which promises so certain results, and is, at the same time, so unobjectionable, as the organization of banking associations, under a general act of Congress, well guarded in its provisions.\n\nTo such associations the government might furnish circulating notes, on the security of United States bonds deposited in the treasury. These notes, prepared under the supervision of proper officers, being uniform in appearance and security, and convertible always into coin, would at once protect labor against the evils of a vicious currency, and facilitate commerce by cheap and safe exchanges.\n\nA moderate reservation from the interest on the bonds would compensate the United States for the preparation and distribution of the notes and a general supervision of the system, and would lighten the burden of that part of the public debt employed as securities. The public credit, moreover, would be greatly improved, and the negotiation of new loans greatly facilitated by the steady market demand for government bonds which the adoption of the proposed system would create."\n\n3. Key Words: finances, suspension of specie payments, United States notes, banking associations, uniform currency\n\nRelevance Determination: Section_58: Relevant\n\n4. Relevance Explanation: The section is relevant to the question about currency reform because it addresses the financial situation, the suspension of specie payments, and the introduction of United States notes. Lincoln suggests the organization of banking associations under an act of Congress as a solution for providing public funds and a safe, uniform currency."""
        japan_question = """2. Section:\n\nText #: 72\n\nSource:  Fourth Annual Message. December 6, 1854.\n\nSummary: In Abraham Lincoln's Fourth Annual Message, he expresses gratitude for good health and abundant harvests. He discusses the United States' foreign relations, including maintaining neutrality in Mexico's civil war and involvement in various projects such as a river survey in Central America and the overland telegraph between America and Europe. He mentions the friendly relations with South American states, the ongoing civil war in the Spanish part of San Domingo, and the improvement in relations with Liberia. Lincoln also discusses the situation in China, where rebellion has been suppressed, and Japan, where progress has been made in performing treaty stipulations.\n\nKey Words: Fourth Annual Message, December 6, 1864, Abraham Lincoln, foreign relations, Mexico, civil war, Central America, river survey, overland telegraph, South American states, San Domingo, Liberia, China, Japan, treaty stipulations.\n\nFull Text:\n\n"Fellow-citizens of the Senate December 6, 1864 and House of Representatives:\n\nAgain the blessings of health and abundant harvests claim our profoundest gratitude to Almighty God.\n\nThe condition of our foreign affairs is reasonably satisfactory. \n\nMexico continues to be a theatre of civil war. While our political relations with that country have undergone no change, we have, at the same time, strictly maintained neutrality between the belligerents. \n\nAt the request of the states of Costa Rica and Nicaragua, a competent engineer has been authorized to make a survey of the river San Juan and the port of San Juan. It is a source of much satisfaction that the difficulties which for a moment excited some political apprehensions, and caused a closing of the inter-oceanic transit route, have been amicably adjusted, and that there is a good prospect that the route will soon be reopened with an increase of capacity and adaptation. We could not exaggerate either the commercial or the political importance of that great improvement.\n\nIt would be doing injustice to an important South American state not to acknowledge the directness, frankness, and cordiality with which the United States of Colombia have entered into intimate relations with this government. A claims convention has been constituted to complete the unfinished work of the one which closed its session in 1861. .\n\nThe new liberal constitution of Venezuela having gone into effect with the universal acquiescence of the people, the government under it has been recognized, and diplomatic intercourse with it has opened in a cordial and friendly spirit. The long-deferred Aves Island claim has been satisfactorily paid and discharged. \n\nMutual payments have been made of the claims awarded by the late joint commission for the settlement of claims between the United States and Peru. An earnest and cordial friendship continues to exist between the two countries, and such efforts as were\n\nin my power have been used to remove misunderstanding and avert a threatened war between Peru and Spain. \n\nOur relations are of the most friendly nature with Chile, the Argentine Republic, Bolivia, Costa Rica, Paraguay, San Salvador, and Hayti.\n\nDuring the past year no differences of any kind have arisen with any of those republics, and, on the other hand, their sympathies with the United States are constantly expressed with cordiality and earnestness.\n\nThe claim arising from the seizure of the cargo of the brig Macedonian in 1821 has been paid in full by the government of Chile. \n\nCivil war continues in the Spanish part of San Domingo, apparently without prospect of an early close.\n\nOfficial correspondence has been freely opened with Liberia, and it gives us a pleasing view of social and political progress in that Republic. It may be expected to derive new vigor from American influence, improved by the rapid disappearance of slavery in the United States.\n\nI solicit your authority to furnish to the republic a gunboat at moderate cost, to be reimbursed to the United States by instalments. Such a vessel is needed for the safety of that state against the native African races; and in Liberian hands it would be more effective in arresting the African slave trade than a squadron in our own hands. The possession of the least organized naval force would stimulate a generous ambition in the republic, and the confidence which we should manifest by furnishing it would win forbearance and favor towards the colony from all civilized nations.\n\nThe proposed overland telegraph between America and Europe, by the way of Behring's Straits and Asiatic Russia, which was sanctioned by Congress at the last session, has been undertaken, under very favorable circumstances, by an association of American citizens, with the cordial good-will and support as well of this government as of those of Great Britain and Russia. Assurances have been received from most of the South American States of their high appreciation of the enterprise, and their readiness to co-operate in constructing lines tributary to that world-encircling communication. I learn, with much satisfaction, that the noble design of a telegraphic communication between the eastern coast of America and Great Britain has been renewed with full expectation of its early accomplishment.\n\nThus it is hoped that with the return of domestic peace the country will be able to resume with energy and advantage its former high career of commerce and civilization.\n\nOur very popular and estimable representative in Egypt died in April last. An unpleasant altercation which arose between the temporary incumbent of the office and the government of the Pacha resulted in a suspension of intercourse. The evil was promptly corrected on the arrival of the successor in the consulate, and our relations with Egypt, as well as our relations with the Barbary powers, are entirely satisfactory.\n\nThe rebellion which has so long been flagrant in China, has at last been suppressed, with the co-operating good offices of this government, and of the other western commercial states. The judicial consular establishment there has become very difficult and onerous, and it will need legislative revision to adapt it to the extension of our commerce, and to the more intimate intercourse which has been instituted with the government and people of that vast empire. China seems to be accepting with hearty good-will the conventional laws which regulate commercial and social intercourse among the western nations.\n\nOwing to the peculiar situation of Japan, and the anomalous form of its government, the action of that empire in performing treaty stipulations is inconstant and capricious. Nevertheless, good progress has been effected by the western powers, moving with enlightened concert. Our own pecuniary claims have been allowed, or put in course of settlement, and the inland sea has been reopened to commerce. There is reason also to believe that these proceedings have increased rather than diminished the friendship of Japan towards the United States."\n\n3.Key Words: Japan, treaty stipulations, foreign relations, Fourth Annual Message.\n\nRelevance Determination: Section_72: Relevant\n\Relevance Explanation: In this section of Lincoln's Fourth Annual Message, he discusses the situation in Japan. He mentions that progress has been made in performing treaty stipulations and that the inland sea has been reopened to commerce. He also states that the proceedings have likely increased the friendship of Japan towards the United States."""
        railroad_question = """2. Section: \n\nText #: 71\n\nSource: Third Annual Message. December 8, 1863.\n\n"But why any proclamation now upon this subject? This question is beset with the conflicting views that the step might be delayed too long or be taken too soon. In some States the elements for resumption seem ready for action, but remain inactive, apparently for want of a rallying point---a plan of action. Why shall A adopt the plan of B, rather than B that of A? And if A and B should agree, how can they know but that the general government here will reject their plan? By the proclamation a plan is presented which may be accepted by them as a rallying point, and which they are assured in advance will not be rejected here. This may bring them to act sooner than they otherwise would.\n\nThe objections to a premature presentation of a plan by the national Executive consists in the danger of committals on points which could be more safely left to further developments. Care has been taken to so shape the document as to avoid embarrassments from this source. Saying that, on certain terms, certain classes will be pardoned, with rights restored, it is not said that other classes, or other terms, will never be included. Saying that reconstruction will be accepted if presented in a specified way, it is not said it will never be accepted in any other way.\n\nThe movements, by State action, for emancipation in several of the States, not included in the emancipation proclamation, are matters of profound gratulation. And while I do not repeat in detail what I have hertofore so earnestly urged upon this subject, my general views and feelings remain unchanged; and I trust that Congress will omit no fair opportunity of aiding these important steps to a great consummation.\n\nIn the midst of other cares, however important, we must not lose sight of the fact that the war power is still our main reliance. To that power alone can we look, yet for a time, to give confidence to the people in the contested regions, that the insurgent power will not again overrun them. Until that confidence shall be established, little can be done anywhere for what is called reconstruction. Hence our chiefest care must still be directed to the army and navy, who have thus far borne their harder part so nobly and well. And it may be esteemed fortunate that in giving the greatest efficiency to these indispensable arms, we do also honorably recognize the gallant men, from commander to sentinel, who compose them, and to whom, more than to others, the world must stand indebted for the home of freedom disenthralled, regenerated, enlarged, and perpetuated."\n\nSummary: In this portion of the Third Annual Message, President Lincoln addresses the importance of presenting a plan for the resumption of national authority within States where it has been suspended. He argues that by providing a rallying point, states can act sooner to initiate reconstruction. He also expresses his satisfaction with the movements towards emancipation in states not covered by the Emancipation Proclamation and urges Congress to support these efforts. Lincoln emphasizes that the war power, represented by the army and navy, is still the primary means to establish confidence in contested regions and prevent the insurgent power from overrunning them. He acknowledges the essential role of the military in securing freedom and promoting reconstruction.\n\n3. Key Words: None directly related to railroad construction.\n\nRelevance Determination: Section_71: Irrelevant\n\n5. Relevance Explanation: The section does not address the topic of railroad construction. Instead, it focuses on the resumption of national authority within states, emancipation movements in states not covered by the Emancipation Proclamation, and the importance of the army and navy in securing freedom and promoting reconstruction."""


        examples = [
            {"question": "1. Question: What points does Lincoln make about currency reform?", "output": currency_question},
            {"question": "1. Question Does Lincoln discuss Japan, and if so, what is the context of this discussion?", "output": japan_question},
            {"question": "1. Question: What points does Lincoln make about railroad construction?", "output": railroad_question}
        ]

        prompt_instructions = "You are an AI expert on presidential history with a specialty on the life and times of Abraham Lincoln. In this exercise you are given a user supplied question, a Section of the Text, and a Method for determining the Section’s relevance to the Question. Your objective is to determine whether that Section of the text is directly and specifically relevant to the user question. You will use the Method below to fulfill this objective, taking each step by step.\n\nHere is your Method.\nMethod: Go step by step in answering the question.\n1. Question: You will be provided with a user question.\n2. Section: You will be given a section of the text from a speech by Abraham Lincoln, accompanied by a summary of that section and keywords associated with the text.\n3. Key Words: Identify key words in the Section that are specifically and directly related to the Question. Such key words could include specific locations, events, or people mentioned in the Section.\n4. Relevance Determination: Based on your review of the earlier steps in the Method, determine whether the section is relevant, and gauge your confidence (high, medium, low, or none) in this determination. High determination is specifically and directly related to the Question. If the section is relevant and ranked high, write ‘'Section_x: Relevant'. Otherwise, if the section is not relevant and the determination is less than high, write 'Section_x: Irrelevant'.\n5. Relevance Explanation: Based on your review in the earlier steps in the Method, explain why the Section’s relevance to the Question.\nLet’s begin.\n"

        example_prompt = SystemMessagePromptTemplate.from_template(prompt_instructions)

        human_message_prompt = HumanMessagePromptTemplate.from_template("Question: {question}\nKey Terms:")

        chat_prompt = ChatPromptTemplate.from_messages([example_prompt, human_message_prompt])

        chat = ChatOpenAI(temperature=0, model_name=model_select)
        chain = LLMChain(llm=chat, prompt=chat_prompt)

        r_check_1 = chain.run(question=str(submission_text + "\n2. Section:\n " + combined1))
        #print(r_check_1)

        r_check_2 = chain.run(question=str(submission_text + "\n2. Section:\n " + combined2))
        #print(r_check_2)

        r_check_3 = chain.run(question=str(submission_text + "\n2. Section:\n " + combined3))
        #print(r_check_3

        st.markdown("**Step 2 complete: Nicolay's has made relevancy checks on the text sections.**")

        # combined function for combining sections + outputs, and then filtering via regex for relevant sections

        combined_df = pd.DataFrame(columns=['output', 'r_check'])
        combined_df['output'] = [combined1, combined2, combined3]
        combined_df['r_check'] = [r_check_1, r_check_2, r_check_3]

        for i in range(num_rows):
        # Get the current row
            row = combined_df.iloc[i]

            # working code - don't DELETE

            with st.expander(label="Relevance Check " + str(i+1) + ":", expanded=False):
                    # Display each cell in the row as a separate block of text
                st.markdown("**1. Question:**")
                st.write(submission_text)
                #st.markdown("**2. Relevance Check:")
                #st.markdown(combined_text_1)
                st.markdown("**Text Information:**")
                #st.write(row['similarities'])

                #combined_text = row['combined']
                #combined_text = combined_text.replace('\\n\\n', '\n\n')  # Convert plain string to actual newline characters
                #text_lines = combined_text.split('\n\n')

                #for line in text_lines:
                    #st.markdown(line.replace('\n', '<br>'))  # Replace '\n' with '<br>' for line breaks in markdown

                # end working code - don't DELETE
                #combined_text_0 = row['output']
                #combined_text_0 = combined_text_0.replace('\\n\\n', '\n\n')  # Convert plain string to actual newline characters

                # Split the combined_text into sections
                #text_num, source, summary, keywords, full_text = combined_text_0.split('\n\n', 4)

                # Remove the repeated labels from the values
                #text_num = text_num.replace("Text #:", "").strip()
                #source = source.replace("Source:", "").strip()
                #summary = summary.replace("Summary:", "").strip()
                #keywords = keywords.replace("Keywords:", "").strip()
                #full_text = full_text.replace("Full Text:", "").strip()

                # Format each section with bold labels
                #formatted_text_num = "**Text #:** {}".format(text_num)
                #formatted_source = "**Source:** {}".format(source)
                #formatted_summary = "**Summary:** {}".format(summary)
                #formatted_keywords = "**Keywords:** {}".format(keywords)

                # Display the formatted sections
                #st.markdown(formatted_text_num)
                #st.markdown(formatted_source)
                #st.markdown(formatted_summary)
                #st.markdown(formatted_keywords)

                # Display the 'Full_Text' section with proper line breaks
                #st.markdown("**Full Text:**")
                #text_lines = full_text.split('\n')
                #for line in text_lines:
                    #st.markdown(line.replace('\n', '<br>'))

                #combined_text_1 = row['r_check']
                #combined_text_1 = combined_text_1.replace('\\n\\n', '\n\n')

                # Split the combined_text into sections
                #relevance_determination, relevance_explanation = combined_text_1.split('\n\n', 1)

                # Remove the repeated labels from the values
                #relevance_determination = relevance_determination.replace("3. Relevance Determination: ", "").strip()
                #relevance_explanation = relevance_explanation.replace("4. Relevance Explanation: ", "").strip()

                #formatted_relevance_determination = "**3. Relevance Determination:** {}".format(relevance_determination)
                #formatted_relevance_explanation = "**4. Relevance Explanation:** {}".format(relevance_explanation)

                #st.markdown(formatted_relevance_determination)
                #st.markdown(formatted_relevance_explanation)

                # working code - don't DELETE

            # begin snippet



                combined_text_0 = row['output']
                combined_text_0 = combined_text_0.replace('\\n\\n', '\n\n')  # Convert plain string to actual newline characters

                text_num, source, summary, keywords, full_text = combined_text_0.split('\n\n', 4)

                text_num = text_num.replace("Text #:", "").strip()
                source = source.replace("Source:", "").strip()
                summary = summary.replace("Summary:", "").strip()
                keywords = keywords.replace("Keywords:", "").strip()
                full_text = full_text.replace("Full Text:", "").strip()

                formatted_text_num = "**Text #:** {}".format(text_num)
                formatted_source = "**Source:** {}".format(source)
                formatted_summary = "**Summary:** {}".format(summary)
                formatted_keywords = "**Keywords:** {}".format(keywords)

                st.markdown(formatted_text_num)
                st.markdown(formatted_source)
                st.markdown(formatted_summary)
                st.markdown(formatted_keywords)

                st.markdown("**Full Text:**")
                text_lines = full_text.split('\n')
                for line in text_lines:
                    st.markdown(line.replace('\n', '<br>'))

                combined_text_1 = row['r_check']
                combined_text_1 = combined_text_1.replace('\\n\\n', '\n\n')

                # Try to split the text into three sections
                #split_text = combined_text_1.split('\n\n', 2)

                # Check if the split was successful
                # existing code
                # begin snippet
                #if len(split_text) == 3:
                #    _, relevant_keywords, relevance_explanation = split_text
                #else:
                    # If the split wasn't successful, try splitting the text differently
                #    split_text = combined_text_1.split('\n\n', 1)
                #    if len(split_text) == 2:
                #        relevant_keywords, relevance_explanation = split_text
                #    else:
                        # If splitting is still unsuccessful, set empty values to avoid errors
                #        relevant_keywords, relevance_explanation = "", ""

                #relevant_keywords = relevant_keywords.replace("3. Key Words: ", "").strip()
                #relevance_explanation = relevance_explanation.replace("4. Relevance Explanation: ", "").strip()

                #formatted_relevant_keywords = "**3. Key Words:** {}".format(relevant_keywords)

                # Extract relevance determination from the beginning of the relevance explanation
                #relevance_determination = relevance_explanation.split('\n', 1)[0].strip().replace("4. Relevance Determination: ", "")
                # Remove relevance determination from the relevance explanation
                #relevance_explanation = relevance_explanation.replace(relevance_determination, "").strip()

                #formatted_relevance_determination = "**4. Relevance Determination:** {}".format(relevance_determination)
                #formatted_relevance_explanation = "**5. Relevance Explanation:** {}".format(relevance_explanation)

                #st.markdown(formatted_relevant_keywords)
                #st.markdown(formatted_relevance_determination)
                #st.markdown(formatted_relevance_explanation)

                st.markdown(combined_text_1)
# end snippet

        #st.dataframe(combined_df)
        st.markdown("**Next step: Answer the user question with supporting quotations from the relevant texts.**")

        # Use the re.IGNORECASE flag to make the regular expression case-insensitive
        #regex = re.compile(r'Relevance Explanation.*?(relevant)', re.IGNORECASE | re.DOTALL)
        # Replace the original regex pattern with the updated one
        #regex = re.compile(r'Relevance Determination.*?:\s*(Relevant)', re.IGNORECASE | re.DOTALL)
        # Modify the regex pattern to include both "Relevance Determination" and "Relevance Explanation"
        #regex = re.compile(r'Relevance (?:Determination|Explanation).*?:\s*(Relevant)', re.IGNORECASE | re.DOTALL)
        # Modify the regex pattern to include the case when there is no "Relevance Determination" or "Relevance Explanation" string
        #regex = re.compile(r'Section_.*?:\s*(Relevant)(?:\.|,).*?Relevance Explanation.*', re.IGNORECASE | re.DOTALL)
        # Modify the regex pattern to include the optional "Relevance Determination" string followed by any characters and a newline character
        # Modify the regex pattern to include the optional "Key Words:" string followed by any characters and a newline character
        # Modify the regex pattern to accommodate multiple newline characters between "Relevance Determination:" and "Section_"
        # Update the regex pattern to make the entire pattern case-insensitive
        #regex = re.compile(r'(?i)Section_.*?:\s*(Relevant)(?:\s*\(.+?\))?(?:\.|,)', re.DOTALL)


        # Apply the regex pattern to the 'r_check' column and store the results in a new 'mask' column
        #combined_df['mask'] = combined_df['r_check'].str.contains(regex)
        # Apply the regex pattern to the 'r_check' column using the str.match() function
        #combined_df['mask'] = combined_df['r_check'].str.match(regex)


        # Create a second mask to capture "this is relevant"
        #combined_df['second_mask'] = combined_df['r_check'].str.contains(r'this section is relevant', flags=re.IGNORECASE)

        # Combine the two masks using the bitwise OR operator (|) and store the result in the 'mask' column
        #combined_df['mask'] = combined_df['mask'] | combined_df['second_mask']

        # Filter the combined dataframe to include only rows where the 'mask' column is True
        #relevant_df = combined_df.loc[combined_df['mask']].copy()

        # Modified regex pattern
        # Updated regex pattern
        #regex = re.compile(r'(?i)(?:\d+\.?\s*)?Relevance Determination:\s*(?:\n|\r\n)?\s*Section_\s*\d+\s*[:=]\s*(?:\n|\r\n)?\s*(Relevant)(?:\s*\(.+?\))?(?:\.|,)?', re.DOTALL)
        regex = re.compile(r'(?i)(?:\d+\.?\s*)?Relevance Determination:\s*(?:\n|\r\n)?\s*(High\s*)?Section_\s*\d+\s*[:=]\s*(?:\n|\r\n)?\s*(Relevant)(?:\s*\(.+?\))?(?:\.|,)?', re.DOTALL)


        # Apply the regex pattern to the 'r_check' column using the str.contains() function
        combined_df['mask'] = combined_df['r_check'].str.contains(regex)

        # Create a second mask to capture "this section is relevant"
        combined_df['second_mask'] = combined_df['r_check'].str.contains(r'this section is relevant', flags=re.IGNORECASE)

        # Combine the two masks using the bitwise OR operator (|) and store the result in the 'mask' column
        combined_df['mask'] = combined_df['mask'] | combined_df['second_mask']

        # Filter the combined dataframe to include only rows where the 'mask' column is True
        relevant_df = combined_df.loc[combined_df['mask']].copy()







        # Check if there are any rows in the relevant_df dataframe
        if relevant_df.empty:
            # If there are no rows, print the desired message
            st.write("No relevant sections identified.")
        else:
            # Otherwise, continue with the rest of the script

            def combine_strings(row):
                return row['output'] + '\nKey Terms\n' + row['r_check']

            # Use the apply function to apply the combine_strings function to each row of the relevant_df dataframe
            # and assign the result to the 'combined_string' column
            relevant_df['combined_string'] = relevant_df.apply(combine_strings, axis=1)

            final_sections = relevant_df['combined_string']
            #final_sections.to_csv('final_sections.csv')

            evidence_df = pd.DataFrame(final_sections)

            evidence = '\n\n'.join(evidence_df['combined_string'])
            #evidence_df.to_csv('evidence.csv')

            #print(evidence)

            # Filter the relevant_df dataframe to include only the 'output' column
            output_df = relevant_df[['output']]

            # Convert the dataframe to a dictionary
            output_dict = output_df.to_dict('records')

            # Extract the values from the dictionary using a list comprehension
            output_values = [d['output'] for d in output_dict]

            # Print the output values to see the results
            #print(output_values)

            # begin quotation identiftication and answer prompt

            cherokee_question = "2. Text:\n\n\nText Number: 59\nSource: Second Annual Message. December 1, 1863\n\nThe Indian tribes upon our frontiers have, during the past year, manifested a spirit of insubordination, and, at several points, have engaged in open hostilities against the white settlements in their vicinity. The tribes occupying the Indian country south of Kansas, renounced their allegiance to the United States, and entered into treaties with the insurgents. Those who remained loyal to the United States were driven from the country. The chief of the Cherokees has visited this city for the purpose of restoring the former relations of the tribe with the United States. He alleges that they were constrained, by superior force, to enter into treaties with the insurgents, and that the United States neglected to furnish the protection which their treaty stipulations required.\n\n3. Compose Initial Answer: Lincoln regarded the Cherokee as a tribe that had been forced into renouncing their allegiance to the United States and entering into treaties with the Confederacy due to superior force and neglect on the part of the United States.\n\n4. Identify Supporting Quote: \"The chief of the Cherokees has visited this city for the purpose of restoring the former relations of the tribe with the United States. He alleges that they were constrained, by superior force, to enter into treaties with the insurgents, and that the United States neglected to furnish the protection which their treaty stipulations required.\" (Second Annual Message. December 1, 1863. (Text 59)\n\n5. Combined Answer with Supporting Quote: Lincoln discusses the Cherokee in his Second Annual Message of December 1, 1863. Lincoln notes notes the visit of the Cherokee chief to Washington D.C. “for the purpose of restoring the former relations of the tribe with the United States.” The Cherokee were “constrained, by superior force, to enter into treaties with the Confederacy.” Furthermore, the chief alleged “that the United States neglected to furnish the protection which their treaty stipulations required.” (Second Annual Message. December 1, 1863. Text 59)\n",
            japan_question = "2. Text\n\nText Number#: 72: \nSource: Fourth Annual Message. December 6, 1864.\n\nOwing to the peculiar situation of Japan, and the anomalous form of its government, the action of that empire in performing treaty stipulations is inconstant and capricious. Nevertheless, good progress has been effected by the western powers, moving with enlightened concert. Our own pecuniary claims have been allowed, or put in course of settlement, and the inland sea has been reopened to commerce. There is reason also to believe that these proceedings have increased rather than diminished the friendship of Japan towards the United States.\n\nCompose Initial Answer: Yes, Lincoln discusses Japan in his Fourth Annual Message of December 6, 1854, stating that the peculiar situation and anomalous form of government of Japan have made their actions in performing treaty stipulations inconsistent and capricious. However, he notes that progress has been made by the western powers in working together and that our own pecuniary claims have been allowed or put in settlement. Additionally, the inland sea has been reopened to commerce, and these proceedings have likely increased Japan's friendship towards the United States.\n\nIdentify Supporting Quote: \"Owing to the peculiar situation of Japan, and the anomalous form of its government, the action of that empire in performing treaty stipulations is inconstant and capricious... There is reason also to believe that these proceedings have increased rather than diminished the friendship of Japan towards the United States.\" (Fourth Annual Message, December 6, 1864. Text Number 72).\n\nCombined Answer with Supporting Quote: Yes, Lincoln discusses Japan in his Fourth Annual Message of December 6, 1864. Lincoln acknowledged that \"the action of [Japan] in performing treaty stipulations is inconstant and capricious\" due to their \"peculiar situation\" and \"anomalous form of government.\" However, he also noted that \"good progress has been effected by the western powers, moving with enlightened concert,\" as evidenced by the settlement of the United States' pecuniary claims and the reopening of the inland sea to commerce. Lincoln further suggested that these efforts had \"increased rather than diminished the friendship of Japan towards the United States.\" Thus, this message reflects Lincoln's recognition of the importance of international cooperation and diplomacy in navigating complex political and cultural landscapes such as that of Japan during the Late Tokugawa period. (Fourth Annual Message, December 6, 1864. Text Number 72).\n"

            examples = [
                {"question": "1. Question: How did Lincoln regard the Cherokee?", "output": cherokee_question},
                {"question": "1. Question: Does Lincoln discuss Japan, and if so, what is the context of this discussion?", "output": japan_question}
            ]

            prompt_instructions ="You are an AI question-answerer and quotation-selector. The focus of your expertise is interpreting the historic writings of Abraham Lincoln. In this exercise you will first be given a user question, a Section of a Lincoln writing, and a Method for answering the question and supporting it with an appropriate quotation from the Section. In following this Method you will complete each step by step until finished.\nHere is your Method.\nMethod: Go step by step in the question.\n1. Question: You will be provided with a user question.\n2. Text: You will be given a section from a Text written by Abraham Lincoln. The Text contains the Text Number, the Source of the Text, and the original prose by Lincoln. \n3. Compose Initial Answer: Based on the Question and information provided in the Text, compose a historically accurate Initial Answer to that Question. The Initial Answer should be incisive, brief, and well-written.\n4. Identify Supporting Quote: Based on the Answer, select a Supporting Quote from the Text that supports that Answer. Select the briefest and most relevant Supporting Quote possible. You can also use paraphrasing to further shorten the Supporting Quote. Provide a citation at the end of the Supporting Quote, in the following manner: (Source, Text Number).\n5. Combined Answer with Supporting Quote: Rewrite the Initial Answer to incorporate the Supporting Quote. This Combined Answer should be historically accurate, and demonstrating a writing style that is incisive, brief, and well-written. All Quotes used should be cited using the method above.\n\nLet’s begin.\n"

            example_prompt = SystemMessagePromptTemplate.from_template(prompt_instructions)

            human_message_prompt = HumanMessagePromptTemplate.from_template("Question: {question}\nKey Terms:")

            chat_prompt = ChatPromptTemplate.from_messages([example_prompt, human_message_prompt])

            chat = ChatOpenAI(temperature=0, model_name=model_select)
            chain = LLMChain(llm=chat, prompt=chat_prompt)

                # Create an empty list to store the final_analysis results
            final_analysis_results = []

                # Iterate over the output_values list
            for output_value in output_values:
                # Run the final_analysis step and store the result in a variable
                final_analysis = chain.run(submission_text+output_value)
                # Add the final_analysis result to the list
                final_analysis_results.append(final_analysis)

            # Create a Pandas dataframe from the output_values list
            final_analysis_df = pd.DataFrame({'output_values': output_values, 'final_analysis': final_analysis_results})

            # Save the dataframe to a CSV file
            #final_analysis_df.to_csv('final_analysis.csv', index=False)

            st.subheader("Nicolay's Final Analysis:")
            st.markdown("**Step 3 complete: Here are Nicolay's analysis of Lincoln's speeches based on your question. Click on the dataframe boxes below to see the full outputs.**")
            #st.dataframe(final_analysis_df)
            #st.write('\n\n')

            #for i in range(len(final_analysis_df)):
            # Get the current row
                #row = final_analysis_df.iloc[i]

            for i in range(len(final_analysis_df)):
                # Get the current row
                row = final_analysis_df.iloc[i]

                # working code - don't DELETE

                with st.expander(label="Nicolay's Response: " + str(i+1) + ":", expanded=False):
                    # Display each cell in the row as a separate block of text
                    st.markdown("**1. Question:**")
                    st.write(submission_text)
                    st.write("**2. Answer:**")

                    combined_text_x = row['final_analysis']
                    combinex_text_x = combined_text_x.replace('\\n\\n', '\n\n')  # Convert plain string to actual newline characters

                    # Find the index of "Combined Answer with Supporting Quote:" and display the text after it
                    start_index = combined_text_x.find("Combined Answer with Supporting Quote:") + len("Combined Answer with Supporting Quote:")
                    answer_text = combined_text_x[start_index:].strip()
                    st.markdown(answer_text)


                # working code - don't DELETE

                #with st.expander(label="Nicolay's Response: " + str(i) + ":", expanded=False):
                    # Display each cell in the row as a separate block of text
                    #st.markdown("**1. Question:**")
                    #st.write(submission_text)
                    #st.write("**2. Answer:**")

                    #combined_text_x = row['final_analysis']
                    #combinex_text_x = combined_text_x.replace('\\n\\n', '\n\n')  # Convert plain string to actual newline characters

                    #st.markdown(combined_text_x)

                    #st.markdown("**3. Text Information:**")

                    combined_text_0 = row['output_values']
                    combined_text_0 = combined_text_0.replace('\\n\\n', '\n\n')  # Convert plain string to actual newline characters

                    text_num, source, summary, keywords, full_text = combined_text_0.split('\n\n', 4)

                    text_num = text_num.replace("Text #:", "").strip()
                    source = source.replace("Source:", "").strip()
                    summary = summary.replace("Summary:", "").strip()
                    keywords = keywords.replace("Keywords:", "").strip()
                    full_text = full_text.replace("Full Text:", "").strip()

                    formatted_text_num = "**Text #:** {}".format(text_num)
                    formatted_source = "**Source:** {}".format(source)
                    formatted_summary = "**Summary:** {}".format(summary)
                    formatted_keywords = "**Keywords:** {}".format(keywords)

                    st.markdown(formatted_text_num)
                    st.markdown(formatted_source)
                    st.markdown(formatted_summary)
                    st.markdown(formatted_keywords)

                    st.markdown("**Full Text:**")
                    text_lines = full_text.split('\n')
                    for line in text_lines:
                        st.markdown(line.replace('\n', '<br>'))




    if search_method == semantic_search:
        embeddings_search()
    else:
        ask_nicolay()
