from langchain.prompts import PromptTemplate
from langchain.prompts import FewShotPromptTemplate
from langchain.chains import LLMChain
from langchain import OpenAI, SerpAPIWrapper, Wikipedia
from langchain.agents import initialize_agent, Tool
from langchain.agents.react.base import DocstoreExplorer
import openai
import csv
from datetime import datetime as dt
import pandas as pd
import numpy as np
from openai.embeddings_utils import get_embedding, cosine_similarity
import os
import re
import streamlit as st
import pygsheets
from google.oauth2 import service_account
import ssl
import base64


from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts import FewShotPromptTemplate

import os
import pandas as pd
import numpy as np
import re

from openai.embeddings_utils import get_embedding, cosine_similarity
import openai
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

#from IPython.display import HTML


scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']

credentials = service_account.Credentials.from_service_account_info(
                    st.secrets["gcp_service_account"], scopes = scope)

gc = pygsheets.authorize(custom_credentials=credentials)


st.set_page_config(
    page_title="Nicolay: An AI Search Tool for the Speeches of Abraham Lincoln (version 0.0)",
    layout='wide',
    page_icon='🔍'
)

os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
openai.api_key = os.getenv("OPENAI_API_KEY")
#os.environ["SERPAPI_API_KEY"] = st.secrets["serpapi_api_key"]

st.title("Nicolay: An AI Search Tool for the Speeches of Abraham Lincoln")

def button_one():
    st.write("This application uses OpenAI's GPT AI models to answer questions about the collected speeches of Abraham Lincoln. Choose one of the options below, and pose a question about the text.")
    semantic_search = "Semantic Search: Enter a question, and recieve sections of Lincoln's speeches that are the most closely related."
    #ask_a_paragraph = "Ask a Paragraph: Internet Search. Select a Section from the text, and then pose a question. GPT-3 will search the internet to answer your question."
    #ask_wiki = "Ask a Paragraph: Wikipedia. Select a Section from the text, and then pose a question. GPT-3 will search Wikipedia to answer your question."
    ask_a_source = "Ask Nicolay: Pose a question about Lincoln's speeches, and a GPT AI model will share answers drawn from the text."


    search_method = st.radio("Choose a method:", (semantic_search, ask_a_source))
    model_choice = st.selectbox("Choose an AI model:", ('GPT-3.5', 'GPT-4'), index=1)
    #section_number = st.number_input('Select a section number if you have selected Ask a Paragraph. You can find the section numbers to the bottom left, or through a semantic search.', step=1)
    submission_text = st.text_area("Enter your question below. ")
    submit_button_1 = st.button(label='Click here to submit your question.')
    if submit_button_1:

        st.subheader("Nicolay's analysis is underway. It can take a minute or two for every step of the process to be completed, so thank you for your patience. Nicolay's progress will be documented below.")

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

            # Create an expander for the current row, with the label set to the row number
                with st.expander(label="Text Section  " + str(i) + ":", expanded=True):
                        # Display each cell in the row as a separate block of text
                    st.markdown("**Question:**")
                    st.write(submission_text)
                    st.markdown("**Below is a section of the text along with its semantic similarity score. It is one of the three highest scoring sections in the text.**")
                    st.write(row['similarities'])

                    combined_text = row['combined']
                    text_lines = combined_text.split('\n')

                    #for line in text_lines:
                        #st.text(line)

                    for line in text_lines:
                        st.markdown(line)



        # Write the DataFrame to a CSV file
        #results_df.to_csv('results_df.csv', index=False, columns=["similarities", "combined"])
        #st.subheader("The steps below illustrate Nicolay's reasoning on this question.")
        #st.write("Step 1 complete: Nicolay identified the most semantically similar text sections.")
        #st.dataframe(results_df)
        #st.write("Next step: relevancy check")

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

            st.write("Step 1 complete - identified the most semantically similar text sections.")
            st.dataframe(results_df)
            st.write("Next step - relevancy check.")

            ## k-shot prompts for relevance
            currency_question = """1. Question\n\n"What points does Lincoln make about currency reform?"\n\n2. Section:\n\nText #: 58: Source: Second Annual Message. December 1, 1862\n\n"The new commercial treaty between the United States and the Sultan of Turkey has been carried into execution.\n\nA commercial and consular treaty has been negotiated, subject to the Senate's consent, with Liberia; and a similar negotiation is now pending with the republic of Hayti. A considerable improvement of the national commerce is expected to result from these measures.\n\nOur relations with Great Britain, France, Spain, Portugal, Russia, Prussia, Denmark, Sweden, Austria, the Netherlands, Italy, Rome, and the other European states, remain undisturbed. Very favorable relations also continue to be maintained with Turkey, Morocco, China and Japan.\n\nDuring the last year there has not only been no change of our previous relations with the independent states of our own continent, but, more friendly sentiments than have heretofore existed, are believed to be entertained by these neighbors, whose safety and progress, are so intimately connected with our own. This statement especially applies to Mexico, Nicaragua, Costa Rica, Honduras, Peru, and Chile.\n\nThe commission under the convention with the republic of New Granada closed its session, without having audited and passed upon, all the claims which were submitted to it. A proposition is pending to revive the convention, that it may be able to do more complete justice. The joint commission between the United States and the republic of Costa Rica has completed its labors and submitted its report.\n\nI have favored the project for connecting the United States with Europe by an Atlantic telegraph, and a similar project to extend the telegraph from San Francisco, to connect by a Pacific telegraph with the line which is being extended across the Russian empire.\n\nThe Territories of the United States, with unimportant exceptions, have remained undisturbed by the civil war, and they are exhibiting such evidence of prosperity as justifies an expectation that some of them will soon be in a condition to be organized as States, and be constitutionally admitted into the federal Union.\n\nThe immense mineral resources of some of those Territories ought to be developed as rapidly as possible. Every step in that direction would have a tendency to improve the revenues of the government, and diminish the burdens of the people. It is worthy of your serious consideration whether some extraordinary measures to promote that end cannot be adopted. The means which suggests itself as most likely to be effective, is a scientific exploration of the mineral regions in those Territories, with a view to the publication of its results at home and in foreign countries---results which cannot fail to be auspicious.\n\nThe condition of the finances will claim your most diligent consideration. The vast expenditures incident to the military and naval operations required for the suppression of the rebellion, have hitherto been met with a promptitude, and certainty, unusual in similar circumstances, and the public credit has been fully maintained. The continuance of the war, however, and the increased disbursements made necessary by the augmented forces now in the field, demand your best reflections as to the best modes of providing the necessary revenue, without injury to business and with the least possible burdens upon labor.\nThe suspension of specie payments by the banks, soon after the commencement of your last session, made large issues of United States notes unavoidable. In no other way could the payment of the troops, and the satisfaction of other just demands, be so economically, or so well provided for. The judicious legislation of Congress, securing the receivability of these notes for loans and internal duties, and making them a legal tender for other debts , has made them an universal currency; and has satisfied, partially, at least, and for the time, the long felt want of an uniform circulating medium, saving thereby to the people, immense sums in discounts and exchanges.\n\nA return to specie payments, however, at the earliest period compatible with due regard to all interests concerned, should ever be kept in view. Fluctuations in the value of currency are always injurious, and to reduce these fluctuations to the lowest possible point will always be a leading purpose in wise legislation. Convertibility, prompt and certain convertibility into coin, is generally acknowledged to be the best and surest safeguard against them; and it is extremely doubtful whether a circulation of United States notes, payable in coin, and sufficiently large for the wants of the people, can be permanently, usefully and safely maintained.\n\nIs there, then, any other mode in which the necessary provision for the public wants can be made, and the great advantages of a safe and uniform currency secured?\n\nI know of none which promises so certain results, and is, at the same time, so unobjectionable, as the organization of banking associations, under a general act of Congress, well guarded in its provisions.\n\nTo such associations the government might furnish circulating notes, on the security of United States bonds deposited in the treasury. These notes, prepared under the supervision of proper officers, being uniform in appearance and security, and convertible always into coin, would at once protect labor against the evils of a vicious currency, and facilitate commerce by cheap and safe exchanges.\n\nA moderate reservation from the interest on the bonds would compensate the United States for the preparation and distribution of the notes and a general supervision of the system, and would lighten the burden of that part of the public debt employed as securities. The public credit, moreover, would be greatly improved, and the negotiation of new loans greatly facilitated by the steady market demand for government bonds which the adoption of the proposed system would create."\n\nSummary: In his Second Annual Message, Abraham Lincoln discusses the execution of the new commercial treaty between the United States and the Sultan of Turkey, as well as commercial and consular treaties with Liberia and Haiti. He describes the favorable relations maintained with European and other foreign states and the improved relations with neighboring countries in the Americas. Lincoln also addresses the financial situation, noting the suspension of specie payments and the introduction of United States notes as a temporary measure. He suggests the organization of banking associations under an act of Congress as a solution for providing public funds and a safe, uniform currency. Furthermore, he mentions the importance of developing the mineral resources in the Territories and the potential benefits of an Atlantic telegraph connecting the United States with Europe.\nKeywords: Abraham Lincoln, Second Annual Message, December 1 1862, commercial treaty, Sultan of Turkey, Liberia, Haiti, foreign relations, Americas, finances, suspension of specie payments, United States notes, banking associations, mineral resources, Territories, Atlantic telegraph.\n\n3. Key Words:”\n\ncurrency reform\n\n4. Relevance Determination: Section 58: Relevant\n\n5. Relevance Explanation: The section is directly and specifically relevant to the question because it contains key words such as "specie payments” and "banking associations" which are directly related to the question. Additionally, the background knowledge and context of the speech provide further evidence of the section's relevance to the question."""
            railroad_question = """2. Section: \n\nText #: 71: Source: Third Annual Message. December 8, 1863.\n\n"But why any proclamation now upon this subject? This question is beset with the conflicting views that the step might be delayed too long or be taken too soon. In some States the elements for resumption seem ready for action, but remain inactive, apparently for want of a rallying point---a plan of action. Why shall A adopt the plan of B, rather than B that of A? And if A and B should agree, how can they know but that the general government here will reject their plan? By the proclamation a plan is presented which may be accepted by them as a rallying point, and which they are assured in advance will not be rejected here. This may bring them to act sooner than they otherwise would.\n\nThe objections to a premature presentation of a plan by the national Executive consists in the danger of committals on points which could be more safely left to further developments. Care has been taken to so shape the document as to avoid embarrassments from this source. Saying that, on certain terms, certain classes will be pardoned, with rights restored, it is not said that other classes, or other terms, will never be included. Saying that reconstruction will be accepted if presented in a specified way, it is not said it will never be accepted in any other way.\n\nThe movements, by State action, for emancipation in several of the States, not included in the emancipation proclamation, are matters of profound gratulation. And while I do not repeat in detail what I have hertofore so earnestly urged upon this subject, my general views and feelings remain unchanged; and I trust that Congress will omit no fair opportunity of aiding these important steps to a great consummation.\n\nIn the midst of other cares, however important, we must not lose sight of the fact that the war power is still our main reliance. To that power alone can we look, yet for a time, to give confidence to the people in the contested regions, that the insurgent power will not again overrun them. Until that confidence shall be established, little can be done anywhere for what is called reconstruction. Hence our chiefest care must still be directed to the army and navy, who have thus far borne their harder part so nobly and well. And it may be esteemed fortunate that in giving the greatest efficiency to these indispensable arms, we do also honorably recognize the gallant men, from commander to sentinel, who compose them, and to whom, more than to others, the world must stand indebted for the home of freedom disenthralled, regenerated, enlarged, and perpetuated."\n\nSummary: In this portion of the Third Annual Message, President Lincoln addresses the importance of presenting a plan for the resumption of national authority within States where it has been suspended. He argues that by providing a rallying point, states can act sooner to initiate reconstruction. He also expresses his satisfaction with the movements towards emancipation in states not covered by the Emancipation Proclamation and urges Congress to support these efforts. Lincoln emphasizes that the war power, represented by the army and navy, is still the primary means to establish confidence in contested regions and prevent the insurgent power from overrunning them. He acknowledges the essential role of the military in securing freedom and promoting reconstruction.\n\nKeywords: Third Annual Message, December 8, 1863, Abraham Lincoln, national authority, reconstruction, rallying point, Emancipation Proclamation, war power, army, navy, contested regions, insurgent power, freedom.\n\n4. Relevance Determination: Section_71: Irrelevant\n\n5. Relevance Explanation: The Section is irrelevant because it does not address the user's question about points Lincoln makes on railroad construction. The content of the Section focuses on other topics such as national authority, reconstruction, emancipation, and the role of the military during the Civil War."""


            examples = [
                {"question": "1. Question: What points does Lincoln make about currency reform?", "output": currency_question},
                {"question": "1. Question: What points does Lincoln make about railroad construction?", "output": railroad_question}
            ]

            prompt_instructions = "You are an AI expert on presidential history with a speciality on the life and times of Abraham Lincoln. In this exercise you are given a user supplied question, a Section of the Text, a Semantic Similarity Score, and a Method for determining the Section’s relevance to the Question. Your objective is to determine whether that Section of the text is directly and specifically relevant to the user question. You will use the Method below to fulfill this objective, taking each step by step.\n\nHere is your Method.\nMethod: Go step by step in answering the question.\n1. Question: You will be provided with a user question.\n2. Section: You will be given a section of the text from a speech by Abraham Lincoln, accompanied by a summary of that section and keywords associated with the text.\n3. Key Words: Identify key words in the Section that are specifically and directly related to the Question. Such key words could include specific locations, events, or people mentioned in the Section.\n4. Relevance Determination: Based on your review of the earlier steps in the Method, determine whether the section is relevant, and gauge your confidence (high, medium, low, or none) in this determination. High determination is specifically and directly related to the Question. If the section is relevant and ranked high, write ‘'Section_x: Relevant'. Otherwise, if the section is not relevant and the determination is less than high, write 'Section_x: Irrelevant'.\n5. Relevance Explanation: Based on your review in the earlier steps in the Method, explain why the Section’s relevance to the Question.\nLet’s begin.\n"

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

            st.write("Step 2 complete: Nicolay's has made relevancy checks on the text sections.")

            # combined function for combining sections + outputs, and then filtering via regex for relevant sections

            combined_df = pd.DataFrame(columns=['output', 'r_check'])
            combined_df['output'] = [combined1, combined2, combined3]
            combined_df['r_check'] = [r_check_1, r_check_2, r_check_3]

            st.dataframe(combined_df)
            st.write("Next step: Answer the user question with supporting quotations from the relevant texts.")

            # Use the re.IGNORECASE flag to make the regular expression case-insensitive
            regex = re.compile(r'Relevance Explanation.*?(relevant)', re.IGNORECASE | re.DOTALL)

            # Apply the regex pattern to the 'r_check' column and store the results in a new 'mask' column
            combined_df['mask'] = combined_df['r_check'].str.contains(regex)

            # Create a second mask to capture "this is relevant"
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
                evidence_df.to_csv('evidence.csv')

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

                cherokee_question = "1. Question: How did Lincoln regard the Cherokee?\n\n2. Text:\n\n\nText Number: 59\nSource:  Second Annual Message. December 1, 1863\n\nThe Indian tribes upon our frontiers have, during the past year, manifested a spirit of insubordination, and, at several points, have engaged in open hostilities against the white settlements in their vicinity. The tribes occupying the Indian country south of Kansas, renounced their allegiance to the United States, and entered into treaties with the insurgents. Those who remained loyal to the United States were driven from the country. The chief of the Cherokees has visited this city for the purpose of restoring the former relations of the tribe with the United States. He alleges that they were constrained, by superior force, to enter into treaties with the insurgents, and that the United States neglected to furnish the protection which their treaty stipulations required.\n\n3. Compose Initial Answer: Lincoln regarded the Cherokee as a tribe that had been forced into renouncing their allegiance to the United States and entering into treaties with the Confederacy due to superior force and neglect on the part of the United States.\n\n4. Identify Supporting Quote: \"The chief of the Cherokees has visited this city for the purpose of restoring the former relations of the tribe with the United States. He alleges that they were constrained, by superior force, to enter into treaties with the insurgents, and that the United States neglected to furnish the protection which their treaty stipulations required.\" (Second Annual Message. December 1, 1863. (Text 59)\n\n5. Combined Answer with Supporting Quote: Lincoln discusses the Cherokee in his Second Annual Message of December 1, 1863. Lincoln notes notes the visit of the Cherokee chief to Washington D.C. “for the purpose of restoring the former relations of the tribe with the United States.” The Cherokee were “constrained, by superior force, to enter into treaties with the Confederacy.” Furthermore, the chief alleged “that the United States neglected to furnish the protection which their treaty stipulations required.” (Second Annual Message. December 1, 1863. Text 59)\n",
                japan_question = "1. Question: Does Lincoln discuss Japan, and if so, what is the context of this discussion?\n\n2. Text\n\nText #: 72: \nSource:  Fourth Annual Message. December 6, 1864.\n\nOwing to the peculiar situation of Japan, and the anomalous form of its government, the action of that empire in performing treaty stipulations is inconstant and capricious. Nevertheless, good progress has been effected by the western powers, moving with enlightened concert. Our own pecuniary claims have been allowed, or put in course of settlement, and the inland sea has been reopened to commerce. There is reason also to believe that these proceedings have increased rather than diminished the friendship of Japan towards the United States.\n\nCompose Initial Answer: Yes, Lincoln discusses Japan in his Fourth Annual Message of December 6, 1854, stating that the peculiar situation and anomalous form of government of Japan have made their actions in performing treaty stipulations inconsistent and capricious. However, he notes that progress has been made by the western powers in working together and that our own pecuniary claims have been allowed or put in settlement. Additionally, the inland sea has been reopened to commerce, and these proceedings have likely increased Japan's friendship towards the United States.\n\nIdentify Supporting Quote: \"Owing to the peculiar situation of Japan, and the anomalous form of its government, the action of that empire in performing treaty stipulations is inconstant and capricious... There is reason also to believe that these proceedings have increased rather than diminished the friendship of Japan towards the United States.\" (Fourth Annual Message, December 6, 1864. Text Number 72).\n\nCombined Answer with Supporting Quote: Yes, Lincoln discusses Japan in his Fourth Annual Message of December 6, 1864. Lincoln acknowledged that \"the action of [Japan] in performing treaty stipulations is inconstant and capricious\" due to their \"peculiar situation\" and \"anomalous form of government.\" However, he also noted that \"good progress has been effected by the western powers, moving with enlightened concert,\" as evidenced by the settlement of the United States' pecuniary claims and the reopening of the inland sea to commerce. Lincoln further suggested that these efforts had \"increased rather than diminished the friendship of Japan towards the United States.\" Thus, this message reflects Lincoln's recognition of the importance of international cooperation and diplomacy in navigating complex political and cultural landscapes such as that of Japan during the Late Tokugawa period. (Fourth Annual Message, December 6, 1864. Text Number 72).\n"

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
                final_analysis_df.to_csv('final_analysis.csv', index=False)

                st.subheader("Nicolay's Final Analysis:")
                st.write("Step 3 complete: Here are Nicolay's analysis of Lincoln's speeches based on your question. Click on the dataframe boxes below to see the full outputs.")
                st.dataframe(final_analysis_df)
                st.write('\n\n')



            #for result in final_analysis_results:
              #print(result)

            #def rearrange_combined_output(combined_output):
                #lines = combined_output.strip().split("\n")

                #source_line = next(line for line in lines if "Source:" in line)
                #source_line_parts = source_line.split(":", 2)
                #source_line_formatted = f"{source_line_parts[0]}:{source_line_parts[1]}\n{source_line_parts[2].strip()}"

                #summary_line_start = next(i for i, line in enumerate(lines) if "Summary:" in line)
                #keywords_line_start = next(i for i, line in enumerate(lines) if "Keywords:" in line)
                #summary_lines = lines[summary_line_start:keywords_line_start]
                #keywords_line = lines[keywords_line_start]

                #full_text_start = next(i for i, line in enumerate(lines) if "Text" in line) + 1
                #full_text_end = summary_line_start - 1
                #full_text = "\n".join(lines[full_text_start:full_text_end])

                #source_data = f"""Source Data:

            #{source_line_formatted}

            #{' '.join(summary_lines)}

            #{keywords_line}

            #Full Text:
            #{full_text}
            #"""
                #return source_data

            #source_data = rearrange_combined_output(combined1)
            #print(source_data)


        if search_method == semantic_search:
            embeddings_search()
        else:
            ask_nicolay()

def button_two():
    #Rank Bacon_bot Responses
    with col1:
        st.write("Rank the AI's Interpretation:")
        sh1 = gc.open('AAS_temp')

        wks1 = sh1[0]
        submission_text = wks1.get_value('F2')
        output = wks1.get_value('G2')
        prompt_text = wks1.get_value('D2')
        st.subheader('Your Question')
        st.write(submission_text)
        st.subheader("The AI's Answer:")
        st.write(initial_analysis)
        st.subheader("The AI's Interpretation:")

        with st.form('form2'):
            accuracy_score = st.slider("Is the AI's answer accuracte?", 0, 10, key='accuracy')
            text_score = st.slider("Are the text sections the AI selected appropriate to the question?", 0, 10, key='text')
            interpretation_score = st.slider("How effective was the AI's interpretation of the texts?", 0, 10, key='interpretation')
            coherence_rank = st.slider("How coherent and well-written is the reply?", 0,10, key='coherence')
            st.write("Transmitting the rankings takes a few moments. Thank you for your patience.")
            submit_button_2 = st.form_submit_button(label='Submit Ranking')

            if submit_button_2:
                sh1 = gc.open('AAS_outputs_temp')
                wks1 = sh1[0]
                df = wks1.get_as_df(has_header=True, index_column=None, start='A1', end=('K2'), numerize=False)
                name = df['user'][0]
                submission_text = df['question'][0]
                output = df['initial_analysis'][0]
                combined_df = df['combined_df'][0]
                relevant_texts = df['evidence'][0]
                now = dt.now()
                ranking_score = [accuracy_score, text_score, interpretation_score, coherence_rank]
                ranking_average = mean(ranking_score)

                def ranking_collection():
                    d4 = {'user':["0"], 'user_id':[user_id],'question':[submission_text], 'output':[initial_analysis], 'accuracy_score':[accuracy_score], 'text_score':[text_score],'interpretation_score':[interpretation_score], 'coherence':[coherence_rank], 'overall_ranking':[ranking_average], 'date':[now]}
                    df4 = pd.DataFrame(data=d4, index=None)
                    sh4 = gc.open('AAS_rankings')
                    wks4 = sh4[0]
                    cells4 = wks4.get_all_values(include_tailing_empty_rows=False, include_tailing_empty=False, returnas='matrix')
                    end_row4 = len(cells4)
                    wks4.set_dataframe(df4,(end_row4+1,1), copy_head=False, extend=True)

                ranking_collection()
                st.write('Rankings recorded - thank you! Feel free to continue your conversation with Francis Bacon.')





st.write("Select the 'Ask Bacon' button to ask the AI questions. Select 'Rank Bacon' to note your impressions of its responses.")


pages = {
        0 : button_one,
        1 : button_two,
    }

if "current" not in st.session_state:
    st.session_state.current = None

if st.button("Ask Nicolay"):
        st.session_state.current = 0
if st.button("Ignore This Button"):
    st.session_state.current = 1

if st.session_state.current != None:
    pages[st.session_state.current]()
