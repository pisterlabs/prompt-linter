import cohere
import streamlit as st
from nlpSummarizer import formattingForSummarizer
import PyPDF2
import re
import math
import time

co = cohere.Client(st.secrets["COHERE_API_KEY"]) 

# with open('content1.txt', 'r', encoding='utf-8') as file:
#     text = file.read()
st.header('SSW-Summarizer')
st.write('High End Summarizer Using Cohere Endpoints - Summarize any documents upto 1 Million words')


def split_text_into_sentences(text):
    # Split text into sentences using regular expressions
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
    return sentences

def summarzerMain(text) :
    responses = []

    text1 = text[:len(text)//2]
    text2 = text[len(text)//2:]
    # start_index = 15 * 5000
    # end_index = (15 + 1) * 5000
    # st.write(text[start_index:end_index])
    # st.write(text)
    
    total_chunks = math.ceil(len(text) / 5000)

    summarizer_prompt  = "You should summarize the given content into meanigfull summaries, withour loosing the context."

    # Add a header for the summary
    with st.sidebar:
        if st.button('Generate Summary'):
            print("___________ START __________")

            start_time = time.time()

            with st.spinner('Generating response...'):

                st.write(' \n Total chunks :', total_chunks)
                if len(text) >= 10000 and len(text) <= 50000:
                    st.write("Make a coffee, take a sip and come back...") 

                if len(text) >= 50000 and len(text) <= 100000:
                    st.write("Watch an anime, finish a episode ...") 

                if len(text) >= 100000 :
                    st.write("Go for a movie, come back during the intervals ...")
                

                # Split the text into chunks of 5000 tokens and process each chunk
                for i in range(total_chunks - 1):
                    start_index = i * 5000
                    end_index = (i + 1) * 5000
                    chunk_text = text[start_index:end_index]
                    # Process the current chunk
                    response = co.summarize( 
                        text=chunk_text,
                        length='short',
                        format='bullets',
                        model='summarize-xlarge',
                        additional_command=summarizer_prompt,
                        temperature=0.3,
                    ) 

                    # Append the summary to the responses list
                    responses.append(response.summary)
                    print("CHUNK", i, "COMPLETED")
                    print(response.summary)



            # Add a header for the summary
            st.markdown("<h3 style='color: green;'>Summary:</h3>", unsafe_allow_html=True)
            # After summarizing add these responses into co.chat and make it upto 300 words
            print("PHASE 1 Completed______________________")
            print("PHASE 2 ONGOING")

            chatResponses = ' '.join(responses)
            # st.write('Word Count for chatResponses:', len(chatResponses))

            summarization_chunks = math.ceil(len(chatResponses) / 6000)
            generated_responses = []

            # Split the chatResponses into chunks of 5000 tokens and process each chunk
            for i in range(summarization_chunks):
                start_index = i * 6000
                end_index = (i + 1) * 6000
                generated_chunk_text = chatResponses[start_index:end_index]

                # Construct the prompt template for the current chunk
                prompt_template2 = (
                    "Summarize the given content into n number of paragraph response of each. "
                    "The content is: " + generated_chunk_text + "Give Response in a VERY EFFICIENT FORMAT, Preferably in BULLETS"
                )

                # Generate response for the current chunk
                generate_response = co.generate(
                    model='command',
                    prompt=prompt_template2,
                    max_tokens=400,
                    temperature=0.3,
                    k=0,
                    stop_sequences=[],
                    truncate='END',
                    return_likelihoods='NONE'
                )
                generated_responses.append(generate_response.generations[0].text)
                print("CHUNK-GENERATION :- ", i, "COMPLETED")
            
            # Extract the text from the generated responses
            # final_generated_text = [response.text for response in generated_responses]
            # st.write("WORD LENFTH OF GENERATIONS : ", len(generated_responses), "\n")
            # st.write("GENERATIONS : ", generated_responses, "\n")
            # st.write(final_generated_text)
            final_generated_response = ' '.join(generated_responses)

            prompt_template3 = (
                    "Summarize the given content into a response of multiple paragraphs with `MORE THAN 500 WORDS`. "
                    "The content is: " + final_generated_response + "Give Response in a VERY EFFICIENT FORMAT, Preferably in BULLETS, Dont give any other information other than the summary."
                )
            
            final_generation = co.generate(
                model='command',
                prompt=prompt_template3,
                num_generations=3,
                max_tokens=400,
                temperature=0.3,
                k=0,
                stop_sequences=[],
                truncate='END',
                return_likelihoods='NONE'
            )

            end_time = time.time()  
            elapsed_time = ( end_time - start_time )/60
            st.write("Summarization took: ", round(elapsed_time, 2), " minutes.")                

            # st.write(response)
            print("PHASE 2 Completed______________________")
            # print('Prediction: {}'.format(response.generations[0].text))
            
            prediction_text = final_generation.generations[0].text
            st.write(f'Prediction: {prediction_text}')
            st.write('Word Count:', len(prediction_text))

            print("___________ END __________")
            # count words in summary
            # count = len(response.summary.split())
            # st.write('Word Count:', count)


def main ():
   
  uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf"]) 

  if 'read_more_clicked' not in st.session_state:
    st.session_state.read_more_clicked = False

  if uploaded_file is not None:
    file_extension = uploaded_file.name.split(".")[-1]
    
    if file_extension.lower() == "pdf":
        # Read text from the PDF file
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        pdf_text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num] 
            pdf_text += page.extract_text()

        # Process the PDF text (you might need additional logic for PDF-specific formatting)
        st.markdown("<h4 style='color: #9D9D9D;'>Uploaded Content (PDF):</h4>", unsafe_allow_html=True)
        # Split text into sentences
        sentences = split_text_into_sentences(pdf_text)
        st.write('Word Count:', len(pdf_text.split()))
        # st.write(math.ceil(len(pdf_text) / 5000))
        # st.write(math.ceil(len(pdf_text.split()) / 5000))

        # Display limited content with "Read More" functionality
        num_displayed_sentences = 2
        if len(sentences) > num_displayed_sentences:
            displayed_content = " ".join(sentences[:num_displayed_sentences])
            st.write(displayed_content)
            # if st.button("Read More"):
            #     st.write(" ".join(sentences))
            #     st.session_state.read_more_clicked = False
        else:
            st.write(pdf_text)
        processed_text = formattingForSummarizer(pdf_text)

    elif file_extension.lower() == "txt":
        # Read text from the TXT file
        text = uploaded_file.read().decode('utf-8')

        # Process the text
        st.markdown("<h4 style='color: #9D9D9D;'>Uploaded Content (PDF):</h4>", unsafe_allow_html=True)
        st.write('Word Count:', len(text.split()))
        # st.write(text)
        sentences = split_text_into_sentences(text)
        num_displayed_sentences = 2
        if len(sentences) > num_displayed_sentences:
            displayed_content = " ".join(sentences[:num_displayed_sentences])
            st.write(displayed_content)

        else:
            st.write(text)
        processed_text = formattingForSummarizer(text)
    else:
        st.write("Unsupported file format: As of now we only have the option to read a TXT or PDF file.")

    summarzerMain(processed_text)


if __name__ == "__main__" :
    main()
   
# Add the summary
