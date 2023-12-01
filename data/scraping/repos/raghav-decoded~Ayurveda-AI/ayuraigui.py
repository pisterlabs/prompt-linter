import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

# Define the path to your trained model
model_path = "/AyurAI/Plant_Recognition/plant_identification_model2.h5"

# Load the trained model
model = tf.keras.models.load_model(model_path)

label_mapping = {
    0: 'Alpinia Galanga (Rasna)\nAlpinia galanga, a type of ginger plant, is known for its aromatic rhizomes and is used in traditional medicine for its various properties and benefits. It is commonly used in culinary dishes as a spice and for its medicinal properties.',
    1: 'Amaranthus Viridis (Arive-Dantu)\nAmaranthus viridis, also known as slender amaranth, is a leafy green vegetable rich in essential nutrients. It is a popular choice in many cuisines and is valued for its nutritional content.',
    2: 'Artocarpus Heterophyllus (Jackfruit)\nArtocarpus heterophyllus, commonly referred to as jackfruit, is a tropical tree known for its large, sweet, and aromatic fruit. Jackfruit is a versatile ingredient in various culinary dishes.',
    3: 'Azadirachta Indica (Neem)\nAzadirachta indica, or neem tree, is well-regarded for its numerous medicinal properties. Neem leaves, oil, and extracts are used in traditional medicine and skincare products.',
    4: 'Basella Alba (Basale)\nBasella alba, also known as Malabar spinach or Basale, is a leafy vegetable commonly used in Indian cuisine. It is valued for its high nutritional content and is known for its cooling properties.',
    5: 'Brassica Juncea (Indian Mustard)\nBrassica juncea, known as Indian mustard, is a mustard plant species. Its seeds are used to make mustard oil, and the leaves are used as a leafy green vegetable in various dishes.',
    6: 'Carissa Carandas (Karanda)\nCarissa carandas, commonly known as Karanda or Christs thorn, is a tropical fruit-bearing shrub. Its fruits are used to make jams, jellies, and traditional remedies.',
    7: 'Citrus Limon (Lemon)\nCitrus limon, or lemon, is a citrus fruit known for its tart flavor and high vitamin C content. Lemons are widely used in cooking, beverages, and for their health benefits.',
    8: 'Ficus Auriculata (Roxburgh fig)\nFicus auriculata, also known as Roxburgh fig or elephant ear fig, is a species of fig tree. It is valued for its edible fruit and is used in traditional medicine.',
    9: 'Ficus Religiosa (Peepal Tree)\nFicus religiosa, commonly known as the peepal tree, is considered sacred in many cultures. It is used for its medicinal properties and as a shade tree.',
    10: 'Hibiscus Rosa-sinensis\nHibiscus rosa-sinensis, also known as the Chinese hibiscus or shoeblack plant, is a flowering shrub with showy blooms. It is valued for its ornamental and medicinal uses.',
    11: 'Jasminum (Jasmine)\nJasminum, commonly known as jasmine, is a fragrant flowering plant used in perfumery and traditional medicine. It is known for its aromatic flowers.',
    12: 'Mangifera Indica (Mango)\nMangifera indica, or mango tree, is a tropical fruit tree known for its sweet and juicy fruits. Mangoes are widely enjoyed in various culinary dishes and desserts.',
    13: 'Mentha (Mint)\nMentha, commonly referred to as mint, is a popular herb known for its refreshing flavor and aromatic leaves. Mint is used in culinary dishes, beverages, and for its medicinal properties.',
    14: 'Moringa Oleifera (Drumstick)\nMoringa oleifera, often called drumstick tree, is a highly nutritious plant. Its leaves, pods, and seeds are used in cooking and as a source of essential nutrients.',
    15: 'Muntingia Calabura (Jamaica Cherry-Gasagase)\nMuntingia calabura, also known as Jamaica cherry or gasagase, is a small tree bearing sweet and juicy red fruits. It is valued for its fruit and traditional uses.',
    16: 'Murraya Koenigii (Curry)\nMurraya koenigii, commonly known as curry tree, is a tropical tree used for its aromatic leaves, which are a key ingredient in many Indian and Southeast Asian dishes.',
    17: 'Nerium Oleander (Oleander)\nNerium oleander, or oleander, is an ornamental shrub with beautiful but toxic flowers. It is used in landscaping and has limited traditional medicinal uses.',
    18: 'Nyctanthes Arbor-tristis (Parijata)\nNyctanthes arbor-tristis, known as parijata or night-flowering jasmine, is a small tree with fragrant white flowers. It is considered sacred in some cultures.',
    19: 'Ocimum Tenuiflorum (Tulsi)\nOcimum tenuiflorum, commonly known as tulsi or holy basil, is a sacred herb in Hinduism. It is used in cooking, traditional medicine, and religious rituals.',
    20: 'Piper Betle (Betel)\nPiper betle, also known as betel leaf or paan, is a tropical plant used for its aromatic leaves, which are chewed with areca nut and slaked lime in some cultures.',
    21: 'Plectranthus Amboinicus (Mexican Mint)\nPlectranthus amboinicus, commonly known as Mexican mint or Cuban oregano, is a herb with aromatic leaves used in cooking and traditional medicine.',
    22: 'Pongamia Pinnata (Indian Beech)\nPongamia pinnata, also known as Indian beech or pongam tree, is a tree valued for its oil, seeds, and traditional uses in various parts of Asia.',
    23: 'Psidium Guajava (Guava)\nPsidium guajava, or guava, is a tropical fruit tree known for its sweet and nutritious fruits. Guavas are enjoyed fresh and used in culinary dishes and beverages.',
    24: 'Punica Granatum (Pomegranate)\nPunica granatum, commonly known as pomegranate, is a fruit-bearing shrub with juicy and antioxidant-rich seeds. Pomegranates are used in cooking and for their health benefits.',
    25: 'Santalum Album (Sandalwood)\nSantalum album, or sandalwood tree, is known for its fragrant heartwood, which is used to extract sandalwood oil. It is valued in perfumery, religious rituals, and traditional medicine.',
    26: 'Syzygium Cumini (Jamun)\nSyzygium cumini, commonly known as jamun or Java plum, is a fruit tree with sweet and tangy purple-black fruits. Jamun is enjoyed fresh and used in various culinary preparations.',
    27: 'Syzygium Jambos (Rose Apple)\nSyzygium jambos, known as rose apple, is a fruit-bearing tree with sweet and aromatic fruits. Rose apples are eaten fresh and used in fruit salads.',
    28: 'Tabernaemontana Divaricata (Crape Jasmine)\nTabernaemontana divaricata, commonly known as crape jasmine, is an ornamental shrub with fragrant white flowers. It is valued for its beauty and limited traditional uses.',
    29: 'Trigonella Foenum-graecum (Fenugreek)\nTrigonella foenum-graecum, or fenugreek, is an herb known for its aromatic seeds and leaves. Fenugreek seeds are used in cooking and traditional medicine.',
}

class AyurAIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AyurAI")

        # Create tabs
        self.create_tabs()

        # Create a frame for the table
        # self.table_frame = ttk.Frame(root)
        # self.table_frame.pack(pady=10)

    def create_tabs(self):
        tab_control = ttk.Notebook(self.root)

        # Tab 1: Ayurvedic Formulation Recommender
        tab1 = ttk.Frame(tab_control)
        tab_control.add(tab1, text="Formulation Recommender")

        symptoms_label = ttk.Label(tab1, text="Enter Symptoms (separated by spaces):")
        symptoms_label.pack()

        self.symptoms_entry = ttk.Entry(tab1)
        self.symptoms_entry.pack()

        recommend_button = ttk.Button(tab1, text="Recommend", command=lambda: self.recommend_formulation(tab1))
        recommend_button.pack()

        self.recommendation_result = tk.Text(tab1, height=10, width=40)
        self.recommendation_result.pack()

        # Create a frame for the table
        self.table_frame = ttk.Frame(tab1)
        self.table_frame.pack(pady=10)

        tab_control.pack(expand=1, fill="both")

        # Tab 2: Plant Recognition
        tab2 = ttk.Frame(tab_control)
        tab_control.add(tab2, text="Plant Recognition")

        upload_label = ttk.Label(tab2, text="Upload Plant Image:")
        upload_label.pack()

        self.upload_button = ttk.Button(tab2, text="Upload", command=self.upload_image)
        self.upload_button.pack()

        self.recognition_result = ttk.Label(tab2, text="")
        self.recognition_result.pack()

        tab_control.pack(expand=1, fill="both")
        
        from tkinter import scrolledtext

        # Tab 3: Ayurvedic Consultant Chatbot
        tab3 = ttk.Frame(tab_control)
        tab_control.add(tab3, text="Chatbot")

        chat_label = ttk.Label(tab3, text="Ask a Question:")
        chat_label.pack()

        self.chat_input = ttk.Entry(tab3)
        self.chat_input.pack()

        send_button = ttk.Button(tab3, text="Send", command=self.ask_chatbot)
        send_button.pack()

        # Create a scrolled text widget for the chat history
        self.chat_output = scrolledtext.ScrolledText(tab3, height=10, width=40)
        self.chat_output.pack()

        tab_control.pack(expand=1, fill="both")

        # Tab 4: Medicine Information
        tab4 = ttk.Frame(tab_control)
        tab_control.add(tab4, text="Medicine Information")

        medicine_label = ttk.Label(tab4, text="Enter Medicine Name:")
        medicine_label.pack()

        self.medicine_entry = ttk.Entry(tab4)
        self.medicine_entry.pack()

        display_button = ttk.Button(tab4, text="Display Info", command=self.display_medicine_info)
        display_button.pack()

        # Create a Treeview widget for displaying the medicine data in a table
        self.tree = ttk.Treeview(tab4, columns=("Title", "Price", "Link"), show="headings")
        self.tree.heading("Title", text="Title")
        self.tree.heading("Price", text="Price")
        self.tree.heading("Link", text="Link")

        # Configure the Treeview for vertical scrolling
        vsb = ttk.Scrollbar(tab4, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)

        # Pack the Treeview and vertical scrollbar
        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        tab_control.pack(expand=1, fill="both")

    def display_medicine_info(self):
        # Get the medicine name from the entry
        medicine_name = self.medicine_entry.get()

        import requests
        from bs4 import BeautifulSoup

        def scrape_ayurkart(search_query):
            ayurkart_url = f"https://www.ayurkart.com/search?q={search_query}"
            print(ayurkart_url)

            # Send an HTTP GET request to the URL
            response = requests.get(ayurkart_url)

            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')

                # Extract data using class selectors
                price_elements = soup.find_all('span', class_='product-price__price')
                title_elements = soup.find_all('a', class_='list-view-item__title')
                link_elements = soup.find_all('a', class_='list-view-item__title')   

                # Clear existing data in the Treeview
                self.tree.delete(*self.tree.get_children())

                # Populate the Treeview with the fetched information
                for price_element, title_element, link_element in zip(price_elements, title_elements, link_elements):
                    price = price_element.get_text()
                    title = title_element.get_text()
                    link = 'https://www.ayurkart.com' + link_element['href'] if link_element else "Link not found"
                    self.tree.insert("", "end", values=(title, price, link))

                # if __name__ == "__main__":
                    # search_query = "vyaghradi kashaya"
        scrape_ayurkart(medicine_name)

    def ask_chatbot(self):
        import subprocess
        # Connect this function to your backend for chatbot interaction
        question = self.chat_input.get()

        # Display user's question in the chat history
        self.display_message("You: " + question + "\n")
        
        # Call your backend function here and update the chat window
        answer = self.get_chatbot_response(question)

        # Display chatbot's response in the chat history
        self.display_message("AyurGPT: " + answer + "\n")

        subprocess.run(["say", answer])

    def get_chatbot_response(self, user_input):

        def create_chatbot(file_path, chain_type, k, llm_name, api_key):
            # Load documents from a PDF file
            loader = PyPDFLoader(file_path)
            documents = loader.load()

            # Split documents into smaller chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            docs = text_splitter.split_documents(documents)

            # Create embeddings using OpenAI GPT-3.5
            embeddings = OpenAIEmbeddings(openai_api_key=api_key)

            # Create a vector database from the documents
            db = DocArrayInMemorySearch.from_documents(docs, embeddings)

            # Define a retriever for similarity search
            retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})

            # Create a chatbot chain
            qa = ConversationalRetrievalChain.from_llm(
                llm=ChatOpenAI(model_name=llm_name, temperature=0, openai_api_key=api_key), 
                chain_type=chain_type, 
                retriever=retriever, 
                return_source_documents=True,
                return_generated_question=True,
            )

            return qa

        api_key = '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'
        llm_name = 'gpt-3.5-turbo'

        # Example usage:
        file_path = '/AyurAI/chatbot/intro_ayurveda.pdf'
        chain_type = 'stuff'
        k = 3

        chatbot = create_chatbot(file_path, chain_type, k, llm_name, api_key)

        # Interaction loop
        chat_history = []  # Initialize an empty chat history

        # Create a dictionary with the user's question and the chat history
        input_dict = {
            "question": user_input,
            "chat_history": chat_history
        }
    
        # Pass the input dictionary to the chatbot
        response = chatbot(input_dict)
        
        # Extract and print just the answer
        answer = response.get("answer", "Chatbot: I don't know the answer to that question.")
        
        # Limit the response to a single sentence
        answer = answer.split('.')[0] + '.'
    
        print(answer)
        
        # Update the chat history with the user's question and the chatbot's response
        # chat_history.append(user_input)
        # chat_history.append(answer)
        return answer
    
    def display_message(self, message):
        # Function to display messages in the chat history
        self.chat_output.insert(tk.END, message + "\n")
        self.chat_output.yview(tk.END)  # Scroll to the bottom of the chat history

    def recommend_formulation(self,tab):
        # Get user input from the entry field
        user_input = self.symptoms_entry.get()

        # Call your Ayurvedic formulation recommendation logic here
        recommendations = recommend_ayurvedic_formulation(user_input)

        # Clear the previous recommendations and table
        self.recommendation_result.delete(1.0, tk.END)
        self.clear_table()

        if isinstance(recommendations, list):
            # Display the recommendations in the text box
            for recommendation in recommendations:
                self.recommendation_result.insert(tk.END, f"- {recommendation}\n")

            # Display the table of formulations and details
            self.display_formulation_table(recommendations)
        else:
            # Display a message if no recommendations were found
            self.recommendation_result.insert(tk.END, recommendations)

    def clear_table(self):
        # Clear the table
        for widget in self.table_frame.winfo_children():
            widget.destroy()

    def display_formulation_table(self, formulations):
        df1 = pd.read_csv('/AyurAI/Formulation_Recommender/Formulation-Indications.csv')
        # Create a boolean mask to filter rows where the second column matches any element in closest_formulations
        mask = df1.iloc[:, 0].isin(formulations)

        # Use the mask to select the rows that match the condition
        filtered_df = df1[mask]

        # Create a Treeview widget for the table
        table = ttk.Treeview(self.table_frame, columns=list(df1.columns), show="headings")

        # Set headings for the table columns
        for column in df1.columns:
            table.heading(column, text=column)

        # Insert data into the table
        for index, row in filtered_df.iterrows():
            table.insert("", "end", values=list(row))

        # Pack the table
        table.pack()

    def recognize_plant(self):
        # Connect this function to your backend for plant recognition
        print()
        # Call your backend function here and update self.recognition_result

    def upload_image(self):
        file_path = filedialog.askopenfilename(title="Select an Image File")
        if file_path:
            predicted_label, confidence = self.predict_plant(file_path)
            self.recognition_result.config(text=f"Predicted Label: {predicted_label}\nConfidence: {confidence:.2f}")

    def preprocess_image(self, image_path):
        image = load_img(image_path, target_size=(224, 224))
        image_array = img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)
        preprocessed_image = preprocess_input(image_array)
        return preprocessed_image

    def predict_plant(self, image_path):
        preprocessed_image = self.preprocess_image(image_path)
        predictions = model.predict(preprocessed_image)
    
        # Map model's numeric predictions to labels
        predicted_label_index = np.argmax(predictions)
        predicted_label = label_mapping.get(predicted_label_index, "Unknown")
        confidence = predictions[0][predicted_label_index]
    
        return predicted_label, confidence


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def recommend_ayurvedic_formulation(user_input):
    df1 = pd.read_csv('/AyurAI/Formulation_Recommender/Formulation-Indications.csv')

    formulations_lst = list(df1['Name of Medicine'])

    original_list = list(df1['Main Indications'])

    processed_list = []

    for item in original_list:
        # Remove spaces and newline characters, convert to lowercase
        processed_item = ''.join(item.split()).lower()
        processed_list.append(processed_item)

    # List of lists of symptoms
    list_of_symptoms = processed_list

    # Flatten the list of lists and split the symptoms using commas and spaces
    flat_symptoms = [symptom.replace(',', ' ').split() for symptoms in list_of_symptoms for symptom in symptoms.split(',')]

    # Get unique symptoms as a list
    unique_symptoms = list(set(symptom for sublist in flat_symptoms for symptom in sublist))

    data = {
        "Formulation": formulations_lst,
        "Symptoms": processed_list,
    }

    symptoms = pd.read_csv('/AyurAI/Formulation_Recommender/ayurvedic_symptoms_desc.csv')

    symptoms['Symptom'] = symptoms['Symptom'].str.lower()

    def symptoms_desc(symptom_name):
        row = symptoms[symptoms['Symptom'] == symptom_name.lower()]
    #     print(row)
        if not row.empty:
            description = row.iloc[0]['Description']
            print(f'Description of "{symptom_name}": {description}')
        else:
            print(f'Symptom "{symptom_name}" not found in the DataFrame.')

    def symptoms_lst_desc(user_symptoms):
        for item in user_symptoms:
    #         print(item)
            symptoms_desc(item)

    import difflib

    # Your list of correct words (assuming you have a list called unique_symptoms)
    correct_words = unique_symptoms

    def correct_symptoms(symptoms):
        corrected_symptoms = []
        for symptom in symptoms:
            corrected_symptom = difflib.get_close_matches(symptom, correct_words, n=1, cutoff=0.6)
            if corrected_symptom:
                corrected_symptoms.append(corrected_symptom[0])
            else:
                corrected_symptoms.append(symptom)
        return corrected_symptoms

    input_symptoms = user_input.split()
    user_symptoms = correct_symptoms(input_symptoms)
    print(f"Did you mean: {', '.join(user_symptoms)}")

    symptoms_lst_desc(user_symptoms)
    user_symptoms_str = " ".join(user_symptoms)  # Convert user symptoms to a single string

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Create a TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Fit and transform the symptom text data into numerical features
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['Symptoms'])

    # Transform user symptoms into TF-IDF format
    user_symptoms_tfidf = tfidf_vectorizer.transform([user_symptoms_str])

    # Calculate cosine similarity between user's symptoms and all formulations
    similarities = cosine_similarity(user_symptoms_tfidf, tfidf_matrix)

    # Set a threshold for similarity score (adjust as needed)
    similarity_threshold = 0.5  # You can adjust this value

    # Find all formulations with similarity scores above the threshold
    matching_indices = [i for i, sim in enumerate(similarities[0]) if sim > similarity_threshold]

    final_lst = []

    if not matching_indices:
        final_lst = ["No matching formulations for the provided symptoms"]
        print("No matching formulations found for the provided symptoms.")
    else:
        closest_formulations = df.iloc[matching_indices]["Formulation"]
        print("Closest Formulations:")
        final_lst = closest_formulations.tolist()
        print(closest_formulations.tolist())

    # For now, returning a placeholder message
    return final_lst

def main():
    root = tk.Tk()
    app = AyurAIApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
