import pandas as pd
import openai
import dotenv
import os

dotenv.load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


df = pd.read_csv("C:/Users/malco/OneDrive/Documents/GitHub/Auto-GPT/GPT-Tools/book_extraction/flash_decks/like_switch_aggr.csv")

#for value in data frame column card_definition has \n filter into a new data frame


df = df[df["card_definition"].str.contains("\n")]


reduced_df = pd.DataFrame(columns=["card_name", "card_definition"])

for index, row in df.iterrows():
    
    #try:
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages= [{"role": "system", "content": 
                    "You take in information and condense it into one high quality flashcard. You produce highly a detailed flashcard with a term name and a definition in the format: Term: <Card Name> \n\n Definition: <Card Definition> "}, 
                    {"role": "user", "content": f"Please produce a flashcard on {row['card_name']} from the following content: \n {row['card_definition']}"}],
    )

    
    
    text = response["choices"][0]["message"]["content"]

    print(text)
    term, definition = text.split("\n\n")

    term = term.replace("Term: ", "")
    definition = definition.replace("Definition: ", "")

    reduced_df = pd.concat([reduced_df, pd.DataFrame({"card_name": [term], "card_definition": [definition]})], ignore_index=True)

    reduced_df["card_definition"] = reduced_df["card_definition"].apply(lambda x: x.strip())

# save reduced_df to a csv file
reduced_df.to_csv("C:/Users/malco/OneDrive/Documents/GitHub/Auto-GPT/GPT-Tools/book_extraction/flash_decks/like_switch_reduced.csv")

        

    #except Exception as e:
    #    continue
        



    




