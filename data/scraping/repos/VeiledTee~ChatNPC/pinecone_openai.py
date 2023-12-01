import json

import openai
import pinecone

from chat import extract_name, get_information, load_file_information, run_query_and_generate_answer

if __name__ == "__main__":
    GRAMMAR: dict = {
        "lower": "poor",
        "middle": "satisfactory",
        "high": "formal",
    }
    WINDOW: int = 3  # how many sentences are combined
    STRIDE: int = 2  # used to create overlap, num sentences we 'stride' over
    for i in range(7):
        # Open the file and load its contents into a dictionary
        with open("Text Summaries/characters.json", "r") as f:
            names = json.load(f)

        CHARACTER: str = list(names.keys())[i]
        # CHARACTER: str = random.choice(list(names.keys()))
        # CHARACTER: str = "Caleb Brown"
        PROFESSION, SOCIAL_CLASS = get_information(CHARACTER)
        print(CHARACTER, PROFESSION)
        # print(names)
        DATA_FILE: str = f"Text Summaries/Summaries/{names[CHARACTER]}.txt"

        INDEX_NAME: str = "thesis-index"
        NAMESPACE: str = extract_name(DATA_FILE).lower()

        QUERY: str = "Can chromafluke fish be found near Ashbourne?"

        file_data = load_file_information(DATA_FILE)

        with open("keys.txt", "r") as key_file:
            api_keys = [key.strip() for key in key_file.readlines()]

            pinecone.init(
                api_key=api_keys[1],
                environment=api_keys[2],
            )

        final_answer = run_query_and_generate_answer(
            namespace=NAMESPACE,
            data=file_data,
            receiver=CHARACTER,
            job=PROFESSION,
            status=SOCIAL_CLASS,
            query=QUERY,
            index_name=INDEX_NAME,
        )

        print(final_answer)
