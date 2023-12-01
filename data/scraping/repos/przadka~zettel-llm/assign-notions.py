import os
import openai
import pandas as pd
import time


openai.api_key = os.getenv("OPENAI_API_KEY")

REQUESTS_PER_MINUTE = 10
SLEEP_INTERVAL = 60.0 / REQUESTS_PER_MINUTE

## SYSTEM_MSG = "You are a helpful assistant. Assign labels to the following quote based on the provided list."

SYSTEM_MSG = '''
Assistant is a superhuman zettelkasten maintainer tasked with assigning labels to zettels. This zettelkasten system is related to philosophy and it is going to be used by philosophers.

Example 1: 
===
Author: Hans Moravec
Title: "Robot: Mere machine to transcendent mind"
Quote: "The nonlinear equations of general relativity are notoriously hard to solve, only the simplest cases have been examined, and there is no theory of quantum gravity at all. That several plausible time machines have emerged in the bit of territory that has been explored is a hopeful indication that the vast unexplored vistas contain better ones, based more on subtle constructions than on brute-force spacetime bending."

Labels assigned: theory of relativity,	time travel
===

Example 2: 
===
Author: Susan Blackmore
Title: "The meme machine"
Quote: "Memes do not yet have precise copying machinery as DNA has. They are still evolving their copying machines, and this is what all the technology is for. […] [205] As Dawkins put it, the new replicator is “still drifting clumsily about in its primeval soup” (Dawkins 1976, p. 192). That soup is the soup of human culture, human artefacts, and human-made copying systems. You and I are living during the stage at which the replication machinery for the new replicator is still evolving, and has not yet settled down to anything like a stable form. The replication machinery includes all those meme-copying devices that fill my home, from pens and books to computers and hi-fi. […] Bearing in mind the dangers of comparing memes and genes, we can speculate that the same process works in both cases, producing a uniform high-fidelity copying system capable of creating a potentially infinite number of products. The genes have settled down, for the most part, to an exquisitely high-fidelity digital copying system based on DNA. The memes have not yet reached such a high-quality system and will probably not settle on one for a long time yet."

Labels assigned: mind and body
===
'''

def assign_notions(author, title, quote, allowed_labels):
    """Assign new labels using OpenAI."""
    
    allowed_labels_str = '\n'.join(allowed_labels)

    # Construct the user message
    user_msg = f'''
    Here is a new quote you need to assign labels to:

    ===
    Author: {author}
    Title: "{title}"
    Quote: "{quote}"

    ===
    List of possible labels have been narrowed down to, and they are sorted by relevance:

    ===
    {allowed_labels_str}
    ===

    Please assign labels to the new quote. Try to assign as few labels as possible, 
    preferably one, two or three. Use only the labels from the list above. 
    Output only the labels, nothing else. 
    Sort labels from the most relevant to the least relevant.
    '''

    print(f"System message: {SYSTEM_MSG}")
    print(f"User message: {user_msg}")

    try:
        response = openai.ChatCompletion.create(
            model='gpt-4',
            messages=[
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": user_msg}
            ],
            max_tokens=30,
        )
        
        return response['choices'][0]['message']['content'].strip()

    except openai.error.InvalidRequestError as e:
        print(f"Error while processing the request: {e}")
        return "ERROR_TRIGGERED_CONTENT_MANAGEMENT_POLICY"



def main():
    """Main function to process queries."""

    df = pd.read_csv("queries.csv")

    for index, row in df.iterrows():
        print(f"Processing row {index}...")
        author = row["author(s)"]
        title = row["title of the source"]
        quote = row["quotation"]
        allowed_labels = [label.strip() for label in row["Merged Notions"].split(",")]
        print(f"Allowed labels: {allowed_labels}")
        assigned_labels = assign_notions(author, title, quote, allowed_labels)
        print(f"Assigned labels: {assigned_labels}")
        # write assigned labels to the dataframe, under the column "Assigned Labels"
        df.at[index, "Assigned Labels"] = assigned_labels
        print("\n")

        # Sleep for the calculated interval to respect the rate limit
        time.sleep(SLEEP_INTERVAL)

    # Save the dataframe with assigned labels back to a new CSV
    df.to_csv("queries_with_labels.csv", index=False)



if __name__ == "__main__":
    main()