import openai
import pandas as pd
import db_handler

# training_data = db_handler.read_table_posts()
df = pd.read_csv("/home/jod/fesso/testing/processed_text.csv")

# df = df.head(10)


def get_questions(context):
    try:
        response = openai.Completion.create(
            engine="davinci-instruct-beta-v3",
            prompt=f"Escreva perguntas, preferencialmente sobre Omar Aziz, baseadas no texto abaixo\n\nTexto: {context}\n\nPerguntas:\n1.",
            temperature=0,
            max_tokens=250,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["\n\n"]
        )
        return response['choices'][0]['text']
    except:
        return ""


df['questions']= df["context"].apply(get_questions)
df['questions'] = "1." + df['questions']
print(df[['questions']].values[0][0])
df.to_csv("teste2.csv", index=False)