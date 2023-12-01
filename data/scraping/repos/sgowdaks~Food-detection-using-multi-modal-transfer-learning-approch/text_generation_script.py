import openai
import csv
import time

# Set up the OpenAI API client
openai.api_key = "key"

# Set up the model and prompt
model_engine = "gpt-3.5-turbo"

o_path = "/home/shivani/work/data/new_challenge_senences.tsv"
i_path = "/home/shivani/work/data/new_challenge_data.tsv"

with open(i_path, "r") as input:
    with open(o_path, "w") as output:
        csv_reader = csv.reader(input, delimiter="\t")
        csv_writer = csv.writer(output, delimiter="\t")
        for i, words in enumerate(csv_reader):
            #prompt = brand + " is a " + type_ + " provided with brief description of " + describe + " now generate a sentences that contains below with some quantity: I ate, " +  brand + "Today"
            prompt = "Given all these categories " + str(words[0]) + " give a sentence how a human would log their food using all of these categories"
            completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", 
            messages=[{"role": "user", "content": prompt}]
            )

            response = completion.choices[0].message.content
            
            #davinci model
            # categories = words[0]
            # completion = openai.Completion.create(
            #     engine=model_engine,
            #     prompt=prompt,
            #     max_tokens=1024,
            #     n=1,
            #     stop=None,
            #     temperature=0.5,
            # )
            # response = completion.choices[0].text
            # response = response.strip("\n")

            csv_writer.writerow([response, str(words[0])])
            if i % 40 == 0:
                time.sleep(200)







