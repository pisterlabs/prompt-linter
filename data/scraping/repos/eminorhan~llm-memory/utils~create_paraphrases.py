import os
import time
import openai
import jsonlines

# replace with your own key
os.getenv("OPENAI_API_KEY")

# read and save file names
read_fname = 'data/seen_data_3.jsonl'
save_fname = 'data/paraphrase_seen_3.jsonl'

prefix = "Please paraphrase the given sentence by expressing it differently but preserving the original meaning as much as possible. Here are some examples to help guide you:\n\n" \
"Sentence: In terms of self-reliance and resilience, the average adult in 1967 would be a massive outlier in 2022.\n" \
"Paraphrase: People living in 1967 were very different from people living today in terms of their self-sufficiency and their toughness in the face of adversity.\n\n" \
"Sentence: The girl was weak minded so that it was only with the greatest difficulty that she could cover her moves, in fact she never could do so with success.\n" \
"Paraphrase: Because of her feeblemindedness, the girl wasn't successful at hiding her activities.\n\n" \
"Sentence: People often ask me: Can open borders work if only one country does it? If the country is the U.S., I have little doubt.\n" \
"Paraphrase: In my view, an open borders immigration policy can work successfully in the U.S.\n\n" \
"Sentence: Sharing your angry feelings is an effective way to dominate the social world, but a terrible way to discover the truth or sincerely convince others.\n" \
"Paraphrase: Angry outbursts may be effective for gaining power, but they're not a good way to find out truths about the world or to persuade other people.\n\n" \
"Sentence: If you fail to work up the courage to make this small request, you have learned something about yourself: The other person does not inspire you.\n" \
"Paraphrase: If you can't even bring yourself to ask for this small favor from the other person, you don't really love them.\n\n" \
"Sentence: "

postfix = "\nParaphrase:"

data = []

with jsonlines.open(read_fname) as reader:
    for obj in reader:
      sent = obj['sent']
      prompt = prefix + sent + postfix

      response = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt,
        max_tokens=128,
        temperature=0.99
      )
      resp = response["choices"][0]["text"]
      resp = resp.replace("\n", "")
      print(sent)
      print(resp)
      print("\n")
      data.append({"sent": resp, "sent_id": obj['sent_id'], "url_id": obj['url_id']})
      time.sleep(1)  # to prevent rate limit error

# save paraphrases to file
with jsonlines.open(save_fname, mode='w') as writer:
       writer.write_all(data)