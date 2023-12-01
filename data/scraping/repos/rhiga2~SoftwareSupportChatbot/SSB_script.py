import sys, fitz, pickle, os
import openai 
import wandb

question = ''
#question += 'Q: How do I relate a filepart in iManage Records Manager?'
#question += 'Q: How do I view an electronic rendition in iManage Records Manager?'
#question += 'Q: How do I cancel a delivery request in imanage records manager?'
question += 'Q: Can you list the steps to printing an individual label in imanage records manager?'
#question += 'Q: How do I view records that are checked out in imanage records manager?'
#question = 'Q: How do I view pending delivery requests in imanage records manager?'
#question = 'Q: Can a file part contain other file parts in imanage records manager?'
#question = 'Q: How do I perform a full text search in imanage records manager?'
#keyword = 'relating'
#keyword = 'renditions'
keyword = 'label'
#keyword = 'pending delivery'
#keyword = 'file part'
#keyword = 'simple searching'

fname = 'C:\dev\pdfconversion\IRM Web Client User Guide (Legal Version) 10.3.3.pdf'

if not os.path.exists('C:\dev\pdfconversion\IRMWEB_toc.pickle'):    
    doc = fitz.open(fname)  # open document
    out = open(fname + ".txt", "wb")  # open text output
    toclist = doc.get_toc(True)
    with open('C:\dev\pdfconversion\IRMWEB_toc.pickle', 'wb') as handle:
        pickle.dump(toclist, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    toclist = pickle.load(open('C:\dev\pdfconversion\IRMWEB_toc.pickle', 'rb'))

llist = list()

for i in range(len(toclist)):
    toclist[i] = str(toclist[i]).lower()

filteredl = list(filter(lambda a: keyword in a, toclist))

pagel = []

for s in filteredl:
    sl = s.split(',')
    pagel.append(int(sl[2].replace(' ','').replace(']','')))


pdfdoc = fitz.open(fname)  # open document
p = pdfdoc.load_page(int(pagel[0]) - 1) #why isn't intellisense displaying load_page? I installed fitz and pymupf
text = p.get_text()

outputtext = ''

for line in text.splitlines():
    if len(line) > 5:
        outputtext += line
        outputtext += '\n'

text_file = open(f'c:\dev\pdfconversion\pagetext.txt', 'wt')
n = text_file.write(outputtext)
text_file.close()

openai.api_key = os.getenv('OPENAIKEY')
#run = wandb.init(project='SoftwareSupportChatbot')
#prediction_table = wandb.Table(columns=["prompt", "completion"])

gpt_prompt=''
gpt_prompt += outputtext 
#gpt_prompt += '\n'
#gpt_prompt += 'Answer the question as truthfully as possible and if you are unsure of the answer say "Sorry I don''t know"'
gpt_prompt += '\n'
gpt_prompt += question
gpt_prompt += '\n'
gpt_prompt += 'A:'

response = openai.Completion.create(
  engine="text-davinci-002",
  prompt=gpt_prompt,
  temperature=0.0,
  max_tokens=104,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0
)

print(response['choices'][0]['text'])

#prediction_table.add_data(gpt_prompt,response['choices'][0]['text'])
#wandb.log({'predictions': prediction_table})
#wandb.finish()

test=''
    
