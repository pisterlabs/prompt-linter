import openai
import os 
import json
import random
import time
from pydantic import BaseModel
from typing import Optional, List

    
    
openai.api_key = os.environ['OPENAI_API_KEY']
openai.organization = os.environ['OPENAI_API_ORG']

MAX_RETRIES = 10

class Verse(BaseModel):
    bsb: dict
    macula: dict
    target: dict
    alignment: Optional[list] = None
    alt: Optional[str] = None

### PARSE ARGS ###

import argparse

# Create the argument parser
parser = argparse.ArgumentParser(description='Script to parse command line arguments.')

parser.add_argument('--run_name', type=str, default='default_run', help='Name of the run')
parser.add_argument('--data_path', type=str, default='/Users/ryderwishart/translators-copilot/data/bible/spapddpt.json', help='Path to the data file')
parser.add_argument('--model', type=str, default='gpt-3.5-turbo-instruct', help='Name of the model')
parser.add_argument('--n', type=int, default=0, help='Number of verses to sample')
parser.add_argument('--randomize', action='store_true', help='Randomize the order of the verses')
parser.add_argument('--pseudo_english_only', action='store_true', help='Only generate pseudo-english translations')
parser.add_argument('--ids_file_path', type=str, default=None, help='Path to the txt file containing vref ids')

# Parse the arguments
args = parser.parse_args()

# Print the parsed arguments
print(f"Run Name: {args.run_name}")
print(f"Data Path: {args.data_path}")
print(f"Model: {args.model}")
print(f"Pseudo-English Only: {args.pseudo_english_only}")
specified_ids = []
if args.ids_file_path:
    with open(args.ids_file_path, 'r') as f:
        specified_ids = [line.strip() for line in f.readlines()]
    print(f"Specified IDs: {specified_ids}")
if args.n == 0:
    print("Number of Verses to Sample: ALL")
else:
    print(f"Number of Verses to Sample: {args.n}")
    print(f"Randomize: {args.randomize}")

### ALIGNMENT FUNCTIONS ###

def generate_pseudo_english_prompt(verse: Verse):
    """
    "bridge": "For it is not the hearers of the law who are righteous before God, but it is the doers of the law who will be declared righteous.",
    "macula": "οὐ  γὰρ  οἱ  ἀκροαταὶ  νόμου  δίκαιοι  παρὰ  τῷ  Θεῷ, ἀλλ’  οἱ  ποιηταὶ  νόμου  δικαιωθήσονται.",
    "target": "Ol man i harim nating lo, ol i no kamap stretpela man long ai bilong God. Nogat. Ol man i bihainim lo, ol dispela man tasol bai God i kolim ol stretpela man.",
    "alt": "All men hear nothing law, them no become straight-fella man in eye belong God. No. All men follow law, them this-fella man only then God call them straight-fella man.",
    """
    bsb, macula, target = verse['bsb']['content'], verse['macula']['content'], verse['target']['content']
    return f'''## Context:
Tok Pisin can be quasi-phonetically transposed into pseudo english like this:
| tok pisin | pseudo | english |
| --- | --- | --- |
| bekim tok | back-him talk | answer |
| marasin bilong kilim jem | medicine belong kill'em germ | antiseptic |
| bai God i ken kisim bek ol | buy God he can get him back all | that they may be saved |

For example:

English Phrase: For it is not the hearers of the law who are righteous before God, but it is the doers of the law who will be declared righteous.",
Source Phrase: οὐ  γὰρ  οἱ  ἀκροαταὶ  νόμου  δίκαιοι  παρὰ  τῷ  Θεῷ, ἀλλ’  οἱ  ποιηταὶ  νόμου  δικαιωθήσονται.",
Target Phrase: Ol man i harim nating lo, ol i no kamap stretpela man long ai bilong God. Nogat. Ol man i bihainim lo, ol dispela man tasol bai God i kolim ol stretpela man.",
Pseudo-English Phrase: All men hear nothing law, them no become straight-fella man in eye belong God. No. All men follow law, them this-fella man only then God call them straight-fella man.",

## Instruction:

Can you create pseudo-english out of the following Tok Pisin sentence (reference translations are included for clarity):

English Phrase: {bsb}
Source Phrase: {macula}
Target Phrase: {target}
Pseudo-English Phrase: '''

def generate_broad_greek_alignment_prompt(verse: Verse):
    bsb, macula, target, alt = verse['bsb']['content'], verse['macula']['content'], verse['target']['content'], verse['alt']
    prefatory_content = '''Here are some general facts to note about Tok Pisin:\nTok Pisin is a creole language. Ensure correct affix attachment; follow SVO order; respect compound verbs/processes.\nTranslation style:\nThe Tok Pisin translation is a dynamic equivalence translation aiming to convey the meaning of the source text in a natural and idiomatic way in the target culture. While trying to maintain the general structure and intent of the original text, adjustments can be made to produce a more natural translation. If a translated verse seems to be more literalistic rather than dynamic, adjust the alignment as needed.'''
    
    try:
        return f'''{prefatory_content}
Here is a sentence:
English: For it is not the hearers of the law who are righteous before God, but it is the doers of the law who will be declared righteous.
Greek: οὐ  γὰρ  οἱ  ἀκροαταὶ  νόμου  δίκαιοι  παρὰ  τῷ  Θεῷ, ἀλλ’  οἱ  ποιηταὶ  νόμου  δικαιωθήσονται.
Target: Ol man i harim nating lo, ol i no kamap stretpela man long ai bilong God. Nogat. Ol man i bihainim lo, ol dispela man tasol bai God i kolim ol stretpela man.
Pseudo-English: All men hear nothing law, them no become straight-fella man in eye belong God. No. All men follow law, them this-fella man only then God call them straight-fella man.

Here is a phonological, semantic, orthographic alignment of that sentence:
```
[
    {{
        "English phrase": "For it is not the hearers of the law",
        "Greek phrase": "οὐ γὰρ οἱ ἀκροαταὶ νόμου",
        "Pseudo-English phrase": "All men hear nothing law,",
        "Target phrase": "Ol man i harim nating lo"
    }},
    {{
        "English phrase": "who are righteous before God,",
        "Greek phrase": "δίκαιοι παρὰ τῷ Θεῷ,",
        "Pseudo-English phrase": "them no become straight-fella man in eye belong God.",
        "Target phrase": "ol i no kamap stretpela man long ai bilong God."
    }},
    {{
        "note": "This alignment simply marks where the polarity shift occurs in the sentence.",
        "English phrase": "but it is",
        "Greek phrase": "ἀλλ’",
        "Pseudo-English phrase": "No.",
        "Target phrase": "Nogat."
    }},
    {{
        "English phrase": "the doers of the law",
        "Greek phrase": "οἱ ποιηταὶ νόμου",
        "Pseudo-English phrase": "All men follow law,",
        "Target phrase": "Ol man i bihainim lo"
    }},
    {{
        "English phrase": "who will be declared righteous.",
        "Greek phrase": "δικαιωθήσονται.",
        "Pseudo-English phrase": "them this-fella man only then God call them straight-fella man.",
        "Target phrase": "ol dispela man tasol bai God i kolim ol stretpela man."
    }}
]
```

Please also align the following sentence. Avoid including multiple phrases in a single alignment unit. You may need to break phrases  on commas or other major punctuation, including enclosing quotation marks. But you may also need to break a phrase along conjunctions or other words that typically mark the start of a new phrase. Always respond with perfect JSON:

English: {bsb}
Greek: {macula}
Target: {target}
Pseudo-English: {alt}
'''
    except Exception as e:
        print('Error on Greek alignment prompt generation.', e)
        return 'ERROR'

def generate_broad_hebrew_alignment_prompt(verse: Verse):
    bsb, macula, target, alt = verse['bsb']['content'], verse['macula']['content'], verse['target']['content'], verse['alt']
    prefatory_content = '''Here are some general facts to note about Tok Pisin:\nTok Pisin is a creole language. Ensure correct affix attachment; follow SVO order; respect compound verbs/processes.\nTranslation style:\nThe Tok Pisin translation is a dynamic equivalence translation aiming to convey the meaning of the source text in a natural and idiomatic way in the target culture. While trying to maintain the general structure and intent of the original text, adjustments can be made to produce a more natural translation. If a translated verse seems to be more literalistic rather than dynamic, adjust the alignment as needed.'''
    
    try:
        return f'''{prefatory_content}
Here is a sentence:
English: "In the beginning God created the heavens and the earth."
Hebrew: "בְּרֵאשִׁ֖ית  בָּרָ֣א  אֱלֹהִ֑ים  אֵ֥ת  הַשָּׁמַ֖יִם  וְאֵ֥ת  הָאָֽרֶץ׃"
Target: "Bipo bipo tru God i mekim kamap skai na graun na olgeta samting i stap long en."
Pseudo-English: "Before before true God he make come up sky and ground and all thing he stop long him."

Here is a phonological, semantic, orthographic alignment of that sentence:
```
[
    {{
        "English phrase": "In the beginning",
        "Hebrew phrase": "בְּרֵאשִׁ֖ית",
        "Target phrase": "Bipo bipo tru",
        "Pseudo-English phrase": "Before before true"
    }},
    {{
        "English phrase": "God created",
        "Hebrew phrase": "בָּרָ֣א  אֱלֹהִ֑ים",
        "Target phrase": "God i mekim kamap",
        "Pseudo-English phrase": "God he make come up"
    }},
    {{
        "English phrase": "the heavens and the earth.",
        "Hebrew phrase": "אֵ֥ת  הַשָּׁמַ֖יִם וְאֵ֥ת  הָאָֽרֶץ׃",
        "Target phrase": "skai na graun",
        "Pseudo-English phrase": "sky and ground"
    }},
    {{
        "note": "This is a special case where the target phrase is added to clarify the meaning of the source sentence",
        "English phrase": "",
        "Hebrew phrase": "",
        "Target phrase": "na olgeta samting i stap long en.",
        "Pseudo-English phrase": "and all thing he stop long him."
    }}
]
```

Here is a sentence:
English: God called the light “day,” and the darkness He called “night.” And there was evening, and there was morning—the first day. 
Hebrew: וַיִּקְרָ֨א  אֱלֹהִ֤ים׀ לָאוֹר֙  י֔וֹם  וְלַחֹ֖שֶׁךְ  קָ֣רָא  לָ֑יְלָה  וַֽיְהִי־ עֶ֥רֶב  וַֽיְהִי־ בֹ֖קֶר  י֥וֹם  אֶחָֽד׃פ
Target: Tulait em i kolim “De,” na tudak em i kolim “Nait.” Nait i go pinis na moning i kamap. Em i de namba wan.
Pseudo-English: to-light him call “Day,” and to-dark him call “Night.” Night go finish and morning come-up. Him day number one.
                    
Here is a phonological, semantic, orthographic alignment of that sentence:
```
[
    {{
        "English phrase": "God called the light",
        "Hebrew phrase": "וַיִּקְרָ֨א  אֱלֹהִ֤ים׀ לָאוֹר֙",
        "Target phrase": "Tulait em i kolim",
        "Pseudo-English phrase": "to-light him call"
    }},
    {{
        "English phrase": "“day,”",
        "Hebrew phrase": "י֔וֹם",
        "Target phrase": "“De,”",
        "Pseudo-English phrase": "“Day,”"
    }},
    {{
        "English phrase": "and the darkness",
        "Hebrew phrase": "וְלַחֹ֖שֶׁךְ",
        "Target phrase": "na tudak",
        "Pseudo-English phrase": "and to-dark"
    }},
    {{
        "English phrase": "He called “night.”",
        "Hebrew phrase": "לָ֑יְלָה",
        "Target phrase": "em i kolim “Nait.”",
        "Pseudo-English phrase": "him call “Night.”"
    }},
    {{
        "English phrase": "And there was evening",
        "Hebrew phrase": "וַֽיְהִי־ עֶ֥רֶב",
        "Target phrase": "Nait i go pinis",
        "Pseudo-English phrase": "Night go finish"
    }},
    {{
        "English phrase": "and there was morning—",
        "Hebrew phrase": "וַֽיְהִי־ בֹ֖קֶר",
        "Target phrase": "na moning i kamap.",
        "Pseudo-English phrase": "and morning come-up."
    }},
    {{
        "English phrase": "the first day.",
        "Hebrew phrase": "י֥וֹם  אֶחָֽד׃",
        "Target phrase": "Em i de namba wan.",
        "Pseudo-English phrase": "Him day number one."
    }}
]
```

Please also align the following sentence. Avoid including multiple phrases in a single alignment unit. You may need to break phrases on commas or other major punctuation, including enclosing quotation marks. But you may also need to break a phrase along conjunctions or other words that typically mark the start of a new phrase. Always respond with perfect JSON:

English: {bsb}
Hebrew: {macula}
Target: {target}
Pseudo-English: {alt}
'''
    except Exception as e:
        return 'ERROR'
    
book_idx = {'GEN': 1, 'EXO': 2, 'LEV': 3, 'NUM': 4, 'DEU': 5, 'JOS': 6, 'JDG': 7, 'RUT': 8, '1SA': 9, '2SA': 10,
 '1KI': 11, '2KI': 12, '1CH': 13, '2CH': 14, 'EZR': 15, 'NEH': 16, 'EST': 17, 'JOB': 18, 'PSA': 19, 'PRO': 20,
 'ECC': 21, 'SNG': 22, 'ISA': 23, 'JER': 24, 'LAM': 25, 'EZK': 26, 'DAN': 27, 'HOS': 28, 'JOL': 29, 'AMO': 30,
 'OBA': 31, 'JON': 32, 'MIC': 33, 'NAM': 34, 'HAB': 35, 'ZEP': 36, 'HAG': 37, 'ZEC': 38, 'MAL': 39, 'MAT': 40,
 'MRK': 41, 'LUK': 42, 'JHN': 43, 'ACT': 44, 'ROM': 45, '1CO': 46, '2CO': 47, 'GAL': 48, 'EPH': 49, 'PHP': 50,
 'COL': 51, '1TH': 52, '2TH': 53, '1TI': 54, '2TI': 55, 'TIT': 56, 'PHM': 57, 'HEB': 58, 'JAS': 59, '1PE': 60,
 '2PE': 61, '1JN': 62, '2JN': 63, '3JN': 64, 'JUD': 65, 'REV': 66}

def generate_broad_alignment_prompt(data_element):
    reference = data_element['vref']
    if book_idx[reference[:3]] < 40:
        return generate_broad_hebrew_alignment_prompt(data_element)
    else:
        return generate_broad_greek_alignment_prompt(data_element)

def align(prompt):
    system_prompt = "You are LangAlignerGPT. Analyze the user-supplied samples below and follow any instructions the user gives. Always respond with perfect JSON.\n"
    
    if not args.model == 'gpt-4':
        
        formatted_prompt = (f'{system_prompt}'
                            f'{prompt}')
        
        for i in range(MAX_RETRIES):
            try:
                response = openai.Completion.create(
                model=args.model,
                prompt=formatted_prompt,
                temperature=0.1,
                max_tokens=1200,
                )
                return response['choices'][0]['text']
            except (openai.error.APIConnectionError, openai.error.APIError) as e:
                print('Error in alignment:', e)
                if i < MAX_RETRIES - 1:  # i is zero indexed
                    continue
                else:
                    return {"error": str(e)}
            
    else:
        messages = [
            {"role": 'system', "content": system_prompt},
            {"role": 'user', 'content': prompt}
        ]
        for i in range(MAX_RETRIES):
            try:
                response = openai.ChatCompletion.create(
                    model=args.model,
                    messages=messages,
                    temperature=0.1,
                )
                generated_texts = [
                    choice.message["content"] for choice in response["choices"]
                ]
                return generated_texts[0]
            except (openai.error.APIConnectionError, openai.error.APIError) as e:
                print('Error in alignment:', e)
                if i < MAX_RETRIES - 1:  # i is zero indexed
                    continue
                else:
                    return {"error": str(e)}

### CREATE OUTPUT DIR ###
import datetime
import os

output_dir = f'data/alignments/{args.run_name}'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
### LOAD DATA ###

if args.n == 0:
    with open(args.data_path, 'r') as f:
        json_data = json.load(f)
else:
    with open(args.data_path, 'r') as f:
        if args.randomize:
            json_data: List[Verse] = random.sample(json.load(f), args.n)
        else:
            json_data: List[Verse] = json.load(f)[:args.n]

# Post-filter json_data by verses in specified_ids array
if specified_ids:
    json_data = [verse for verse in json_data if verse['vref'] in specified_ids]

### ALIGN ###

output_file_path = f'{output_dir}/alignments_{os.path.basename(args.data_path)}_{args.run_name}_{args.model}_{datetime.datetime.now().strftime("%Y%m%d-%H%M")}.jsonl'

for verse in json_data:
    pseudo_english_prompt = generate_pseudo_english_prompt(verse)
    pseudo_english_text = align(pseudo_english_prompt)
    verse['alt'] = pseudo_english_text
    print('pseudo-english:', pseudo_english_text)
    prompt = generate_broad_alignment_prompt(verse)
    
    if not args.pseudo_english_only:
        try:
            print('aligning', verse['vref'])
            response = align(prompt)
            output = json.loads(response) # FIXME: use pyjson5 ?
            verse['alignment'] = output
        except json.JSONDecodeError as e:
            verse['alignment'] = f'Error: Maximum retries exceeded. {e}'
            verse['error'] = 'true'
        except Exception as e:
            print('Error on alignment.', e)
            verse['alignment'] = f'Error: {e}'
            verse['error'] = 'true'
    with open(output_file_path, 'a') as f:
        f.write(json.dumps(verse, ensure_ascii=False))
        f.write('\n')
