import openai
openai.api_key = open('key.txt').read().strip()
import regex as re



OPS = ['→', '↔', '¬', '⊕', '∨', '∧', '∀', '∃']


def get_syntax(fol):
    prompt = open('prompt/get_syntax.txt', encoding='utf-8').read()
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt+fol,
        temperature=0,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["\\n"]
        )

    text = response['choices'][0]['text'].strip()
    text = text[text.find('Output: ')+8:]
    for op in OPS:
        if text.count(op) != fol.count(op):
            return None
    return text


def find_p(fol):
    all_p = re.findall(r'[^∀∃][a-z0-9]+\(', fol)
    all_p = [re.sub(r'[^a-zA-Z0-9]', '', i) for i in all_p]
    all_p = list(dict.fromkeys(all_p))
    return all_p

def find_c(fol):
    all_c = []
    record = ''
    recording = False
    for c in fol:
        if c == '(':
            recording = True
            record = ''
        elif c == ')':
            recording = False
            if record:
                record = record.split(',')
                for c in record:
                    if c not in all_c and c not in {'x', 'y', 'z'}:
                        all_c.append(c)
            record = ''
        elif recording:
            record += c
    return all_c
        
        


def get_syntax_rule(fol):
    '''
    get pure syntax of a fol premise
    '''
    ps = find_p(fol)
    cs = find_c(fol)
    to_replace = {n:f'P{i}' for i, n in enumerate(ps)}
    to_replace.update({n:f'C{i}' for i, n in enumerate(cs)})
    for k, v in sorted(to_replace.items(), key=lambda x: len(x[0]), reverse=True):
        fol = fol.replace(k, v)
    
    if not all([i in OPS+['P', 'C', '(', ')', ',','1','2','0','3','4','5','6','7','8','9','x','y','z'] for i in fol]):
        print(fol)
        return None
    
    return fol




        
