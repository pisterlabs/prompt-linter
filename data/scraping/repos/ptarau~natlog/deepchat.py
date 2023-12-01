import os
import webbrowser
import openai
import tiktoken
from natlog import Natlog, natprogs, lconsult
import json

MAX_TOKENS = 300 # 1 << 12  # make shorter if needed e.g. 300

openai.api_key = os.getenv("OPENAI_API_KEY")


class ChatMind:
    def __init__(self, max_toks=MAX_TOKENS, avatar=None):
        self.model = "gpt-3.5-turbo"

        self.sys_prompt = dict(role='system',
                               content='You are a helpful assistant. Your answers are as concise as possible.')
        self.max_toks = max_toks
        self.avatar = avatar

        if not self.resume():
            self.short_mem = dict()
            self.long_mem = dict()
            self.toks = []

    def to_message(self, quest):
        mes = [self.sys_prompt]
        for (q, a) in self.short_mem.items():
            qd = dict(role='user', content=q)
            ad = dict(role='assistant', content=a)
            mes.extend([qd, ad])
        mes.append(dict(role='user', content=quest))
        return mes

    def get_state(self):
        return dict(long_mem=self.long_mem,
                    short_mem=self.short_mem,
                    toks=self.toks)

    def set_state(self, state):
        self.long_mem = state['long_mem']
        self.short_mem = state['short_mem']
        self.toks = state['toks']

    def store_name(self):
        if self.avatar is None: return None
        return 'states/' + self.avatar + '.json'

    def resume(self):
        fname = self.store_name()
        if fname is not None and exists_file(fname):
            state = from_json(fname)
            #print('!!!!',json.dumps(state,indent=4))
            self.set_state(state)
            return True
        return False

    def persist(self):
        fname = self.store_name()
        if fname is None: return
        state = self.get_state()
        to_json(state, fname)

    def already_answered(self,quest):
        answer = self.short_mem.get(quest,None)
        if answer is not None: return answer
        answer = self.long_mem.get(quest,None)
        return answer

    def __repr__(self):
        return json.dumps(self.get_state(), indent=2)

    def ask(self, quest):
        if quest == 'quit':
            self.persist()
            return 'bye'

        answered = self.already_answered(quest)
        if answered is not None:
            return answered

        self.trim_context(quest)

        # print(len(self.answers), len(self.short_mem), len(self.toks))

        assert len(self.short_mem) == len(self.toks)

        mes = self.to_message(quest)

        r = openai.ChatCompletion.create(
            model=self.model,
            messages=mes
        )
        result = r['choices'][0]['message']['content']
        answer = result.split('\n')[-1]

        t = r['usage']['total_tokens']
        self.toks.append(t)

        self.short_mem[quest] = answer

        return answer

    def trim_context(self, quest):
        if len(self.toks) == 0: return

        total_toks = sum(self.toks)
        avg_toks = total_toks / len(self.toks)

        quest_toks = count_toks(quest)
        tok_estimate = total_toks + quest_toks + 2 * avg_toks  # conservative ...

        if tok_estimate > self.max_toks:
            k, v = dict_trim(self.short_mem)
            self.long_mem[k] = v
            self.toks = self.toks[1:]


# tools

def count_toks(text):
    enc = tiktoken.get_encoding("gpt2")
    toks = enc.encode(text)
    return len(toks)


def exists_file(fname):
    return os.path.exists(fname)


def ensure_path(fname):
    """
    makes sure path to directory and directory exist
    """
    d, _ = os.path.split(fname)
    os.makedirs(d, exist_ok=True)


def to_json(obj, fname, indent=2):
    """
    serializes an object to a json file
    assumes object made of array and dicts
    """
    ensure_path(fname)
    with open(fname, "w") as outf:
        json.dump(obj, outf, indent=indent)


def from_json(fname):
    """
    deserializes an object from a json file
    """
    with open(fname, "rt") as inf:
        obj = json.load(inf)
        return obj


def dict_trim(d):
    for k, v in d.items():
        break
    d.pop(k)
    return k, v


def run_natlog(natprog="deepchat.nat"):
    n = Natlog(file_name=natprog,
               with_lib=natprogs() + "lib.nat", callables=globals())
    next(n.solve('initialize.'))
    # n.repl()
    for x in n.solve('chat.'):
        print()


def test_deepchat():
    cm = ChatMind(avatar='you')

    answer = cm.ask('What was the warmest temperature measured on Mars?')
    answer = cm.ask('The same question for planet Venus?')
    answer = cm.ask('And what about dwarf planet Ceres?')
    answer = cm.ask('Is it much colder on Pluto?')
    print(f'ANSWER: {answer}\n')
    answer = cm.ask('quit')
    print(answer,'\n')

    print('STATE:\n',cm)


if __name__ == "__main__":
    #test_deepchat()
    run_natlog()
