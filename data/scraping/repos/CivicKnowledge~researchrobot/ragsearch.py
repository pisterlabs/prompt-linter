import json
import pandas as pd
from functools import cache
from tabulate import tabulate
from researchrobot.cmech.classify import resolve_cache
import logging
import dataclasses

from researchrobot.openai_tools.completions import openai_one_completion
from openai.types.chat import ChatCompletion

logger = logging.getLogger(__name__)

from researchrobot.openai_tools import costs


def compose_examples(self, sec, example_file='./example.csv'):
    """Compose an example for the system prompt"""
    ex = pd.read_csv(example_file)

    resp = []

    out = []

    for idx, r in ex.iterrows():
        t = sec.job_search(r.title, r.summary).head(5)

        out.append(self.format_exp(r, t))

        d = {
            'job': "jc_" + str(r.exp_id),
            'reasoning': r.reasoning,
            'soc': json.loads(r.soc),
            'quality': r.quality
        }

        resp.append(json.dumps(d))

    return out, resp


def decode_response(r):
    records = []

    if r.startswith('[{'):
        # We got a JSON list, instead of json lines
        return json.loads(r)

    for e in r.splitlines():
        try:
            j = json.loads(e)
            records.append(j)
        except json.JSONDecodeError:
            continue

    if len(records) == 1 and isinstance(records[0], list):
        # We got a single-line JSON list, instead of json lines
        return records[0]
    else:
        return records


def unpack_response(self, r):
    d = {}

    recs = decode_response(r)

    for j in recs:
        j['soc'] = [(k, self.data.title_map.get(k)) for k in j['soc']]
        exp_id = int(j['job'].replace('jc_', ''))
        del j['job']
        d[exp_id] = j

    return d


@dataclasses.dataclass
class Request:
    pid: str
    education: list
    experiences: list
    searches: list
    prompt: str

    @property
    def request_id(self):
        if len(self.experiences) == 1:
            return f'{self.pid};{self.experiences[0].exp_id}'
        else:
            return f'{self.pid};{self.experiences[0].exp_id}-{self.experiences[-1].exp_id}'

    def format_table(self, request, exp, search,
                     head=7, rag_style='plain', **table_args):

        t = search.head(head)

        if rag_style == 'table':
            table = tabulate(t, headers=t.columns,
                             maxcolwidths=[10, 10, 10, 20, 40, 10, 10, 10, 10, 10],
                             tablefmt="grid")
        elif rag_style == 'plain':
            t = t[['detailed', 'title', 'text']].rename(columns={'detailed': 'soc'})
            table = tabulate(t, headers=t.columns, maxcolwidths=[10, 10, 40, 100], tablefmt="plain")
        elif rag_style == 'list':
            table = 'Candidate SOC codes:\n\n' + \
                    ''.join([f"  * {r.detailed} {r.title}: {r.text}\n" for idx, r in t.iterrows()])
        else:
            table = t.to_json(orient='records')

        return table

    def format_candidate(self, request, exp, search, rag_style='plain', **table_args):

        t = self.format_table(request, exp, search, rag_style=rag_style, **table_args)

        return f"experience id: exp_{exp.exp_id}\nTitle: {exp.title}\nDescription:\n{exp.summary}\n\n" + \
            "Candidate job matches:\n\n" + \
            t + '\n'

    def candidates(self, formatter=None, head=7, **formatter_args):

        if formatter is None:
            formatter = self.format_candidate

        return [formatter(self, exp, search, head=head, **formatter_args)
                for exp, search in zip(self.experiences, self.searches)]


@dataclasses.dataclass
class CompletedExperience:
    """Object holding the request for a RAG completion, and the response,
    with accessors for the data in the response"""

    request: Request
    response: ChatCompletion

    @property
    def content(self):
        return self.response.choices[0].message.content.strip()

    @property
    def response_json(self):
        return decode_response(self.content)

    @property
    def response_df(self):

        ragscores = {
            'good': [1.15, 1.10, 1.07],
            'fair': [1.10, 1.07, 1.03],
            'poor': [1.05, 1.02, 1]
        }

        rows = []
        for e in self.response_json:
            for i, s in enumerate(e['soc']):
                rows.append(
                    {
                        'detailed': s,
                        'exp_id': int(e['job'].replace('jc_', '')),
                        'rag_rank': i,
                        'rag_score': ragscores[e['quality']][i] * .5,
                        'rag_quality': e['quality'],
                        'reasoning': e['reasoning']
                    }
                )
        return pd.DataFrame(rows)

    @property
    def experiences_df(self):
        return pd.DataFrame(self.request.experiences)

    @property
    def search_df(self):
        return pd.concat(s.assign(exp_id=i) for i, s in enumerate(self.request.searches))

    @property
    def cost(self):
        usage = self.response.usage
        ctok = usage.completion_tokens
        ptok = usage.prompt_tokens
        model = self.response.model

        return costs[model][0] * ctok / 1000 + costs[model][1] * ptok / 1000

    @property
    def df(self):
        t = self.search_df.merge(self.response_df, on=['detailed', 'exp_id'], how="left") \
            .rename(columns={'score': 'search_score'})
        t['rag_score'] = t.rag_score.fillna(1)
        t['score'] = t.search_score * t.rag_score
        idx = t.groupby('exp_id').search_score.nlargest(5).reset_index(level=0, drop=True).index
        t = t.loc[idx]
        t['pid'] = self.request.pid

        t['rank'] = t.groupby('exp_id').score.rank(ascending=False, method='first')

        return t.sort_values(['exp_id', 'score'], ascending=[True, False])

    @property
    def simple_df(self):
        return self.df[['pid', 'exp_id', 'detailed', 'score', 'rank', 'title']].copy()

    def review(self):
        from tabulate import tabulate

        g = self.simple_df.groupby('exp_id')
        out = []
        for idx, r in self.experiences_df.iterrows():
            s = g.get_group(r.exp_id)
            cols = ['detailed', 'score', 'title']
            out.append(f"{r.title}\n{r.summary}\n\n"
                       + (tabulate(s[cols], headers=['code', 'score', 'title'], tablefmt='plain')))

        return out


def make_example():
    """Generate an example for the system prompt"""

    ex_pid = 'PzaMlWLnVEV-wsTR2Tsllg_0000'

    ex = exp_df[exp_df.pid == ex_pid]

    rows = []
    for idx, r in ex.iterrows():
        t = sec.job_search(r.title, r.summary)

        soc = t.head(3).sample(3).detailed.to_list()
        d = {
            "reasoning": "Financial advisor is very close to financial representative",
            "soc": json.dumps(soc),
            "quality": "fair"
        }
        rows.append(d)

    out = pd.DataFrame(rows, index=ex.index)
    t = pd.concat([ex, out], axis=1).drop(columns=['id', 'name', 'web', 'company_size', 'text', 'role', 'embeddings'])
    t.to_csv('example.csv')


class RAGJobSearch:
    """Uses the search classifier to produce a set of candidate classifications
    for a job experience, then calls open AI to select the best one"""

    # noinspection PyShadowingNames
    def __init__(self, cache, sec, data, rag_style='plain',
                 example_file='example.csv'):
        self.cache = resolve_cache(cache)
        self.data = data
        self.sec = sec
        self.rag_style = rag_style

        self.example_file = example_file

    @staticmethod
    def request_key(pid):
        return f"request/{pid}"

    @staticmethod
    def response_key(pid):
        return f"response/{pid}"

    @property
    def complete_pids(self):
        """PIDs that have been requested and have a response"""
        return list(self.cache.sub('response').list())

    @property
    def completed_responses(self):

        from collections import namedtuple

        request = None

        for pid in self.complete_pids:
            req = self.cache[self.request_key(pid)]
            resp = self.cache[self.response_key(pid)]

            yield CompletedExperience(Request(**req), resp)

    @property
    def incomplete_pids(self):
        """PIDS that have a request and no response"""
        return list(set(self.cache.sub('request').list()) - set(self.complete_pids))

    def search_experience(self, exp):

        r = self.sec.job_search(exp.title, exp.summary)

        return r

    def search_profile(self, profile):

        r = self.sec.job_search(profile.title, profile.summary)

        return r

    @staticmethod
    def response_blocks(r):

        blocks = {}

        for exp_id, e in r.items():
            o = f"Job Title Matches\n-----------------\n\nMatch quality: {e['quality']}; {e['reasoning']}\n"
            for soc, title in e['soc']:
                o += f"  {soc} {title}\n"

            blocks[exp_id] = o

        return blocks

    def format_exp(self, exp, search_results, head=7):
        """Format a single job experience and its search results"""
        return f"Job code: jc_{exp.exp_id}\nTitle: {exp.title}\nDescription:\n{exp.summary}\n\n" + \
            "Candidate job matches:\n\n" + \
            self.format_search_results(search_results, head=head) + '\n'


    def system_prompt(self):

        ex, resp = self.compose_examples()
        nl = "\n"

        sp_file = Path(__file__).resolve().parent / "system_prompt.txt"
        sp = sp.file.read_text()



        # noinspection PyMethodMayBeStatic
    def prompt(self, request: Request):

        nl = '\n'

        return (
                "Please determine the SOC codes for the following jobs  and return you classification" +
                " in JSON lines format, one JSON document per line. \n\n" +
                ("\n\n".join(request.candidates)) +
                "\n\nJSON Lines Output\n" +
                "================="
        )

    def get_pid(self, experiences):

        if isinstance(experiences, pd.DataFrame):

            pids = experiences.pid.unique()

            assert len(pids) == 1, "Only one person's experiences can be processed at a time"

            pid = pids[0]

            return pid
        else:
            return experiences.pid

    def prepare_request(self, education, experiences, prompt_f=None):

        pid = self.get_pid(experiences)

        if isinstance(experiences, pd.DataFrame):
            experiences = [exp for idx, exp in experiences.iterrows()]
        else:
            experiences = [experiences]

        searches = [self.sec.job_search(exp.title, exp.summary) for exp in experiences]

        request = Request(pid, education, experiences, searches, prompt=None)

        # if prompt_f is  None:
        #    prompt_f = self.prompt

        # request.prompt = prompt_f(request)

        return request

    def run_prepared_completion(self, request_key):

        _, pid = request_key.split('/')

        key = f"response/{pid}"

        if key in self.cache:
            return key

        d = self.cache[request_key]

        self.cache[key] = openai_one_completion(d['prompt'], system=self.system_prompt(),
                                                model='gpt-3.5-turbo-16k', json_mode=False,
                                                return_response=True, max_tokens=600)

        return key

    # noinspection PyUnusedLocal
    def run_completion(self, education, experiences, force=False):

        request_key, d = self.prepare_completion(education, experiences, fast_return=False, force=force)

        return self.run_prepared_completion(request_key)

    def run_experiences(self, experiences):
        """Run multiple experiences"""
        from tqdm.auto import tqdm

        for gn, g in tqdm(experiences.groupby('pid')):
            self.run_completion(None, g)
