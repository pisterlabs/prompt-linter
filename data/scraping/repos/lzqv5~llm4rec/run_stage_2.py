from tqdm.notebook import tqdm
from time import time, sleep
from ast import literal_eval
import os, json, argparse
import re
from langchain.output_parsers import StructuredOutputParser, ResponseSchema 
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate 
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate, 
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

os.environ["OPENAI_API_KEY"] = "API_KEY" 
summarizer = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.3, n=1, max_tokens=512)

# output parser
class JsonExtractor():

    def __init__(self) -> None:
        self.pattern_json_extractor = re.compile(r'{[^\{\}]+}')
        self.pattern_remove_space = re.compile(r'\n\s{0,}')
        self.pattern_refine = re.compile(r'(?<![a-zA-Z])\'(?=[a-zA-Z])|(?<=[a-zA-Z])\'(?![a-zA-Z])')
        self.keys = ["overall", "education", "education comment", "experience", "experience comment", "skill", "skill comment"]
        self.descriptions = ["Total score", "Educational score", "comment on the educational score", "work experience score", "comment on the work experience score", "skill score", "comment on the skill score"]
        self.response_schemas = [ResponseSchema(name=name, description=desc) for name, desc in zip(self.keys, self.descriptions)]
        self.output_parser = StructuredOutputParser(response_schemas=self.response_schemas)
    
    def _extract(self, text: str) -> str:
        return self.pattern_json_extractor.search(text).group()
    
    def _remove_space(self, text: str) -> str:
        return self.pattern_remove_space.sub('', text)
    
    def _refine_property_name_quotes(self, text: str) -> str:
        return self.pattern_refine.sub('"', text)
    
    def _convert_to_md_json(self, text: str) -> str:
        return '```json'+text+'```'
    
    def __call__(self, text: str) -> dict:
        text = self._remove_space(self._extract(text))
        try:
            return literal_eval(text)
        except json.decoder.JSONDecodeError:
            print('wrong text format:\n', text)
            return literal_eval(self._refine_property_name_quotes(text))

def build_chat_prompt(prompt_idx=0):
    human_message_prompt = """You are a professional HR who can determine whether a job description and a curriculum vitae match and provide comments.

    You are provided with inputs in the following format:
    ```
    job description:
    a paragraph of text

    curriculum vitae:
    a paragraph of text
    ```

    Your task is to output the following:
    1. An overall matching score between the curriculum vitae and the job description (0-100).
    2. A matching score between the educational background in the curriculum vitae and the job description (0-100).
    3. A matching score between the work experience in the curriculum vitae and the job description (0-100).
    4. A matching score between the professional skills in the curriculum vitae and the job description (0-100).

    The results should be represented in a JSON format with the following key-value pairs:
    ```
    "overall": a number between 0 and 100,
    "education": a number between 0 and 100,
    "education comment": a detailed paragraph of comment on the education score
    "experience": a number between 0 and 100,
    "experience comment": a detailed paragraph of comment on the experience score
    "skill": a number between 0 and 100,
    "skill comment": a detailed paragraph of comment on the skill score
    ```

    """
    if prompt_idx == 0:
        human_message_prompt += """You should pay attention to the content of ``work experience'', ``job description'', and ``job requirements'' in the job description during the matching process. You also need to pay attention to the content of ``work experience'', ``educational background'', and ``work experience'' in the curriculum vitae.
        The basis for matching should be dynamically generated based on the input job description and curriculum vitae content.
        You should provide any other information or feedback about the matching process and automatically handle any errors or missing information that may exist in the curriculum vitae. If there are errors, you should skip the missing information and continue to complete the matching.
        Your first response should be 'Understood.'.
        """
    elif prompt_idx == 1:
        human_message_prompt += """customized emphasis 1"""
    else:
        human_message_prompt += """customized emphasis 2"""
    # more customized emphases if needed.
    assistant_message_prompt = "Understood."
    input_prompt = """job description:
    {jd}

    curriculum vitae:
    {cv}

    Please output the results in JSON format.
    """
    human_message_prompt_template = HumanMessagePromptTemplate.from_template(human_message_prompt)
    assistant_message_prompt_template = AIMessagePromptTemplate.from_template(assistant_message_prompt)
    input_prompt_template = HumanMessagePromptTemplate.from_template(input_prompt)
    chat_prompt= ChatPromptTemplate.from_messages([human_message_prompt_template, assistant_message_prompt_template, input_prompt_template])
    return chat_prompt

def build_comment_prompt():
    summarize_prompt = """You are a professional HR. Now, several experts have commented on the same curriculum vitae. You need to generate a short comment based on these comments to describe the strengths and weaknesses of this curriculum vitae.

    The results should be represented in a JSON format with the following key-value pairs:
    ```
    "comments": a comment paragraph
    ```

    The following are the comments of experts:
    ```
    {comments}
    ```

    Please output the results in JSON format.
    """
    prompt= ChatPromptTemplate.from_messages([HumanMessagePromptTemplate.from_template(summarize_prompt)])
    return prompt

cmts_prompt = build_comment_prompt()

def rebuild_output(mean_score, outputs, output_parser):
    flag = 1e3
    idx = 0
    for i, output in enumerate(outputs):
        diff = abs(output.get('overall',0)-mean_score)
        if diff<flag:
            flag = diff
            idx = i
    picked_output = outputs[idx]
    picked_output['overall'] = mean_score

    edu_cmts = ''
    exp_cmts = ''
    ski_cmts = ''
    for i in range(len(outputs)):
        edu_cmts += 'education comment' + outputs[i]['education comment'] + '\n'
        exp_cmts += 'experience comment' + outputs[i]['experience comment'] + '\n'
        ski_cmts += 'skill comment' + outputs[i]['skill comment'] + '\n'
    edu_cmts = summarizer(cmts_prompt.format_prompt(comments=edu_cmts).to_messages())
    picked_output['education comment'] = output_parser(edu_cmts.content)
    exp_cmts = summarizer(cmts_prompt.format_prompt(comments=exp_cmts).to_messages())
    picked_output['experience comment'] = output_parser(exp_cmts.content)
    ski_cmts = summarizer(cmts_prompt.format_prompt(comments=ski_cmts).to_messages())
    picked_output['skill comment'] = output_parser(ski_cmts.content)
    return picked_output

def single_session_processing(generation: list, output_parser: JsonExtractor):
    n = len(generation)
    outputs = [output_parser(generation[i].text) for i in range(n)]
    mean_score = sum([d.get('overall', 0) for d in outputs])/n
    return rebuild_output(mean_score, outputs, output_parser)

def sessions_processing(generations: list, output_parser: JsonExtractor):
    return [single_session_processing(gen, output_parser) for gen in generations]

def resorting(jd, cv, candidates, chat_model, chat_prompt, output_parser,  rate_limit_per_min=3500, tokens_per_min=90000, token_per_req=4000, delta=2.5):
    if not isinstance(chat_prompt, list):
        n = chat_model.n
        rpm = min(rate_limit_per_min, tokens_per_min//token_per_req)
        step = rpm//n
        messages_list = [[chat_prompt.format_prompt(jd=jd, cv=cand).to_messages() for cand in candidates[i:i+step]] for i in range(0,len(candidates),step)]
        generations = []
        for messages in messages_list:  # 1 batch/min
            start_time = time()
            responses_of_generation = chat_model.generate(messages)
            generations.extend(responses_of_generation.generations)
            end_time = time()
            diff = end_time-start_time
            if diff < 60:
                print("sleeping...")
                sleep(60-diff)

        outputs = sessions_processing(generations, output_parser)
        cands_new = [item for item in zip(candidates,outputs)]
    else:   
        prompt_nums = len(chat_prompt)
        candidates_nums = len(candidates)
        n = chat_model.n
        rpm = min(rate_limit_per_min, tokens_per_min//token_per_req)
        step = rpm//n
        outputs = []
        for prompt in chat_prompt:
            messages_list = [[prompt.format_prompt(jd=jd, cv=cand).to_messages() for cand in candidates[i:i+step]] for i in range(0,candidates_nums,step)]
            generations = []
            for messages in messages_list:  # 1 batch/min
                start_time = time()
                responses_of_generation = chat_model.generate(messages)
                generations.extend(responses_of_generation.generations)
                end_time = time()
                diff = end_time-start_time
                if diff < 60:
                    print("sleeping...")
                    sleep(60-diff)
            outputs.append(sessions_processing(generations, output_parser))

        mean_scores = [sum([outputs[i][j].get('overall',0) for i in range(prompt_nums)])/prompt_nums for j in range(candidates_nums)]
        outputs_new = [rebuild_output(mean_score=mean_scores[j], outputs=[outputs[i][j] for i in range(prompt_nums)], output_parser=output_parser) for j in range(candidates_nums)]
        cands_new = [item for item in zip(candidates, outputs_new)]

    cands_new.sort(key=lambda item:item[1].get('overall',0),reverse=True)
    return [jd, cv, cands_new]
            
def resort_self_consistency(chat_model, recs, output_parser: JsonExtractor):
    chat_prompt = build_chat_prompt(prompt_idx=0)
    resorted_recs = []
    for idx, (jd, cv, cands, _) in enumerate(recs):
        print(f"idx: {args.start_idx+idx} - processing...")
        resorted_recs.append(resorting(jd, cv, cands, chat_model, chat_prompt, output_parser))
        with open(f'data/recs_self_consistency_(cv{args.cv_len}_jd500)_temperature_{args.temperature}_n_{args.n}.json', 'w') as f:
            json.dump(resorted_recs, f, ensure_ascii=False)
    return resorted_recs

def resort_ensemble(chat_model, recs, output_parser: JsonExtractor, prompt_ids=[0,1,2], rate_limit_per_min=3, delta=2.5):
    chat_prompts = [build_chat_prompt(prompt_idx=idx) for idx in prompt_ids]
    resorted_recs = []
    for idx, (jd, cv, cands, _) in enumerate(recs):
        # start_time = time()
        print(f"idx: {args.start_idx+idx} - processing...")
        resorted_recs.append(resorting(jd, cv, cands, chat_model, chat_prompts, output_parser))
        with open(f'data/recs_ensemble_(cv{args.cv_len}_jd500)_temperature_{args.temperature}_n_{args.n}.json', 'w') as f:
            json.dump(resorted_recs, f, ensure_ascii=False)
    return resorted_recs

parser = argparse.ArgumentParser(description='Pipeline Settings')
parser.add_argument('--cv_len', type=int, required=True, help='set the token_num of cv')
parser.add_argument('--start_idx', type=int, required=True, help='')
parser.add_argument('--temperature', type=float, required=True, help='')
parser.add_argument('--n', type=int, required=True, help='')
args = parser.parse_args()

if __name__ == '__main__':
    # load llm
    chat_diverse = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=args.temperature, n=args.n, max_tokens=512)
    chat_ensemble_only = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=args.temperature, n=args.n, max_tokens=512)

    with open(f'data/output_from_stage_1.json', 'r') as f:
        recs = json.load(f)

    output_extractor = JsonExtractor()
    recs = recs[args.start_idx:]

    print("Self-consistency ...")
    recs_self_consistency = resort_self_consistency(chat_model=chat_diverse, recs=recs, output_parser=output_extractor)
    print("Done.")
    print("Self-consistency results saving...")
    with open(f'data/recs_self_consistency_(cv{args.cv_len}_jd500)_temperature_{args.temperature}_n_{args.n}.json', 'w') as f:
        json.dump(recs_self_consistency, f, ensure_ascii=False)

    # print("Ensemble-only ...")
    # recs_ensemble =resort_ensemble(chat_model=chat_ensemble_only, recs=recs, output_parser=output_extractor, prompt_ids=[0,1,2])
    # print("Done.")
    # print("Ensemble-only results saving...")
    # with open(f'data/recs_ensemble_(cv{args.cv_len}_jd500)_temperature_{args.temperature}_n_{args.n}.json', 'w') as f:
    #     json.dump(recs_ensemble, f, ensure_ascii=False)

    # print("Combined ...")
    # recs_combined = resort_ensemble(chat_model=chat_diverse, recs=recs, output_parser=output_extractor, prompt_ids=[0,1,2])
    # print("Done.")
    # print("Combined results saving...")
    # with open(f'data/recs_combined_(cv{args.cv_len}_jd500)_temperature_{args.temperature}_n_{args.n}.json', 'w') as f:
    #     json.dump(recs_combined, f, ensure_ascii=False)

    print('All finished')
