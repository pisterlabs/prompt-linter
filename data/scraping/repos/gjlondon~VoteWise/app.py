import hashlib
import json
import os
import time
from pathlib import Path
from urllib.parse import quote, unquote

from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from canonicaljson import encode_canonical_json
from environs import Env
from flask import Flask, render_template, request, jsonify, session, make_response
from flask import redirect, url_for
from flask_cors import CORS
from flask_wtf.csrf import CSRFProtect
from langchain import OpenAI, PromptTemplate
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory

from constants import OAKLAND_MAYOR_CANDIDATES
from forms import IntakeForm
from models import VoterInfo
from models import VoterInfoDecoder
from prompts import OAKLAND_MAYOR_ISSUES, MAYOR_SCORING_PROMPT_TEMPLATE, MAYOR_OVERALL_RECOMMENDATION_PROMPT_TEMPLATE
import sms

env = Env()
# Read .env into os.environ
env.read_env()

anthropic = Anthropic()
llm = ChatOpenAI(model_name='gpt-3.5-turbo-16k')
memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=2000)
new_memory_by_race = {}
app = Flask(__name__)
app.secret_key = env.str("SECRET_KEY")
CORS(app)
csrf = CSRFProtect(app)

with open(os.path.join(app.root_path, 'static', 'ballot.json'), 'r') as f:
    ballot_data = json.load(f)

with open(os.path.join(app.root_path, 'static', 'ballot-descriptions.json'), 'r') as f:
    ballot_descriptions = json.load(f)


def races():
    new_list = (list(ballot_data.keys()) + list(ballot_data.get('Propositions').keys()))
    return new_list


def escaped_races():
    return [quote(item) for item in races()]

def save_voter_info(voter_info_json):
    print('saving voter info')
    json_string = json.dumps(voter_info_json)
    md5_hash = hashlib.md5(json_string.encode()).hexdigest()
    
    voter_info_dir = Path(f"./data/voter_infos")
    if not voter_info_dir.exists():
        # create voter_info_dir
        voter_info_dir.mkdir(parents=True, exist_ok=True)
    
    voter_info_path = voter_info_dir / f"{md5_hash[0:10]}.json"
    if not voter_info_path.exists():
        with open(voter_info_path, 'w') as f:
            json.dump(voter_info_json, f)
    print('saved voter info')

def load_voter_info():
    vote_info = 'generic.json'
    voter_info_dir = Path(f"./data/voter_infos")
    voter_info_path = voter_info_dir / f"{vote_info}"
    with open(voter_info_path, 'r') as f:
        voter_info_json = json.load(f)
    return voter_info_json


@app.route('/chat2')
def chat2():
    return render_template('chat.html')

@app.route('/landing')
def landing():
    return render_template('landing.html')

@app.route('/skip-intake', methods=['GET'])
def skip_intake(be_conservative=True):
    liberal_answers = {'zip_code': '94666', 'party_affiliation': 'democrat',
                       'political_issues': ['healthcare', 'housing'],
                       'housing': '5', 'economy': '1', 'environment': '5', 'immigration': '2', 'income_inequality': '5',
                       'transportation': '4', 'education': '2', 'healthcare': '5', 'public_safety': '2',
                       'taxation': '5'}

    conservative_answers = {
        "address_zip_code": "94608",
        "city": "Oakland",
        "economy": "5",
        "education": "1",
        "environment": "1",
        "healthcare": "1",
        "housing": "1",
        "immigration": "5",
        "income_inequality": "1",
        "party_affiliation": "republican",
        "political_issues": [
            "economy",
            "public_safety"
        ],
        "public_safety": "1",
        "state": "CA",
        "street_address": "4109 West",
        "taxation": "1",
        "transportation": "1"
    }

    if be_conservative:
        data = conservative_answers
    else:
        data = liberal_answers

    voter_info = VoterInfo.from_vals(**data).for_llm()
    print(f"updating session with {voter_info}")
    session['voter_info'] = voter_info
    session.modified = True
    return redirect(url_for('race'))

    # response = make_response('', 204)
    # response.mimetype = 'application/json'
    # return response

@app.route('/')
def home():
    return render_template('landing.html')

@app.route('/intake', methods=['GET', 'POST'])
def index():
    form = IntakeForm()
    if form.validate_on_submit():
        voter_info = VoterInfo()
        form_data = form.data
        # remove csrf token and any other unnecessary WTF-form inserted fields
        # just preserve the actual user-inputted data
        form.populate_obj(voter_info)
        session['voter_info'] = voter_info.for_llm()

        session.modified = True
        print(form_data)
        print("in form")
        return redirect(url_for('race'))
        return jsonify(form_data)
    else:
        # Setting default values when the form is first loaded (not submitted)
        form.street_address.data = '9777 Golf Links Rd'
        form.city.data = 'Oakland'
        form.state.data = 'CA'
        form.address_zip_code.data = '94605'
        form.full_address.data = '9777 Golf Links Rd, Oakland, CA 94605'
        form.political_issues.data = ['education', 'economy']

    return render_template('intake_form.html', form=form, google_api_key = env.str('GOOGLE_MAPS_API_KEY'))


# read the race and candidate parameters from the request, save them as kv into the session variable, and redirect to the next race
# input: race = encoded race name, candidate = encoded candidate name
@app.route('/confirm', methods=['GET'])
def confirm():
    race = request.args.get('race')
    candidate = request.args.get('candidate')
    if 'choices' not in session:
        session['choices'] = {}

    session['choices'][unquote(race)] = candidate

    session.modified = True

    race_index = escaped_races().index(race)
    next_race = escaped_races()[race_index + 1] if race_index < len(races()) - 1 else None
    return redirect(url_for('race', race_name=next_race))
    # get the index of this race from races    


@app.route('/pdf', methods=['GET', 'POST'])
@csrf.exempt
def pdf():
    if request.method == 'POST':
        print(request.data)  # print raw data
        print(request.form)  # print form data
        phone_number = request.form.get('phone_number')
        if phone_number:
            phone_number = sms.normalize_number(phone_number)
            # Call your desired Python function here
            msg1 = "Hello There! I'm happy I could help you figure out who to vote for and which issues to support by understanding who you are and what matters most to you. Because you care about environmental protection and job creation, make sure to vote on November 5th. "
            msg1_video = "http://votewise.radiantmachines.com/static/msg1.mp4"
            sms.send(phone_number, text=msg1, media=msg1_video)
            sms.save_to_json(phone_number, session, ["REMIND"])
            msg2 = "Hi There, Don’t forget to Vote for VoteWise - the world’s first AI-powered personalized voter guide that is helping to ensure everyone votes and every vote counts. Help us use AI to deepen democracy and build a better world. Go here to vote for VoteWise. "
            msg2_video = "http://votewise.radiantmachines.com/static/msg2.mp4"
            #sms.send(phone_number, text=msg2, media=msg2_video )
            return jsonify(success=True)
        else:
            return jsonify(success=False, message="Phone number not provided"), 400

    choices = session.get('choices', {})
    sorted_races = sorted(races(), key=lambda x: x not in choices)
    return render_template('pdf.html', races=sorted_races, choices=choices)

@app.route('/done', methods=['GET'])
def done():
    choices = session.get('choices', {})
    sorted_races = sorted(races(), key=lambda x: x not in choices)
    return render_template('done.html', races=sorted_races, choices=choices)


def score_candidates_for_mayor(candidate_summaries, voter_info_json, voter_info_hash):
    voter_info_dir = Path(f"./data/{voter_info_hash}/oakland_mayor")
    if not voter_info_dir.exists():
        # create voter_info_dir
        voter_info_dir.mkdir(parents=True, exist_ok=True)

    overall_candidate_scores = {}
    for candidate in OAKLAND_MAYOR_CANDIDATES:
        # load candidate scores saved inside voter_info_dir
        candidate_score_path = voter_info_dir / f"candidate_score_{candidate[:5]}.json"
        if candidate_score_path.exists():
            with open(candidate_score_path, 'r') as f:
                candidate_scoring = json.load(f)
        else:
            print(f"start scoring {candidate}")

            oakland_mayor_scoring_prompt = MAYOR_SCORING_PROMPT_TEMPLATE.format(**{
                "oakland_mayor_issues": OAKLAND_MAYOR_ISSUES,
                "voter_info_json": voter_info_json,
                "candidate_summary": candidate_summaries[candidate],
            })
            mayor_scoring_completion = anthropic.completions.create(
                model="claude-2",
                max_tokens_to_sample=15_000,
                prompt=f"{HUMAN_PROMPT} {oakland_mayor_scoring_prompt}{AI_PROMPT}"
            )

            # Print the completion
            candidate_scoring = mayor_scoring_completion.completion
            print(candidate_scoring)

            # save candidate scores in voter_info_dir
            with open(candidate_score_path, 'w') as f:
                json.dump(candidate_scoring, f)

        overall_candidate_scores[candidate] = candidate_scoring

    return overall_candidate_scores


LANGUAGE = "english"

def extract_key_from_json(json_data, key, human_readable=False):
    prompt = f"""
    You have been passed some poorly formatted json. You want to extract the data associated with a particular key.
    
    Return JUST the data, no preamble, no introduction. Just go straight into the data.
    
    Here is the data:
    ```
    {json_data}
    ```
    
    Here is the key:
    ```
    {key}
    
    Your reply should be exclusively in the language of {LANGUAGE}.
    ```
    """

    if human_readable:
        prompt += """\n
        Also reformat the data to be in HTML that can inserted directly into a web page, and will then be human readable in natural language.
         Try to use the HTML to retain any formatting or structure that ads
        clarity, but remember this HTML will be inserted directly into a webpage DOM by javascript so it should not cause any rendering problems when inserted.
        Remove any extraneous characters or quotation marks.
        
        For any lists, make sure to use native HTML lists (ul, ol, li) to represent them. Make the HTML as human readable and visually appealing as possible. 
        Give any headers in lists appropriate but subtle visual accenting.
        """

    # retrieve response from cache if it already exists (using same caching to disk pattern)
    # otherwise make live call

    encoded = encode_canonical_json({"request": prompt})
    hashed_prompt = hashlib.sha256(encoded).hexdigest()

    cache_path = Path(os.path.join(app.root_path, 'data', f'extract_{hashed_prompt}'))
    if cache_path.exists():
        with open(cache_path, 'r') as f:
            return json.load(f)['response']

    completion = anthropic.completions.create(
        model="claude-2",
        max_tokens_to_sample=15_000,
        prompt=f"{HUMAN_PROMPT} {prompt}{AI_PROMPT}"
    )

    # Print the completion
    print(completion.completion)
    with open(cache_path, 'w') as f:
        json.dump({"response": completion.completion}, f)

    return completion.completion


def parse_wrapped_json(input_str: str):
    recommendation = input_str
    # json.loads(recommendation.split('```')[1].strip("json"))
    lines = input_str.split('```')
    core = lines[1]

    json_data = json.loads(core)

    return json_data


def formulate_mayor_recommendation(candidate_scores, voter_info_json, voter_info_hash):
    voter_info_dir = Path(f"./data/{voter_info_hash}/oakland_mayor")
    if not voter_info_dir.exists():
        # create voter_info_dir
        voter_info_dir.mkdir(parents=True, exist_ok=True)

    overall_candidate_scores = candidate_scores
    language = json.loads(voter_info_json).get('language', LANGUAGE)
    language = LANGUAGE
    recommendation_path = voter_info_dir / f"{language}_recommendation.json"
    if recommendation_path.exists():
        with open(recommendation_path, 'r') as f:
            recommendation = json.load(f)
    else:
        print(f"start recoomending for {voter_info_json}")

        recommendation_template = MAYOR_OVERALL_RECOMMENDATION_PROMPT_TEMPLATE
        recommendation_prompt = recommendation_template.format(**{
            "oakland_mayor_issues": OAKLAND_MAYOR_ISSUES,
            "voter_info_json": voter_info_json,
            "overall_candidate_scores": overall_candidate_scores,
            "language": language,
        })

        mayor_overall_recommendation_completion = anthropic.completions.create(
            model="claude-2",
            max_tokens_to_sample=5000,
            prompt=f"{HUMAN_PROMPT} {recommendation_prompt}{AI_PROMPT}",
            stop_sequences=["\n\nHuman:"],
        )

        recommendation = mayor_overall_recommendation_completion.completion
        print(recommendation)

        with open(recommendation_path, 'w') as f:
            json.dump(recommendation, f)

    return recommendation


def hash_json_object(json_object):
    canonical_json_str = encode_canonical_json(json_object)
    return hashlib.sha256(canonical_json_str).hexdigest()


# Example usage remains the same as above

#
# @app.route('/race/<race_name>/recommendation')
# def race_recommendation(race_name):
#     voter_info_json = session.get('voter_info')
#     voter_info_hash = hash_json_object(voter_info_json)
#
#     voter_info = VoterInfoDecoder().decode(voter_info_json)
#     # switch on race name
#     # if mayor, then run mayor flow
#     if race_name == "mayor":
#         # load candidate summaries from ../data/big_candidate_summaries.json
#         with open(os.path.join(app.root_path, 'data', 'big_candidate_summaries.json'), 'r') as f:
#             candidate_summaries = json.load(f)
#         candidates_scores = score_candidates_for_mayor(candidate_summaries, voter_info_json, voter_info_hash)
#         recommendation = formulate_mayor_recommendation(candidates_scores, voter_info_json, voter_info_hash)
#
#         recommended_candidate_data = recommendation
#     else:
#         recommended_candidate_data = {
#             "name": "Jane Smith",
#             "reason": "Jane Smith cares about children's ability to study remotely, which aligns with your values."
#         }
#
#     decoded_race_name = None if race_name is None else unquote(race_name)
#
#     return render_template('race.html', races=races(),
#                            recommended_candidate=recommended_candidate_data,
#                            current_race=decoded_race_name,
#                            ballot_data=races,
#                            quote=quote)
#

# Over the top routing magic
@app.route('/race/', defaults={'race_name': None}, methods=['GET'])
@app.route('/race/<race_name>', methods=['GET'])
def race(race_name):
    decoded_race_name = races()[0] if race_name is None else unquote(race_name)
    race_description = ballot_descriptions[
        decoded_race_name] if decoded_race_name in ballot_descriptions else "A description of this race is not yet available in this demo version of VoteWise."

    voter_info_json = session.get('voter_info')
    voter_info_hash = hash_json_object(voter_info_json)

    voter_info = VoterInfoDecoder().decode(voter_info_json)
    # switch on race name
    # if mayor, then run mayor flow
    if False and "mayor" in race_name.lower():
        # load candidate summaries from ../data/big_candidate_summaries.json
        with open(os.path.join(app.root_path, 'data', 'big_candidate_summaries.json'), 'r') as f:
            candidate_summaries = json.load(f)
        candidates_scores = score_candidates_for_mayor(candidate_summaries, voter_info_json, voter_info_hash)
        recommendation = formulate_mayor_recommendation(candidates_scores, voter_info_json, voter_info_hash)

        recommended_candidate_data = {
            "name": extract_key_from_json(recommendation, "recommendation"),
            "reason": extract_key_from_json(recommendation, "justification", human_readable=True)
        }
    else:
        recommended_candidate_data = {
            "name": "Jane Smith",
            "reason": "Jane Smith cares about children's ability to study remotely, which aligns with your values."
        }
    # import ipdb; ipdb.set_trace()

    recommended_candidate_data = {
        "name": "Jane Smith",
        "reason": "Jane Smith cares about children's ability to study remotely, which aligns with your values."
    }

    all_candidates = ballot_data[decoded_race_name]
    return render_template('race.html', races=races(),
                           recommended_candidate=recommended_candidate_data,
                           current_race=decoded_race_name,
                           ballot_data=races,
                           race_description=race_description,
                           quote=quote,
                           voter_info_json=voter_info_json,
                           all_candidates = all_candidates
                           )

@app.route('/race/', defaults={'race_name': None}, methods=['POST'])
@app.route('/race/<race_name>/recommendation', methods=['POST'])
@csrf.exempt
def race_recommendation(race_name):

    decoded_race_name = races()[0] if race_name is None else unquote(race_name)
    race_description = ballot_descriptions[
        decoded_race_name] if decoded_race_name in ballot_descriptions else "A description of this race is not yet available in this demo version of VoteWise."
    voter_info_json = session.get('voter_info')

    # read the demo field from env
    demo = env.bool("DEMO", False)
    if demo:
        print('Demo Mode; loading generic voter info')
        voter_info_json = load_voter_info()
        time.sleep(3)
    else:
        save_voter_info(voter_info_json)

    voter_info_hash = hash_json_object(voter_info_json)

    voter_info = VoterInfoDecoder().decode(voter_info_json)
    # switch on race name
    # if mayor, then run mayor flow
    if "mayor" in race_name.lower():
        # load candidate summaries from ../data/big_candidate_summaries.json
        with open(os.path.join(app.root_path, 'data', 'big_candidate_summaries.json'), 'r') as f:
            candidate_summaries = json.load(f)
        candidates_scores = score_candidates_for_mayor(candidate_summaries, voter_info_json, voter_info_hash)
        recommendation = formulate_mayor_recommendation(candidates_scores, voter_info_json, voter_info_hash)

        recommended_candidate_data = {
            "name": extract_key_from_json(recommendation, "recommendation"),
            "reason": extract_key_from_json(recommendation, "justification", human_readable=True)
        }
    elif "senator" in race_name.lower():
        recommended_candidate_data = {
            "name": "Alex Padilla, Democratic",
            "reason": """
         <ul>
    <li><strong>Democratic Primary Winner:</strong> Padilla was the chosen candidate in the Democratic primary, indicating that he resonated with a significant number of Democratic voters in the state.</li>

    <li><strong>Affordable Housing and Criminal Justice:</strong> Padilla has shown a commitment to affordable housing and criminal justice reform. For instance, he supported the "Fighting Homelessness Through Services and Housing Act", which aligns with the interests of many California Democrats who are concerned about the state's housing crisis.</li>

    <li><strong>Public Service Experience:</strong> Padilla has a long history of public service, from the Los Angeles City Council to the State Senate, and then as California's Secretary of State. This experience gives him a deep understanding of the issues facing Californians and the legislative process.</li>

    <li><strong>Democratic Leader Endorsements:</strong> While endorsements from Democratic leaders might seem partisan, they also indicate that Padilla has the respect and support of his peers, which can be crucial in getting legislation passed.</li>

    <li><strong>Clean Record:</strong> Padilla has maintained a clean record throughout his public and private career, which speaks to his integrity and commitment to ethical conduct.</li>

    <li><strong>Climate Action and Renewable Energy:</strong> As Chair of the State Senate Committee on Energy, Utilities, and Communications, Padilla was instrumental in developing California's renewable energy and climate policies. This aligns with the environmental concerns of many California Democrats.</li>

    <li><strong>Voting Rights Advocacy:</strong> As California's Secretary of State, Padilla worked to make elections more accessible and inclusive, and protected the integrity of voting systems. This commitment to democratic processes is likely to resonate with Democrats who value voting rights.</li>

    <li><strong>Immigration Reform:</strong> Padilla, the son of Mexican immigrants, has shown a commitment to immigration reform. His first bill in the Senate sought to create a pathway to citizenship for undocumented immigrants who served as essential workers during the COVID-19 pandemic. This stance aligns with the views of many California Democrats who support comprehensive immigration reform.</li>
</ul>
                      """

        }
    else:

        recommended_candidate_data = {
            "name": "No Recommendation Available",
            "reason": """
            <ul><p>This demo of VoteWise only provides recommendations for a limited selection of races.</p> <p>In future versions, we aim to provide
            reliable recommendations for every race on your ballot.</p>
                      </ul>
                      """

        }

    # add recommendation to the session
    session['recommendation'] = recommended_candidate_data
    session.modified = True

    # time.sleep(4)
    return jsonify({"response": True, "message": recommended_candidate_data})


@app.route('/race/<race_name>/chat/<recommendation>', methods=['POST'])
@csrf.exempt
def chat(race_name, recommendation):
    data = request.get_json()
    text = data.get('data')
    voter_info = session.get('voter_info')
    race = unquote(race_name)

    print(voter_info)

    # retrieve recommendation from session
    recommendation = session.get('recommendation')



    # if memory has key race_name, use that memory, otherwise create a new memory
    if race_name not in new_memory_by_race:
        new_memory_by_race[race_name] = ConversationSummaryBufferMemory(llm=llm, memory_key="chat_history",
                                                                        return_messages=True)

    race_memory = new_memory_by_race[race_name]

    # Convert the Python object to a JSON string
    escaped_voter_info = json.dumps(voter_info)

    # Escape the curly braces
    escaped_voter_info = escaped_voter_info.replace('{', '{{').replace('}', '}}')

    print(escaped_voter_info)

    language = json.loads(voter_info).get('selected_language', LANGUAGE)

    with open(os.path.join(app.root_path, 'data', 'big_candidate_summaries.json'), 'r') as f:
        candidate_summaries = json.load(f)

    escaped_candidate_summaries = json.dumps(candidate_summaries).replace('{', '{{').replace('}', '}}')

    prompt = f"""
                    You are a helpful voting assistant. You made the following recommendation:
                    {recommendation['name']}
                    for the following race:
                    {race}

                    This was your justification:
                    {recommendation['reason']}
                    
                    Here's info about the voter:
                    {escaped_voter_info}
                    
                    And here is information about the candidates:
                    {escaped_candidate_summaries}

                    Always reply in {language}.
                    
                    Connect any answer you give back to the overall goal of helping the voter decide how they should vote in this election, in order to select the
                    candidate who best represents their priorities, interests, and values (based on the voter summary you've been provided). Your response should
                    try to volunteer information that will be helpful in connecting the voter's question or statement to relevant information about the candidates
                    or the race (only include information about the candidates that is relevant to the voter's question or statement).
                    
                """
    prompt += """

    Current conversation:
    {chat_history}
    Human: {input}
    AI Assistant:
    """

    print(prompt)
    local_prompt = PromptTemplate(input_variables=["chat_history", "input"], template=prompt)

    try:
        conversation = ConversationChain(llm=llm, memory=race_memory, prompt=local_prompt)
        output = conversation.predict(input=text,)
        race_memory.save_context({"input": text}, {"output": output})
        return jsonify({"response": True, "message": output})
    except Exception as e:
        print(e)
        error_message = f'Error: {str(e)}'
        return jsonify({"message": error_message, "response": False})


@app.route('/mayor', methods=['GET', 'POST'])
@csrf.exempt
def handle_issue_chat():
    # check if the request was a POST and if so, get the data
    if request.method == 'POST':
        data = request.get_json()
        text = data.get('data')
    else:
        text = ""

    voter_info_json = session.get('voter_info')
    voter_info = VoterInfoDecoder().decode(voter_info_json)

    # a prompt to instruct the LLM to make recommendations for my friend on who to vote for in the 2022 oakland
    # mayoral election should be written in a way that takes a json summarizing the voter and their values and makes
    # sure the LLM does not refuse to make a recommendation for any reason

    template = INITIAL_RECOMMENDATION_PROMPT

    # Create a PromptTemplate with necessary inputs
    prompt = PromptTemplate(
        input_variables=[
            "race",
            "voter_zip_code",
            "voter_info_summary",
            "race_info",
        ],
        template=template
    )

    # Use the PromptTemplate to construct the query
    constructed_query = prompt.format(**{
        "race": "2022 Oakland Mayor Election",
        "voter_zip_code": voter_info.address_zip_code,
        "voter_info_summary": voter_info_json,
        "race_info": OAKLAND_MAYOR_CANDIDATE_INFO,
    })

    print(constructed_query)

    try:
        conversation = ConversationChain(llm=llm, memory=memory)
        output = conversation.predict(input=constructed_query)
        memory.save_context({"input": constructed_query}, {"output": output})
        return jsonify({"response": True, "message": output})
    except Exception as e:
        print(e)
        error_message = f'Error: {str(e)}'
        return jsonify({"message": error_message, "response": False})


@app.route('/privacy_policy.html')
def privacy_policy():
    return render_template('privacy_policy.html')


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
