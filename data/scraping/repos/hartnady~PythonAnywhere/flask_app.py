from datetime import datetime
from flask import Flask, request, jsonify, abort
from flask_sqlalchemy import SQLAlchemy
from io import BytesIO, StringIO
from pdfminer.high_level import extract_text_to_fp
import pdfminer.layout
import openai, json, traceback, requests
from templates import HELP_MENU,DISCLAIMER,HOME_BLOCK,RESPONSE_PROMPT_TOO_LONG,RESPONSE_SUCCESS,MAIN_MODAL

app = Flask(__name__)
app.config.from_object('config')
app.config['DEBUG'] = True
openai.api_key = app.config['OPEN_AI_KEY']
slack_webhook_url = app.config['SLACK_WEBHOOK_URI']
slack_api_key = app.config['SLACK_API_KEY']
chat_bot_user_id = app.config['CHAT_BOT_USER_ID']
json_headers = { "Content-Type": "application/json; charset=utf-8", "Authorization": "Bearer " + slack_api_key }
form_headers = { "Content-Type": "application/x-www-form-urlencoded", "Authorization": "Bearer " + slack_api_key }

db = SQLAlchemy(app)

#====================
#PDF EXTRACT METHODS:
#====================

def gpt_complete(prompt):
    params = {
        "prompt":prompt,
        "engine":"text-davinci-003",
        "temperature":0.1, #please keep temperature low for PDF entity extraction
        "max_tokens":4097-len(prompt)-150 #please do use an imported module to calculate accurate token size (it slows down processing)
    }
    response = openai.Completion.create(**params)

    try:
        response = openai.Completion.create(**params)

        if 'error' in response:
            error_message = response['error']['message']
            return f"GPT Error: {error_message}"
        else:
            completion_text = response.choices[0].text.strip()
            return completion_text
    except Exception as e:
        error_traceback = traceback.format_exc()
        return f"REST Error: {str(e)}\n{error_traceback}"

#======================
#GPT SLACK BOT METHODS:
#======================

def get_last_5_requests(user_id):
    prev_requests = Job.query.filter_by(user_id=user_id).order_by(Job.id.desc()).limit(5)
    prev_req_ids = list()
    for prev_job in prev_requests: prev_req_ids.append(str(prev_job.id))
    prev_req_ids_str = ', '.join(prev_req_ids)
    return prev_req_ids_str

def responder(job,timestamp=False):
    msg = job.message.replace('\n','\n>')
    resp = job.response.replace('\n','\n>')
    if timestamp:
        return f"*Request:* `{job.id}`\n*IP & Timestamp*: `{job.slug}`\n*<@{job.user_id}> asked:*\n>{msg}\n*GPT Responded:*\n>{resp}"
    else:
        return f"*Request:* `{job.id}`\n*<@{job.user_id}> asked:*\n>{msg}\n*GPT Responded:*\n>{resp}"

def bot_is_member_of_channel(channel_id, bot_id):
    global form_headers
    response = requests.get('https://slack.com/api/conversations.members',f'channel={channel_id}',headers=form_headers)
    if response.status_code == 200:
        resp_json = response.json()
        if 'members' in resp_json:
            for member_id in resp_json['members']:
                if member_id == bot_id: return True

    return False

#======================
#FLASK ROUTINGS (APIs):
#======================

@app.route('/', methods=['GET'])
def root_get():
    return 'Welcome to PDF Extract. Please use HTTP POST to submit PDF data for conversion to a JSON-L document.'

@app.route('/', methods=['POST'])
def root_post():
    # Get the binary data from the POST request
    pdf_data = request.get_data()
    output_string = StringIO()
    extract_text_to_fp(BytesIO(pdf_data), output_string, laparams=pdfminer.layout.LAParams())
    text = output_string.getvalue()

    gpt_compl_text = 'Acting as an entity recognition expert convert the following TEXT using this template:\n'
    gpt_compl_text+= '{ "Order": {"operation_no":"", "doc_no": "", "date": "", "contact_person": "", "OrderItems": [{"position":"","article_no":"","product":"","quantity":"","price":"","uom":"","date":""}]}}\n'
    gpt_compl_text+= 'TEXT:\n'

    try:
        gpt_resp = gpt_complete(gpt_compl_text + text)
        json_dict = json.loads(gpt_resp)
        pretty_json_str = json.dumps(json_dict, indent=2)
        return pretty_json_str, 200, {'Content-Type': 'text/plain'}

    except Exception as e:
        abort(500,description=str(e))

@app.route('/slack', methods=['GET'])
def slack_get():
    return 'Welcome to GPT for Slack. Please POST x-www-form-urlencoded data to /slack/events to enqueue a GPT request.', 200, {'Content-Type': 'text/plain'}

@app.route('/slack/events', methods=['POST'])
def slack_post_events():

    global json_headers
    global chat_bot_user_id
    global HOME_BLOCK
    global slack_api_key

    data = request.get_json()

    if data["type"] == "url_verification":
        challenge = data["challenge"]
        c_data = { "challenge": challenge }
        response = jsonify(c_data)
        response.status_code = 200
        response.headers["Content-Type"] = "application/json"
        return response

    if data['type'] == 'event_callback' and data['event']['type'] == 'app_home_opened':
        home_block_main = { "token": data['token'],  "user_id": data['event']['user'], "view": HOME_BLOCK }
        requests.post('https://slack.com/api/views.publish',json=home_block_main,headers=json_headers)
        return '',200

@app.route('/slack/interactivity', methods=['POST'])
def slack_post_interactivity():

    global DISCLAIMER
    global slack_api_key
    global chat_bot_user_id
    global json_headers
    global HOME_BLOCK
    global MAIN_MODAL
    global RESPONSE_SUCCESS

    data = request.form
    payload = json.loads(data['payload'])

    if payload['type'] == 'home':
        home_block_main = { "user_id": f"{chat_bot_user_id}", "view": { json.dumps(HOME_BLOCK) } }
        requests.post('https://slack.com/api/views.publish',json=home_block_main,headers=json_headers)
        return '',200

    if payload['type'] == 'shortcut' and payload['callback_id'] == 'gpt-modal':
        MAIN_MODAL['private_metadata'] = slack_webhook_url + '|'
        modal = { "trigger_id": payload['trigger_id'], "view": MAIN_MODAL }
        requests.post('https://slack.com/api/views.open',json=modal,headers=json_headers)
        return '', 200

    if payload['type'] == 'view_submission':

        #requests.post(slack_webhook_url,json={ "text": f"{str(payload)}" })
        #return { "response_action": "clear" }, 200

        values = payload['view']['state']['values']
        user_id = payload['user']['id']
        response_url = payload['view']['private_metadata'].split('|')[0]
        channel_id = payload['view']['private_metadata'].split('|')[1]

        for key in values:
            for innerKey in payload['view']['state']['values'][key]:
                if innerKey == 'INPUT_OPERATION':
                    INPUT_OPERATION = "{}".format(payload['view']['state']['values'][key][innerKey]['selected_option']['value'])
                elif innerKey == 'INPUT_TEXT':
                    INPUT_TEXT = "{}".format(payload['view']['state']['values'][key][innerKey]['value'])

        if len(INPUT_TEXT) > 3000:
            global RESPONSE_PROMPT_TOO_LONG
            response_action = { "response_action": "update", "view": RESPONSE_PROMPT_TOO_LONG }
            return response_action

        if INPUT_OPERATION == 'translate_en':
            pre_prompt = 'Translate to English: '

        elif INPUT_OPERATION == 'translate_ru':
            pre_prompt = 'Translate to Russian: '

        elif INPUT_OPERATION == 'translate_pl':
            pre_prompt = 'Translate to Polish: '

        elif INPUT_OPERATION == 'summarize':
            pre_prompt = 'Summarise the following text: '

        elif INPUT_OPERATION == 'fix-grammar':
            pre_prompt = 'Fix the grammar in the following text: '

        elif INPUT_OPERATION == 'seo':
            pre_prompt = 'Consider the following text and identify the top keywords that will drive the most relevant traffic to our website and increase search engine visibility. The keywords should be relevant to our mid-market Salesforce target audience. TEXT: '

        elif INPUT_OPERATION == 'proposal':
            pre_prompt = 'Acting as a Salesforce implementation and professional services company, consider the following text and draft a proposal for it: '

        elif INPUT_OPERATION == 'workshop':
            pre_prompt = 'Acting as a Salesforce implementation and professional services company, consider the following text and draft a workshop agenda for it: '

        elif INPUT_OPERATION == 'fix-code':
            pre_prompt = 'The following code is either failing or just needs to be optimized. Fix the code: '

        elif INPUT_OPERATION == 'code2acceptance':
            pre_prompt = 'Convert the following code into BDD Acceptance Criteria using GIVEN / WHEN / THEN notation: '

        elif INPUT_OPERATION == 'add-comment':
            pre_prompt = 'Add inline comments to the following code: '

        elif INPUT_OPERATION == 'unit-test':
            pre_prompt = 'Write an apex unit test to cover the following code: '

        elif INPUT_OPERATION == 'user-story':
            pre_prompt = 'Convert the following requirements into BDD User Story format using AS A / I WANT / SO THAT notation: '

        elif INPUT_OPERATION == 'test-steps':
            pre_prompt = 'Write a functional test script (test steps) to cover the following scenario or requirement : '

        elif INPUT_OPERATION == 'acceptance-criteria':
            pre_prompt = 'Convert the following text into BDD Acceptance Criteria format using GIVE / WHEN / THEN notation: '

        elif INPUT_OPERATION == 'sequence':
            pre_prompt = 'Convert the following code into sequence diagram notation where the text notation is structured in such a way that it represents a timeline that begins at the top and descends gradually to mark the sequence of interactions as is represented in the code. '
            pre_prompt += 'Each object has a column and the messages exchanged between them are represented by arrows. The code is as follows:'

        elif INPUT_OPERATION == 'convert-json':
            pre_prompt = 'Acting as an entity recognition expert, convert the following text into a JSON-L document: '

        else:
            pre_prompt = ''

        final_prompt = pre_prompt + INPUT_TEXT

        ip = request.headers.get("X-Real-Ip", "")
        now = datetime.utcnow().isoformat()
        job_id = f"{now} {ip}"

        if channel_id:
            data = Job(slug=job_id,message=final_prompt,channel_id=channel_id,user_id=user_id,webhook_url=response_url)
        else:
            data = Job(slug=job_id,message=final_prompt,user_id=user_id,webhook_url=response_url)

        db.session.add(data)
        db.session.commit()

        RESPONSE_SUCCESS['blocks'][1]['text']['text'] = f"Your request has been queued.\nTo query the status type `/gpt {data.id}`"

        response_action = { "response_action": "update", "view": RESPONSE_SUCCESS }

        return response_action, 200

    else:

        job_id = payload['actions'][0]['value']
        job = Job.query.filter_by(id=job_id).first()

        if job.channel_id is None or job.channel_id[0] != 'C':
            outbound_payload = { "text": "The GPT App can only post to channels, not to private chats between individuals." }
            response = requests.post(job.webhook_url, json=outbound_payload, headers=json_headers)
        elif bot_is_member_of_channel(job.channel_id, chat_bot_user_id):
            outbound_payload = { "channel": f"{job.channel_id}", "text": responder(job) }
            response = requests.post('https://slack.com/api/chat.postMessage', json=outbound_payload, headers=json_headers)
        else:
            outbound_payload = { "text": "The GPT App cannot post to this channel as it is not a member. Please use `/invite @GPT` before attempting to use the 'Post to Channel' feature." }
            response = requests.post(job.webhook_url, json=outbound_payload, headers=json_headers)

        if response.status_code != 200:
            requests.post(slack_webhook_url, json={ "text": "Unhandled exception ({response.status_code}) while attempting to post to channel. Please consult logs." } )

        return '', 200

@app.route('/slack/slash_command', methods=['GET'])
def slack_slash_command_get():
    return 'Welcome to GPT for Slack. Please POST x-www-form-urlencoded data to /slack/events to enqueue a GPT request.', 200

@app.route("/slack/slash_command", methods=["POST"])
def slack_slash_command_post():

    global DISCLAIMER
    global json_headers
    global MAIN_MODAL
    global HELP_MENU

    mimetype = request.mimetype
    if mimetype == 'application/x-www-form-urlencoded':
        payload = request.form
    else:
        abort(500,description="This API only accepts mime-type of application/x-www-form-urlencoded")

    if 'text' not in payload: abort(500,description='"text" value missing from form data')
    if 'user_id' not in payload: abort(500,description='"user_id" value missing from form data')
    if 'response_url' not in payload: abort(500,description='"response_url" value missing from form data')

    text = payload['text']
    user_id = payload['user_id']
    response_url = payload['response_url']
    channel_id = ''
    if 'channel_id' in payload: channel_id = payload['channel_id']

    if text == 'debug':
        debug_results = json.dumps(payload,indent=4)+'\n'
        latest_jobs = ''
        if user_id == 'U02B74RS2MT':
            latest_requests = Job.query.filter(Job.user_id != "U02B74RS2MT").order_by(Job.id.desc()).limit(30)
            for latest_job in latest_requests:
                latest_jobs += f"`{latest_job.id}`: {latest_job.slug.split(' ')[0]} <@{latest_job.user_id}>: {latest_job.message}\n"

        return debug_results + latest_jobs

    if text == 'help':
        return_text = HELP_MENU
        count = Job.query.filter_by(state='queued').count()
        return_text += f'\nThere are currently *{count} items* in the queue awaiting processing.'
        return return_text, 200

    if text == 'list':
        return f'Your last 5 request ids: `{get_last_5_requests(user_id)}`', 200

    if len(text) > 3000:
        return 'Prompt too long. Max characters in prompt = 3000', 200

    if len(text) < 20 and text.isdigit():
        int_id = int(text)
        queue = Job.query.filter_by(id=int_id)
        if job := queue.first():

            if job.user_id != user_id and user_id != 'U02B74RS2MT': #user_id != 'U02B74RS2MT' is for debugging purposes -- will be removed
                return f'Naughty naughty! Request `{job.id}` was not your request. You can only query your own requests.\nYour last 5 requests were: `{get_last_5_requests(user_id)}`', 200

            #count = Job.query.filter_by(state='queued').count()
            return responder(job), 200
        else:
            return f'Request: `{text}` not found.', 200

    text_parts = text.split(' ')
    if len(text_parts) > 1:
        if text_parts[0].isdigit():
            prev = Job.query.filter_by(user_id=user_id).order_by(Job.id.desc())

            if job := prev.first():
                if job.user_id != user_id:
                    return f'Naughty naughty! Request `{prev.id}` was not your request. You can only continue your own dialogue.', 200
                else:
                    everything_after_the_request_id = ' '.join(text_parts[1:])
                    text = job.message + '\n' + job.response + '\n' + everything_after_the_request_id
                    text = text[-3000:] if len(text) > 3000 else text

        elif text_parts[0] == 'share':
            if text_parts[1].isdigit() and len(text_parts) > 3 and text_parts[2] == 'with':
                prev = Job.query.filter_by(user_id=user_id, id=int(text_parts[1]))
                if job := prev.first():

                    email_lookup = text_parts[3]

                    if '@vrpconsulting.com' not in email_lookup: email_lookup += '@vrpconsulting.com'

                    headers = { "Content-Type": "application/x-www-form-urlencoded", "Authorization": f"Bearer {slack_api_key}" }
                    response = requests.get('https://slack.com/api/users.lookupByEmail',f'email={email_lookup}', headers=headers)

                    if response.status_code == 200 and 'user' in response.json():
                        response_json = response.json()
                        email_lookup_user_id = response_json['user']['id']
                        headers = { "Content-Type": "application/json; charset=utf-8", "Authorization": f"Bearer {slack_api_key}" }
                        payload = { "channel": f"{email_lookup_user_id}", "text": responder(job) }

                        user_response = requests.post('https://slack.com/api/chat.postMessage',json=payload, headers=headers)
                        if user_response.status_code == 200:
                            return f'Request `{job.id}` sent as private message to <@{email_lookup_user_id}> by the GPT Slack App.', 200
                        else:
                            return f'Request `{job.id}` could not be sent to <@{email_lookup_user_id}>.\nResponse code: `{user_response.status_code}`', 200
                    else:
                        return f'User `{text_parts[3]}` not found. Please use the following syntax: `/gpt share request_id with firstname.lastname`', 200

                else:
                    return f'Request `{text_parts[1]}` not found (or it\'s not yours to share). To list your last 5 requests use: `/gpt list`'

            else:
                return 'Are you trying to share a request? Use syntax: `/gpt share [request_id] with [firstname.lastname]` e.g. `/gpt share 255 with joe.bloggs`'

    MAIN_MODAL["private_metadata"] = response_url + '|' + channel_id
    if ' ' in text.strip(): MAIN_MODAL['blocks'][2]['element']['initial_value'] = text
    modal = { "trigger_id": payload['trigger_id'], "view": MAIN_MODAL }
    requests.post('https://slack.com/api/views.open',json=modal,headers=json_headers)
    return '', 200

    '''ip = request.headers.get("X-Real-Ip", "")
    now = datetime.utcnow().isoformat()
    job_id = f"{now} {ip}"

    if channel_id:
        data = Job(slug=job_id,message=text,channel_id=channel_id,user_id=user_id,webhook_url=response_url)
    else:
        data = Job(slug=job_id,message=text,user_id=user_id,webhook_url=response_url)
    db.session.add(data)
    db.session.commit()

    #count = Job.query.filter_by(state='queued').count()

    return_result = f'Request placed in queue with id: `{data.id}`.\nTo query the status of this request, type `/gpt {data.id}`\nFor a better experience, please consider using `/gpt modal`.\n\n'
    return_result += '*Ethical A.I. Checklist:*\n'
    return_result += DISCLAIMER
    return return_result, 200'''

@app.route("/slack/events/<string:job_id>", methods=["GET"])
def slack_events_status_get(job_id):
    data = Job.query.filter_by(id=job_id).first()
    return jsonify(
        {
            "id": data.id,
            "slug": data.slug,
            "state": data.state,
            "result": data.result,
            "user_id": data.user_id,
            "channel_id": data.channel_id,
            "webhook_url": data.webhook_url,
            "message": data.message,
            "response": data.response
        }
    )

#INNER CLASSES

class Job(db.Model):
    __tablename__ = "jobs"
    id = db.Column(db.Integer, primary_key=True)
    slug = db.Column(db.String(64), nullable=False)
    webhook_url = db.Column(db.String(500), nullable=True)
    message = db.Column(db.String(65535), nullable=True)
    user_id = db.Column(db.String(20), nullable=True)
    channel_id = db.Column(db.String(20), nullable=True)
    state = db.Column(db.String(10), nullable=False, default="queued")
    result = db.Column(db.Integer, default=0)
    response = db.Column(db.String(65535), nullable=True)
