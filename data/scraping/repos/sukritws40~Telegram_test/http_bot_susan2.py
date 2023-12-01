#!/usr/bin/env python
# coding: utf-8

import traceback
import json
import os
import re
import sys
import time
import urllib2
import Queue
from argparse import ArgumentParser
from threading import Thread
from unidecode import unidecode
from flask import Flask, request, jsonify
import random
from StringIO import StringIO

from bottools.git import get_short_git_version
from mongodb_alana.history import History
from mongodb_alana.ner_db import Ner_db
from intents.intents import Intents
from intents.sent_transform import normalize_name_info, normalize_tell_me_about_request, \
    normalize_time_request, profanity_filter
from reranker.filter import BatchFilter
from reranker.preprocess import Preprocessor
from reranker.reranker import Reranker
#from reranker.vw import VwReranker
#from reranker.neural_ranker import NeuralReranker
from reranker.postprocess import Postprocessor
from stanford_np.postag_extractor import POSTagExtract
from mongodb_alana.conf_scores import conf_scores
from ner_util import StanfordNERWrapper
from reranker.bots import COHERENCE_BOT_URL

app = Flask(__name__)

# all the bots + hostnames
BOT_LIST = {
    'evi': 'localhost:5010',  # devel evi
    'persona+eliza': 'localhost:5011',
    'facts': 'localhost:5014',
    'news_api': '34.202.124.91:5011', # DEV VERSION
    'wiki': '34.202.124.91:5101', # DEV VERSION
    'weather': 'localhost:5015',
    'intro': 'localhost:5016',
    'coherence': COHERENCE_BOT_URL,
    'rapport': 'localhost:5017',
}

# list of priority bots, in the order of priority
PRIORITY_BOTS = ['rapport', 'intro', 'topic_quiz_game', 'facts', 'weather', 'persona', 'evi']

# bots ignored when calling "ask_all_bots"
NONDEFAULT_BOTS = ['coherence']


history = History('../mongodb_alana/mongo_info.json')

nerdb = Ner_db('../mongodb_alana/mongo_info.json')

cscores = conf_scores('../mongodb_alana/mongo_info.json')

# Create filters
preprocessor = Preprocessor()
basic_filter = BatchFilter({'profanities': True,
                            'profanities_strict': True,
                            'profanities_nostrict_bots': set(['persona', 'evi', 'topic_quiz_game']),
                            'max_length': 300,
                            'min_length': 3,
                            'never_filter': 'never_filter.txt',
                            'ban_punct_only': True,
                            'no_repeat_bots': {'news_api': 0.5,
                                               'wiki': 0.5,
                                               'persona': 0.9,
                                               'eliza': 0.9,
                                               # 'evi': 0.5,
                                               'coherence': 0.7},
                            'repeat_hist_length': 30,
                            'min_repq_length': 3})
postprocessor = Postprocessor()

# Initialize parsers
tagger = StanfordNERWrapper()
postagger = POSTagExtract()

# Instantiate the reranker
reranker = Reranker()
#reranker = VwReranker('train-07.16bit-1pass-RR_CR_BR_BC_BF_BCR-logistic.vwmodel')
#reranker = NeuralReranker('learning_to_rank_model', tagger)

# initialize custom intents
intents = Intents('intents.yaml')

# backoff bot settings
BACKOFF_BOTS = {
    'coherence':  {'name': 'coherence',
                   'prob': 10,
                   'params_override': {}},
    # 'fun_fact': {'name': 'facts',
    #              'prob': 2,
    #              'params_override': {'q': 'tell me a fun fact'}},
    # 'joke': {'name': 'facts',
    #          'prob': 1,
    #          'params_override': {'q': 'tell me a joke'}},
}
# this will create a list where each of the backoff bots' names is repeated prob times,
# thus imitating a probability distribution (e.g. if 'prob' for the 'coherence' bot is 5,
# the list will contain 'coherence' repeated 5 times
BACKOFF_BOTS_DISTRO = [rep_botname
                       for key, val in BACKOFF_BOTS.iteritems()
                       for rep_botname in ([key] * val['prob'])]

# seed the random number generator (otherwise backoff seems to prefer facts)
random.seed()

# Git version
VERSION = get_short_git_version(os.path.dirname(os.path.realpath(__file__)))
# ASR confidence threshold
ASR_THRESHOLD = 0.25
print >> sys.stderr, 'Bucket version: %s' % VERSION


@app.route('/')
def answer():

    printbuf = StringIO()  # buffer for printing to stderr

    get_debug = bool(request.args.get('debug'))
    sessionID = request.args.get('sid')
    utt_conf_score = float(request.args.get('cs'))
    question = request.args.get('q')
    if sessionID is None:
        print >> printbuf, '!!!NO SESSION ID'
        sessionID = 'EMPTY'
    if question is None:
        print >> printbuf, '!!!EMPTY REQUEST in session ', sessionID
        question = ''
    question = unidecode(question)

    try:
        ret = get_answer(sessionID, question, utt_conf_score, printbuf, get_debug)
    except Exception as err:
        traceback.print_exc(file=printbuf)
        ret = 'Error: ' + str(err), 500, {'Content-Type': 'text/plain'}

    print >> sys.stderr, printbuf.getvalue()
    return ret


def get_answer(sessionID, question, utt_conf_score, printbuf, get_debug=False):
    # remember the user question
    raw_asr = question
    user_name = ''

    # Get user's name from db if exists
    try:
        user_name = history.get_user_name(sessionID)
    except TypeError:
        pass

    # Get user's preferences from db if exists
    try:
        preferences = history.get_user_pref(sessionID)
    except TypeError:
        pass

    # get dialogue context
    dialogue_history = history.get_dialogue_history(sessionID)
    last_system_utt = (dialogue_history['dialogue'][-2]['utterance']
                       if len(dialogue_history['dialogue']) >= 2 else '')
    turn_no = str(len(dialogue_history['dialogue']))  # current turn number

    # Get last topic mentioned
    topic, _ = history.get_topic(sessionID)

    # get the bot that responded last
    responded_bot = ''
    if len(dialogue_history['dialogue']) >= 2:
        responded_bot = (dialogue_history['dialogue'][-2]['bot']
                         if dialogue_history['dialogue'][-2]['actor'] == 'system'
                         else dialogue_history['dialogue'][-1]['bot'])

    # preprocess (remove "Alexa", transform Y/N replies, indirect questions)
    question = preprocessor.preprocess(last_system_utt, question, user_name)

    # match intents
    intent, intent_param = intents.match_intents(question)

    print >> printbuf  # To add some space between the responses
    print >> printbuf, "SESSION:", sessionID
    print >> printbuf, 'ASR RAW:', raw_asr
    print >> printbuf, 'INTENT:', intent

    # Check if user asked to repeat, and repeat last system's response if needed
    if intent == 'repeat':
        return handle_repeat_intent(last_system_utt)
    # normalize time requests so that they go to Evi
    elif intent == 'time':
        question = normalize_time_request()
    # handle stop requests (1st: ignore, goes to persona; 2nd: end dialogue)
    elif intent == 'stop' and is_second_stop_request(dialogue_history):
        return handle_stop_intent()
    elif intent == 'dont_tell_about':
        question, topic = handle_dont_tell_about(sessionID, intent_param, topic, turn_no)
    # handle "let's talk about X"
    elif intent == 'tell_me_about' and intent_param:
        question = normalize_tell_me_about_request(question, intent_param)
    elif intent == 'name':
        question = normalize_name_info(intent_param)
    
    # Extract names and locations from utterance
    if intent != 'name':
        ner_data = tagger.get_entities(question)
    else:
        ner_data = tagger.get_entities('')
 
    # filter important parts-of-speech from utterance
    pos_filtered = postagger.extract_np(question)

    # Save both in db
    nerdb.save_entities(ner_data, pos_filtered, sessionID)

    # Replace question with NER
    noner_question = question
    question, entities = nerdb.resolve_anaphora_name(question, sessionID)
 
    # Prepare NPs it for forwarding
    pos_filtered = ', '.join(pos_filtered)

    # get average score across all recognized words
    # utt_conf_score = cscores.get_score(sessionID)

    print >> printbuf, "Q:", question
    print >> printbuf, "confidence score: ", utt_conf_score
    print >> printbuf, "E:", entities
    print >> printbuf, "POS:", pos_filtered
    print >> printbuf, "TOPIC:", topic

    # Prompt user to repeat if below threshold
    if utt_conf_score < ASR_THRESHOLD or question.startswith("LOW_ASR"):
        return handle_bad_asr(raw_asr, printbuf)

    # forward question to all bots in list
    response = None
    bot_params = {'q': question,
                  'nnq': noner_question,
                  'p': pos_filtered,
                  'e': json.dumps(entities),
                  'sid': sessionID,
                  'lb': responded_bot,
                  'u': user_name,
                  't': topic,
                  'pr': ','.join(preferences),
                  'n': turn_no,
                  'i': intent if intent else ''}
    bucket, news_id = ask_all_bots(dialogue_history, bot_params, printbuf)

    # select priority reply
    try:
        for priority_bot in PRIORITY_BOTS:
            for bot in bucket:
                if (bot['value'] and
                        (bot['bot'] == priority_bot or
                         bot['bot'].startswith(priority_bot + '-'))):
                    response = [[1, bot['value'][0], bot['bot']]]
                    raise StopIteration
    except StopIteration:
        pass  # this just means that a priority reply has been selected

    # If newsAPI will handle the next response as well (XX multiturn) then return that one.
    try:
        for bot in bucket:
            if (bot['bot'].startswith('news_api') or bot['bot'].startswith('wiki')) and ('tag' in bot):
                if bot['tag'] == 'NEWS_LOCKED' and bot['value']:
                    print >> printbuf, "NEWS_LOCKED", bot
                    n_id = get_news_id(bot['value'][0], news_id)
                    print >> printbuf, "News_id: ",n_id
                    response = [[1, bot['value'][0], bot['bot'],
                                 (n_id if n_id else 'NO_LAST_NEWS')]]
    except StopIteration:
        pass  # this just means that a priority reply has been selected
    # TODO: add the ADD_FEEDBACK tag

    # if we don't have a priority reply, use the ranker
    if not response:
        response = reranker.re_rank(
            sessionID,
            bucket,
            dialogue_history,
            pos_filtered, entities,
            None if get_debug else 10  # just return 10-best unless debugging is turned on
        )

    # log top 10 responses in the bucket
    n_id = ''
    if response:
        print >> printbuf, "BUCKET:\n", "\n".join(["%f %s %s" % (r[0], r[2], r[1])
                                                   for r in response[:10]])
        # get the news_id of the selected response if it comes from NewsAPI
        if news_id and (response[0][2].startswith("news_api") or response[0][2].startswith('wiki')):
            n_id = get_news_id(response[0][1], news_id)

    else:  # backoff action
        response = backoff_action(dialogue_history, bot_params, printbuf)

    # default: return plain response (for Alexas)
    if n_id != '':
        response[0].append(n_id)
    response = response[0] if len(response) else [0.0, '', 'empty_bucket']

    # update user name in case it has been recognized in this turn
    bot_params['u'] = history.get_user_name(sessionID)
    # add drivers
    response[1] = postprocessor.add_drivers(response[1], bot_params, printbuf)
    # run NER (including drivers) and save results
    ner_data = tagger.get_entities(response[1], filter_names=entities)
    print >> printbuf, "E:", ner_data
    nerdb.save_entities(ner_data, None, sessionID)
    # add user names, detokenize
    response[1] = postprocessor.postprocess(response[1], bot_params, printbuf) or 'Sorry, I don\'t know what to say'

    # print the actual returned (+ postprocessed) response to the console
    print >> printbuf, "R: " + response[1]

    if get_debug:
        debug_info = {'question rewrite': question,
                      'entities q/r': [bot_params['e'], json.dumps(ner_data)],
                      'NPs': bot_params['p'],
                      'intent': bot_params['i'],
                      'topic': bot_params['t'],
                      'preferences': bot_params['pr']}
        response.append(debug_info)
    return jsonify(response), 200, {'Content-Type': 'application/json'}


def get_news_id(text, news_id):
    try:
        return [n['ID'] for n in news_id if n['text'] in text][0]
    except IndexError:
        return ''


def handle_bad_asr(question, printbuf):
    # first filter the user question for profanity
    question = re.sub("^LOW_ASR", '', question)
    if profanity_filter.check_sentence(question):
        text = "I am sorry. I think I heard you said " + question + ". Could you repeat that please?"
    else:
        text = "I am sorry. I did not hear what you said. Could you repeat that please?"

    print >> printbuf, "R: " + text
    return jsonify([1.0, text, 'bad_asr']), 200, {'Content-Type': 'application/json'}


def handle_repeat_intent(last_system_utt):
    """Handle repeats -- repeat last utterance, prepending "I said"."""
    # the .strip is to avoid multiple 'I said' in case of back to back repeat requests
    text = "I said, " + re.sub(r'^I said, ', '', last_system_utt)
    return jsonify([1.0, text, 'repeat_intent']), 200, {'Content-Type': 'application/json'}


def is_second_stop_request(dialogue_history):
    """Check if this is the 2nd request to stop the dialogue in a row (basically checks if the
    previous user utterance also matched the stop intent). Assumes that the current user utterance
    matches the stop intent."""
    dialogue = dialogue_history['dialogue']
    if len(dialogue) < 3:
        return False
    return intents.match_intents(dialogue[-3]['utterance'])[0] == 'stop'


def handle_stop_intent():
    """Return 'STOP_INTENT_REQUESTED' which is handled by the Lambda function."""
    return jsonify([1.0, 'STOP_INTENT_REQUESTED', 'stop_intent']), 200, {'Content-Type': 'text/plain'}


def handle_dont_tell_about(session_id, bad_topic, cur_topic, turn_no):
    """Clear the current topic if the user does not want to talk about it, normalize the sentence
    to "let\'s talk about something else"."""
    if cur_topic == bad_topic:
        history.set_topic(session_id, 'EMPTY', turn_no)
        cur_topic = ''
    return 'tell me about something else', cur_topic


def backoff_action(dialogue_history, params, printbuf):
    """A backoff action -- trying to get another response in case the bucket is empty.
    Now triggering the coherence bot."""
    bucket = Queue.Queue()
    tries = 0
    response = {'value': []}
    # try to get a filtered, non-empty result (up to 10 tries)
    while not response['value']:
        tries += 1
        # select a random bot based on the backoff bots probability distribution
        bot = BACKOFF_BOTS[random.choice(BACKOFF_BOTS_DISTRO)]
        # set up its parameters (based on current parameters and pre-set overrides)
        bot_params = {k: v for k, v in params.iteritems()}
        bot_params.update(bot['params_override'])
        # call the bot & get its answer
        call_bot(bucket, bot['name'], bot_params, printbuf)
        response = bucket.get()
        print >> printbuf, "EMPTY BUCKET -- BACKOFF:\n", "%f %s %s" % (1.0, response['bot'], response['value'][0])
        # filter the result, if we haven't tried filtering too many times
        if tries <= 10:
            response['value'] = basic_filter.filter(response['value'], response['bot'],
                                                    dialogue_history['dialogue'], printbuf)
    # return the response (imitate the ranker for the format)
    response = [[1.0, response['value'][0], response['bot']]]
    return response


def ask_all_bots(dialogue_history, params, printbuf):
    """Call all bots in BOT_LIST and ask the given question, wait for their replies,
    filter replies using basic filter.

    @param dialogue_history: current dialogue history ('context' parameter will be handled here!)
    @param params: all bot call parameters
    @returns: the bucket with individual bots' answers
    """
    # use last 2 sentences from dialogue history as context
    params['c'] = ' '.join([item['utterance'] for item in dialogue_history['dialogue'][-3:-1]])
    # create a placeholder for responses
    bucket = Queue.Queue()
    # launch threads calling all bots
    thread_list = []
    for bot_name in BOT_LIST.iterkeys():
        if bot_name in NONDEFAULT_BOTS:
            continue
        try:
            thread_list.append(Thread(target=call_bot, args=(bucket, bot_name, params, printbuf)))
            thread_list[-1].start()
        except Exception as errtxt:
            print >> printbuf, errtxt

    # wait for threads to finish
    for t in thread_list:
        t.join()

    # collect all bots' responses
    values = []
    news_id = []
    while not bucket.empty():
        v = bucket.get()
        # For the responses given by the NewsAPI bot, we need the mapping between ids and news texts in order to
        # provide to the multi-turn news the id of the selected news response by the reranker (if any)
        if (v['bot'].startswith('news_api') or v['bot'].startswith('wiki')):
            try:
                # perform same filtering as all caditates but without removing any candidate (for correct retrieval later)
                for t in v['value']:
                    t['text'] = basic_filter.normal_filter.normalize_encoding(t['text'])

                news_id.extend(v['value'])
                # now that we stored the mapping we don't need the news_ids anymore, so go back to usual newsAPI format
                v['value'] = [item['text'] for item in v['value']]
            except TypeError:
                print >> printbuf, "[ERROR] Type Error (possible thread mixup) in news"

        v['value'] = basic_filter.filter(v['value'], v['bot'], dialogue_history['dialogue'], printbuf)
        values.append(v)

    return values, news_id


def call_bot(bucket, bot_name, params, printbuf):
    """Ask a specific bot to respond to the given question (in the given context).
    @param bucket: a Queue.Queue thread-safe object to store the response
    @param bot_name: bot name, used for URL lookup in BOT_LIST
    @param params: all bot call parameters
    """
    # build the URL, url-encode everything
    call_url = ('http://%s/?' % BOT_LIST[bot_name] +
                '&'.join(['%s=%s' % (key, urllib2.quote(val.encode('UTF-8')))
                          for key, val in params.iteritems()]))
    try:
        answer = urllib2.urlopen(call_url, timeout=5).read()
        answer = json.loads(answer)
    except Exception, errtxt:
        print >> printbuf, 'Call to bot %s (%s) failed. Reason: %s' % (bot_name, call_url, errtxt)
        return

    # Add to bucket
    for an in answer:
        bucket.put(an)


@app.route('/ping')
def ping():
    return 'OK', 200, {'Content-Type': 'text/plain'}


def cli(no_history=False):
    """Run in a CLI session. Stores history in DynamoDB by default; if no_history is True,
    history is not stored and each utterance is treated as a separate dialogue."""
    # start a CLI dialogue session (open empty history)
    print >> sys.stderr, 'CLI session started.'
    session_id = 'CLI-%s' % time.strftime('%Y-%m-%d--%H-%M-%S')
    print >> sys.stderr, session_id
    history.start_db_session(session_id, bucket_version=VERSION)
    nerdb.start_db_ner(session_id)

    while True:
        # read the question
        question = sys.stdin.readline()
        question = question.strip()
        if not question:
            break  # exit on empty line

        # add the question to session history (reset beforehand if we're ignoring history)
        history.update_db(session_id, 'user', question, clear=no_history)

        # create context for the Flask app to handle the question as if coming from Echo/Telegram
        ctx = '/?q=' + urllib2.quote(question) + '&sid=' + session_id + '&cs=1.0'
        with app.test_request_context(ctx):
            # handle the question here (trigger the main "answer" function, take just the reply)
            # answer()'s 1st returned value is a Request object which holds JSON-encoded info
            # in the data variable
            parts = json.loads(answer()[0].data)
            score = parts[0]
            reply_text = parts[1]
            bot = parts[2]

            news_id = None
            if len(parts) > 3:
                news_id = parts[3]
            # score, reply_text, bot = json.loads(answer()[0].data)

            # print out the result
            print reply_text
            sys.stdout.flush()
            # store it in DB, if we're not ignoring history
            if not no_history:
                history.update_db(session_id, 'system', reply_text, score=score, bot=bot)
                if news_id:
                    history.update_news_id(session_id, news_id)

        if reply_text == 'STOP_INTENT_REQUESTED':
            break


if __name__ == '__main__':
    ap = ArgumentParser(description='Bot bucket')
    ap.add_argument('-s', '--server', action='store_true',
                    help='Run as a (multi-threaded) server.')
    ap.add_argument('-p', '--port', type=int, default=5000,
                    help='Port on which to run as a server (default: 5000)')
    ap.add_argument('-n', '--no-history', action='store_true',
                    help='For CLI sessions only: suppress history and treat each ' +
                    'utterance individually.')
    args = ap.parse_args()

    if args.server:
        app.run(host='0.0.0.0', port=args.port, threaded=True)
    else:
        cli(args.no_history)
