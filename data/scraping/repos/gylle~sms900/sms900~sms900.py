""" The main bot module for sms900 """
from collections import deque
import json
import logging
import queue
import re
import traceback
import uuid
from os import mkdir, path

import sqlite3

import twilio
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException

from sms900.phonebook import PhoneBook, SMS900InvalidAddressbookEntry
from sms900.ircthread import IRCThread
from sms900.http_interface import HTTPThread
from sms900.indexer import Indexer
from sms900.openai import OpenAI


class SMS900InvalidNumberFormatException(Exception):
    """ Simple exception to use for invalid number inputs"""
    pass


class SMS900():
    """ The main class to use """
    def __init__(self, configuration_path):
        """ The init method for the main class.
        Should probably add an example or two here.
        """
        self.configuration_path = configuration_path
        self.events = queue.Queue()
        self.config = None
        self.dbconn = None
        self.irc_thread = None
        self.pb = None
        self.openai = None
        self.openai_history = deque(maxlen=100)

    def run(self):
        """ Starts the main loop"""
        self._load_configuration()
        self._init_database()
        self.pb = PhoneBook(self.dbconn)
        self.indexer = Indexer()

        logging.info("Starting IRCThread thread")
        self.irc_thread = IRCThread(self,
                                    self.config['server'],
                                    self.config['server_port'],
                                    self.config['nickname'],
                                    self.config['channel'])
        self.irc_thread.start()

        try:
            if 'openai_api_key' in self.config:
                self.openai = OpenAI(self.config)
        except Exception as err:
            logging.info("Failed to initialize openai: %s", err)

        logging.info("Starting webserver")
        http_thread = HTTPThread(self, ('0.0.0.0', self.config['http_server_port']))
        http_thread.start()

        logging.info("Starting main loop")
        self._main_loop()

    def _load_configuration(self):
        with open(self.configuration_path, 'r') as file:
            self.config = json.load(file)

        # FIXME: Check that we got everything we'll be needing

    def _init_database(self):
        self.dbconn = sqlite3.connect('sms900.db', isolation_level=None)
        conn = self.dbconn.cursor()

        try:
            conn.execute(
                "create table phonebook ("
                "  id integer primary key,"
                "  nickname text UNIQUE,"
                "  number text UNIQUE"
                ")"
            )

            conn.execute(
                "create table phonebook_email ("
                "  email text primary key collate nocase,"
                "  nickname text"
                ")"
            )
        except sqlite3.Error as err:
            logging.info("Failed to create table(s): %s", err)

    def queue_event(self, event_type, data):
        """ queues event, stupid doc string i know """
        event = {'event_type': event_type}
        event.update(data)
        self.events.put(event)

    def on_privmsg_received(self, hostmask, channel, msg):
        nickname = self._get_nickname_from_hostmask(hostmask)

        if not msg.startswith('!'):
            self.openai_history.append({
                'nickname': nickname,
                'channel': channel,
                'msg': msg,
                'type': 'irc',
            })

            if self.config['nickname'] in msg:
                self.queue_event('TRIGGER_COMPLETION', {})

    def openai_set_prompt(self, prompt):
        self.openai.set_prompt(prompt)

    def openai_reset_history(self):
        self.openai_history.clear()

    def _main_loop(self):
        while True:
            event = self.events.get()
            self._handle_event(event)

    def _handle_event(self, event):
        try:
            logging.info('EVENT: %s', event)

            if event['event_type'] == 'SEND_SMS':
                sender_hm = event['hostmask']
                number = self._get_num_from_nick_or_num(event['number'])
                nickname = self._get_nickname_from_hostmask(sender_hm)
                # FIXME: Check the sender
                msg = "<%s> %s" % (nickname, event['msg'])

                self._send_sms(number, msg)
            elif event['event_type'] == 'ADD_PB_ENTRY':
                nickname = event['nickname']
                if event['number']:
                    number = self._get_canonicalized_number(event['number'])

                    self.pb.add_number(nickname, number)
                    self._send_privmsg(self.config['channel'],
                                       'Added %s with number %s' % (nickname, number))

                elif event['email']:
                    email = event['email']
                    self.pb.add_email(nickname, email)
                    self._send_privmsg(self.config['channel'],
                                       'Added email %s for %s' % (email, nickname))

            elif event['event_type'] == 'DEL_PB_ENTRY':
                if event['nickname']:
                    nickname = event['nickname']
                    oldnumber = self.pb.get_number(nickname)

                    self.pb.del_entry(nickname)
                    self._send_privmsg(self.config['channel'],
                                       'Removed contact %s (number: %s)' % (nickname, oldnumber))

                elif event['email']:
                    email = event['email']
                    nickname = self.pb.get_nickname_from_email(email)

                    self.pb.del_email(email)
                    self._send_privmsg(self.config['channel'],
                                       'Removed %s for contact %s' % (email, nickname))

            elif event['event_type'] == 'LOOKUP_CARRIER':
                number = event['number']
                number = self._get_canonicalized_number(number)
                self._lookup_carrier(number)
            elif event['event_type'] == 'REINDEX_ALL':
                self._reindex_all()
            elif event['event_type'] == 'SMS_RECEIVED':
                number = event['number']
                sms_msg = event['msg']
                try:
                    sender = self.pb.get_nickname(number)
                except SMS900InvalidAddressbookEntry:
                    sender = number

                msg = '<%s> %s' % (sender, sms_msg)
                self._send_privmsg(self.config['channel'], msg)

                self.openai_history.append({
                    'nickname': sender,
                    'channel': self.config['channel'],
                    'msg': sms_msg,
                    'type': 'sms',
                })

                if self.config['nickname'] in sms_msg:
                    self.queue_event('TRIGGER_COMPLETION', {})

            elif event['event_type'] == 'GITHUB_WEBHOOK':
                self._handle_github_event(event['data'])
            elif event['event_type'] == 'MAILGUN_INCOMING':
                self._handle_incoming_mms(event['data'])
            elif event['event_type'] == 'TRIGGER_COMPLETION':
                if self.openai:
                    context = self._openai_get_relevant_context(event)

                    response = self.openai.generate_response(
                        self.config['channel'],
                        self.config['nickname'],
                        context
                    )
                    if response:
                        self.openai_history.append({
                            'nickname': self.config['nickname'],
                            'channel': self.config['channel'],
                            'msg': response,
                            'type': 'irc',
                        })

                        self._openai_parse_response_commands(response)

                        self._send_privmsg(self.config['channel'], response)
                else:
                    logging.info("openai not configured")

        except (SMS900InvalidNumberFormatException,
                SMS900InvalidAddressbookEntry) as err:
            self._send_privmsg(self.config['channel'], "Error: %s" % err)
        except Exception as err:
            self._send_privmsg(self.config['channel'], "Unknown error: %s" %
                               err)
            traceback.print_exc()

    def _openai_get_relevant_context(self, data):
        default_limit = 20
        context = list(self.openai_history)

        if 'include_all_length' in data:
            limit = max(min(data['include_all_length'], default_limit), 1)
        else:
            limit = default_limit
            context = [x for x in context
                       if self.config['nickname'] in x['msg'] or x['nickname'] == self.config['nickname']]

        return context[-limit:]

    def _openai_parse_response_commands(self, response):
        m = re.findall(r'\|SMS:([^:]+):([^|]+)\|', response)
        for sms in m:
            self.queue_event('SEND_SMS', {
                'hostmask': 'sms900!fakehostmask',
                'number': sms[0],
                'msg': sms[1],
            })


    def _send_sms(self, number, message):
        logging.info('Sending sms ( %s -> %s )', message, number)

        try:
            client = Client(self.config['twilio_account_sid'],
                            self.config['twilio_auth_token'])

            message_data = client.messages.create(to=number,
                                                  from_=self.config['twilio_number'],
                                                  body=message,)

            self._send_privmsg(self.config['channel'],
                               "Sent %s sms to number %s"
                               % (message_data.num_segments, number))
        except TwilioRestException as err:
            self._send_privmsg(self.config['channel'],
                               "Failed to send sms: %s" % err)

    def _lookup_carrier(self, number):
        logging.info('Looking up number %s', number)

        try:
            client = Client(self.config['twilio_account_sid'],
                            self.config['twilio_auth_token'])

            number_data = client.lookups.v1.phone_numbers(number).fetch(type=['carrier'])

            self._send_privmsg(self.config['channel'],
                               '%s is %s, carrier: %s'
                               % (number,
                                  number_data.carrier['type'],
                                  number_data.carrier['name']))
        except TwilioRestException as err:
            self._send_privmsg(self.config['channel'],
                               "Failed to lookup number: %s" % err)

    def _send_privmsg(self, target, msg):
        self.irc_thread.send_privmsg(target, msg)

    def _get_canonicalized_number(self, number):
        match = re.match(r'^\+[0-9]+$', number)
        if match:
            logging.info('number %s already canonicalized, returning as is',
                         number)
            return number

        match = re.match(r'^0(7[0-9]{8})$', number)
        if match:
            new_number = '+46%s' % match.group(1)
            logging.info('number %s was canonicalized and returned as %s',
                         number,
                         new_number)
            return new_number

        raise SMS900InvalidNumberFormatException("%s is not a valid number" % number)

    def _get_num_from_nick_or_num(self, number_or_name):
        try:
            number_or_name = self._get_canonicalized_number(number_or_name)

        except SMS900InvalidNumberFormatException:
            try:
                number_or_name = self.pb.get_number(number_or_name)
            except SMS900InvalidAddressbookEntry as err2:
                raise SMS900InvalidNumberFormatException(
                    "%s is not a valid number or existing nickname: %s"
                    % (number_or_name, err2))

        return number_or_name

    def _get_nickname_from_hostmask(self, hostmask):
        nick = re.match(r'^([^\!]+)', hostmask)
        if not nick:
            # FIXME
            return hostmask
        else:
            return nick.group(1)

    def _handle_github_event(self, data):
        try:
            repository_name = data['repository']['full_name']

            for commit in data['commits']:
                self._send_privmsg(
                    self.config['channel'],
                    "[%s] %s (%s)" % (
                        repository_name,
                        commit['message'],
                        commit['author']['username']
                    )
                )

        except KeyError as err:
            logging.exception("Failed to parse data from github webhook, reason: %s", err)

    def _handle_incoming_mms(self, data):
        rel_path = str(uuid.uuid4())
        save_path = path.join(
            self.config['mms_save_path'],
            rel_path
        )

        mkdir(save_path)

        [sender, files] = self._parse_mms_data(data, save_path)

        sender = self._map_mms_sender_to_nickname(sender)

        base_url = "%s/%s" % (
            self.config['external_mms_url'],
            rel_path
        )

        mms_summary, summary_contains_all = self._get_mms_summary(base_url, files)
        if mms_summary:
            self._send_privmsg(
                self.config['channel'],
                "[MMS] <%s> %s" % (sender, mms_summary)
            )

        if not summary_contains_all:
            self._send_privmsg(
                self.config['channel'],
                "[MMS] <%s> Received %d file(s): %s" % (sender, len(files), base_url)
            )

        self.indexer.generate_local_index(save_path)
        self.indexer.generate_global_index(self.config['mms_save_path'])

    def _reindex_all(self):
        self.indexer.reindex_all(self.config['mms_save_path'])

    def _map_mms_sender_to_nickname(self, sender):
        m = re.match('^([^<]*<)?([^<]+@[^>]+)>?', sender)
        if not m:
            return sender

        email = m.group(2)

        try:
            return self.pb.get_nickname_from_email(email)
        except SMS900InvalidAddressbookEntry:
            logging.exception("No nickname found for email %s", email)

        m = re.match('^([0-9]+)\@', email)
        if m:
            try:
                number = self._get_canonicalized_number("+" + m.group(1))
                return self.pb.get_nickname(number)
            except SMS900InvalidAddressbookEntry:
                logging.exception("No number found for %s" , m.group(1))
            except SMS900InvalidNumberFormatException:
                logging.exception("Weirdly formatted number: %s", m.group(1))

        return email

    def _parse_mms_data(self, data, save_path):
        payload = data['payload']

        if data['type'] == 'form-data':
            return self._parse_form_mms_data(payload, save_path)
        elif data['type'] == 'urlencoded':
            return self._parse_urlencoded_mms_data(payload, save_path)
        else:
            return [None, []]

    def _parse_form_mms_data(self, payload, save_path):
        files = []
        sender = None
        i = 0

        for part in payload.parts:
            disposition = part.headers[b'Content-Disposition'].decode('utf-8', 'ignore')

            m = re.search('name="([^"]+)"', disposition, re.IGNORECASE)
            if m:
                disposition_name = m.group(1)

                if disposition_name == 'from':
                    sender = part.content.decode('utf-8', 'ignore')
                    continue

                elif disposition_name in ['body-plain', 'subject']:
                    contents = part.content.decode('utf-8', 'ignore')
                    if not len(contents.strip()):
                        continue

                    filename = path.join(save_path, '%d-%s.txt' % (i, disposition_name))
                    i += 1

                    with open(filename, 'w') as f:
                        f.write(contents)

                    files.append(filename)
                    continue

            m = re.search('filename="([^"]+)"', disposition, re.IGNORECASE)
            if m:
                filename = path.join(save_path, '%d-%s' % (i, m.group(1)))
                i += 1

                with open(filename, 'wb') as f:
                    f.write(part.content)

                files.append(filename)
                continue

            if b'Content-Type' in part.headers:
                # Probably something we need to handle better, but
                # let's just dump it in a file for now.
                filename = path.join(save_path, '%d-unknown' % i)
                i += 1

                with open(filename, 'wb') as f:
                    f.write(part.content)

                files.append(filename)
                continue

            logging.info("Ignoring: %s" % disposition)

        return [sender, files]

    def _parse_urlencoded_mms_data(self, payload, save_path):
        files = []
        sender = None
        i = 0

        if 'sender' in payload:
            for _sender in payload['sender']:
                sender = _sender
                break

        for text_type in ['body-plain', 'subject']:
            if text_type in payload:
                for body in payload[text_type]:
                    if not len(body.strip()):
                        continue

                    filename = path.join(save_path, '%d-%s.txt' % (i, text_type))
                    i += 1

                    with open(filename, 'w') as f:
                        f.write(body)

                    files.append(filename)

        return [sender, files]

    def _get_mms_summary(self, base_url, files):
        try:
            text = None
            text_was_cut = False
            img_url = None

            # Find the first text and the first image file, if any
            for full_path in files:
                match = re.search(r'\.([^.]+)$', full_path)
                if match:
                    extension = match.group(1).lower()

                    if not img_url:
                        if extension in ['jpg', 'jpeg', 'png']:
                            filename = path.basename(full_path)
                            img_url = "%s/%s" % (base_url, filename)

                    if not text:
                        if extension in ['txt']:
                            with open(full_path, 'r', encoding='utf-8',
                                      errors='ignore') as file:
                                text = file.read()

                            # Only show one line
                            text_lines = text.splitlines()
                            if len(text_lines):
                                text = text_lines[0]

                                if len(text_lines) > 1:
                                    text_was_cut = True
                                    text += " [%d lines]" % len(text_lines)

            if text or img_url:
                parts = [p for p in [text, img_url] if p]
                message = ", ".join(parts)
                summary_contains_all = (len(parts) == len(files)) and not text_was_cut
                return message, summary_contains_all

        except Exception:
            traceback.print_exc()

        return None, False
