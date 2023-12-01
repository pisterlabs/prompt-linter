#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: i2cy(i2cy@outlook.com)
# Project: CodyBot2
# Filename: addons
# Created on: 2023/2/16

import time
import json
import asyncio
from nonebot import get_bot, logger
from nonebot.adapters.onebot.v11 import Bot, MessageSegment, Message
from .session import SessionGPT35
from .api import get_chat_response
from .config import APIKEY_LIST
from .memory import Memory, ExtraTypes
from .utils import TimeStamp, GPTResponse, extract_json_and_purge_cody_response
from . import get_user_session, get_group_session


class AddonBase:

    def __init__(self, session: SessionGPT35):
        """
        basic addon object parent
        :param session: SessionGPT35
        """

        self.addon_name = "AddonBase"
        self.session = session
        self.bot = self.session.bot

    def log(self, log_message: str):
        """
        add one line of log information to system logger
        :param log_message: str
        :return:
        """
        self.session.log(f'[{self.addon_name}] {log_message}')

    def alarm_callback(self, *args, **kwargs):
        """
        basic alarm callback function. calls when registered alarm in session triggered.
        :param args: Any
        :param kwargs: Any
        :return: Any
        """
        pass

    def user_msg_post_proc_callback(self):
        """
        basic user message post-processing callback method. calls when a user send message to Cody
        :return: None
        """
        pass

    def cody_msg_post_proc_callback(self):
        """
        basic cody message post-processing callback method. calls when received a new feedback from gpt
        :return: None
        """
        pass


class DefaultsAddon(AddonBase):
    """
    this addon contains following features by default:
    1. user's name updating
    2. user's name deletion
    3. user's impression updating
    4. user's impression data maintaining
    5. address book update (in status section)
    6. reach anyone in impression database
    """

    def is_silenced(self, user_id: int) -> bool:
        """
        return whether if selected user is silenced in current session
        :param user_id: int
        :return:
        """
        # get user impression data frame
        frame = self.session.impression.get_individual(user_id)

        # get result
        ret = False
        flag_name = f'SI_{self.session.id}'
        if flag_name in frame.additional_json:
            ret = frame.additional_json[flag_name]

        return ret

    def set_silenced(self, user_id: int):
        """
        set a user conversation status to silenced in current session
        :param user_id: int
        :return:
        """
        # get user impression data frame
        frame = self.session.impression.get_individual(user_id)

        flag_name = f'SI_{self.session.id}'
        # update flag
        frame.additional_json.update({flag_name: True})

        # write modified json dict in impression database
        self.session.impression.update_individual(
            user_id,
            additional_json=frame.additional_json
        )

    def set_feedback_required(self, activate: bool, user_id: int, feedback_source: SessionGPT35, ts: float,
                              feecback_topic: str):
        """
        set private session of specified user feedback required active or disabled
        :param feecback_topic: str
        :param ts: float, timestamp
        :param activate: bool
        :param user_id: int
        :param feedback_source: SessionGPT35
        :return:
        """
        # get user impression data frame
        frame = self.session.impression.get_individual(user_id)

        flag_name = f'FBR_{self.session.id}'
        # update flag
        frame.additional_json.update({flag_name: {'source': feedback_source.id,
                                                  'is_group': feedback_source.is_group,
                                                  'active': activate,
                                                  'timestamp': ts,
                                                  'topic': feecback_topic}})

    def set_active(self, user_id: int):
        """
        set a user conversation status to active which means 'not silenced' in current session
        :param user_id: int
        :return:
        """
        # get user impression data frame
        frame = self.session.impression.get_individual(user_id)

        flag_name = f'SI_{self.session.id}'
        # update flag
        frame.additional_json.update({flag_name: False})

        # write modified json dict in impression database
        self.session.impression.update_individual(
            user_id,
            additional_json=frame.additional_json
        )

    def extract_json_from_cody_response(self) -> dict or None:
        """
        extract json dict from cody's message
        :return: dict
        """
        # copy message content from session.conversation
        msg = self.session.conversation.cody_msg

        json_text, purged = extract_json_and_purge_cody_response(msg)

        # decode and transfer
        ret = None
        if len(json_text):
            try:
                ret = json.loads(json_text)
            except Exception as err:
                # log error when failed to decode json and return None
                self.log("[ERROR] failed to decode json text from Cody's response, {}, json text: {}".format(
                    err, json_text
                ))

        return ret

    def extract_user_id_with_no_impression_description(self) -> list:
        """
        extract all ID of users in conversation
        :return: list(int user_ID)
        """
        users = []
        users_with_impression = []
        # iterate conversations for user ID and impressions
        for i, ele in enumerate(self.session.conversation.conversation_extra):
            if ele['type'] == ExtraTypes.user_msg_info:
                if ele['user_id'] not in users:
                    # if users not recorded, append it in record
                    users.append(ele['user_id'])

                # decode user info json
                user_info = json.loads(self.session.conversation.conversation[i]['content'])

                if "previous impression" in user_info:
                    # append user with impression in conversation
                    users_with_impression.append(ele['user_id'])

        # find out users without impression
        ret = [ele for ele in users if ele not in users_with_impression]

        return ret

    def extract_latest_msg_segment_to_summary(self, time_sec: float = 300, max_msg_pair_count: int = 10) -> str:
        """
        extract latest message pairs
        :param time_sec: float
        :param max_msg_pair_count: int
        :return: str
        """
        now = time.time()
        ret = ""
        seg_count = 0
        for i, ele in enumerate(self.session.conversation.conversation_extra[::-1]):
            if now - ele['timestamp'] > time_sec:
                # break if message timestamp too old
                break

            if ele['type'] == ExtraTypes.user_msg:
                # in case message type is user's input
                ret += f"{ele['name']}: {self.session.conversation.conversation[i]}\n"

            elif ele['type'] == ExtraTypes.cody_msg:
                # in case message type is cody's feedback
                ret += f"Cody: {extract_json_and_purge_cody_response(self.session.conversation.conversation[i])[1]}"

            if seg_count >= max_msg_pair_count * 2:
                # break if messages exceeded maximum size
                break

    def user_msg_post_proc_callback(self):
        """
        update impression, interaction timestamp data and ensure it is in conversation
        :return: None
        """
        # update status massage
        all_users = self.session.impression.list_individuals()
        # forming information in CSV format
        info = []
        for user_id in all_users:
            # fetch impression frame
            frame = self.session.impression.get_individual(user_id)
            ts = frame.last_interact_timestamp

            # calculate duration since last contact
            ts_now = time.time()
            duration = ts_now - frame.last_interact_timestamp.timestamp

            if duration > 63072000:
                # if the user is never contacted (or no cantact in 2 years), skip
                continue

            # forming location text
            if frame.last_interact_session_is_group:
                loc = "group chat(group ID: {})".format(frame.last_interact_session_ID)
            else:
                loc = "private chat"

            # forming time text
            time_till_now = frame.last_interact_timestamp.till_now_str()

            # assembling last contact text
            if frame.last_interact_session_ID == -1:
                last_contact = "no record"
            else:
                last_contact = "in {} {}".format(loc, time_till_now)

            info.append("\"{}\",{},\"{}\",\"{}\",\"{}\"".format(
                frame.name,  # user's name
                frame.id,  # user QQ ID
                frame.title,  # user's relationship to Cody
                last_contact,  # last contact text
                frame.impression.replace("\n", " ")  # impression text
            ))

        self.session.conversation.status_messages.update(
            {
                "address_book": "persons‘ information you know listed in CVS format as follows:\n"
                                "Name,User_ID,Relationship_to_You,Last_Contact,Your_Impression_to_Him/Her\n"
                                "{}".format("\n".join(info))
            }
        )

        # copy current user info
        current_user_id = self.session.conversation.user_msg_extra['user_id']
        current_user_name = self.session.conversation.user_msg_extra['name']

        # update user interaction time and location
        ts: TimeStamp = self.session.conversation.user_msg_extra['timestamp']
        session_id = self.session.id
        is_group = self.session.is_group
        self.session.impression.update_individual(
            current_user_id,
            last_interact_session_ID=session_id,
            last_interact_session_is_group=is_group,
            last_interact_timestamp=int(ts)
        )

        # update user conversation status for 'FBR'(feed back required) flag
        if not self.session.is_group:
            # update only if current session is not group session
            frame = self.session.impression.get_individual(self.session.id)
            # fetch all FBR flags
            all_FBR = [ele for ele in frame.additional_json.keys() if 'FBR_' in ele]

            for fbr in all_FBR:
                # process all FBR flags one-by-one
                is_FBR = {'active': False}
                flag_name = fbr
                if flag_name in frame.additional_json:
                    # fetch flag in impression database if flag exists
                    is_FBR = frame.additional_json[flag_name]

                if is_FBR['active']:
                    if time.time() - is_FBR['timestamp'] > 3600 * 24 * 2:
                        # remove FBR flag if user is not responding in 2 days
                        frame.additional_json.pop(flag_name)
                        self.session.impression.update_individual(self.session.id,
                                                                  additional_json=frame.additional_json)
                        continue

                    # check and update feedback in other session
                    duration = time.time() - is_FBR['timestamp']
                    feedback_summary = self.extract_latest_msg_segment_to_summary(duration, max_msg_pair_count=20)
                    if feedback_summary == "":
                        # skip if no summary
                        continue

                    # fetch target session
                    if is_FBR['is_group']:
                        # fetch target group session
                        session = get_group_session(is_FBR['source'])
                    else:
                        # fetch target user session
                        session = get_group_session(is_FBR['source'])

                    # check and toggle busy flag of target session
                    session.busy_check()
                    session.is_busy = True

                    # locating index of target session's memory recall info segment
                    info_i = -1
                    for i, ele in enumerate(session.conversation.conversation_extra):
                        if (ele['type'] == ExtraTypes.sys_msg
                                and 'sub_type' in ele and ele['sub_type'] == 'reach_feedback'):
                            # fetch feedback system info message segment
                            if ele['timestamp'] == is_FBR['timestamp']:
                                # record feedback target index
                                info_i = i
                                self.log("successfully located target feedback index ({}) in session {}".format(
                                    info_i, session.id
                                ))

                    if info_i == -1:
                        # when target feedback system info segment is not found (forgotten by system)
                        frame.additional_json.pop(flag_name)
                        self.session.impression.update_individual(self.session.id,
                                                                  additional_json=frame.additional_json)
                        session.is_busy = False
                        continue

                    # update target feedback system info
                    session.conversation.conversation[info_i] = (f"feedbacks from {current_user_name} after you reached"
                                                                 f" him:\n{feedback_summary}")

                    # using openai to determine whether if topic of remind is close
                    payload = [
                        {'role': 'system',
                         'content': "You will receive a sequence of conversation as follows, determine whether if the "
                                    "topic related to '{}' is ended, return in JSON format, e.g. {}, {}".format(
                             is_FBR['topic'], "'ended': 0", "'ended': 1")},
                        {'role': 'user',
                         'content': feedback_summary}
                    ]

                    # try to generate response from openai
                    for i in APIKEY_LIST:
                        try:
                            key = APIKEY_LIST.get_api()
                            feedback, status = get_chat_response(key, payload,
                                                                 temperature=0.7, presence_p=0.0, frequency_p=0.0)
                            if status:
                                # try to decode json text
                                feedback_json, purged = extract_json_and_purge_cody_response(feedback)
                                feedback_json = json.loads(feedback_json)
                                if feedback_json['ended'] == 1:
                                    # remove FBR flag
                                    frame.additional_json.pop(flag_name)
                                    self.session.impression.update_individual(self.session.id,
                                                                              additional_json=frame.additional_json)
                                    session.is_busy = False
                                    continue
                        except Exception as err:
                            self.log(
                                f"error while trying to get feedback status from openai, using key \"{key}\", {err}")

                    # release busy flag
                    session.is_busy = False

        # get users without impression
        no_imp_users = self.extract_user_id_with_no_impression_description()

        # prepare commanding prompts
        command = "Summarize your current impression of {} based on the previous conversation above and past " \
                  "impressions in second person and return text start with \"Your impression of {} is\". return:"

        # update impression data in impression database for every no impression user
        for user in no_imp_users:
            if self.is_silenced(user):
                # skip if user is silenced in this session
                continue
            # get username
            user_name = self.session.impression.get_individual(user).name
            # generate list of prompts for openai
            prompts = self.session.conversation.to_list()
            # modify prompts, add system message for commanding
            prompts.append(
                {
                    'role': 'system',
                    'content': command.format(user_name, user_name)
                }
            )

            feedback: GPTResponse = GPTResponse()
            status = False
            # generate impression text
            for i in APIKEY_LIST:
                feedback, status = get_chat_response(
                    APIKEY_LIST.get_api(),
                    prompts,
                    temperature=0.6,
                    frequency_p=0.1
                )
                if status:
                    # break with status=True when succeed
                    break

            if status:
                # update impression text if succeed
                self.session.impression.update_individual(
                    user,
                    impression=feedback.message
                )
                # set silenced flag
                self.set_silenced(user)

                # log message
                self.log("updated impression text of user {}({}), content: {}".format(
                    user_name, user, feedback.message
                ))
            else:
                # log error message
                self.log("[ERROR] failed to update impression text for user {}({}), failed to communicate "
                         "with OpenAI API.".format(user_name, user))

        # update impression description in conversation
        if current_user_id in no_imp_users or len(self.session.conversation) == 0:
            self.set_active(current_user_id)
            # update impression description in user_msg_info
            imp = self.session.impression.get_individual(current_user_id).impression
            self.session.conversation.user_msg_info.update(
                {
                    "previous impression": imp
                }
            )

    def cody_msg_post_proc_callback(self):
        """
        decode emotion feelings, name update, reach someone
        :return: None
        """
        # try to get json text from
        res = self.extract_json_from_cody_response()

        if res is None:
            # if no json message decoded, skip
            return

        # get essential information of conversation
        is_group = self.session.is_group
        user_id = self.session.conversation.cody_msg_extra['user_id']
        timestamp = self.session.conversation.cody_msg_extra['timestamp']
        group_id = self.session.id

        # decode keywords
        for key in res:
            if key == "feeling":
                # feeling process
                # TODO: add feeling processing system
                pass
            elif key == "add_name":
                # update name in impression database
                old_frame = self.session.impression.get_individual(user_id)

                if res[key].upper in [old_frame.name, *old_frame.alternatives]:
                    # skip if the name already exists
                    continue

                if "UNKNOWN" in old_frame.name.upper():
                    # if user's name is unknown, replace it with current updated name
                    self.session.impression.update_individual(user_id, name=res[key])
                else:
                    # else alter the old name to alternatives
                    self.session.impression.update_individual(
                        user_id,
                        name=res[key],
                        alternatives=old_frame.alternatives.append(old_frame.name)
                    )
            elif key == "del_name":
                # delete a name from impression database
                old_frame = self.session.impression.get_individual(user_id)

                if res[key] == old_frame.name:
                    # deleting current default username, and replace it with alternatives
                    self.session.impression.update_individual(
                        user_id,
                        name=old_frame.alternatives[0],
                        alternatives=old_frame.alternatives[1:]
                    )
                elif res[key] in old_frame.alternatives:
                    # deleting alternative names
                    old_frame.alternatives.remove(res[key])
                    self.session.impression.update_individual(
                        user_id,
                        alternatives=old_frame.alternatives
                    )

            elif key == "reach":
                # reach for someone else in impression database
                if "reach_reason" not in res:
                    # skip reach command without reach_reason keyword
                    continue
                # preprocessing username, reason, latest 10 messages in 10 min
                username = res['reach'].upper()
                reach_reason = res['reach_reason']
                addtional_msg = self.extract_latest_msg_segment_to_summary()

                matched_frames = []
                # matching database
                for ele in self.session.impression.list_individuals():
                    frame = self.session.impression.get_individual(ele)
                    if frame.name.upper() == username and self.session.id != frame.id:
                        matched_frames.append(frame)

                if len(matched_frames) == 1:
                    # matched one, get target session
                    session = get_user_session(matched_frames[0].id)

                    # busy check target session
                    session.busy_check()
                    session.is_busy = True

                    # modify target session
                    session.conversation.conversation.append({'role': 'system',
                                                              'content': f"You need to form a message for "
                                                                         f"\"{username}\" with following reason:\n"
                                                                         f"{reach_reason}\n"
                                                                         f"additional information from other chat "
                                                                         f"session as follows:\n{addtional_msg}"
                                                              })
                    session.conversation.conversation_extra.append({
                        'type': ExtraTypes.sys_msg,
                        'sub_type': 'reach_call',
                        'timestamp': timestamp
                    })

                    # add feedback info system message segment in current session
                    self.session.conversation.conversation_extra.append({
                        'type': ExtraTypes.sys_msg,
                        'sub_type': 'reach_feedback',
                        'timestamp': timestamp
                    })
                    self.session.conversation.conversation.append({
                        'role': 'system',
                        'content': f'feedbacks from {username} after you reached him:\nNone'
                    })

                    status = False
                    # try to generate response from openai
                    for i in APIKEY_LIST:
                        try:
                            key = APIKEY_LIST.get_api()
                            feedback, status = get_chat_response(key, session.conversation.to_list())
                            if status:
                                # add to conversation history if succeed
                                session.conversation.add_cody_message(feedback)
                        except Exception as err:
                            self.log(f"error while using key \"{key}\", {err}")

                    # notify target user through api
                    if status:
                        # notify user if succeed
                        try:
                            json_text, purged = extract_json_and_purge_cody_response(feedback)
                            async_object = self.session.bot.send_private_msg(user_id=matched_frames[0].id,
                                                                             message=purged)
                            asyncio.run(async_object)

                            # set FBR flag active in target user impression database
                            self.set_feedback_required(True, matched_frames[0].id, self.session,
                                                       ts=timestamp.timestamp, feecback_topic=reach_reason)
                        except Exception as err:
                            self.log(f"error while sending message though nonebot API, {err}")

                    # release busy flag of target session
                    session.is_busy = False

                elif len(matched_frames) > 1:
                    # matched multiple, ask cody to contact which
                    # TODO: finish reach processing when multiple user matched in database, basic logic:
                    #  1. modify current session conversation(add system prompts of listed matched users), add busy flag
                    #  2. generate list to get Cody's structured response
                    #  3. if Cody is not sure, ask user to contact which
                    #  4. get target session or cancel
                    pass


# TODO: reconstruct ReminderAddon to fit new addon base

class ReminderAddon(AddonBase):

    def __init__(self, session_class):
        addon_text = "Cody will remember a schedule when {} said, and never remember a schedule that has existed in " \
                     "Cody's memory. Cody can only use programmatic command formatted like \"[SC|<ADD/EDIT/REMOVE>" \
                     "|<unique integer schedule number>|<time of schedule>|<full description of event without " \
                     "subject>^#]\" to remember, edit or remove a schedule, time format like %Y-%m-%d %H:%M:%S. " \
                     "Conversation example: \"Human: 嗨。;Cody: 你好呀。;Human: 我明天早上9点有个会，帮我记一下。;Cody: [SC|ADD|1|2019-08-20 " \
                     "9:00:00|attend a meeting^#]好的，已经记下啦。\". Cody will always use plain text when quoting instead of " \
                     "programmatic command format. Never use programmatic command format when retelling or quoting."

        if session_class.is_group:
            super().__init__(session_class, "", name_format_count=0, priority=2)
        else:
            super().__init__(session_class, addon_text, name_format_count=3, priority=2)

        self.reminders = {}
        self.alarm_id_header = "cody_reminder"

    def update_status_callback(self, status_text: str) -> str:
        if not self.session.is_group:
            reminder_sequence = ""
            reminder_ids = list(self.reminders.keys())
            reminder_ids.sort()
            if not len(reminder_ids):
                reminder_sequence = "None"
            else:
                for id in reminder_ids:
                    content = self.reminders[id]
                    alarm_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(content['alarm']))
                    reminder_sequence += f"ID:{id},Deadline:{alarm_time},Event:{content['text']}; "

            status_text += "\n"
            status_text += "(All schedules for {} in Cody's memory, and Cody will never use programmatic command " \
                           "to remember these again: {})".format(self.session.name, reminder_sequence)

        return status_text

    async def action_retell(self, reminder_id: int):
        reminder = self.reminders[reminder_id]
        preset = self.session.static_preset
        preset = self.session.generate_preset_with_addons(preset)
        status_header = self.session.generate_status_text_for_chat()
        alarm_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(reminder['alarm']))
        mixed_time_header = self.session.generate_time_header_for_chat(
            "Cody need to remind {} gently in plain text for schedule with ID:{},Time:{},Event:{}. And Cody will no "
            "longer need to remember this schedule.".format(
                self.session.name, reminder_id, alarm_time, reminder['text'])
        )

        self.session.check_and_forget_conversations()

        token_len, prompt = self.session.generate_prompts(preset, status_header, mixed_time_header, CODY_HEADER, '')

        status, res, warning_text = await self.session.generate_GPT3_feedback(
            prompt, self.session.name, f"{mixed_time_header}{CODY_HEADER}")

        if status:
            feedback = res
            self.reminders.pop(reminder_id)
        else:
            feedback = "[Cody正尝试提醒您的一个计划日程，但出于某种原因失败了]"

        event = FriendMessage.parse_obj(
            {
                'self_id': 0,
                'type': 0,
                'messageChain': '',
                'sender': {
                    'id': self.session.session_id,
                    'nickname': self.session.name,
                    'remark': self.session.name
                }
            }
        )

        await get_bot().send(event, feedback)

    def update_response_callback(self, resp: str) -> str:

        if self.session.is_group:
            return resp

        def convert_time(time_str) -> int:
            timeArray = time.strptime(time_str, "%Y-%m-%d %H:%M:%S")
            ts = int(time.mktime(timeArray))
            return ts

        add_count = 0
        edit_count = 0
        remove_count = 0
        failure_count = 0

        while True:
            start_ost = resp.find("SC|")
            if start_ost != -1:
                end_ost = resp[start_ost:].find("^#") + start_ost + 2
                cmd_text = resp[start_ost:end_ost]
                logger.debug("Reminder Command Detected: {}".format(cmd_text))

                try:
                    cmd = cmd_text[:-2].split("|")
                    action = cmd[1].upper()
                    reminder_id = int(cmd[2])

                    if "ADD" in action:
                        reminder_ts = convert_time(cmd[3])
                        self.reminders.update({reminder_id: {"alarm": reminder_ts,
                                                             "text": cmd[4]}})
                        add_count += 1
                        self.session.registered_alarms.update(
                            {
                                f"{self.alarm_id_header}_{reminder_id}": [
                                    reminder_ts,  # 定时任务时间戳
                                    self.action_retell,  # 定时任务回调函数
                                    (reminder_id,)  # 定时任务回调函数参数
                                ]
                            }
                        )

                    elif "EDIT" in action:
                        reminder_ts = convert_time(cmd[3])
                        self.reminders.update({reminder_id: {"alarm": reminder_ts,
                                                             "text": cmd[4]}})
                        edit_count += 1
                        self.session.registered_alarms.update(
                            {
                                f"{self.alarm_id_header}_{reminder_id}": [
                                    reminder_ts,  # 定时任务时间戳
                                    self.action_retell,  # 定时任务回调函数
                                    (reminder_id,)  # 定时任务回调函数参数
                                ]
                            }
                        )

                    elif "REMOVE" in action:
                        self.reminders.pop(reminder_id)
                        remove_count += 1
                        self.session.registered_alarms.pop(
                            f"{self.alarm_id_header}_{reminder_id}"
                        )

                    resp_f = resp[:start_ost]
                    if resp_f[-1] in ("[", "【"):
                        resp_f = resp_f[:-1]
                    resp_b = resp[end_ost:]
                    if resp_b[0] in ("]", "】"):
                        resp_b = resp_b[1:]

                    resp = resp_f + resp_b
                except Exception as err:
                    logger.error("Failed when processing reminder command, {}".format(err))
                    failure_count += 1
                    break
            else:
                break

        if add_count:
            resp += " [添加了{}项日程]".format(add_count)
        if edit_count:
            resp += " [编辑了{}项日程]".format(edit_count)
        if remove_count:
            resp += " [移除了{}项日程]".format(remove_count)
        if failure_count:
            resp += " [{}个日程操作失败]".format(failure_count)

        return resp
