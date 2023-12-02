
import asyncio
import datetime
from actions.game.gamemodehandler import GameModeHandler
from actions.game.competition.competitionmodehandler import CompetitionModeHandler
from actions.session.sessionhandler import SessionHandler
from actions.timestamps.timestamphandler import TimestampHandler
from actions.common.common import get_credentials, get_requested_slot, get_random_person, get_dp_inmemory_db, ben_is_typing, ask_openai, get_countdown_value, ben_is_typing_2, setup_logging
from actions.achievements.achievementshandler import AchievementHandler
import logging
logger = setup_logging()


class GoalHandler:
    '''
    Goalhandler to set goal
    '''
    async def set_competive_goal(self, slot_value, tracker, dispatcher):
        try:
            timestamp_handler = TimestampHandler()
            competition_mode_handler = CompetitionModeHandler()
            session_handler = SessionHandler()
            # get infos
            timestamp, loop, quest_id, _ = await timestamp_handler.get_timestamp(tracker.sender_id, 'waiting')
            countdown = get_countdown_value(quest_id, loop)
            now = datetime.datetime.now().timestamp()
            requested_slot = get_requested_slot(tracker)

            # group gives goal
            if requested_slot and 'competive_goal' in requested_slot and "#" in slot_value and now >= timestamp + countdown:
                await competition_mode_handler.telegram_bot_send_message('text', tracker.sender_id, "Ich prüfe einmal eurer Ziel 🔍👀...")

                await ben_is_typing_2(tracker.get_slot('countdown'), competition_mode_handler)
                # openai evaluates and redefines the passed goal from user
                role = "Du bist ein Experte, wenn es um Ziele geht. Du befolgst die Zielsetzung anhand der SMART-Regel.\
                Das Szenario ist ein Quiz-Spiel über Inhalte der Einführung in die Wirtschaftsinformatik, wobei der Fokus auf der gemeinsamen Erarbeitung der Lösung geht.\
                Das Quiz-Spiel beinhaltet insgesamt 6 Fragen (2x Single-Choice, 2x Multiple-Choide und 2x Offene Fragen).\
                Die Spieler haben für jede Frage einne bestimmten Zeitraum, um sich zu besprechen. Der Zeitraum ist abhängig von der Frageart und kann zwischen 60 und 100 Sekunden betragen.\
                Nach Ablauf der Zeit, muss ein Spieler die Antwort geben. Das Team tritt dabei gegen ein anderes Team an.\
                Verbessere das kommende Ziel und begründe kurz warum du es so änderst. Antworte nur mit einem Satz."
                goal = ask_openai(role, slot_value)
                random_user = get_random_person(tracker.get_slot("my_group"))

                btn_lst = [
                    {"title": "Ziel gesetzt🎯\nLass uns anfangen. 🚀",
                        "payload": '/set_goal{"goal":"set_goal"}'}
                ]

                dispatcher.utter_message(text="Ich habe mir erlaubt euer Ziel zu verbessern. Mein Vorschlag ist:\n%s\n\n👋 %s bitte drücke den Button, um das Spiel zu starten." % (goal, random_user['username']), buttons=btn_lst)
                return {'KLMK_competive_goal': None, "goal": goal}

            # group aggree with the redefined goal from openai
            elif "set_goal" in slot_value:
                filter = session_handler.get_session_filter(tracker)
                achievement_handler = AchievementHandler()
                achievement = 'GOAL'
                # group get badge
                if await achievement_handler.insert_achievement(filter, achievement):
                    badges = get_dp_inmemory_db("./badges.json")
                    await competition_mode_handler.telegram_bot_send_message('photo', tracker.sender_id, badges[achievement])
                dispatcher.utter_message(response="utter_waiting_of_opponent")
                return {"KLMK_competive_goal": slot_value, "random_person": None, "flag": None,  "countdown": None, "activated_reminder_comp": None}

            elif now < timestamp + countdown:
                print("Before Submitting: KLMK_competive_goal")
                return {'KLMK_competive_goal': None}
            else:
                return {'KLMK_competive_goal': slot_value}
        except Exception as e:
            logger.exception(e)

    async def set_non_competive_goal(self, slot_value, tracker, dispatcher):
        try:
            timestamp_handler = TimestampHandler()
            game_mode_handler = GameModeHandler()
            session_handler = SessionHandler()
            # get infos
            timestamp, loop, quest_id, _ = await timestamp_handler.get_timestamp(tracker.sender_id, 'waiting')
            countdown = get_countdown_value(quest_id, loop)
            now = datetime.datetime.now().timestamp()
            requested_slot = get_requested_slot(tracker)

            # user/group gives goal
            if requested_slot and now >= timestamp + countdown and 'non_competive_goal' in requested_slot and "#" in slot_value:
                await game_mode_handler.telegram_bot_send_message('text', tracker.sender_id, "Ich prüfe einmal eurer Ziel 🔍👀...")
                await ben_is_typing(tracker.get_slot('countdown'), game_mode_handler)

                # openai evaluates and redefines the passed goal from user
                role = "Du bist ein Experte, wenn es um Ziele geht. Du befolgst die Zielsetzung anhand der SMART-Regel. Das Szenario ist ein Quiz-Spiel über Inhalte der Einführung in die Wirtschaftsinformatik, wobei der Fokus auf der gemeinsamen Erarbeitung der Lösung geht. Das Quiz-Spiel beinhaltet insgesamt 6 Fragen (2x Single-Choice, 2x Multiple-Choide und 2x Offene Fragen). Die Spieler haben für jede Frage einen bestimmten Zeitraum, um sich zu besprechen. Der Zeitraum ist abhängig von der Frageart und kann zwischen 60 und 100 Sekunden betragen. Nach Ablauf der Zeit, muss ein Spieler die Antwort geben. Verbessere das kommende Ziel und begründe kurz warum du es so änderst. Antworte nur mit einem Satz."
                goal = ask_openai(role, slot_value)
                btn_lst = [
                    {"title": "Ziel gesetzt🎯\nLass uns anfangen. 🚀",
                        "payload": '/set_goal{"goal":"set_goal"}'}
                ]
                if loop == "quiz_form_OKK":
                    # just user mode
                    dispatcher.utter_message(
                        text="Ich habe mir erlaubt dein Ziel zu verbessern. Mein Vorschlag ist:\n%s\n👋 bitte drücke den Button, um das Spiel zu starten" % (goal), buttons=btn_lst)
                else:
                    # group mode
                    random_user = get_random_person(
                        tracker.get_slot("my_group"))
                    dispatcher.utter_message(text="Ich habe mir erlaubt euer Ziel zu verbessern. Mein Vorschlag ist:\n%s\n\n👋 %s bitte drücke den Button, um das Spiel zu starten." % (
                        goal, random_user['username']), buttons=btn_lst)
                return {'KL_non_competive_goal': None, "goal": goal}
            elif "set_goal" in slot_value:
                # user/ group affirm redefined goal
                filter = session_handler.get_session_filter(tracker)
                await game_mode_handler.set_goal_and_status('non_competive_goal', tracker.get_slot('goal'), 'evaluated', filter,  True)
                achievement_handler = AchievementHandler()
                # badge
                achievement = 'GOAL'
                if await achievement_handler.insert_achievement(filter, achievement):
                    badges = get_dp_inmemory_db("./badges.json")
                    await game_mode_handler.telegram_bot_send_message('photo', tracker.sender_id, badges[achievement])
                if tracker.get_slot('countdown'):
                    await ben_is_typing_2(tracker.get_slot('countdown'), game_mode_handler)
                return {'KL_non_competive_goal': 'set_goal', "random_person": None, "flag": None,  "countdown": None}
            else:
                return {'KL_non_competive_goal': None}
        except Exception as e:
            logger.exception(e)
