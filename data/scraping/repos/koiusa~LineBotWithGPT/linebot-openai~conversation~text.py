import openai
from database.channel import channel
from database.histoly import histoly
from linebot.models import (TextSendMessage)
from common.context import eventcontext


class textresponce:
    event_context = None
    channel = None
    histoly = None
    current = None
    targets = None

    def __init__(self, event_context: eventcontext):
        self.event_context = event_context
        self.channel = channel(self.event_context)
        self.histoly = histoly(self.event_context)

    def get_message(self):
        self.targets = self.channel.get_target_channels()
        # self.run_sync()
        msg = None

        self.current = self.channel.get_record()
        actionid = self.current.get("actionid")
        if actionid == 1:
            msg = self.run_prompt()
        elif actionid == 3:
            msg = self.run_delete()
        elif actionid == 4:
            msg = self.run_memory()
        elif actionid == 5:
            msg = self.run_deleteprompt()
        elif actionid == 6:
            msg = self.run_ignorechannel()
        elif actionid == 7:
            msg = self.run_aimchannel()
        else:
            msg = self.run_conversation()

        self.run_reset()
        return msg

    def is_userchannel(self):
        return self.current.get("type") == "user"

    def run_sync(self):
        self.channel.sync()
        for target in self.targets:
            self.channel.sync_ref(target.get("channelId"))

    def run_reset(self):
        if not self.is_userchannel():
            self.channel.reset()
        else:
            for target in self.targets:
                self.channel.reset_ref(target.get("channelId"))

    def run_delete(self):
        text = self.event_context.line_event.message.text
        msg = None
        if text.lower() == "y":
            if not self.is_userchannel():
                self.histoly.delete_histoly()
            else:
                for target in self.targets:
                    self.histoly.delete_histoly_ref(target.get("channelId"))
            msg = "削除完了しました"
        else:
            msg = "削除キャンセルしました"
        return msg

    def run_prompt(self):
        text = self.event_context.line_event.message.text
        if not self.is_userchannel():
            self.channel.add_prompt(text)
        else:
            for target in self.targets:
                self.channel.add_prompt_ref(target.get("channelId"), text)
        msg = "AIの役割を設定しました"
        return msg

    def run_deleteprompt(self):
        text = self.event_context.line_event.message.text
        msg = None
        if text.lower() == "y":
            if not self.is_userchannel():
                self.channel.add_prompt(None)
            else:
                for target in self.targets:
                    self.channel.add_prompt_ref(target.get("channelId"), None)
            msg = "AIの役割を削除しました"
        else:
            msg = "削除キャンセルしました"
        return msg

    def run_memory(self):
        text = self.event_context.line_event.message.text

        if text.isdecimal():
            num = int(text)
            if num > 10:
                num = 10
            if not self.is_userchannel():
                self.channel.add_memory(num)
            else:
                for target in self.targets:
                    self.channel.add_memory_ref(target.get("channelId"), num)
            msg = "記憶数を[{}]に設定しました".format(num)
        else:
            msg = "設定キャンセルしました"
        return msg

    def run_ignorechannel(self):
        text = self.event_context.line_event.message.text
        if text.isdecimal():
            num = int(text)
            channels = self.channel.get_channels()
            if len(channels) > num:
                self.channel.add_setting(channels[num].get('channelId'), False)
                # self.event_context.line_bot_api.push_message(
                #     channels[num].get('channelId'), TextSendMessage(text="[ユーザー：{}]の操作対象から削除されました。".format(
                #         channels[num].get('userId'))))
                msg = "[{}]を操作対象から削除しました。".format(num)
            else:
                msg = "削除キャンセルしました"
        else:
            msg = "削除キャンセルしました"
        return msg

    def run_aimchannel(self):
        text = self.event_context.line_event.message.text
        if text.isdecimal():
            num = int(text)
            channels = self.channel.get_channels()
            if len(channels) > num:
                self.channel.add_setting(channels[num].get('channelId'), True)
                # self.event_context.line_bot_api.push_message(
                #     channels[num].get('channelId'), TextSendMessage(text="[ユーザー：{}]の操作対象に設定されました。".format(
                #         channels[num].get('userId'))))
                msg = "[{}]を操作対象に設定しました。".format(num)
            else:
                msg = "設定キャンセルしました"
        else:
            msg = "設定キャンセルしました"
        return msg

    def run_conversation(self):
        text = self.event_context.line_event.message.text
        userid = self.event_context.line_event.source.user_id

        self.histoly.add_histoly(userid, text)

        conversation = self.histoly.get_histoly(self.current.get("memory"))
        prompt = self.histoly.to_prompt(
            conversation, self.current.get("prompt"))
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=prompt
        )
        # 受信したテキストをCloudWatchLogsに出力する
        print(completion.choices[0].message.content)
        msg = completion.choices[0].message.content.lstrip()

        self.histoly.add_histoly("bot", msg)
        return msg
