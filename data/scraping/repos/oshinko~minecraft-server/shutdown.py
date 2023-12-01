import datetime
import json
import os
import traceback
from typing import Annotated

from .. import core


class Execute:
    idle_timeout = datetime.timedelta(hours=1)

    def __init__(self, rcon, log_file, do_webhook=core.do_webhook,
                 shutdown=core.shutdown):
        self.rcon = rcon
        self.log_file = log_file
        self.do_webhook = do_webhook
        self.shutdown = shutdown

    def now(self):
        return datetime.datetime.now()

    def stop_and_shutdown(self):
        with self.rcon.connect() as conn:
            conn.stop()

        self.shutdown()

    def __call__(self):
        raise NotImplementedError


class StrictExecute(Execute):
    def __call__(self):
        with self.rcon.connect() as conn:
            if conn.list():
                return

        specified_lines = []
        feature_line_types = []
        feature_line_dts = []

        with self.log_file.open() as log:
            for log_type, m, *captured in log.parse():  # 一行ずつ解析
                if log_type is core.LogType.UNSPECIFIED:
                    continue

                t, *_ = captured
                dt = datetime.datetime.combine(log.date, t)

                if log_type in (
                    core.LogType.SERVER_STARTING,
                    core.LogType.PLAYER_ENTERED,
                    core.LogType.PLAYER_EXITED
                ):
                    feature_line_types.append(log_type)
                    feature_line_dts.append(dt)

                specified_lines.append((log_type, dt))

        if feature_line_types[-1] is core.LogType.PLAYER_EXITED:
            # 直近が EXITED
            if self.now() - feature_line_dts[-1] >= self.idle_timeout:
                # 十分な時間が経っていたら停止
                self.do_webhook(f'最後のプレイヤー切断から十分な時間 ({self.idle_timeout}) '
                                'が経っているので停止します。')
                self.stop_and_shutdown()
        elif feature_line_types[-1] is core.LogType.PLAYER_ENTERED:
            # 直近が ENTERED
            if self.now() - feature_line_dts[-1] >= self.idle_timeout:
                # 十分な時間が経っていたら停止
                self.do_webhook(f'最後のログエントリから十分な時間 ({self.idle_timeout}) '
                                'が経っているので停止します。')
                self.stop_and_shutdown()
            else:  # 十分な時間が経っていない
                ...  # 直前の rcon.list() 以降にログインした僅かな可能性を考慮し、何もしない


class OpenAIExecute(Execute):
    enable = bool(os.environ.get('OPENAI_API_KEY'))
    model = os.environ.get('OPENAI_MODEL', 'gpt-4-0613')

    @property
    def token_encoding(self):
        if (not self.model.startswith('gpt-3.5') and
                not self.model.startswith('gpt-4')):
            return 'p50k_base'

        return 'cl100k_base'

    @property
    def max_tokens(self):
        max_tokens = os.environ.get('OPENAI_MODEL_MAX_TOKENS')

        if max_tokens:
            return int(max_tokens)

        # ref: https://platform.openai.com/docs/models/gpt-4
        if self.model.startswith('gpt-4-32k'):
            return 32768

        if self.model.startswith('gpt-4'):
            return 8192

        # ref: https://platform.openai.com/docs/models/gpt-3-5
        if self.model.startswith('gpt-3.5-turbo-16k'):
            return 16384

        if self.model.startswith('gpt-3.5-turbo'):
            return 4096

        raise ValueError(f'model {self.model} is unsupported')

    def create_chat_completions(self, *args, **kwargs):
        import openai

        return openai.ChatCompletion.create(*args, **kwargs)

    @core.chat_function('システムをシャットダウンします。', class_member=True)
    def stop_and_shutdown(self, reason: Annotated[str, '理由 (日本語)']):
        self.do_webhook(f"""
次の理由により、システムをシャットダウンします:
```txt
{reason}
```
""".strip())

        return super().stop_and_shutdown()

    def __call__(self):
        with self.rcon.connect() as conn:
            n_players = len(conn.list())

        import tiktoken

        log_text = self.log_file.read_text()

        messages = [
            {
                'role': 'system',
                'content': f"""
あなたはマインクラフトサーバーの管理者です。
サーバーの現在の接続数と {self.log_file.date} のログを確認し、条件を満たしていれば、システムをシャットダウンして下さい。

シャットダウンの条件:

- 現在接続中のプレイヤー数が 0 名である
- 全プレイヤーがサーバーからログアウトして {self.idle_timeout} 以上経過している
- 現在接続中のプレイヤー数とログに矛盾がある場合、最後のログエントリから {self.idle_timeout} 以上経過している
""".strip()
            },

            {
                'role': 'user',
                'content': f"""
現在接続中のプレイヤー数: {n_players}

{self.log_file.date} のログ内容:

```log
{log_text}
```

現在日時: {self.now().isoformat()}
""".strip()
            }
        ]

        functions = core.chat_function.defs

        encoding = tiktoken.get_encoding(self.token_encoding)

        # JSON であれば、トークン数の推定に十分なはず
        message_json = json.dumps(dict(messages=messages, functions=functions),
                                  ensure_ascii=False)
        n_tokens = len(encoding.encode(message_json))

        if n_tokens > self.max_tokens:
            core.logger.info(f'トークン数 ({n_tokens}) が、モデルの許容トークン数 '
                             f'({self.model}: {self.max_tokens}) を超えたので、'
                             'StrictExecute を実行します。')

            return StrictExecute(self.rcon, self.log_file, self.do_webhook,
                                 self.shutdown)()

        resp = self.create_chat_completions(model=self.model,
                                            messages=messages,
                                            functions=functions)

        assistant_message = resp.choices[0].message

        function = assistant_message.get('function_call')

        if function:
            f_kwargs = function.get('arguments')

            if f_kwargs:
                f_kwargs = json.loads(f_kwargs)

            f = getattr(core.chat_function, function.name)
            f_resp = f(self, **f_kwargs)

            messages.append(assistant_message)

            messages.append({
                'role': 'function',
                'name': function.name,
                'content': json.dumps(f_resp)
            })

            resp = self.create_chat_completions(model=self.model,
                                                messages=messages)

            assistant_message_content = resp.choices[0].message.get('content')

            if assistant_message_content:
                self.do_webhook(assistant_message_content)


if __name__ == '__main__':
    Execute = OpenAIExecute if OpenAIExecute.enable else StrictExecute
    execute = Execute(core.Rcon(), core.latest_log)

    try:
        execute()
    except (
        core.RconException,
        FileNotFoundError  # おそらくログファイルが存在しない
    ):
        stacktrace = traceback.format_exc()
        execute.do_webhook(stacktrace)
