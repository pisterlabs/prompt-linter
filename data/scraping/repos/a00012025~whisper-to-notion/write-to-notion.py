# import datetime
import sys
import os
import re
from notion_client import Client
import openai

filter_out_words = [
    "[Start speaking]",
    "[END PLAYBACK]",
    "[AUDIO OUT]",
    "[VIDEO PLAYBACK]",
    "[BLACK_AUDIO]",
    "[音声なし]"
]
end_characters = [
    ".",
    "?",
    "!",
    "。"
]

translate = False
notion = Client(auth=os.environ["NOTION_TOKEN"])
page_id = "04646995367c426fb7942d9d37800392"

page = notion.pages.retrieve(page_id)
ansi_escape = re.compile(r'\x1B[@-_][0-?]*[ -/]*[@-~]')

# pprint(page)
translation_prompt = "請翻譯以下文字成為繁體中文，只輸出翻譯後的文字，不要輸出原文 \n ```\n %s \n``` #lang:zh-tw\n"
last_line, last_translated_line, last_block_id = "", "", ""
while True:
    try:
        line = sys.stdin.readline()
        line = ansi_escape.sub('', line)
        for word in filter_out_words:
            line = line.replace(word, "")
        line = line.strip().replace("\n", "")
        if not line:
            continue

        combine = False
        if translate and (len(last_line) > 0 and last_line[-1] not in end_characters and last_translated_line[-1] not in end_characters):
            # combine this line with last line
            combine = True
            if last_line.endswith("--"):
                last_line = last_line[:-2]
            line = last_line + " " + line

        # translate to Chinese
        if translate:
            response = openai.ChatCompletion.create(
                model="gpt-4", messages=[{"role": "user", "content": translation_prompt % line}])
            # type: ignore
            translated_content = response.choices[0].message.content
        else:
            translated_content = " "

        # if not combined, append to the end of the Notion page
        if not combine:
            result = notion.blocks.children.append(page_id, children=[
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"type": "text", "text": {"content": line+"\n"+translated_content if translate else line}}],
                    },
                }
            ])
            last_block_id = result["results"][0]["id"]  # type: ignore
        else:
            # if combined, replace the last block with the combined one
            result = notion.blocks.update(last_block_id, paragraph={
                "rich_text": [{"type": "text", "text": {"content": line+"\n"+translated_content}}],
            })
        last_line = line
        last_translated_line = translated_content
        # print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), end=" ")
        # print(f"Added new line to Notion page")
    except KeyboardInterrupt:
        print("Terminating...")
        break
