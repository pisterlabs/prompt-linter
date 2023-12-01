import getpass
import tkinter as tk
import tkinter.messagebox as message
from datetime import datetime, timedelta
from tkinter import filedialog

import pandas as pd

from keyword_explorer.tkUtils.ToolTip import ToolTip
from keyword_explorer.Apps.AppBase import AppBase
from keyword_explorer.OpenAI.OpenAIComms import OpenAIComms
from keyword_explorer.TwitterV2.TwitterV2Counts import TwitterV2Counts, TwitterV2Count
from keyword_explorer.tkUtils.Buttons import Buttons
from keyword_explorer.tkUtils.DataField import DataField
from keyword_explorer.tkUtils.DateEntryField import DateEntryField
from keyword_explorer.tkUtils.ListField import ListField
from keyword_explorer.tkUtils.TextField import TextField

from typing import List


class KeywordExplorer(AppBase):
    oai:OpenAIComms
    tvc:TwitterV2Counts
    prompt_text_field:TextField
    response_text_field:TextField
    keyword_text_field:TextField
    start_date_field:DateEntryField
    end_date_field:DateEntryField
    regex_field:DataField
    max_chars_field:DataField
    token_list:ListField
    engine_list:ListField
    sample_list:ListField
    query_options_field:DataField
    action_buttons:Buttons
    action_buttons2:Buttons

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("KeywordExplorer")

    def setup_app(self):
        self.app_name = "KeywordExplorer"
        self.app_version = "3.17.2023"
        self.geom = (850, 790)
        self.oai = OpenAIComms()
        self.tvc = TwitterV2Counts()

        if not self.oai.key_exists():
            message.showwarning("Key Error", "Could not find Environment key 'OPENAI_KEY'")
        if not self.tvc.key_exists():
            message.showwarning("Key Error", "Could not find Environment key 'BEARER_TOKEN_2'")


    def build_app_view(self, row:int, main_text_width:int, main_label_width:int) -> int:
        param_text_width = 20
        param_label_width = 15
        row += 1

        lf = tk.LabelFrame(self, text="GPT")
        lf.grid(row=row, column=0, columnspan = 2, sticky="nsew", padx=5, pady=2)
        self.build_gpt(lf, main_text_width, main_label_width)

        lf = tk.LabelFrame(self, text="GPT Params")
        lf.grid(row=row, column=2, sticky="nsew", padx=5, pady=2)
        self.build_gpt_params(lf, param_text_width, param_label_width)
        row += 1

        lf = tk.LabelFrame(self, text="Twitter")
        lf.grid(row=row, column=0, columnspan = 2, sticky="nsew", padx=5, pady=2)
        self.build_twitter(lf, main_text_width, main_label_width)

        lf = tk.LabelFrame(self, text="Twitter Params")
        lf.grid(row=row, column=2, columnspan = 2, sticky="nsew", padx=5, pady=2)
        self.build_twitter_params(lf, param_text_width, param_label_width)

        self.end_date_field.set_date()
        self.start_date_field.set_date(d = (datetime.utcnow() - timedelta(days=10)))

    def build_gpt(self, lf:tk.LabelFrame, text_width:int, label_width:int):
        row = 0
        self.prompt_text_field = TextField(lf, row, "Prompt:\nDefault style:\nHere's a list of X:\n1)", text_width, height=5, label_width=label_width)
        self.prompt_text_field.set_text("Here's a short list of popular pets:\n1)")
        ToolTip(self.prompt_text_field.tk_text, "The prompt that the GPT will use to generate text from")
        row = self.prompt_text_field.get_next_row()

        self.response_text_field = TextField(lf, row, 'Response', text_width, height=10, label_width=label_width)
        ToolTip(self.response_text_field.tk_text, "The response from the GPT will be displayed here")
        row = self.response_text_field.get_next_row()

        self.max_chars_field = DataField(lf, row, 'Max chars:', text_width, label_width=label_width)
        self.max_chars_field.set_text('30')
        ToolTip(self.max_chars_field.tk_entry, "The maximum allowable length for a line in the response. \nLonger lines will be discarded")
        row = self.max_chars_field.get_next_row()

        self.regex_field = DataField(lf, row, 'Parse regex', text_width, label_width=label_width)
        self.regex_field.set_text(r"\n[0-9]+\)|\n[0-9]+|[0-9]+\)")
        ToolTip(self.regex_field.tk_entry, "The regex used to parse the GPT response. Editable")
        row = self.regex_field.get_next_row()

        self.action_buttons = Buttons(lf, row, "Actions", label_width=label_width)
        b = self.action_buttons.add_button("New prompt", self.new_prompt_callback, width=-1)
        ToolTip(b, "Sends the prompt to the GPT-3")
        b = self.action_buttons.add_button("Extend prompt", self.extend_prompt_callback, width=-1)
        ToolTip(b, "Uses the previous prompt and response as the new prompt")
        b = self.action_buttons.add_button("Parse response", self.parse_response_callback, width=-1)
        ToolTip(b, "Applies the regex in the 'Parse Regex' field to each line in the resonse text \nthat is longer than Max Chars and places the parsed results in the 'Test Keyords' area below")
        row = self.action_buttons.get_next_row()

    def build_gpt_params(self, lf:tk.LabelFrame, text_width:int, label_width:int):
        row = 0
        self.token_list = ListField(lf, row, "Tokens", width=text_width, label_width=label_width, static_list=True)
        self.token_list.set_text(text='32, 64, 128, 256')
        self.token_list.set_callback(self.set_tokens_callback)
        ToolTip(self.token_list.tk_list, "Sets the maxumum number of tokens that the GPT can use in a response")
        row = self.token_list.get_next_row()

        engine_list = self.oai.list_models(exclude_list = [":", "ada", "embed", "similarity", "code", "edit", "search", "audio", "instruct", "2020", "if", "insert", "whisper"])
        engine_list = sorted(engine_list, reverse=True)
        self.engine_list = ListField(lf, row, "Engines", width=text_width, label_width=label_width, static_list=True)
        self.engine_list.set_text(list=engine_list)
        self.engine_list.set_callback(self.set_engine_callback)
        ToolTip(self.engine_list.tk_list, "Sets the GPT engine. Includes the original and most recent engines")
        row = self.engine_list.get_next_row()
        #
        # lbl = tk.Label(lf, text="Tokens", width=label_width, bg="red")
        # lbl.grid(row=row, column=0, sticky="w", padx=2, pady=2)

    def build_twitter(self, lf:tk.LabelFrame, text_width:int, label_width:int):
        row = 0
        self.keyword_text_field = TextField(lf, row, 'Test Keyword(s)', text_width, height=10, label_width=label_width)
        ToolTip(self.keyword_text_field.tk_text,
                "List of terms to search.\nTerms can have spaces or be combined with OR:\nNorth Korea\nSouth Korea\nNorth Korea OR South Korea")
        row = self.keyword_text_field.get_next_row()

        self.start_date_field = DateEntryField(lf, row, 'Start Date', text_width, label_width=label_width)
        row = self.start_date_field.get_next_row()
        self.end_date_field = DateEntryField(lf, row, 'End Date', text_width, label_width=label_width)
        row = self.end_date_field.get_next_row()
        self.action_buttons2 = Buttons(lf, row, "Actions", label_width=label_width)
        b = self.action_buttons2.add_button("Clear", self.clear_counts_callbacks, width=-1)
        ToolTip(b, "Clears any old data from the plot")
        b = self.action_buttons2.add_button("Test Keyword", self.test_keyword_callback, width=-1)
        ToolTip(b, "Query Twitter for each keyword and plot")
        b = self.action_buttons2.add_button("Plot", self.plot_counts_callback, width=-1)
        ToolTip(b, "Plot the current data")
        b = self.action_buttons2.add_button("Save", self.save_callback, width=-1)
        ToolTip(b, "Save the results as an xlsx file")
        b = self.action_buttons2.add_button("Launch Twitter", self.launch_twitter_callback, width=-1)
        ToolTip(b, "Open tabs in the default browser for each term over the time period")
        row = self.action_buttons2.get_next_row()

    def build_twitter_params(self, lf:tk.LabelFrame, text_width:int, label_width:int):
        row = 0
        self.sample_list = ListField(lf, row, "Sample", width=text_width, label_width=label_width, static_list=True)
        self.sample_list.set_text(text='day, week, month')
        self.sample_list.set_callback(self.set_time_sample_callback)
        self.set_time_sample_callback()
        ToolTip(self.sample_list.tk_list, "The sampling period\nWeek and month are subsamples")
        row = self.sample_list.get_next_row()

        self.query_options_field = DataField(lf, row, 'Query Options', text_width, label_width=label_width)
        self.query_options_field.set_text("lang:en -is:retweet")
        ToolTip(self.query_options_field.tk_entry, "TwitterV2 args. Default is English (en), and no retweets\nMore info is available here:\ndeveloper.twitter.com/en/docs/twitter-api/tweets/search/integrate/build-a-query")
        row = self.query_options_field.get_next_row()

    def set_experiment_text(self, l:List):
        self.prompt_text_field.clear()
        pos = 0
        for s in reversed(l):
            self.prompt_text_field.add_text(s+"\n")
            pos += 1

    def save_experiment_text(self, filename:str):
        s = self.prompt_text_field.get_text()
        with open(filename, mode="w", encoding="utf8") as f:
            f.write(s)

    def set_engine_callback(self, event:tk.Event = None):
        engine_str = self.engine_list.get_selected()
        self.oai.engine = engine_str
        self.engine_list.set_label("Engines\n{}".format(engine_str))

    def set_tokens_callback(self, event:tk.Event = None):
        token_str = self.token_list.get_selected()
        self.token_list.set_label("Tokens\n({})".format(token_str))

    def set_time_sample_callback(self, event:tk.Event = None):
        sample_str = self.sample_list.get_selected()
        self.sample_list.set_label("Sample\n({})".format(sample_str))

    def adjust_tokens(self) -> int:
        tokens = self.token_list.get_selected()
        tint = int(tokens)
        return tint

    def new_prompt_callback(self):
        prompt = self.prompt_text_field.get_text()
        response = self.get_gpt3_response(prompt)
        self.response_text_field.set_text(response)

    def extend_prompt_callback(self):
        prompt = "{}{}".format(self.prompt_text_field.get_text(), self.response_text_field.get_text())
        self.prompt_text_field.set_text(prompt)
        response = self.get_gpt3_response(prompt)
        self.response_text_field.set_text(response)

    def parse_response_callback(self):
        split_regex = self.regex_field.get_text()
        response_list = self.response_text_field.get_list(split_regex)
        print(response_list)

        if len(response_list) > 1:
            s:str = ""
            max_chars = self.max_chars_field.get_as_int()
            for r in response_list:
                if len(r) < max_chars:
                    s += r+"\n"
            #s = '\n'.join(response_list)
            self.keyword_text_field.set_text(s.strip())
        else:
            message.showwarning("Parse Error",
                                "Could not parse [{}]".format(self.response_text_field.get_text()))

    def test_keyword_callback(self):
        l = self.keyword_text_field.get_list("\n")
        print(l)
        start_dt = self.start_date_field.get_date()
        end_dt = self.end_date_field.get_date()

        key_list = []
        for keyword in l:
            if len(keyword) > 2:
                key_list.append(keyword)
        if len(key_list) == 0:
            message.showwarning("Keyword too short",
                                "Please enter something longer than [{}] in the text area".format(keyword))
            return

        tweet_options = self.query_options_field.get_text()
        granularity = self.sample_list.get_selected()
        log_dict = {"granularity":granularity, "twitter_start": start_dt.strftime("%Y-%m-%d"), "twitter_end":end_dt.strftime("%Y-%m-%d")}
        for keyword in key_list:
            if granularity == 'day':
                self.tvc.get_counts(keyword, start_dt, end_time=end_dt, granularity=granularity, tweet_options=tweet_options)
                print("testing keyword {} between {} and {} - granularity = {}".format(keyword, start_dt, end_dt, granularity))
            elif granularity == 'week':
                self.tvc.get_sampled_counts(keyword, start_dt, end_time=end_dt, skip_days=7, tweet_options=tweet_options)
                print("testing keyword {} between {} and {} - skip_days = {}".format(keyword, start_dt, end_dt, 7))
            elif granularity == 'month':
                self.tvc.get_sampled_counts(keyword, start_dt, end_time=end_dt, skip_days=30, tweet_options=tweet_options)
                print("testing keyword {} between {} and {} - skip_days = {}".format(keyword, start_dt, end_dt, 30))
            else:
                self.dp.dprint("test_keyword_callback() unable to handle granularity = {}".format(granularity))
                return

            tvc:TwitterV2Count
            for tvc in self.tvc.count_list:
                print(tvc.to_string())

        for k, v in self.tvc.totals_dict.items():
            log_dict[k] = v
        self.log_action("test_keyword", log_dict)
        self.tvc.plot()

    def clear_counts_callbacks(self):
        self.tvc.reset()

    def plot_counts_callback(self):
        self.tvc.plot()

    def save_callback(self):
        default = "{} {}.xlsx".format(self.experiment_field.get_text(), datetime.now().strftime("%B_%d_%Y_(%H_%M_%S)"))
        filename = filedialog.asksaveasfilename(filetypes=(("Excel files", "*.xlsx"),("All Files", "*.*")), title="Save Excel File", initialfile=default)
        if filename:
            print("saving to {}".format(filename))
            df1 = self.get_description_df(self.prompt_text_field.get_text(), self.response_text_field.get_text())
            df2 = self.tvc.to_dataframe()
            with pd.ExcelWriter(filename) as writer:
                df1.to_excel(writer, sheet_name='Experiment')
                df2.to_excel(writer, sheet_name='Results')
                writer.save()
                self.log_action("save", {"filename":filename})

    def launch_twitter_callback(self):
        key_list = self.keyword_text_field.get_list("\n")
        start_dt = self.start_date_field.get_date()
        end_dt = self.end_date_field.get_date()
        self.log_action("Launch_twitter", {"twitter_start": start_dt.strftime("%Y-%m-%d"), "twitter_end":end_dt.strftime("%Y-%m-%d"), "terms":" ".join(key_list)})
        self.tvc.launch_twitter(key_list, start_dt, end_dt)

    def get_description_df(self, probe:str, response:str) -> pd.DataFrame:
        now = datetime.now()
        now_str = now.strftime("%B_%d_%Y_(%H:%M:%S)")
        token_str = self.token_list.get_selected()
        engine_str = self.engine_list.get_selected()
        sample_str = self.sample_list.get_selected()

        description_dict = {'name':getpass.getuser(), 'date':now_str, 'probe':probe, 'response':response, 'sampling':sample_str, 'engine':engine_str, 'tokens':token_str}
        df = pd.DataFrame.from_dict(description_dict, orient='index', columns=['Value'])
        return df

    def get_gpt3_response(self, prompt:str) -> str:
        """
        Method that takes a prompt and gets the response back via the OpenAI API
        :param prompt: The prompt to be sent to the GPT-3
        :return: The GPT-3 Response
        """
        if len(prompt) < 3:
            self.dp.dprint("get_gpt3_response() Error. Prompt too short: '{}'".format(prompt))
            return ""

        # print(prompt)
        self.oai.max_tokens = self.adjust_tokens()
        self.oai.set_engine(name = self.engine_list.get_selected())
        results = self.oai.get_prompt_result(prompt, False)
        self.dp.dprint("\n------------\ntokens = {}, engine = {}\nprompt = {}".format(self.oai.max_tokens, self.oai.engine, prompt))
        self.log_action("gpt_prompt", {"tokens":self.oai.max_tokens, "engine":self.oai.engine, "prompt":prompt})

        # clean up before returning
        s = results[0].strip()
        s =  self.clean_list_text(s)
        self.log_action("gpt_response", {"gpt_text":s})
        return s

    def clean_list_text(self, s:str) -> str:
        """
        Convenience method to clean up list-style text. Useful for a good chunk of the GPT-3 responses for
        the style of prompt that I've been using
        :param s: The string to clean up
        :return: The cleaned-up string
        """
        lines = s.split("\n")
        line:str
        par =""
        for line in lines:
            s = line.strip()
            if s != "":
                par = "{}\n{}".format(par, s)
        return par.strip()

def main():
    app = KeywordExplorer()
    app.mainloop()

if __name__ == "__main__":
    main()