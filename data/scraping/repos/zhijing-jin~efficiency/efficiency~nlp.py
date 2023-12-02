from __future__ import unicode_literals, print_function
import time
import multiprocessing



from efficiency.log import show_var



class NLP:
    def __init__(self, disable=['ner', 'parser', 'tagger', "lemmatizer"]):
        import spacy

        self.nlp = spacy.load('en_core_web_sm', disable=disable)
        try:
            self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))
        except:
            self.nlp.add_pipe('sentencizer')

    def detokenize(self, text):
        # Reference: Use the model https://github.com/julianmichael/jjm/blob/master/jjm/src/ling/Text.scala
        pass

    def sent_tokenize(self, text):
        doc = self.nlp(text)
        sentences = [str(sent).strip() for sent in doc.sents]
        return sentences

    def word_tokenize(self, text, lower=False):  # create a tokenizer function
        if text is None: return text
        text = ' '.join(text.split())
        if lower: text = text.lower()
        toks = [tok.text for tok in self.nlp.tokenizer(text)]
        return ' '.join(toks)

    @staticmethod
    def sent_bleu(ref_list, hyp):
        from nltk.translate import bleu
        from nltk.translate.bleu_score import SmoothingFunction
        smoothie = SmoothingFunction().method4
        refs = [ref.split() for ref in ref_list]
        hyp = hyp.split()
        return bleu(refs, hyp, smoothing_function=smoothie)


class Translator:
    def __init__(self, cache_file='.cache_trans_lookup.csv', load_translator=True):
        self.cache_file = cache_file
        self.cache = self.load_cache()
        if load_translator:
            from googletrans import Translator
            self.translator = Translator()

    def load_cache(self):
        from collections import defaultdict
        cache = defaultdict(dict)
        from efficiency.log import fread
        data = fread(self.cache_file, verbose=False)
        data = [i for i in data if (i['input'] != i['output']) and i['input'].strip() and i['output'].strip()]
        for i in data:
            cache[(i['src_lang'], i['tgt_lang'])][i['input']] = i['output']
            cache[(i['tgt_lang'], i['src_lang'])][i['output']] = i['input']

        return cache

    def save_cache(self, text, translated_text, src_lang, tgt_lang):
        if text not in self.cache[(src_lang, tgt_lang)]:
            datum = [{
                'src_lang': src_lang,
                'tgt_lang': tgt_lang,
                'input': text,
                'output': translated_text,
            }]
            self.cache[(src_lang, tgt_lang)][text] = translated_text
            self.cache[(tgt_lang, src_lang)][translated_text] = text

            from efficiency.log import write_dict_to_csv
            write_dict_to_csv(datum, self.cache_file, mode='a')

    def raw_translate(self, text, src_lang, tgt_lang):
        def successful_pass():
            translated_text = self.translator.translate(text, src=src_lang, dest=tgt_lang).text

            if translated_text != text and translated_text.strip():
                return translated_text

        if_success = successful_pass()
        if if_success:
            return if_success
        else:
            import time
            time.sleep(10)
            if_success = successful_pass()
            if if_success:
                return if_success
            else:
                print(
                    "[Error] Translation failed. You have very likely reached the limit of the `googletrans' "
                    "library. Try to wait for a while or change your IP address. Alternatively, you can also edit the "
                    "source code here to change it to Google cloud API by setting up your credentials.")
                print(src_lang, tgt_lang, text)
                import pdb;
                pdb.set_trace()

    def translate(self, text, src_lang='en', tgt_lang='de', verbose=True, enable_cache=True):
        if src_lang == tgt_lang:
            return text

        this_cache = self.cache[(src_lang, tgt_lang)]
        if enable_cache and (text in this_cache):
            translated_text = this_cache[text]
        else:
            translated_text = self.raw_translate(text, src_lang, tgt_lang)
            self.save_cache(text, translated_text, src_lang, tgt_lang)

        if verbose:
            from efficiency.log import show_var
            show_var(['text', 'translated_text', ])
        return translated_text

    @staticmethod
    def print_countries(langs):
        langs = sorted(langs)

        import pycountry
        rows = [
            {'lang': lang,
             'country': pycountry.languages.get(alpha_2=lang).name if pycountry.languages.get(alpha_2=lang) else None
             }
            for lang in langs
        ]
        import pandas as pd
        df = pd.DataFrame(rows, index=None)
        print(df)
        import pdb;
        pdb.set_trace()
        return df

    @staticmethod
    def get_language_list(list_from=['googletrans', 'pycountry', 'langcodes'][0]):
        langs = []
        if list_from == 'googletrans':
            from googletrans import LANGUAGES
            translateable_langs = []
            for language_code, language_name in LANGUAGES.items():
                translateable_langs.append(language_code)
            translateable_langs = sorted(translateable_langs)
            langs = translateable_langs
        elif list_from == 'pycountry':
            import pycountry
            code2family = [{"alpha_3": i.alpha_3, "name": i.name} for i in pycountry.language_families]

            pyc_langs = []
            for language in pycountry.languages:
                name = language.name.replace(' (macrolanguage)', '').strip()
                name = name.split('(', 1)[0].strip()
                item = {'name': name}
                try:
                    item['lang'] = language.alpha_2
                except:
                    try:
                        item['lang'] = language.alpha_3
                    except:
                        pass
                pyc_langs.append(item)
            additional_dict = [
                {'lang': 'zh-cn', 'name': 'Chinese (Simplified)'},
                {'lang': 'zh-tw', 'name': 'Chinese (Traditional)'},
                {'lang': 'iw', 'name': 'Modern Hebrew'},
                {'lang': 'jw', 'name': 'Javanese'},
                {'lang': 'me', 'name': 'Montenegrin'},
            ]
            pyc_langs.extend(additional_dict)
            from langcodes import Language
            for item in pyc_langs:
                language_code = item['lang']
                # lang = Language.get(language_code)
                # item['family'] = lang.family_name()
            import pandas as pd
            pyc_langs = pd.DataFrame(pyc_langs)
            langs = pyc_langs

        return langs
    
class APICall:
    def __init__(self, start_time, end_time, prompt_tokens, completion_tokens, model):
        self.start_time = start_time
        self.end_time = end_time
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.model = model
    
class APICallTracker:
    def __init__(self):
        self.calls = []
        self.tokens_per_engine = {}
        self.start_time = None
        self.end_time = None
        self.num_tokens = 0
        self.num_in_tokens = 0
        self.num_out_tokens = 0
        self.num_requests = 0
        self.l = multiprocessing.Lock()
    
    def add_call(self, call: APICall):
        with self.l:
            if call.model not in self.tokens_per_engine:
                self.tokens_per_engine[call.model] = [0, 0]
            self.tokens_per_engine[call.model][0] += call.prompt_tokens
            self.tokens_per_engine[call.model][1] += call.completion_tokens

            self.num_tokens += call.prompt_tokens + call.completion_tokens
            self.num_in_tokens += call.prompt_tokens
            self.num_out_tokens += call.completion_tokens
            self.num_requests += 1
            
            if self.start_time is None or call.start_time < self.start_time:
                self.start_time = call.start_time
            if self.end_time is None or call.end_time > self.end_time:
                self.end_time = call.end_time

            self.calls.append(call)
    
    def tokens_per_second(self):
        with self.l:
            if self.start_time is None or self.end_time is None:
                return 0.0

            total_tokens = 0
            for tokens in self.tokens_per_engine.values():
                total_tokens += tokens[0] + tokens[1]
            return total_tokens / (self.end_time - self.start_time)
        
    def requests_per_second(self):
        with self.l:
            if self.start_time is None or self.end_time is None:
                return 0.0

            return len(self.calls) / (self.end_time - self.start_time)
        
class APICallCache:
    def __init__(self, gpt_files, output_file):
        self.l = multiprocessing.Lock()

        self.gpt_files = gpt_files
        self.output_file = output_file

        self.cache = self.load_cache()
    
    # implement dict interface with thread-safe locking
    def __getitem__(self, key):
        with self.l:
            return self.cache[key]
    
    def __setitem__(self, key, value):
        with self.l:
            self.cache[key] = value
        
    def __contains__(self, key):
        with self.l:
            return key in self.cache
    
    def __len__(self):
        with self.l:
            return len(self.cache)
    
    def __iter__(self):
        with self.l:
            return iter(self.cache)
    
    def __delitem__(self, key):
        with self.l:
            del self.cache[key]

    def save_cache(self, question, response_text):
        with self.l:
            if (not (question in self.cache)) and response_text:
                self.cache[question] = response_text
                datum = [{
                    'pred': response_text,
                    'query': question,
                }]
                from efficiency.log import write_dict_to_csv
                write_dict_to_csv(datum, self.output_file, mode='a')

    def load_cache(self):
        with self.l:
            cache = {}
            from efficiency.log import fread
            for file in self.gpt_files:
                data = fread(file, verbose=False)
                cache.update({i[f'query{q_i}']: i[f'pred{q_i}'] for i in data
                            for q_i in list(range(10)) + ['']
                            if f'query{q_i}' in i})
            cache = {k: v for k, v in cache.items() if v}  # there are cases where the response is empty
            return cache

class Chatbot:
    model_version2engine = {
        'gpt4': "gpt-4",
        'gpt3.5': "gpt-3.5-turbo",
        'gpt3': "text-davinci-003",

        'gpt3.043': "text-davinci-003",
        'gpt3.042': "text-davinci-002",
        'gpt3.041': "text-davinci-001",
        'gpt3.04': "davinci",
        'gpt3.03': "curie",
        'gpt3.02': "babbage",
        'gpt3.01': "ada",
    }

    # model -> (1000_input_tokens, 1000_output_tokens, 1000_training_tokens) prices
    engine2pricing = {
        'gpt-4': (0.03, 0.06, None),
        'gpt-4-32k': (0.06, 0.12, None),

        'gpt-3.5-turbo': (0.0015, 0.002, None),
        'gpt-3.5-turbo-16k': (0.003, 0.004, None),

        'gpt-3.5-turbo-0613': (0.0015, 0.002, None),
        'gpt-3.5-turbo-0613-16k': (0.003, 0.004, None),

        'ada': (0.0004, 0.0004, None),
        'babbage': (0.0005, 0.0005, None),
        'curie': (0.002, 0.002, None),
        'davinci': (0.02, 0.02, None),

        'text-davinci-001': (0.02, 0.02, None),
        'text-davinci-002': (0.02, 0.02, None),
        'text-davinci-003': (0.02, 0.02, None),
    }

    def __init__(self, model_version='gpt3.5', max_tokens=100, output_file=None, output_folder='./',
                 system_prompt="You are a helpful assistant.", cache_files=[],
                 openai_key_alias='OPENAI_API_KEY', openai_org_alias='OPENAI_ORG_ID', tracker=None, cache=None):
        import os
        import openai
        self.openai = openai

        if tracker is None:
            tracker = APICallTracker()

        api_key = os.environ[openai_key_alias]
        openai.api_key = api_key
        if openai_org_alias != 'OPENAI_ORG_ID':
            organization_id = os.environ[openai_org_alias]
            openai.organization = organization_id

        self.model_version = model_version
        self.engine = self.model_version2engine.get(model_version, model_version)
        self.max_tokens = max_tokens
        self.set_system_prompt(system_prompt)

        if cache is None:
            output_file = f'{output_folder}/.cache_{model_version}.csv' if output_file is None else output_file
            gpt_files = cache_files + [output_file]

            cache = APICallCache(gpt_files, output_file)

        self.cache = cache

        # self.list_all_models()
        self.clear_dialog_history()

        self.tracker = tracker
    
    def clone(self):
        return Chatbot(
            model_version=self.model_version,
            max_tokens=self.max_tokens,
            system_prompt=self.system_prompt,
            tracker=self.tracker,
            cache=self.cache,
        )

    def set_system_prompt(self, system_prompt):
        if system_prompt is None:
            system_prompt = "You are a helpful assistant."
        self.system_prompt = system_prompt
        self.system_is_default = system_prompt == "You are a helpful assistant."

    def clear_dialog_history(self):
        self.dialog_history = [
            {"role": "system", "content": self.system_prompt},
            # {"role": "user", "content": "Who won the world series in 2020?"},
            # {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        ]

    def dialog_history_to_str(self, pure_completion_mode=False, ):
        dialog_history = [turn for turn in self.dialog_history
                        if not ((turn['role'] == 'system') and self.system_is_default)]

        if not self.if_newer_engine:
            # dialog_history = [turn for turn in dialog_history if not (turn['role'] == 'system')]
            for turn_i, turn in enumerate(dialog_history):
                if turn['role'] == 'system':
                    system_prompt = turn['content']
                    dialog_history.pop(turn_i)
                    next_turn = dialog_history[turn_i]
                    if next_turn['role'] == 'user':
                        dialog_history[turn_i]['content'] = '\n'.join([system_prompt, next_turn['content']])
                        break
            self.dialog_history = dialog_history
        if pure_completion_mode:
            dialog_history_text = '\n'.join(turn['content'] for turn in dialog_history) # TODO: error before but unsure if this is the right fix
        else:
            dialog_history_text = []
            for turn in dialog_history:
                if turn['role'] == 'system':
                    prefix = 'S'
                elif turn['role'] == 'user':
                    prefix = 'Q'
                elif turn['role'] == 'assistant':
                    prefix = 'A'
                else:
                    continue
                this_text = f"{prefix}: {turn['content']}"
                if prefix == 'A':
                    this_text += '\n'
                dialog_history_text.append(this_text)

            dialog_history_text = '\n'.join(dialog_history_text)

            if not self.if_newer_engine:
                if turn['role'] != 'assistant':
                    dialog_history_text += '\nA:'
        return dialog_history_text

    def list_all_models(self):
        model_list = self.openai.Model.list()['data']
        model_ids = [x['id'] for x in model_list]
        model_ids.sort()
        print(model_ids)
        import pdb;
        pdb.set_trace()

    @property
    def _total_cost(self):
        price = 0.
        for call in self.tracker.calls:
            # price += (call.prompt_tokens + call.completion_tokens) / 1000 * self.engine2pricing[self.engine]
            price += (call.prompt_tokens*self.engine2pricing[self.engine][0] + call.completion_tokens*self.engine2pricing[self.engine][1]) / 1000
        return price

    def print_cost_and_rates(self):
        print(f"[Info] Spent ${self._total_cost:.3f} for {self.tracker.num_tokens} tokens (in: {self.tracker.num_in_tokens}, out: {self.tracker.num_out_tokens}) and {self.tracker.num_requests} requests. Throughput: {self.tracker.tokens_per_second():.1f} tokens/s and {self.tracker.requests_per_second():.1f} requests/second.")

    import asyncio

    async def semaphore_gather(self, num, coros, return_exceptions=True):
        import asyncio
        semaphore = asyncio.Semaphore(num)

        async def _wrap_coro(coro):
            async with semaphore:
                return await coro

        return await asyncio.gather(
            *(_wrap_coro(coro) for coro in coros), return_exceptions=return_exceptions
        )

    async def _ask_n(self, questions, num_parallel=100, **kwargs):
        return await self.semaphore_gather(num_parallel, [
            # create a new chatbot for each question, but copy the cache, tracker, model_version, etc.
            self.clone().aask(q, **kwargs)
                for q in questions
        ], return_exceptions=True)

    def ask_n(self, questions, num_parallel=100, **kwargs):
        """
        Ask multiple questions in parallel. This is useful for asking a large number of questions.

        :param questions: a list of questions
        :param num_parallel: max number of requests to send in parallel
        :param kwargs: other arguments to pass to ask()
        :return: list of answers

        Example:
        ```python
        from efficiency.nlp import Chatbot

        bot = Chatbot(model_version='gpt-3.5-turbo')

        predictions = bot.ask_n([f"Write this number: '{i}'. Then 20 sentences about why it's your favorite number?" for i in range(0, 3000)], verbose=0, num_parallel=90)
        print(f'Length of predictions: {len(predictions)}')
        print(f'Sum of chars in predictions: {sum([len(p) for p in predictions])}')

        bot.print_cost_and_rates()
        ```
        """

        import asyncio
        return asyncio.run(self._ask_n(questions, num_parallel=num_parallel, **kwargs))

    def ask(self, *args, delta_time=10, **kwargs):
        def repeat():
            self.print_cost_and_rates()
            import time
            time.sleep(delta_time)

            return self.ask(*args, delta_time=2*delta_time, **kwargs)

        def api_error(e):
            print(f'[Info] openai.error.APIError: {e}. Wait for {delta_time} seconds.')
            return repeat()
        
        def rate_limit_error(e):
            print(f'[Info] openai.error.RateLimitError: {e}. Wait for {delta_time} seconds.')
            '''
            Default rate limits for gpt-4/gpt-4-0314 are 40k TPM and 200 RPM. Default rate limits for gpt-4-32k/gpt-4-32k-0314 are 80k PRM and 400 RPM. 
            https://platform.openai.com/docs/guides/rate-limits/overview
            '''
            return repeat()

        import openai
        try:
            return self.raw_query(*args, **kwargs)
        except openai.error.InvalidRequestError as e:
            print(f'[Error] InvalidRequestError: {e}')
            import pdb;
            pdb.set_trace()
            if len(self.dialog_history) > 10:
                import pdb;
                pdb.set_trace()
            for turn_i, turn in enumerate(self.dialog_history):
                if turn['role'] == 'assistant':
                    turn['content'] = turn['content'][:1000] # TODO: QUES: DAVE: why 1000? @zhijing-jin
        except openai.error.RateLimitError as e:
            return rate_limit_error(e)
        except openai.error.APIError as e:
            return api_error(e)
        except Exception as e:
            print(f'[Error] Unknown exception when calling openai: {e}')
            # raise e # if there is an unknown error, we should stop the program
            return repeat() # sometimes we get: `[Error] Unknown exception when calling openai: The server is overloaded or not ready yet.` So we will just try again...
    
    async def aask(self, *args, delta_time=10, **kwargs):
        async def repeat():
            self.print_cost_and_rates()
            import asyncio
            await asyncio.sleep(delta_time)

            return await self.aask(*args, delta_time=2*delta_time, **kwargs)

        async def api_error(e):
            print(f'[Info] openai.error.APIError: {e}. Wait for {delta_time} seconds.')
            return await repeat()
        
        async def rate_limit_error(e):
            print(f'[Info] openai.error.RateLimitError: {e}. Wait for {delta_time} seconds.')
            '''
            Default rate limits for gpt-4/gpt-4-0314 are 40k TPM and 200 RPM. Default rate limits for gpt-4-32k/gpt-4-32k-0314 are 80k PRM and 400 RPM. 
            https://platform.openai.com/docs/guides/rate-limits/overview
            '''
            return await repeat()

        import openai
        try:
            return await self.araw_query(*args, **kwargs)
        except openai.error.InvalidRequestError as e:
            print(f'[Error] InvalidRequestError: {e}')
            import pdb;
            pdb.set_trace()
            if len(self.dialog_history) > 10:
                import pdb;
                pdb.set_trace()
            for turn_i, turn in enumerate(self.dialog_history):
                if turn['role'] == 'assistant':
                    turn['content'] = turn['content'][:1000] # TODO: QUES: DAVE: why 1000? @zhijing-jin
        except openai.error.RateLimitError as e:
            return await rate_limit_error(e)
        except openai.error.APIError as e:
            return await api_error(e)
        except Exception as e:
            print(f'[Error] Unknown exception when calling openai: {e}')
            # raise e # if there is an unknown error, we should stop the program
            return await repeat() # sometimes we get: `[Error] Unknown exception when calling openai: The server is overloaded or not ready yet.` So we will just try again...

    async def araw_query(self, question,
                system_prompt=None,
                turn_off_cache=False, valid_ways=['cache', 'api_call'],
                continued_questions=False,
                max_tokens=None, stop_sign="\nQ: ",
                model_version=[None, 'gpt3', 'gpt3.5', 'gpt4'][0],
                engine=[None, "text-davinci-003", "gpt-3.5-turbo", "gpt-4-32k-0314", "gpt-4-0314", "gpt-4"][0],
                enable_pdb=False, verbose=1, only_response=True,
                temperature=0.,
                pure_completion_mode=False,
            ):
        if verbose < 0 or verbose > 2:
            raise ValueError('verbose must be 0, 1 or 2. 0=quiet, 1=print cost and rates, 2=print cost, rates and response.')
                
        if temperature < 0. or temperature > 1.:
            raise ValueError('temperature must be between 0 and 1.')

        if temperature != 0. and not turn_off_cache:
            raise ValueError('turn_off_cache must be True when temperature != 0.')
        
        if system_prompt is not None:
            self.set_system_prompt(system_prompt)
        if model_version is not None:
            if model_version != self.model_version:
                turn_off_cache = True
        
        enable_api = 'api_call' in valid_ways
        if model_version is not None:
            engine = self.model_version2engine.get(model_version, model_version)
        elif engine is not None:
            engine = engine
        else:
            engine = self.engine
        self.engine = engine  # to be called in print_cost()

        max_tokens = self.max_tokens if max_tokens is None else max_tokens
        verbose = 2 if enable_pdb else verbose

        if_newer_engine = engine.startswith('gpt-3.5') or engine.startswith('gpt-4')
        self.if_newer_engine = if_newer_engine

        if not continued_questions:
            self.clear_dialog_history()

        self.dialog_history.append({"role": "user", "content": question}, )

        prompt = self.dialog_history_to_str(pure_completion_mode=pure_completion_mode)
        cache_input = prompt

        if enable_pdb:
            print(cache_input)
            import pdb;
            pdb.set_trace()
        if (cache_input in self.cache) & (not turn_off_cache):
            response_text = self.cache[cache_input]
            if not if_newer_engine:
                response_text = str(response_text).split(stop_sign, 1)[0]
            response_text = response_text.strip()
            if verbose: print(f'[Info] Using cache for "{cache_input[:10]}..."')
        elif enable_api:
            start_time = time.time()

            openai = self.openai
            if if_newer_engine:
                response = await openai.ChatCompletion.acreate(
                    model=engine,
                    temperature=0,
                    max_tokens=max_tokens,
                    messages=self.dialog_history,
                )
                response_text = response['choices'][0]['message']['content']
            else:
                response = await openai.Completion.acreate(
                    model=engine,
                    # prompt=[question],
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=0,
                    stop=stop_sign,
                )
                response_text = response['choices'][0]['text']

            end_time = time.time()
            self.tracker.add_call(APICall(start_time, end_time, response['usage']['prompt_tokens'], response['usage']['completion_tokens'], model=engine))

            response_text = response_text.strip()
            if verbose: self.print_cost_and_rates()
        else:
            response_text = ''

        self.dialog_history.append({"role": "assistant", "content": response_text}, )

        if verbose > 1:
            print()
            print(self.dialog_history_to_str(pure_completion_mode=pure_completion_mode))

        if enable_pdb:
            import pdb;
            pdb.set_trace()

        if enable_api:
            if not turn_off_cache:
                self.cache.save_cache(cache_input, response_text)

        if only_response:
            return response_text
        return response_text #, output QUES: why is there an ouptut, you removed it in the previous commits?
    
    def raw_query(self, question,
                system_prompt=None,
                turn_off_cache=False, valid_ways=['cache', 'api_call'],
                continued_questions=False,
                max_tokens=None, stop_sign="\nQ: ",
                model_version=[None, 'gpt3', 'gpt3.5', 'gpt4'][0],
                engine=[None, "text-davinci-003", "gpt-3.5-turbo", "gpt-4-32k-0314", "gpt-4-0314", "gpt-4"][0],
                enable_pdb=False, verbose=1, only_response=True,
                temperature=0.,
                pure_completion_mode=False,
            ):
        if verbose < 0 or verbose > 2:
            raise ValueError('verbose must be 0, 1 or 2. 0=quiet, 1=print cost and rates, 2=print cost, rates and response.')
                
        if temperature < 0. or temperature > 1.:
            raise ValueError('temperature must be between 0 and 1.')

        if temperature != 0. and not turn_off_cache:
            raise ValueError('turn_off_cache must be True when temperature != 0.')
        
        if system_prompt is not None:
            self.set_system_prompt(system_prompt)
        if model_version is not None:
            if model_version != self.model_version:
                turn_off_cache = True
        
        enable_api = 'api_call' in valid_ways
        if model_version is not None:
            engine = self.model_version2engine.get(model_version, model_version)
        elif engine is not None:
            engine = engine
        else:
            engine = self.engine
        self.engine = engine  # to be called in print_cost()

        max_tokens = self.max_tokens if max_tokens is None else max_tokens
        verbose = 2 if enable_pdb else verbose

        if_newer_engine = engine.startswith('gpt-3.5') or engine.startswith('gpt-4')
        self.if_newer_engine = if_newer_engine

        if not continued_questions:
            self.clear_dialog_history()

        self.dialog_history.append({"role": "user", "content": question}, )

        prompt = self.dialog_history_to_str(pure_completion_mode=pure_completion_mode)
        cache_input = prompt

        if enable_pdb:
            print(cache_input)
            import pdb;
            pdb.set_trace()
        if (cache_input in self.cache) & (not turn_off_cache):
            response_text = self.cache[cache_input]
            if not if_newer_engine:
                response_text = str(response_text).split(stop_sign, 1)[0]
            response_text = response_text.strip()
            if verbose: print(f'[Info] Using cache for "{cache_input[:10]}..."')
        elif enable_api:
            start_time = time.time()

            openai = self.openai
            if if_newer_engine:
                response = openai.ChatCompletion.create(
                    model=engine,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    messages=self.dialog_history,
                )
                response_text = response['choices'][0]['message']['content']
            else:
                response = openai.Completion.create(
                    model=engine,
                    # prompt=[question],
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=stop_sign,
                )
                response_text = response['choices'][0]['text']

            end_time = time.time()
            self.tracker.add_call(APICall(start_time, end_time, response['usage']['prompt_tokens'], response['usage']['completion_tokens'], model=engine))

            response_text = response_text.strip()
            if verbose: self.print_cost_and_rates()
        else:
            response_text = ''

        self.dialog_history.append({"role": "assistant", "content": response_text}, )

        if verbose > 1:
            print()
            print(self.dialog_history_to_str(pure_completion_mode=pure_completion_mode))

        if enable_pdb:
            import pdb;
            pdb.set_trace()

        if enable_api:
            if not turn_off_cache:
                self.cache.save_cache(cache_input, response_text)

        if only_response:
            return response_text
        return response_text #, output QUES: why is there an ouptut, you removed it in the previous commits?

def main():
    raw_text = 'Hello, world. Here are two people with M.A. degrees from UT Austin. This is Mr. Mike.'
    nlp = NLP()
    sentences = nlp.sent_tokenize(raw_text)
    words = nlp.word_tokenize(sentences[0], lower=True)
    show_var(['sentences', 'words'])

    chat = Chatbot()
    query = 'What is the best way to learn Machine Learning?'
    response = chat.ask(query)


if __name__ == '__main__':
    main()
