from lyc.data import get_hf_ds_scripts_path
from datasets import load_dataset
import pandas as pd
import pickle
from lyc.utils import DeepLTranslator
from nltk.corpus import wordnet as wn
from nltk import WordNetLemmatizer
from random import choice
import openai
import os

# openai.api_key = os.environ['OPENAI_API_KEY']

class Task:

    available_tasks = ['wsd', 'nli', 'sentiment', 'translation', 'prep']

    def __init__(self, task, save_path, num_instances, pos_included, pos_dfs):
        assert task in self.available_tasks, f"task {task} not available"
        self.task = task
        self.pos_included = pos_included
        self.pos_dfs = pos_dfs

        self.save_path_table = save_path + f"/{task}_{num_instances}.tsv"
        self.save_path_text = save_path + f"/annotation_form_{task}_{num_instances}.md"

        self.wn_pos_map = {'NOUN': 'n', 'VERB': 'v', 'ADJ': 'a', 'ADV': 'r'}
        self.num_instances_per_pos = {pos: len(df.index) for pos, df in pos_dfs.items()}

        self.num_instances = num_instances
        self.pos_included = pos_included
    
    def _sample_instances(self, pos, n=10, new_dfs = None):
        if new_dfs is not None:
            df = new_dfs[pos]
        else:
            df = self.pos_dfs[pos]
        return df.sample(n)
        
    def _num_instances_to_sample_per_pos(self):
        num_instances_per_pos = [self.num_instances_per_pos[pos] for pos in self.pos_included]
        ratios = [num_instances/sum(num_instances_per_pos) for num_instances in num_instances_per_pos]
        return [int(ratio*self.num_instances) for ratio in ratios]
    
    def _get_instances(self, new_col_name = None, process_func = None):
        dfs = {pos: self.pos_dfs[pos] for pos in self.pos_included}

        if new_col_name is None:
            new_dfs = None
        else:
            new_dfs = {}
            for pos, df in dfs.items():
                df[new_col_name] = df.apply(process_func, axis=1)
                new_dfs[pos] = df[df[new_col_name].notna()]

        num_instances_per_pos = self._num_instances_to_sample_per_pos()
        instances = [self._sample_instances(pos, n=num_instances, new_dfs=new_dfs) for pos, num_instances in zip(self.pos_included, num_instances_per_pos)]
        self.instances = pd.concat(instances)

        return self.instances

    def _prepare_resources(self):
        pass

    def _make_question_and_answer(self):
        pass

    def _realise_to_textual_format(self):
        pass

class SentimentTask(Task):

    def __init__(self, task, pos_dfs, save_path, num_instances = 120, save_table = True, pos_included = ['ADV', 'ADJ', 'VERB']):
        super().__init__(task, save_path, num_instances, pos_included, pos_dfs)
        
        self.save_table = save_table

        self._prepare_resources()
        self._make_question_and_answer()
        self._realise_to_textual_format()
    
    def _prepare_resources(self):
        with open('/Users/yucheng/projects/EmpatheticMeta/data/opinion_lexicon/opinion-words.pickle', 'rb') as f:
            self.opinion_lexicon = pickle.load(f)
        self.lemmatizer = WordNetLemmatizer()
    
    def _make_question_and_answer(self):
        pos_included = self.pos_included
        num_instances = self.num_instances
        
        def get_sentiment_label(x):
            pos = x['pos']
            target = x['target']
            target_lemma = self.lemmatizer.lemmatize(target, pos=self.wn_pos_map[pos])
            if target_lemma in self.opinion_lexicon:
                return self.opinion_lexicon[target_lemma]
            else:
                return None

        self.instances = self._get_instances(new_col_name='sentiment', process_func=get_sentiment_label)

        if self.save_table:
            self.instances.to_csv(self.save_path_table, sep='\t', index=False)
            print(f"Table saved to {self.save_path_table}")

        return self.instances
    
    def _realise_to_textual_format(self):
        def realise_to_textual_format(x):
            pos = self.wn_pos_map[x['pos']]
            target = x['target']
            sentence = x['sentence']
            sentiment = x['sentiment']
            return f"What is the sentiment of **\"{target}\"** in the following sentence? Positive (1), Negative (2), or Neutral (0)?\n\n{sentence}\n\nYour answer: \n\n----------------\n"
        
        self.instances['textual_format'] = self.instances.apply(realise_to_textual_format, axis=1)
        with open(self.save_path_text, 'w') as f:
            for text in self.instances['textual_format']:
                f.write(text)
        
        print(f"Textual format of task {self.task} saved to {self.save_path_text}")

class TranslationTask(Task):

    def __init__(self, task, pos_dfs, save_path, num_instances = 160, save_table = True, pos_included = ['ADV', 'ADJ', 'VERB', 'NOUN']):
        super().__init__(task, save_path, num_instances, pos_included, pos_dfs)
        
        self.save_table = save_table

        self._prepare_resources()
        self._make_question_and_answer()
        self._realise_to_textual_format()
    
    def _prepare_resources(self):
        self.translator = DeepLTranslator()
    
    def _make_question_and_answer(self):
        pos_included = self.pos_included
        num_instances = self.num_instances

        instances = self._get_instances()

        def input_format(x):
            pos = x['pos']
            target = x['target']
            sentence = x['vanilla_sentence']
            return sentence.replace(f'[{target}]', target)
        
        sentences_to_translate = instances.apply(input_format, axis=1)
        translations = self.translator.translate(sentences_to_translate, source_lang='EN', target_lang='ZH')
        instances['translation'] = [str(res) for res in translations]

        if self.save_table:
            instances.to_csv(self.save_path_table, sep='\t', index=False)
            print(f"Table saved to {self.save_path_table}")
        
        self.instances = instances
        return instances

    def _realise_to_textual_format(self):
        def realise_to_textual_format(x):
            pos = x['pos']
            target = x['target']
            sentence = x['sentence']
            translation = x['translation']
            return f"Does the translation of **\"{target}\"** in the following example make sense?:\n\n**Original Sentence**: {sentence}\n\n**Translation**: {translation}\n\nYour answer [Correct: 1, Wrong: 0]: \n\n----------------\n\n"
        
        self.instances['textual_format'] = self.instances.apply(realise_to_textual_format, axis=1)
        with open(self.save_path_text, 'w') as f:
            for text in self.instances['textual_format']:
                f.write(text)
        
        print(f"Textual format of task {self.task} saved to {self.save_path_text}")

class WSDTask(Task):

    def __init__(self, task, pos_dfs, save_path, num_instances = 160, max_num_senses = 10, save_table = True, pos_included = ['VERB', 'NOUN']):
        super().__init__(task, save_path, num_instances, pos_included, pos_dfs)
        
        self.save_table = save_table
        self.max_num_senses = max_num_senses

        self._make_question_and_answer()
        self._realise_to_textual_format()

    def _make_question_and_answer(self):
        pos_included = self.pos_included
        num_instances = self.num_instances

        def get_sense_glosses(x):
            pos = self.wn_pos_map[x['pos']]
            target = x['target']
            sentence = x['sentence']

            # Not sure whether needs lemmatisation or not
            synsets = wn.synsets(target, pos=pos)
            if len(synsets) == 0:
                return None
            glosses = [synset.definition() for synset in synsets]
            if len(glosses) > self.max_num_senses:
                return None
            return glosses
        
        instances = self._get_instances(new_col_name='sense_glosses', process_func=get_sense_glosses)
                
        self.instances = instances
        return instances
    
    def _realise_to_textual_format(self):
        def realise_to_textual_format(x):
            pos = x['pos']
            target = x['target']
            sentence = x['sentence']
            sense_glosses = x['sense_glosses']
            gloss_string = [f'- [ ] {gloss}' for index, gloss in enumerate(sense_glosses)]
            gloss_string = '\n'.join(gloss_string)
            return f"What is the meaning of **\"{target}\"** in the following sentence?\n\n{sentence}\n\nYour answer (choose one of the following): \n{gloss_string}\n----------------\n"
        
        self.instances['textual_format'] = self.instances.apply(realise_to_textual_format, axis=1)
        with open(self.save_path_text, 'w') as f:
            for text in self.instances['textual_format']:
                f.write(text)
        
        print(f"Textual format of task {self.task} saved to {self.save_path_text}")

class NLIQuestion(Task):

    def __init__(self, task, pos_dfs, save_path, num_instances = 200, max_lemma_substitues = 12, save_table = True, pos_included = ['ADV', 'ADJ', 'VERB', 'NOUN']):
        super().__init__(task, save_path, num_instances, pos_included, pos_dfs)
        
        self.save_table = save_table
        self.max_lemma_substitues = max_lemma_substitues

        self._prepare_resources()
        self._make_question_and_answer()
        self._realise_to_textual_format()
    
    def _prepare_resources(self):
        self.lemmatizer = WordNetLemmatizer()
    
    def _make_question_and_answer(self):
        pos_included = self.pos_included
        num_instances = self.num_instances

        def get_substitues(x):
            pos = self.wn_pos_map[x['pos']]
            target = x['target']
            sentence = x['sentence']
            
            # Not sure whether needs lemmatisation or not
            synsets = wn.synsets(target, pos=pos)
            target_lemma = self.lemmatizer.lemmatize(target, pos=pos)

            substitues = set([ lemma.name() for synset in synsets for lemma in synset.lemmas()])
            substitues.discard(target_lemma)
            substitues.discard(target)

            if len(substitues) == 0 or len(substitues) > self.max_lemma_substitues:
                return None

            return substitues
        
        def choose_substitues(x, random = False):
            sentence = x['sentence']
            substitues = x['substitues']

            # option 1: Randomly choose one substitue
            if random:
                return choice(list(substitues))

            # option 2: use ChatGPT to choose one substitue
            prompt = f"Choose a word from the given word list that can fit in the place of the bolded word marked with ** in the sentence, return only the chosen word and nothing else:\n - The sentence -: {sentence}\n - word candidates -: {list(substitues)}\n\n"
            try:
                response = openai.ChatCompletion.create(
                    model = 'gpt-3.5-turbo',
                    messages = [
                        {"role": "user", "content": prompt}
                    ]
                )
                sub = response['choices'][0]['message']['content']
                return sub
            except:
                return choice(list(substitues))        
        # instances = self._get_instances(new_col_name='substitues', process_func=get_substitues)
        instances = self._get_instances(new_col_name='substitues', process_func=get_substitues)
        instances['substitue'] = instances.apply(choose_substitues, axis=1)
        
        self.instances = instances
        return instances
    
    def _realise_to_textual_format(self):
        def realise_to_textual_format(x):
            pos = x['pos']
            target = x['target']
            sentence = x['sentence']
            substitue = x['substitue']
            # substitues = x['substitues']
            # return f"{sentence}\n {str(substitues)}\n\n"
            return f"Does the following sentence pair semantically equivalent?\n\n1. {sentence} \n2. {sentence.replace(target, substitue)}\n\nYour answer [Correct: 1, Wrong: 0]: \n\n----------------\n"
        
        self.instances['textual_format'] = self.instances.apply(realise_to_textual_format, axis=1)
        with open(self.save_path_text, 'w') as f:
            for text in self.instances['textual_format']:
                f.write(text)
        
        print(f"Textual format of task {self.task} saved to {self.save_path_text}")

class RelationOfPrep(Task):

    def __init__(self, task, pos_dfs, save_path, num_instances = 80, save_table = True, pos_included = ['ADP']):
        super().__init__(task, save_path, num_instances, pos_included, pos_dfs)
        
        self.save_table = save_table

        self._make_question_and_answer()
        self._realise_to_textual_format()
    
    def _prepare_resources(self):
        self.lemmatizer = WordNetLemmatizer()
    
    def _make_question_and_answer(self):
        pos_included = self.pos_included
        num_instances = self.num_instances
        
        instances = self._get_instances()
        
        self.instances = instances
        return instances
    
    def _realise_to_textual_format(self):
        TopRelationChoices = """- [ ] Location: indicating where something is in relation to something else (e.g. "on the table," "in the box")
- [ ] Time: indicating when something occurs (e.g. "at 5 o'clock," "on Monday")
- [ ] Direction: indicating the movement of something (e.g. "to the store," "from here")
- [ ] Manner: indicating how something is done (e.g. "with a smile," "by bus")
- [ ] Cause or reason: indicating why something happens (e.g. "because of you," "due to the weather")
- [ ] Agent or instrument: indicating the person or thing that causes something to happen (e.g. "by the teacher," "with a pen")
- [ ] Possession: indicating ownership or a relationship of belonging (e.g. "the book of John," "the child's toy")
- [ ] Comparison: indicating similarity or difference (e.g. "like a bird," "unlike you")"""

        ALLRelationChoices = "1. Agent\n2. Patient\n3. Instrument\n4. Location\n5. Time\n6. Manner\n7. Cause\n8. Beneficiary\n9. Source\n10. Goal\n11. Theme\n12. Experiencer\n13. Attribute\n14. Recipient\n15. Stimulus\n16. Product\n17. Instrument\n18. Other"

        def realise_to_textual_format(x):
            pos = x['pos']
            target = x['target']
            sentence = x['sentence']
            return f"What is the relation of **\"{target}\"** in the following sentence?\n\n{sentence}\n\nYour answer (choose one of the following): \n{TopRelationChoices}\n----------------\n"
        
        self.instances['textual_format'] = self.instances.apply(realise_to_textual_format, axis=1)
        with open(self.save_path_text, 'w') as f:
            for text in self.instances['textual_format']:
                f.write(text)
        
        print(f"Textual format of task {self.task} saved to {self.save_path_text}")

class DownstreamQuestion:
    
    ADJ, ADJ_SAT, ADV, NOUN, VERB = "a", "s", "r", "n", "v"

    def __init__(self):
        
        script = get_hf_ds_scripts_path('vua20')
        self.vua = load_dataset(script, data_dir='VUA20')

        self.meta_dfs = self._prepare_dfs(meta=True, split='train')
        self.non_meta_dfs = self._prepare_dfs(meta=False, split='train')

        self.write_questions(self.meta_dfs, save_path = 'VUA20/meta_tasks_questions')
        self.write_questions(self.non_meta_dfs, save_path = 'VUA20/non_meta_tasks_questions')

    def write_questions(self, pos_dfs, save_path):
        # SentimentTask('sentiment', pos_dfs, save_path, save_table=False)
        RelationOfPrep('prep', pos_dfs, save_path)
        # NLIQuestion('nli', pos_dfs, save_path)
        # WSDTask('wsd', pos_dfs, save_path)
        # TranslationTask('translation', pos_dfs, save_path, save_table=False)

        print("All tasks saved to ", save_path)

    def _prepare_dfs(self, meta = True, split = 'test'):

        sorted_by_pos = {}
        flag = 1 if meta else 0

        for i in self.vua[split]:
            pos = i['pos']
            if i['label'] != flag: continue
            if pos not in sorted_by_pos:
                sorted_by_pos[pos] = []
            sorted_by_pos[pos].append(i)
        
        pos_dfs = {}
        for pos, sets in sorted_by_pos.items():
            df = pd.DataFrame(sets)
            pos_dfs[pos] = df
        
        for pos, df in pos_dfs.items():
            if not len(df.index): continue
            target = df.apply(DownstreamQuestion.target_word, axis=1)
            vanilla_sents = df.apply(DownstreamQuestion.sentencer, axis=1, args=(False,))
            sents = df.apply(DownstreamQuestion.sentencer, axis=1)
            df = df.drop(['tokens', 'sent_id'], axis=1)
            df['target'] = target
            df['sentence'] = sents
            df['vanilla_sentence'] = vanilla_sents
            pos_dfs[pos] = df
        
        return pos_dfs
    
    @staticmethod
    def sentencer(x, bold_target=True):
        tokens = x['tokens']
        target = tokens[x['word_index']]
        if bold_target:
            tokens[x['word_index']] = f"**{target}**"
        return ' '.join(tokens)

    @staticmethod
    def target_word(x):
        return x['tokens'][x['word_index']]

        self.pos_dfs = pos_dfs
        self.pos = list(pos_dfs.keys())

        self.wn_pos_map = {'NOUN': 'n', 'VERB': 'v', 'ADJ': 'a', 'ADV': 'r'}
        self.num_instances_per_pos = {pos: len(df.index) for pos, df in pos_dfs.items()}
        self.available_tasks = ['wsd', 'nli', 'sentiment', 'translation', 'prep']
        self._prepare_resources()


if __name__ == '__main__':
    DownstreamQuestion()