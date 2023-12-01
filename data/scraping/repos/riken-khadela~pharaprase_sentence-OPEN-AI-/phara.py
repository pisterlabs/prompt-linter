from django.core.management.base import BaseCommand
from home.models import Text, ParaphrasedText
import openai, os, re, threading, time
openai.api_key = '<API_KEY>'

class Command(BaseCommand):
    help = '''this commands helps to use paraphrase sentence'''
    def generate_prompt(self,sent):
        return """paraphrase extremely 50 times the following sentence "{}"
        """.format(
                sent.capitalize()
        )

    
    def handle(self, *args, **options):
        ParaphrasedText.objects.all().delete()
        threads = []
        all = list(Text.objects.all())
        while True :
            for i in range(10):
                if not all : return 
                sentence = all.pop(0)
                t = threading.Thread(target=self.pharaphrase, args=(sentence,))
                t.start()
                threads.append(t) 
                
            for t in threads:
                t.join()

    def pharaphrase(self,sentence):
        for i in range(4):
            try:
                        
                response = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=self.generate_prompt(sentence.text),
                    max_tokens=4000
                    )
                sentence.Paraphrased = True
                sentence.save()
                phara_sentence = re.sub(r'\d+\.', '', response.choices[0].text)
                phara_sentence_li = phara_sentence.split('\n')
                number_count = 1
                print('results :', len(phara_sentence_li), 'in' ,sentence.text)
                for phara_sen in phara_sentence_li :
                    if not phara_sen : continue
                    ParaphrasedText.objects.create(
                        sentence = sentence,
                        response = phara_sen,
                        number = number_count 
                        )
                    
                    number_count += 1
                break
            except Exception as e:
                print('API got a token limit') 
                time.sleep(20)