import os, json, re
import logging
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (ChatPromptTemplate, HumanMessagePromptTemplate)
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from collections import namedtuple

CAITranslationResult = namedtuple(
  'CAITranslationResult',
  ['translation', 'pending', 'text', 'language', 'InputLanguage', 'Flags']
)

class CAIAssistant:
  def __init__(self, promptsFolder=None, openai_api_key=None):
    if promptsFolder is None:
      promptsFolder = os.path.join(os.path.dirname(__file__), '../data')
    self._promptsFolder = promptsFolder
    self.bindAPI(openai_api_key=openai_api_key)
    return

  # Hacky way to switch API key
  def bindAPI(self, openai_api_key):
    self._connected = False
    try:
      promptsFolder = self._promptsFolder
      self._LLM = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key)
      self._translateShallowQuery = LLMChain(
        llm=self._LLM,
        prompt=ChatPromptTemplate.from_messages([
          HumanMessagePromptTemplate(
            prompt=PromptTemplate.from_file(
              os.path.join(promptsFolder, 'translate_shallow.txt'),
              input_variables=['UserInput', 'FastTranslation', 'Language']
            )
          ),
        ]),
      )
      self._translateDeepQuery = LLMChain(
        llm=self._LLM,
        prompt=ChatPromptTemplate.from_messages([
          HumanMessagePromptTemplate(
            prompt=PromptTemplate.from_file(
              os.path.join(promptsFolder, 'translate_deep.txt'),
              input_variables=['UserInput', 'FastTranslation', 'Language', 'Flags', 'InputLanguage']
            )
          ),
        ]),
      )
      self._connected = True
    except Exception as e:
      logging.error('Failed to bind API: ' + str(e))
    return
  
  def _extractParts(self, text):
    text = '\n' + text + '\n' # hack to make it work
    tmp = [x for x in text.split('\n@')]
    # split into parts by :
    tmp = [tuple(y.strip('\n" \t\r\'`{}') for y in x.split(':', maxsplit=1)) for x in tmp]
    # remove empty parts
    tmp = [x for x in tmp if (len(x) == 2) and (len(x[0]) > 0) and (len(x[1]) > 0)]
    return {k: v for k, v in tmp}
  
  def _executePrompt(self, prompt, variables):
    if not self._connected: raise Exception('Not connected to API')

    rawPrompt = prompt.prompt.format_prompt(**variables).to_string()
    logging.info('Raw prompt: ' + rawPrompt)
    res = prompt.run(variables)
    logging.info('Raw result: ' + res)
    res = self._extractParts(res)
    logging.info('Extracted result: ' + json.dumps(res, indent=2))
    flags = {
      k: v.lower() == 'yes'
      for k, v in res.items()
      if v.lower() in ['yes', 'no']
    }
    # remove flags from result
    res = {k: v for k, v in res.items() if k not in flags}
    # add flags as separate variable
    res['Flags'] = flags
    return res
  
  def _translateShallow(self, text, translation, language):
    res = self._executePrompt(
      self._translateShallowQuery,
      {
        'UserInput': text,
        'FastTranslation': translation,
        'Language': language
      }
    )
    translation = res['Translation']
    flags = res['Flags']
    totalIssues = sum([int(v) for v in flags.values()])
    done = (totalIssues < 2)
    return res, translation, done
  
  def _translateDeep(self, text, translation, language, inputLanguage, flags):
    # extract first word from input language, can be separated by space, comma, etc.,
    inputLanguage = re.split(r'[\s,]+', inputLanguage)[0]
    inputLanguage = inputLanguage.strip().capitalize()

    res = self._executePrompt(
      self._translateDeepQuery,
      {
        'UserInput': text,
        'FastTranslation': translation,
        'Language': language,
        'InputLanguage': inputLanguage,
        'Flags': ', '.join([k for k, v in flags.items() if v])
      }
    )
    return res['Translation']
  
  def translate(self, text, fastTranslation, language):
    # run shallow translation
    raw, translation, done = self._translateShallow(
      text=text, translation=fastTranslation, language=language
    )
    translationResult = CAITranslationResult(
      translation=translation, pending=not done,
      text=text, language=language,
      InputLanguage=raw.get('Input language', 'unknown'),
      Flags=raw['Flags'],
    )
    yield translationResult
    if not done: # run deep translation
      yield self.refine(translationResult)
    return
  
  def refine(self, previousTranslation: CAITranslationResult):
    res = self._translateDeep(
      text=previousTranslation.text,
      translation=previousTranslation.translation,
      language=previousTranslation.language,
      inputLanguage=previousTranslation.InputLanguage,
      flags=previousTranslation.Flags,
    )

    translationResult = CAITranslationResult(
      translation=res,
      text=previousTranslation.text, language=previousTranslation.language,
      InputLanguage=previousTranslation.InputLanguage,
      Flags=previousTranslation.Flags,
      pending=False, # no more steps
    )
    return translationResult