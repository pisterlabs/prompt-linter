# encoding: utf-8
import spacy
from coherenceanalyzer.analyzerenglish import CohesionAnalyzerEnglish


# Load language models
nlp_english = spacy.load('en_core_web_md')
# nlp_german = spacy.load('de_core_news_md')

analyzer_english = CohesionAnalyzerEnglish(nlp_english)
# analyzer_german = CohesionAnalyzerEnglish(nlp_german)
