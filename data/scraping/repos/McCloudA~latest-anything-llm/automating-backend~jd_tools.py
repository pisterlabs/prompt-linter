from __future__ import print_function

import openai
#import json
import numpy as np
import time
from sklearn.metrics.pairwise import cosine_similarity
#import re
import six
import pdfminer.settings
pdfminer.settings.STRICT = False
import pdfminer.high_level
import docx
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

#import langchain
#from langchain.text_splitter import TokenTextSplitter


class CheekiFileHandler:
    def __init__(self):
        self.filename = None

    def text_from_file(self, filename):
        self.filename = filename
        full_text = ''
        #if it's a PDF:
        if '.pdf' in self.filename or '.PDF' in self.filename:
            try:
                full_text = pdfminer.high_level.extract_text(self.filename)
            except Exception as e:
                print('PDF Reader Exception: ', e)
        elif '.txt' in self.filename or '.TXT' in self.filename:
            try:
                with open(self.filename, 'r') as file:
                    full_text = file.read()
            except Exception as e:
                print('Text Reader Exception: ', e)
        elif '.docx' in self.filename or '.DOCX' in self.filename:
            try:
                doc = docx.Document(self.filename)
                full_text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
            except Exception as e:
                print('Docx Reader Exception: ', e)
        return full_text

class LLMEmbedder:
    def __init__(self):
        self.api_key = None

    def set_up_embedder(self, api_key):
        self.api_key = api_key
        openai.api_key = self.api_key


    def embed(self, prompt_text):
        # the extra "machinery" here is to keep asking the API until we get a good answer.
        # this thing is a bit janky, and often returns "APIConnectionError: Error communicating with OpenAI: ('Connection aborted.', ConnectionResetError(104, 'Connection reset by peer'))"
        # for essentially no reason.  So we're going to use a rate limiter, and also ask until we get an answer.

        #important:  truncate the text if it's too long for our embedder:
        if len(prompt_text) > 20000:
            prompt_text = prompt_text[0:20000]

        output = None
        while output is None:
            time.sleep(0.4) # avoid hitting rate limiter
            #remember 46,000 seconds is about 12 hours, and 
            # the "rate limit" stated on the OpenAI site is 200 tokens per minute
            #https://platform.openai.com/docs/guides/rate-limits/overview
            try:
                response = openai.Embedding.create(
                    input='Please Read the following text and summarize it in 400 words: ' + prompt_text,
                    model="text-embedding-ada-002"
                )
                output = response['data'][0]['embedding']
            except Exception as e:
                print('Error in API Call, retrying API embedding query: ')
                print(e)
        return output
    
    def chatgpt_simpleresponse(self, prompt_text):
        ret_text = None
        while ret_text is None:
            time.sleep(0.4) # avoid hitting rate limiter
            #remember 46,000 seconds is about 12 hours, and 
            # the "rate limit" stated on the OpenAI site is 200 tokens per minute
            #https://platform.openai.com/docs/guides/rate-limits/overview
            try:
                completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                    {"role": "user", "content":prompt_text}
                    ]
                )
                ret_text = completion['choices'][0]['message']['content'].replace('\n','')
            except Exception as e:
                print('Error in API Call, retrying API query: ')
        return ret_text , completion
    
    # TBD
    # def chatgpt_adaptiveresponse(self, context_text, prompt_text):
    #     # split up the context data (resume or whatever) into bite-sized chunks.
    #     # if there's only one, then just ask the algorithm for the prompt text.
    #     # if more than one, first load up the algorithm with the chunks, then ask.
    #     text_splitter = TokenTextSplitter(chunk_size=50, chunk_overlap=0) # 1000
    #     #context_split = text_splitter.split_documents(context_text)
    #     context_split = text_splitter.create_documents(context_text)
    #     print(context_split)
    #     return context_split

# TODO:  use this as well to make the ClusterComparison
class BOWEmbedder:
    def __init__(self):
        self.api_key = None
        self.stop_words = set(stopwords.words('english'))
        self.extra_stopwords = ['unusednonsense', 'phrenology']

    def set_up_embedder(self, full_text_corpus):
        self.vectorizer = TfidfVectorizer(ngram_range = (1,2))
        #self.vectorizer.fit(self.remove_stopwords([full_text_corpus]))
        #print(full_text_corpus)
        self.vectorizer.fit(self.remove_stopwords(full_text_corpus))

    def remove_stopwords(self, text_in):
        text_out = []
        for i, line in enumerate(text_in):
            #text_out[i] = ' '.join([x for 
            text_out_line = ' '.join([x for 
                x in nltk.word_tokenize(line) if 
                ( x not in self.stop_words ) and ( x not in self.extra_stopwords )])
            text_out.append(text_out_line)
        return text_out

    def embed(self, prompt_text):
        body_embedded = self.vectorizer.transform([prompt_text]).toarray()[0]
        return body_embedded
    

class ContentEmbedder:
    # Here we take in a list of json objects, plus a list of the keys we care about and some vector lengths. For example, we might get:
    #[
    #    {job_id: 'a3hflg98053egh'
    #    job_title:'Legal Assistant 2',
    #    job_description: 'blah blah you need to assist our legal department',
    #    job_required_education: '4 years experience or relevant degree'
    #    },
    #    {<same_stuff_for_the_next_job>}
    #], and ['job_title',job_desxription'], and [100,10].
    # in this case, we'd want to return a json of embeddings for each desired field, and for each job, like so:
    #
    # [
    #    {job_id: 'a3hflg98053egh'
    #     job_title_embed_LLM_dim_all: <vector of all embedded elements for the job title for this job>,
    #     job_title_embed_LLM_dim_100: <reduced vector of 100 embedded elements for the job title for this job>,
    #     job_title_embed_LLM_dim_10: <reduced vector of 10 embedded elements for the job title for this job>,
    #    },
    #    {job_id: 'eoqjnf9un3n9on'
    #     ...
    #     job_title_embed_LLM_dim_10: <reduced vector of 10 embedded elements for the job title for this job>
    #    }
    # ]
    # ultimately we'll use these to compare against each other or against resumes.

    def __init__(self):
        self.dummy = 0

    def add_api_key(self, api_key):
        self.api_key = api_key
        self.llmembedder = LLMEmbedder()
        self.llmembedder.set_up_embedder(api_key=self.api_key)

    def setup_bow_embedder(self,full_text_corpus):
        self.bowembedder = BOWEmbedder()
        self.bowembedder.set_up_embedder(full_text_corpus)


    def embed_content(self, job_input_json_list, relevant_fields, vector_truncations,uid_name):
        if len(job_input_json_list) == 0 or len(relevant_fields) == 0: # or len(vector_truncations) == 0:
            return []
        
        ret_embeddings = []
        for job in job_input_json_list:
            embedded_job = {}
            for field in relevant_fields:
                #go get the embedded vector for this field
                relevant_text = job[field]
                # Use The OpenAI Embedding!
                #full_vector_llm = self.llmembedder.embed(relevant_text)
                # Use the Bag of Words Embedding!
                full_vector_llm = self.bowembedder.embed(relevant_text)

                #print(job)
                embedded_job[uid_name] = job[uid_name]
                embedded_job[field + '_embed_LLM_dim_all'] = full_vector_llm
                try: # only keep this for jobs where it exists, not people:
                    embedded_job['title'] = job['title']
                    embedded_job['locations'] = job['locations']
                except:
                    #do nothing
                    dummy = 0

                # print('full_vector_llm:')
                # print(full_vector_llm)
                for trunc_num in vector_truncations:
                    # HACK: this is just literally truncating the vector.  Now, we could 
                    # instead try to do a smarter dimensional redution here, or we could
                    # rely on the pre-trained power of the LLM embedder.  Think about this later.
                    # the point is to get some shorter vectors to do a quicker throw-out of low
                    # similarity jobs.
                    tl = np.min([trunc_num,len(full_vector_llm)]) # just in case we try to truncate longer than the original vector
                    # print('Truncating to ', str(tl), ' Elements:')
                    trunc_vector_llm = full_vector_llm[0:tl]
                    embedded_job[field + '_embed_LLM_dim_' + str(int(trunc_num))] = trunc_vector_llm
                
            ret_embeddings.append(embedded_job)
        return ret_embeddings

        

class JobScorer:
    # Here we take in many job details, and an applicant's details, and find the best matching jobs for them!
    # To do this, we:
    # - compare the relevant embeddings of the applicant with each job (shortest vector first)
    # - keep the jobs that meet the threshold tests and do the comparison again with the next longest vector
    # - continue this until we have a set of closest jobs, return them in order with their similarity scores,
    #   for the UI to order.

    def __init__(self):
        self.dummy = 0


    def eligible_jobs(self,job_input_json_list,applicant_input_json_list):
        #TODO: ensure locations are in both job and applicant,(i.e. job_input_json_list[0]), and use some trick or library to ensure proximity
        # For now, this is just stubbed out as "all jobs are eligible"
        # To fix this, first ensure that locations are in both of these inputs, and then put them up first in the 
        # relevant comapirsons (?), and then use the last, rather than first, comparison as the main score.
        # should be much faster.

        max_miles_commute = 25
        #print(job_input_json_list[0])
        #print(applicant_input_json_list[0])
        #print()
        return job_input_json_list.copy()

    def best_jobs(self, previously_applied_uuids, job_input_json_list, applicant_input_json_list, relevant_fields_jobs, \
                  relevant_fields_applicant, vector_truncations,uid_name_jobs, uid_name_applicant \
                    ,vector_comparison_threshold_list,full_vector_comparison_threshold):
        
        #NOTE:  We're making an "eligibility" cut here first - only keep the subset of jobs that match location/pay level etc.
        jobs_eligible = self.eligible_jobs(job_input_json_list,applicant_input_json_list)
        # applicant 'locations' and job location, etc

        jobs_remaining = []
        if len(vector_truncations)==0:
            jobs_remaining = jobs_eligible.copy()
        else:
            for job in jobs_eligible:
                job_remaining = True
                for relfield_ind in range(len(relevant_fields_jobs)):
                    relfield_jobs = relevant_fields_jobs[relfield_ind]
                    relfield_applicant = relevant_fields_applicant[relfield_ind]
                    for trunc_ind in range(len(vector_truncations)):
                        trunc = vector_truncations[trunc_ind]
                        thresh = vector_comparison_threshold_list[trunc_ind]
                        #okay, now for this set of comparison fields, for this set of thresholds, what jobs make the grade?
                        # this stuff takes awhile, so only keep doing it if the job has survived all challenges:
                        if job_remaining:
                            #compare and cull!
                            job_vector = job[relfield_jobs + '_embed_LLM_dim_' + str(int(trunc))]
                            applicant_vector = applicant_input_json_list[0][relfield_applicant + '_embed_LLM_dim_' + str(int(trunc))]
                            cosine_angle, job_survives = self.job_similarity_test(job_vector,applicant_vector,thresh)
                            #print(cosine_angle)
                            if job['uuid'] in previously_applied_uuids:
                                # we've already applied to this job, toss it!
                                job_survives = False
                            if job_survives == False:
                                #jobs_remaining.append(job)
                                job_remaining = False
                if job_remaining:
                    jobs_remaining.append(job)
        #now apply the reasoning to the final column/scoring (first comparison column for each):
        jobs_final = []
        jobs_final_scores = []
        for job in jobs_remaining:
            job_remaining = True
            #print('Considering Job: ' , job['title'])
            #original:
            # for relfield_ind in [0]:
            for relfield_ind in range(len(relevant_fields_jobs)):
                relfield_jobs = relevant_fields_jobs[relfield_ind]
                relfield_applicant = relevant_fields_applicant[relfield_ind]
                #no truncation - time to look at full vectors!
                thresh = full_vector_comparison_threshold
                #okay, now for this set of comparison fields, for this set of thresholds, what jobs make the grade?
                # this stuff takes awhile, so only keep doing it if the job has survived all challenges:
                if job_remaining:
                    #compare and cull!
                    #job_vector = job[relfield_jobs + '_embed_LLM_dim_' + str(int(trunc))]
                    #applicant_vector = applicant_input_json_list[0][relfield_applicant + '_embed_LLM_dim_' + str(int(trunc))]
                    job_vector = job[relfield_jobs + '_embed_LLM_dim_all']
                    applicant_vector = applicant_input_json_list[0][relfield_applicant + '_embed_LLM_dim_all']
                    score, job_survives = self.job_similarity_test(job_vector,applicant_vector,thresh)
                    #if job_survives: #job['title']=="Control Technician":
                    #    print('Considering Job: ' , job['title'])
                    #    print('score: ', score)
                    #    print('job survives: ', job_survives)
                    #print(score)
                    #print(job_vector)
                    #print(applicant_vector)
                    if job_survives == False:
                        job_remaining = False
            if job_remaining:
                    job['cheeki_score'] = score
                    jobs_final.append(job)
                    jobs_final_scores.append(score)
        #return the remaining jobs that were good enough, and a score for each.  We can hand these to the UI.
        return jobs_final, jobs_final_scores
            
                
    def job_similarity_test(self, job_vector,applicant_vector,thresh):
        flat_dist = np.linalg.norm(np.array(job_vector) - np.array(applicant_vector))
        cosine_angle = cosine_similarity([np.array(job_vector)],[np.array(applicant_vector)])[0][0]
        # print(cosine_angle)
        #return flat_dist, flat_dist < thresh
        return cosine_angle, cosine_angle > thresh
