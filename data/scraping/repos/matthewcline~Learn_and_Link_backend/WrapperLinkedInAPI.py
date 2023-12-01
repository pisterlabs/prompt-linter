

#from linkedin_api import Linkedin
from custom_linkedin_api import Linkedin
import os
import pickle
import ast
import anthropic
import cohere
import json

USERNAME = "shah.jaidev00@gmail.com"
PASSWORD = os.environ.get('LINKEDIN_PASSWORD')

ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')
COHERE_API_KEY = os.environ.get('COHERE_API_KEY')

claude = anthropic.Client(os.environ.get('ANTHROPIC_API_KEY'))
co = cohere.Client(os.environ.get('COHERE_API_KEY'))



class MyLinkedInAPI:
    def __init__(self, username, password):
        self.username = username
        self.password = password
        self.api = Linkedin(username, password)

    def get_profile_urn(self, profile_response):
        profile_urn_string = profile_response['profile_urn']
        profile_urn = profile_urn_string.split(":")[3]
        return profile_urn
    
    def get_profile_connections(self, profile_urn, keywords=None):
        connections = self.api.get_profile_connections(profile_urn)
        return connections
    
    def get_cleaned_profile_summary(self, profile_name):
        profile_dict= self.api.get_profile(profile_name)

        keys_to_keep = ['industryName', 'lastName', 'firstName', 'geoLocationName', 'headline', 'experience', 'education', 'projects']
        filtered_profile_dict = {key: profile_dict[key] for key in profile_dict if key in keys_to_keep}
        # the values of the dictionary
        filtered_profile_dict_str = str(filtered_profile_dict) 

        #Create a prompt for Claude to summarize the profile
        summarization_prompt = f"Clean up and summarize the below LinkedIn Profile such that all the key parts that are important when networking are retained:\n" \
                    f"{filtered_profile_dict_str}\n" \
                    "Profile Summary: "   
        
        #Call the Claude API to summarize the profile
        max_tokens_to_sample = 1000
        response = claude.completion(
            prompt=f"{anthropic.HUMAN_PROMPT} {summarization_prompt}{anthropic.AI_PROMPT}",
            stop_sequences=[anthropic.HUMAN_PROMPT],
            model="claude-instant-v1",
            max_tokens_to_sample=max_tokens_to_sample,
        )

        resp_summary = response['completion']

        #Return the summary
        return resp_summary

   

    def claude_extract_keywords(self, profile_name):
        profile_dict= self.api.get_profile(profile_name)
        keys_to_keep = ['industryName', 'lastName', 'firstName', 'geoLocationName', 'headline', 'experience', 'education', 'projects']
        filtered_profile_dict = {key: profile_dict[key] for key in profile_dict if key in keys_to_keep}
        
        summarization_prompt = f"Clean up and summarize the below LinkedIn Profile such that all the key parts that are important when networking are retained:\n" \
                    f"{filtered_profile_dict}\n" \
                    "Profile Summary: "   
        
        #Call the Claude API to summarize the profile
        max_tokens_to_sample = 1000
        response = claude.completion(
            prompt=f"{anthropic.HUMAN_PROMPT} {summarization_prompt}{anthropic.AI_PROMPT}",
            stop_sequences=[anthropic.HUMAN_PROMPT],
            model="claude-instant-v1",
            max_tokens_to_sample=max_tokens_to_sample,
        )
        resp_summary = response['completion']

        prompt_for_keywords= f"Given this LinkedIn profile: {resp_summary}, Extract the 5 most relevant keywords from this persons profile that would help him find good connections to network with. Extract the keywords as a list of strings such that the first keyword is the most aligned with the kind of connection the user is looking to make: "
        
        max_tokens_to_sample = 300
        resp_keywords = claude.completion(
            prompt=f"{anthropic.HUMAN_PROMPT} {prompt_for_keywords}{anthropic.AI_PROMPT}",
            stop_sequences=[anthropic.HUMAN_PROMPT],
            model="claude-instant-v1",
            max_tokens_to_sample=max_tokens_to_sample,
        )

        resp_keywords = resp_keywords['completion']
        print("Extracted Keywords")
        print(resp_keywords)
        return resp_keywords


    def get_job_posting(self, job_id):
        job_posting_dict = self.api.get_job(job_id)
        summarization_prompt = f"Clean up the below Job Posting such that all the key parts are retained:\n" \
                    f"{job_posting_dict}\n" \
                    "Job Posting Cleaned: "   
        
        #Call the Claude API to summarize the profile
        max_tokens_to_sample = 3000
        response = claude.completion(
            prompt=f"{anthropic.HUMAN_PROMPT} {summarization_prompt}{anthropic.AI_PROMPT}",
            stop_sequences=[anthropic.HUMAN_PROMPT],
            model="claude-v1",
            max_tokens_to_sample=max_tokens_to_sample,
        )
        resp_job_posting = response['completion']

        print(resp_job_posting)
        return resp_job_posting

 
    def helper_profile_summarize(self, profile_dict):
        keys_to_keep = ['industryName', 'lastName', 'firstName', 'geoLocationName', 'headline', 'experience', 'education', 'projects']
        filtered_profile_dict = {key: profile_dict[key] for key in profile_dict if key in keys_to_keep}
        
        summarization_prompt = f"Clean up and summarize the below LinkedIn Profile such that all the key parts that are important when networking are retained:\n" \
                    f"{filtered_profile_dict}\n" \
                    "Profile Summary: "   
        
        #Call the Claude API to summarize the profile
        max_tokens_to_sample = 1000
        response = claude.completion(
            prompt=f"{anthropic.HUMAN_PROMPT} {summarization_prompt}{anthropic.AI_PROMPT}",
            stop_sequences=[anthropic.HUMAN_PROMPT],
            model="claude-instant-v1",
            max_tokens_to_sample=max_tokens_to_sample,
        )

        resp_summary = response['completion']

        return resp_summary


    def intro_generation(self, candidate_profile_url, lead_profile_url, user_free_form_text):
        candidate_profile_name = candidate_profile_url.split("/")[-1]
        candidate_summary = self.get_cleaned_profile_summary(candidate_profile_name)

        lead_profile_name = lead_profile_url.split("/")[-1]
        lead_summary = self.get_cleaned_profile_summary(lead_profile_name)


        prompt = f"Assume you are a candidate looking for a job.  Write a warm personalized introduction message to the lead profile, that includes commonalities between the two of you, relevance to their own work and title, and and explaining why your skills would be a good fit for his team and the role you are interested in." \
                f"Here are the descriptions for each the candidate and the lead:\n\n" \
                    f"Candidate: {candidate_summary}\n\n" \
                    f"Lead: {lead_summary}" \
                    f"And here is the candidate's aspiration: {user_free_form_text}"
        
        # Call the Claude API to generate an intro
        max_tokens_to_sample: int = 1000
        resp = claude.completion(
            prompt=f"{anthropic.HUMAN_PROMPT} {prompt}{anthropic.AI_PROMPT}",
            stop_sequences=[anthropic.HUMAN_PROMPT],
            model="claude-v1",
            max_tokens_to_sample=max_tokens_to_sample,
        )

        print(resp)

        return resp['completion']

    def recommended_profiles(self,free_form_text, profile_name):
        profile_dict= self.api.get_profile(profile_name)
        profile_urn = self.get_profile_urn(profile_dict)
        cleaned_profile_summary = self.get_cleaned_profile_summary(profile_name)
        keywords = self.claude_extract_keywords(profile_name)
        pass 
    
    def run_k_hop_BFS(self, profile_name, keywords=None):
        print("Kicking off BFS")
        if keywords is None:
            top_5_keywords = self.claude_extract_keywords(profile_name)
            KEYWORDS = [top_5_keywords[0]]
        
        profile_dict= self.api.get_profile(profile_name)
        profile_id = profile_dict['profile_id']
        connections_with_filter = self.api.get_profile_connections( urn_id= profile_id, network_depths=['F','S','O'], keywords = KEYWORDS)

        second_degree_list = []
        visited_set = set()
        for connection in connections_with_filter:
            next_hop_connections = self.api.get_profile_connections(urn_id = connection['urn_id'], network_depths=['F', 'S', 'O'], keywords=KEYWORDS)
            print("For connection: ", connection['name'], " next_hop_connections are: ") 
            visited_set.add(connection['public_id'])
            for next_hop_connection in next_hop_connections:
                public_id = next_hop_connection['public_id']
                if public_id not in visited_set:
                    second_degree_list.append(next_hop_connection)
                    print(public_id)

        print("Length of second degree connections: ", len(second_degree_list))

        k_hop_connections = connections_with_filter + second_degree_list
        print([connection['profile_id'] for connection in k_hop_connections])


    def rerank_summarized_profiles_for_recommendation(self, profile_name, user_free_form_text, top_K = 5, read_from_pickle = False):
        
        if read_from_pickle:
            PATH = "software_k_hop_connections.pickle"
            bfs_recalled_profiles_list = pickle.load(open(PATH, "rb"))

        else:
            bfs_recalled_profiles_list = self.run_k_hop_BFS(profile_name)
        
        bfs_recalled_profiles_jsons_list = [json.dumps(profile) for profile in bfs_recalled_profiles_list if profile["public_id"] != profile_name]


        user_intent_query = user_free_form_text #"Looking to talk to a machine learning team manager at a large tech company for machine learning roles"
        query = user_intent_query    #+ "\n" + user_profile_summarized

        #ensure profile_name is not in the list
        
        print("Calling Cohere rerank to rerank the profiles based on similarity to the user intent")
        reranked_results = co.rerank(query=query, documents=bfs_recalled_profiles_jsons_list, top_n=10, model='rerank-english-v2.0')

        cleaned_reranked_results_dict = {}
        for r in reranked_results[:top_K]:
            profile_dict = r.document
            print(profile_dict)
            summarized_profile = self.helper_profile_summarize(r.document)
            #convert r.document string to dict
            #profile_dict = json.loads(profile)
            print(f"Document: {summarized_profile}")
            print(f"Relevance Score: {r.relevance_score}")
            public_id = ast.literal_eval(profile_dict['text'])['public_id']
            cleaned_reranked_results_dict[public_id] = summarized_profile
            
        return  cleaned_reranked_results_dict


    def helper_summarize_recalled_profiles(self, bfs_recalled_profiles_list):
        cnt = 0
        summarized_bfs_recalled_profiles_list= []
        for profile in bfs_recalled_profiles_list:
            summarized = self.helper_profile_summarize(profile)
            if cnt%10 ==0:
                print(profile["public_id"])
                print(summarized)
                print(cnt)
                cnt += 1

                summarized_bfs_recalled_profiles_list.append(summarized)


        #Save the summarized profiles to a pickle file
        pickle.dump(summarized_bfs_recalled_profiles_list, open("summarized_bfs_recalled_profiles_list.pickle", "wb"))
        return summarized_bfs_recalled_profiles_list
    
def main():
    myapi_wrapper = MyLinkedInAPI(USERNAME, PASSWORD)
    """
    test_job_id = 3595709943
    print("Job Posting")
    cleaned_posting = myapi_wrapper.get_job_posting(test_job_id)
    print(cleaned_posting)

    print("Profile Summary")
    profile_summary = myapi_wrapper.get_cleaned_profile_summary("ranganm")
    print(profile_summary)

    print("Keywords")
    keywords = myapi_wrapper.claude_extract_keywords("ranganm")
    print(keywords)
   

    user_free_form_text = "Looking for opportunities where I can use my applied machine learning skills at scale"
    intro_generated = myapi_wrapper.intro_generation("shahjaidev", "linjuny", user_free_form_text)
    print(intro_generated)
    """

    user_free_form_text = "Looking for opportunities where I can use my applied machine learning skills at scale"
    PROFILE_NAME = 'ethereal-shah-636388276'
    reranked = myapi_wrapper.rerank_summarized_profiles_for_recommendation(PROFILE_NAME, user_free_form_text, top_K = 5, read_from_pickle = True)
    print(reranked)
    print((len(reranked)))





main()

