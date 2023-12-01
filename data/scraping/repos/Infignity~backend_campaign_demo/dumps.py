# '''importing the libraries'''
# from django.db import models


# class Company(models.Model):
#     '''defining the models field'''
#     name = models.TextField()


# # Define the CompanyAnalysis model with a foreign key relationship to Company
# class CompanyAnalysis(models.Model):
#     '''defining the models field'''
#     company = models.ForeignKey(Company, on_delete=models.CASCADE)
#     description = models.TextField()
#     vision_mission = models.TextField()
#     position_statement = models.TextField()
#     key_audiences = models.TextField()
#     brand_image = models.TextField()
#     usp = models.TextField()
#     functionality = models.TextField()
#     innovative_aspects = models.TextField()
#     integration_compatibility = models.TextField()
#     cost_structure = models.TextField()
#     comparative_value = models.TextField()
#     demographics = models.TextField()
#     behavioral_traits = models.TextField()
#     motivations_goals = models.TextField()
#     issues_identified = models.TextField()
#     solutions_offered = models.TextField()
#     current_gaps = models.TextField()
#     mitigating_factors = models.TextField()
#     swot_analysis = models.TextField()
#     channels_overview = models.TextField()
#     efficacy_and_reach = models.TextField()
#     sales_cycle_duration = models.TextField()
#     key_touchpoints = models.TextField()
#     key_players = models.TextField()
#     strengths_weaknesses = models.TextField()
#     differentiation_opportunities = models.TextField()
#     value_proposition = models.TextField()
#     differentiators = models.TextField()
#     segmentation = models.TextField()
#     potential_reach = models.TextField()
#     market_size_growth = models.TextField()


# class CompanyProductInfo(models.Model):
#     '''product information'''
#     company = models.ForeignKey(Company, on_delete=models.CASCADE)
#     vision_and_mission = models.TextField()
#     usp = models.TextField()
#     functionality = models.TextField()
#     innovative_aspects = models.TextField()
#     integration_and_compatibility = models.TextField()
#     cost_structure = models.TextField()
#     comparative_value = models.TextField()
#     demographics = models.TextField()
#     behavioral_traits = models.TextField()
#     motivations_and_goals = models.TextField()
#     issues_identified = models.TextField()
#     solutions_offered = models.TextField()
#     strategy_name = models.CharField(max_length=255)
#     target_company_profile = models.TextField()
#     decision_maker_profile = models.TextField()
#     buyer_persona_details = models.TextField()
#     content_strategy = models.TextField()
#     follow_up_strategy = models.TextField()
#     feedback_loop_integration = models.TextField()
#     person_name = models.CharField(max_length=255)
#     person_job_role = models.CharField(max_length=255)
#     person_company = models.CharField(max_length=255)
#     company_keywords = models.TextField()
#     person_industry = models.CharField(max_length=255)
#     person_skills = models.TextField()
#     person_interests = models.TextField()
#     person_university = models.CharField(max_length=255)
#     linkedin_headline = models.CharField(max_length=255)
#     person_achievement = models.TextField()

#     def __str__(self):
#         return self.company.name


# class CompanyProfile(models.Model):
#     '''company profile'''
#     company = models.ForeignKey(Company, on_delete=models.CASCADE) 
#     # Demographic Characteristics
#     size = models.CharField(max_length=100)
#     industry = models.CharField(max_length=255)
#     location = models.CharField(max_length=255)
#     growth_trajectory = models.CharField(max_length=100)

#     # Technographic Data
#     technological_tools = models.TextField()
#     platforms = models.TextField()
#     systems = models.TextField()

#     # Unstructured Firmographic and Technographic Data
#     unstructured_data = models.TextField() 

#     def __str__(self):
#         return f"{self.size} {self.industry} Company Profile"
    

# class CompanyBlogPost(models.Model):
#     '''company blogs'''
#     company = models.ForeignKey(Company, on_delete=models.CASCADE) 
#     title = models.CharField(max_length=255)
#     problem_description = models.TextField()
#     solution_description = models.TextField()
#     results_description = models.TextField()
#     publication_date = models.DateField(auto_now_add=True)

#     def __str__(self):
#         return str(self.title)


# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework import status
# # program defined functions
# from .web_scraper import WebScrapping, ApolloCompany

# from .open_ai import LangChainAI
# from .models import Company
# from .serializers import CompanySerializer


# # def company_detail_api(request):
# #     try:
# #         company_id = request.session.get('company_id')

# #         if company_id:
# #             company = Company.objects.get(id=company_id)
# #             serializer = CompanySerializer(company)
# #             return Response(serializer.data)
# #         else:
# #             raise Http404("Company ID not found in the session")
# #     except Company.DoesNotExist:
# #         raise Http404("Company does not exist")
# #     except Exception as e:
# #         return Response({'error': str(e)}, 
# # status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# class ScrapeDataAPIView(APIView):
#     ''' scrawl urls and return the https body'''
#     def post(self, request, *args, **kwargs):
#         '''a post method to get url to crawl'''
#         website_url = request.data.get('website_url')
#         linkedin_url = request.data.get('linkedin_url')

#         if not (website_url and linkedin_url):
#             return Response(
#                 {"error": "URL is required"},
#                 status=status.HTTP_400_BAD_REQUEST
#                 )
#         # data = scrape_website(url)
#         company_urls = WebScrapping.get_url_website(website_url)

#         # crawl the data from the 10 urls and pass it to chatgpt to provide
#         # the company informations
#         company_data = WebScrapping.domain_related_route(
#             urls=company_urls,
#             target_domain=website_url
#         )
#         company_data_content = WebScrapping.get_all_content(company_data)

#         # passing the data crawlles to llangchain

#         # use apollo to data
#         linkedin_data = WebScrapping.apolo_request_sesion(linkedin_url)

#         # using apollo search
#         apolo = ApolloCompany(website_url)
#         lang_chain = LangChainAI()
#         organization_data = apolo.get_data()

#         ai_analysis = lang_chain.get_ai_data(website_url, organization_data)
#         print(ai_analysis)
#         data = {
#             'company_data': company_data_content,
#             'linkedin_data': linkedin_data,
#             "organization_data": organization_data,
#             "ai_analysis": ai_analysis
#         }
#         # new_company = Company.objects.create(
#         # homepage_summary="Company homepage summary",
#         # # Other fields...
#         # # )
#         # request.session['company_id'] = new_company.id

#         if company_data and linkedin_data:
#             return Response(
#                 data=data,
#                 status=status.HTTP_200_OK
#                 )
#         else:
#             return Response(
#                 {"error": "Failed to scrape data from the URL."},
#                 status=status.HTTP_500_INTERNAL_SERVER_ERROR
#                 )


# # class PromptAPIView(APIView):
# #     ''' scrawl urls and return the https body'''
# #     def post(self, request, *args, **kwargs):
# #         '''a post method to get url to crawl'''
# #         website_url = request.data.get('website_url')

# #         langChain = LangChainAI()
# #         organization_data = apolo.get_data()

# #         ai_analysis = langChain.get_ai_data(website_url, organization_data)

# #         data = get_ai_data(
# #             website_url,
# #             prompt_template=prompt_templatex
# #         )
# #         if data:
# #             return Response(
# #                 data=data,
# #                 status=status.HTTP_200_OK
# #                 )
# #         else:
# #             return Response(
# #                 {"error": "Failed to get URL data."},
# #                 status=status.HTTP_500_INTERNAL_SERVER_ERROR
# #                 )
        
# #     def get(self, request, *args, **kwargs):
# #         ''' a get method for data crawling'''
# #         return Response({"data": "welcome"}, status=status.HTTP_200_OK)


# def run(self):
#     """the delayed task to scraped data"""
#     # scraping all urls from the companies webpage
#     company_urls = WebScrapping.get_url_website(self.website_url)
#     # get domain domain-related pages and scrape the data
#     company_related_urls = WebScrapping.domain_related_route(
#         urls=company_urls,
#         target_domain=self.website_url
#     )
#     # linkedin_data = WebScrapping.apolo_request_sesion(self.linkedin_url)
#     company_data_content = WebScrapping.get_all_content(
#         company_related_urls
#         )

#     # using apollo search to get companies information
#     apolo = ApolloCompany(self.website_url)
#     organization_data = apolo.get_data()
#     # using langchain ai to analyze the company data
#     lang_chain = LangChainAI()
#     ai_analysis, campaign_data = lang_chain.get_ai_data(
#         company_data_content,
#         organization_data
#     )

#     # Create a list to store the query conditions
#     query_conditions = []

#     # Add conditions for "Job Title"
#     if 'Job Title' in campaign_data:
#         job_title_conditions = [
#             {"data.job_title": {"$regex": f".*{title}.*", "$options": "i"}}
#             for title in campaign_data['Job Title']]
#         query_conditions.extend(job_title_conditions)

#     # Add conditions for "Job Title Role"
#     if 'Job Title Role' in campaign_data:
#         job_title_role_conditions = [
#             {"data.job_title_role": {"$regex": f".*{role}.*", "$options": "i"}}
#             for role in campaign_data['Job Title Role']]
#         query_conditions.extend(job_title_role_conditions)

#     # Use the "$or" condition to combine query conditions
#     search_query = None
#     if query_conditions:
#         search_query = {"$or": query_conditions}
#     # else:
#     #     # Raise an error if job title and job title role are not found
#     #     raise ValueError("Job title or job title role\
#     #                       not found from the company website")

#     # Define the MongoDB client for connecting to the LinkedIn database
#     client = MongoClient(os.environ.get("MONGO_DB_URL"))['Linked_in_db']
#     collection = client['Linked_in_1']

#     # Execute the query and limit the results to the top 10 users
#     # if search_query is not None:
#     #     users = collection.find(search_query).limit(10)
#     #     data_user = [json.loads(json_util.dumps(dc)) for dc in users]

#     projection = {
#         "_id": 0,  # Excluding the "_id" field
#         "data.job_title": 1,
#         "data.job_title_role": 1,
#         "data.job_company_name": 1
#     }
#     if search_query is not None:
#         users = collection.find(
#             search_query, projection
#         ).limit(10)
#         data_user = [json.loads(json_util.dumps(dc)) for dc in users]

#     return {
#         # 'company_data': company_data_content,
#         # 'linkedin_data': linkedin_data,
#         # "organization_data": organization_data,
#         "ai_analysis": ai_analysis,
#         "matched_users": data_user if search_query else [],
#         # "extracted_data_points": campaign_data
#     }
# import re

# text = """
# Problems Addressed: 
# 1.a. Problems Identified:
#     - Difficulty in finding support information and services 
#     - Difficulty in understanding product and use instructions 
#     - Difficulty in obtaining genuine parts for repairs 
#     - Difficulty in understanding terms and conditions 
#     - Limited access to customer service 
#     - Difficulty in accessing Apple support due to language barriers
# 1.b. Solutions Offered:
#     - Video tutorials to help customers understand how to use the products 
#     - Official Customer Service and Apple Support website 
#     - Access to Apple Support app 
#     - Clear and concise terms and conditions 
#     - Access to genuine Apple parts 
#     - Apple Trade-in program 
#     - Access to Apple Service Programs 
#     - Multilingual support 
#     - Gift card scams awareness 

# Buyer Persona: 
# 2.a. Demographics:
#     - Age range: 18 -65 
#     - Gender: Primarily male 
#     - Location: US and International 
# 2.b. Behavioral Traits:
#     - Tech-savvy 
#     - Interested in novelty products 
#     - Likes to be in the know 
# """

# # Define regular expressions to match headers and items
# header_pattern = r'(?P<header>\d+\.[a-zA-Z\s]+:)'
# item_pattern = r'(?P<item>\s+- .+)'

# # Initialize variables to store the extracted data
# data = {}
# current_header = None
# current_items = []

# # Split the text into lines
# lines = text.split('\n')

# # Iterate through each line and extract data
# for line in lines:
#     # Check if the line matches the header pattern
#     header_match = re.match(header_pattern, line)
#     if header_match:
#         # Extract the header and set it as the current header
#         current_header = header_match.group('header').strip(':')
#         current_items = []
#         data[current_header] = current_items
#     elif current_header:
#         # Check if the line matches the item pattern
#         item_match = re.match(item_pattern, line)
#         if item_match:
#             # Add the item to the current list of items
#             current_items.append(item_match.group('item').strip())

# # Print the formatted data
# for header, items in data.items():
#     print(f"{header}:")
#     for item in items:
#         print(item)
# class TestElasticAPI(APIView):

#     def post(self, request):
#         es = Elasticsearch(
#             os.environ.get('ELASTIC_DB_URL'), 
#         )

#         task_id = request.data.get('task_id')
#         key = request.data.get('_id')

#         job_titles = ["cto"]
#         countries = ["united arab emirates"]
#         keywords = ["apple"]

#         def query_should_list(field_name, key_list):
#             olist = []
#             for k in key_list:
#                 olist.append({
#                     "match": {
#                         field_name: k
#                     }
#                 })
#             return olist

#         job_title_should = query_should_list("job_title", job_titles)
#         countries_should = query_should_list("location_country", countries)
#         keywords_should = query_should_list("summary", keywords)

#         search_query = {
#             "bool": {
#                 "should": [
#                     {"bool": {
#                         "should": job_title_should
#                     }},
#                     {"bool": {
#                         "must": countries_should
#                     }},
#                     {
#                         "bool": {
#                             "should": keywords_should
#                         }
#                     }
#                 ]
#             }
#         }

#         resp_data = es.search(
#             index="linked-in",
#             query=search_query,
#             from_=0, size=10)['hits']['hits']
        
#         return Response(
#             data=resp_data,
#             content_type='application/json',
#             status=status.HTTP_200_OK
#         )



import re

# Sample text
sample_text = """
Problem Addressed:
1.a. Problems Identified:
- Lack of security mechanisms for user accounts
- Poor user experience with cumbersome sign-in process
- Limited support for outdated web browsers
- Lack of customer support

1.b. Solutions Offered:
- Establish stronger security mechanisms for user accounts
- Streamline the sign-in process to improve user experience
- Update website to be compatible with modern web browsers
- Provide better customer support via email, phone, and chat 

Buyer Persona:
2.a. Demographics:
- Primarily young adults (18-35) who are tech-savvy and highly engaged
- Middle to higher-income earners

2.b. Behavioral Traits:
- Prefer quick and straightforward experiences
- Likely to use multiple devices and browsers
- Willing to pay for premium products and services

2.c. Motivations & Goals:
- Desire for modern and easy-to-use technology
- Focus on finding the best product for their needs
- Constantly looking for updates to stay ahead of the competition

"""

# Regex pattern to split the text by headers
regex_pattern = r"(\d+\.\w+\. [A-Za-z\s]+):"

# Find all matches and split the text
sections = re.split(regex_pattern, sample_text)[1:]

# Prepare a JSON object with headers as keys
data = {}
current_header = ""

for i, section in enumerate(sections):
    if i % 2 == 0:
        current_header = section.strip()
    else:
        content_lines = section.strip().split('\n')[1:]
        # Remove '-' sign from each content line
        content_without_dash = [line.lstrip('-').strip() for line in content_lines]
        data[current_header] = content_without_dash

# Print the JSON object
import json
print(json.dumps(data, indent=4))


# import re

# # Sample text
# sample_text = """
# Problem Addressed:
# 1.a. Problems Identified:
# - Lack of security mechanisms for user accounts
# - Poor user experience with a cumbersome sign-in process
# - Limited support for outdated web browsers
# - Lack of customer support

# 1.b. Solutions Offered:
# - Establish stronger security mechanisms for user accounts
# - Streamline the sign-in process to improve user experience
# - Update the website to be compatible with modern web browsers
# - Provide better customer support via email, phone, and chat

# Buyer Persona:
# 2.a. Demographics:
# - Primarily young adults (18-35) who are tech-savvy and highly engaged
# - Middle to higher-income earners

# 2.b. Behavioral Traits:
# - Prefer quick and straightforward experiences
# - Likely to use multiple devices and browsers
# - Willing to pay for premium products and services

# 2.c. Motivations & Goals:
# - Desire for modern and easy-to-use technology
# - Focus on finding the best product for their needs
# - Constantly looking for updates to stay ahead of the competition
# """

# # Split the text into lines
# lines = sample_text.strip().split('\n')

# # Initialize variables
# data = {}
# current_main_header = ""
# current_sub_header = ""
# current_body = []

# # Iterate through lines
# for line in lines:
#     line = line.strip()
#     if line.endswith(":"):
#         # Identify main headers
#         current_main_header = line[:-1]
#         data[current_main_header] = {}
#     elif re.match(r"^\d+\.\w+\.", line):
#         # Identify sub-headers
#         current_sub_header = line
#         data[current_main_header][current_sub_header] = []
#     elif line.startswith("- "):
#         # Add content to the current sub-header's body
#         current_body.append(line[2:])
#     else:
#         # Unexpected line, ignore or handle accordingly
#         pass
    
#     # Update the data structure
#     data[current_main_header][current_sub_header] = current_body

# # Print the JSON object
# import json
# print(json.dumps(data, indent=4))
# {
#     "Problem Addressed" : {
#         "1.a. Problems Identified":[
#             "Lack of security mechanisms for user accounts",
#             "Poor user experience with a cumbersome sign-in process",
#             "Limited support for outdated web browsers",
#             "Lack of customer support",
#             "Establish stronger security mechanisms for user accounts",
#             "Streamline the sign-in process to improve user experience",
#             "Update the website to be compatible with modern web browsers",
#             "Provide better customer support via email, phone, and chat",
#             "Primarily young adults (18-35) who are tech-savvy and highly engaged",
#             "Middle to higher-income earners",
#             "Prefer quick and straightforward experiences",
#             "Likely to use multiple devices and browsers",
#             "Willing to pay for premium products and services",
#             "Desire for modern and easy-to-use technology",
#             "Focus on finding the best product for their needs",
#             "Constantly looking for updates to stay ahead of the competition"
#         ],
#         "1.b. Solutions Offered":  [
#             "Lack of security mechanisms for user accounts",
#             "Poor user experience with a cumbersome sign-in process",
#             "Limited support for outdated web browsers",
#             "Lack of customer support",
#             "Establish stronger security mechanisms for user accounts",
#             "Streamline the sign-in process to improve user experience",
#             "Update the website to be compatible with modern web browsers",
#             "Provide better customer support via email, phone, and chat",
#             "Primarily young adults (18-35) who are tech-savvy and highly engaged",
#             "Middle to higher-income earners",
#             "Prefer quick and straightforward experiences",
#             "Likely to use multiple devices and browsers",
#             "Willing to pay for premium products and services",
#             "Desire for modern and easy-to-use technology",
#             "Focus on finding the best product for their needs",
#             "Constantly looking for updates to stay ahead of the competition"
#         ]
#     },
# }

import re

# Sample text
sample_text = """
Problem Addressed:
1.a. Problems Identified:
- Lack of security mechanisms for user accounts
- Poor user experience with a cumbersome sign-in process
- Limited support for outdated web browsers
- Lack of customer support

1.b. Solutions Offered:
- Establish stronger security mechanisms for user accounts
- Streamline the sign-in process to improve user experience
- Update the website to be compatible with modern web browsers
- Provide better customer support via email, phone, and chat

Buyer Persona:
2.a. Demographics:
- Primarily young adults (18-35) who are tech-savvy and highly engaged
- Middle to higher-income earners

2.b. Behavioral Traits:
- Prefer quick and straightforward experiences
- Likely to use multiple devices and browsers
- Willing to pay for premium products and services

2.c. Motivations & Goals:
- Desire for modern and easy-to-use technology
- Focus on finding the best product for their needs
- Constantly looking for updates to stay ahead of the competition
"""

# Split the text into lines
lines = sample_text.strip().split('\n')

# Initialize variables
data = {}
current_main_header = ""
current_sub_header = ""
current_body = []

# Iterate through lines
for line in lines:
    line = line.strip()
    if line.endswith(":"):
        # Identify main headers
        current_main_header = line[:-1]
        data[current_main_header] = {}
    elif re.match(r"^\d+\.\w+\.", line):
        # Identify sub-headers
        current_sub_header = line
        data[current_main_header][current_sub_header] = []
    elif line.startswith("- "):
        # Add content to the current sub-header's body
        current_body.append(line[2:])
    else:
        # Unexpected line, ignore or handle accordingly
        pass
    
    # Update the data structure
    data[current_main_header][current_sub_header] = current_body

# Create a new dictionary to store the flattened output
output_data = {}
for main_header, sub_headers in data.items():
    flattened_body = []
    for sub_header, body in sub_headers.items():
        flattened_body.extend(body)
    output_data[main_header] = flattened_body

# Print the JSON object
import json
print(json.dumps(output_data, indent=4))



import re

# Sample text
sample_text = """
1. Problems Addressed:
a. Problems Identified:
   - Poor customer support
   - Inefficient product repair and replacement
   - Lack of access to safe, reliable, and secure repairs
   - Counterfeit parts and services
   - Gift card scams
b. Solutions Offered:
   - Official Apple support videos
   - Connecting customers to experts via phone, chat, email, etc.
   - Apple Support app
   - My Support feature
   - Apple Care+
   - Apple Trade In
   - Access to safe and reliable repairs
   - Recycling for free
   - Avoiding counterfeit parts
   - Awareness around gift card scams

2. Buyer Persona:
a. Demographics:
   - Age: 18+
   - Gender: Any
   - Education: Any
   - Location: Worldwide
b. Behavioral Traits:
   - Tech-savvy
   - Interested in Apple products
   - Looking for support and advice
c. Motivations & Goals:
   - Access to official Apple support
   - Assistance with their Apple products
   - Solutions for repair and replacement
   - Solutions for protection, security, and privacy
"""

# Split the text into lines
lines = sample_text.strip().split('\n')

# Initialize variables
data = {}
current_main_header = ""
current_sub_header = ""
current_body = []

# Iterate through lines
for line in lines:
    line = line.strip()
    if re.match(r"^\d+\.", line):
        # Main header
        current_main_header = line.split(".")[1].strip()
        data[current_main_header] = {}
    elif re.match(r"^[a-z]\.", line):
        # Sub-header
        current_sub_header = line.strip()
        data[current_main_header][current_sub_header] = []
    elif line.startswith("- "):
        # Add content to the current sub-header's body
        current_body.append(line[2:])
    elif current_sub_header:
        # If line doesn't match any of the above patterns but there's a current sub-header
        # it's considered part of the current sub-header's body
        current_body.append(line)

# Create a new dictionary to store the formatted data
formatted_data = {}
for main_header, sub_headers in data.items():
    sub_header_data = {}
    for sub_header, body in sub_headers.items():
        sub_header_data[sub_header] = body
    formatted_data[main_header] = sub_header_data

# Print the JSON object
import json
print(json.dumps(formatted_data, indent=4))

datax = [
    [
        {
            "_index": "linked-in",
            "_id": "wD3HB9ARRgdNRBBxmT1vjg_0000",
            "_score": 1.0,
            "_ignored": [
                "experience.summary.keyword"
            ],
            "_source": {
                "full_name": "andrea cremaschi",
                "first_name": "andrea",
                "middle_initial": None,
                "middle_name": "None",
                "last_name": "cremaschi",
                "gender": "female",
                "birth_year": None,
                "birth_date": None,
                "linkedin_url": "linkedin.com/in/acremaschi",
                "linkedin_username": "acremaschi",
                "linkedin_id": "24188505",
                "facebook_url": "facebook.com/andrea.cremaschi.33",
                "facebook_username": "andrea.cremaschi.33",
                "facebook_id": "1198114514",
                "twitter_url": "twitter.com/andreacremaschi",
                "twitter_username": "andreacremaschi",
                "github_url": "github.com/andreacremaschi",
                "github_username": "andreacremaschi",
                "work_email": None,
                "mobile_phone": "+447871782520",
                "industry": "computer software",
                "job_title": "software engineer",
                "job_title_role": "engineering",
                "job_title_sub_role": "software",
                "job_title_levels": [],
                "job_company_id": "apple",
                "job_company_name": "apple",
                "job_company_website": "apple.com",
                "job_company_size": "10001+",
                "job_company_founded": "1976",
                "job_company_industry": "consumer electronics",
                "job_company_linkedin_url": "linkedin.com/company/apple",
                "job_company_linkedin_id": "162479",
                "job_company_facebook_url": "facebook.com/authorized.apple",
                "job_company_twitter_url": "twitter.com/apple",
                "job_company_location_name": "cupertino, california, united states",
                "job_company_location_locality": "cupertino",
                "job_company_location_metro": "san jose, california",
                "job_company_location_region": "california",
                "job_company_location_geo": "37.32,-122.03",
                "job_company_location_street_address": "1 apple park way",
                "job_company_location_address_line_2": None,
                "job_company_location_postal_code": "95014",
                "job_company_location_country": "united states",
                "job_company_location_continent": "north america",
                "job_last_updated": "2020-12-01",
                "job_start_date": "2017-11",
                "job_summary": None,
                "location_name": "san francisco, california, united states",
                "location_locality": "san francisco",
                "location_metro": "san francisco, california",
                "location_region": "california",
                "location_country": "united states",
                "location_continent": "north america",
                "location_street_address": None,
                "location_address_line_2": None,
                "location_postal_code": None,
                "location_geo": "37.77,-122.41",
                "location_last_updated": "2020-12-01",
                "linkedin_connections": 464,
                "inferred_salary": 100000,
                "inferred_years_experience": 14,
                "summary": "macOS/iOS Developer presso Apple at Apple",
                "phone_numbers": [
                    "+447871782520",
                    "+14156197514"
                ],
                "emails": [
                    "andreacremaschi@libero.it",
                    "cremaschiandrea@gmail.com"
                ],
                "interests": [],
                "skills": [
                    "ios",
                    "cocoa",
                    "iphone application development",
                    "video",
                    "objective c",
                    "swift",
                    "sviluppo ios",
                    "digital video",
                    "cocoa touch",
                    "sql",
                    "gis",
                    "final cut pro",
                    "photoshop",
                    "postgis",
                    "user experience",
                    "user experience design",
                    "css",
                    "open source",
                    "os x",
                    "mapbox",
                    "open data",
                    "openstreetmap",
                    "haip",
                    "quartz composer",
                    "qlab",
                    "sviluppo web",
                    "mobile applications",
                    "html",
                    "javascript",
                    "mysql",
                    "applicazioni mobili",
                    "user interface design"
                ],
                "location_names": [
                    "london, england, united kingdom",
                    "bergamo, lombardy, italy",
                    "london, greater london, united kingdom",
                    "san francisco, california, united states",
                    "portland, oregon, united states"
                ],
                "regions": [
                    "oregon, united states",
                    "california, united states",
                    "greater london, united kingdom",
                    "lombardy, italy",
                    "england, united kingdom"
                ],
                "countries": [
                    "italy",
                    "united kingdom",
                    "united states"
                ],
                "street_addresses": [],
                "experience": [
                    {
                        "company": {
                            "name": "midapp srl",
                            "size": 10,
                            "id": "midapp-srl",
                            "founded": None,
                            "industry": "information technology and services",
                            "location": {
                                "name": "bergamo, lombardy, italy",
                                "locality": "bergamo",
                                "region": "lombardy",
                                "metro": None,
                                "country": "italy",
                                "continent": "europe",
                                "street_address": None,
                                "address_line_2": None,
                                "postal_code": None,
                                "geo": "45.69,9.66"
                            },
                            "linkedin_url": "linkedin.com/company/midapp-srl",
                            "linkedin_id": "10299258",
                            "facebook_url": None,
                            "twitter_url": None,
                            "website": "midapp.it"
                        },
                        "location_names": [],
                        "end_date": "2015-03",
                        "start_date": "2012-06",
                        "title": {
                            "name": "ios developer, gis data architect",
                            "role": "engineering",
                            "sub_role": None,
                            "levels": []
                        },
                        "is_primary": False,
                        "summary": None
                    },
                    {
                        "company": {
                            "name": "visuality software",
                            "size": None,
                            "id": None,
                            "founded": None,
                            "industry": None,
                            "location": None,
                            "linkedin_url": None,
                            "linkedin_id": None,
                            "facebook_url": None,
                            "twitter_url": None,
                            "website": None
                        },
                        "location_names": [
                            "bergamo, lombardy, italy"
                        ],
                        "end_date": "2012-01",
                        "start_date": "2011-10",
                        "title": {
                            "name": "ios and osx developer",
                            "role": "engineering",
                            "sub_role": None,
                            "levels": []
                        },
                        "is_primary": False,
                        "summary": None
                    },
                    {
                        "company": {
                            "name": "apple",
                            "size": 10001,
                            "id": "apple",
                            "founded": "1976",
                            "industry": "consumer electronics",
                            "location": {
                                "name": "cupertino, california, united states",
                                "locality": "cupertino",
                                "region": "california",
                                "metro": "san jose, california",
                                "country": "united states",
                                "continent": "north america",
                                "street_address": "1 apple park way",
                                "address_line_2": None,
                                "postal_code": "95014",
                                "geo": "37.32,-122.03"
                            },
                            "linkedin_url": "linkedin.com/company/apple",
                            "linkedin_id": "162479",
                            "facebook_url": "facebook.com/authorized.apple",
                            "twitter_url": "twitter.com/apple",
                            "website": "apple.com"
                        },
                        "location_names": [],
                        "end_date": None,
                        "start_date": "2017-11",
                        "title": {
                            "name": "software engineer",
                            "role": "engineering",
                            "sub_role": "software",
                            "levels": []
                        },
                        "is_primary": True,
                        "summary": None
                    },
                    {
                        "company": {
                            "name": "pie mapping",
                            "size": 10,
                            "id": "piemapping-com",
                            "founded": "2004",
                            "industry": "information technology and services",
                            "location": {
                                "name": "united kingdom",
                                "locality": None,
                                "region": None,
                                "metro": None,
                                "country": "united kingdom",
                                "continent": "europe",
                                "street_address": None,
                                "address_line_2": None,
                                "postal_code": None,
                                "geo": None
                            },
                            "linkedin_url": "linkedin.com/company/piemapping-com",
                            "linkedin_id": "2498367",
                            "facebook_url": None,
                            "twitter_url": "twitter.com/piemapping",
                            "website": None
                        },
                        "location_names": [
                            "london, greater london, united kingdom"
                        ],
                        "end_date": "2016-03",
                        "start_date": "2016-01",
                        "title": {
                            "name": "mobile team leader",
                            "role": None,
                            "sub_role": None,
                            "levels": [
                                "manager"
                            ]
                        },
                        "is_primary": False,
                        "summary": "Led a team of one designer and two developers to deploy the MVP of the company’s core product, a routing app for lorry drivers. The app consumes RESTful web services from the backend, shows the driver’s jobs on a map and offers turn by turn instructions (Swift, RxSwift, GEOSwift). First customer was DPD UK (9000 commercial vehicles). Setup a Continuous Integration and Delivery environment using Bitrise and TestFlight."
                    },
                    {
                        "company": {
                            "name": "nt next",
                            "size": 50,
                            "id": "nt-next",
                            "founded": "2007",
                            "industry": "marketing and advertising",
                            "location": {
                                "name": "bergamo, lombardy, italy",
                                "locality": "bergamo",
                                "region": "lombardy",
                                "metro": None,
                                "country": "italy",
                                "continent": "europe",
                                "street_address": None,
                                "address_line_2": None,
                                "postal_code": None,
                                "geo": "45.69,9.66"
                            },
                            "linkedin_url": "linkedin.com/company/nt-next",
                            "linkedin_id": "5360614",
                            "facebook_url": "facebook.com/ntnext.bergamo",
                            "twitter_url": "twitter.com/nt_next",
                            "website": "ntnext.it"
                        },
                        "location_names": [],
                        "end_date": "2015-12",
                        "start_date": "2014-09",
                        "title": {
                            "name": "ios developer, gis data architect",
                            "role": "engineering",
                            "sub_role": None,
                            "levels": []
                        },
                        "is_primary": False,
                        "summary": "Senior iOS developer simplify: remote control for your Candy appliances - https://itunes.apple.com/it/app/candy-simply-fi/id905270470?mt=8"
                    },
                    {
                        "company": {
                            "name": "independent",
                            "size": None,
                            "id": None,
                            "founded": None,
                            "industry": None,
                            "location": None,
                            "linkedin_url": None,
                            "linkedin_id": None,
                            "facebook_url": None,
                            "twitter_url": None,
                            "website": None
                        },
                        "location_names": [
                            "bergamo, lombardy, italy"
                        ],
                        "end_date": "2014-12",
                        "start_date": "2009",
                        "title": {
                            "name": "interaction designer",
                            "role": "design",
                            "sub_role": None,
                            "levels": []
                        },
                        "is_primary": False,
                        "summary": "As interaction designer I worked on a live cinema show. I develop most of the tools used during the show, and two of them grew up in independent open source projects: Syphon Virtual Screen, to share video output between OSX applications: https://github.com/andreacremaschi/Syphon-virtual-screen/issues/3 This project was then used by Casio to enhance their XJ-A and XJ-M projectors with the \"extended desktop\" feature. Syphon1394, to connect and tweak a fireware camera video stream, and share it between apps with Syphon: https://github.com/andreacremaschi/Syphon1394 http://www.michelecremaschi.it/show/meliesandme/"
                    }
                ],
                "education": [
                    {
                        "school": {
                            "name": "università degli studi di torino",
                            "type": "post-secondary institution",
                            "id": "2m80qGuctdRpDURmJRAzCw_0",
                            "location": {
                                "name": "torino, piedmont, italy",
                                "locality": "torino",
                                "region": "piedmont",
                                "country": "italy",
                                "continent": "europe"
                            },
                            "linkedin_url": "linkedin.com/school/universita-degli-studi-di-torino",
                            "facebook_url": "facebook.com/unito.it",
                            "twitter_url": "twitter.com/unito",
                            "linkedin_id": "13892",
                            "website": "unito.it",
                            "domain": "unito.it"
                        },
                        "degrees": [],
                        "start_date": "1973",
                        "end_date": "1975",
                        "majors": [],
                        "minors": [],
                        "gpa": None,
                        "summary": None
                    },
                    {
                        "school": {
                            "name": "università degli studi di padova",
                            "type": "post-secondary institution",
                            "id": "GwNVCSiUN-7qmGRB1IinHA_0",
                            "location": {
                                "name": "padova, veneto, italy",
                                "locality": "padova",
                                "region": "veneto",
                                "country": "italy",
                                "continent": "europe"
                            },
                            "linkedin_url": "linkedin.com/school/university-of-padova",
                            "facebook_url": "facebook.com/universitapadova",
                            "twitter_url": "twitter.com/unipadova",
                            "linkedin_id": "13881",
                            "website": "unipd.it",
                            "domain": "unipd.it"
                        },
                        "end_date": "2005",
                        "start_date": "2002",
                        "gpa": None,
                        "degrees": [],
                        "majors": [],
                        "minors": [],
                        "summary": None
                    },
                    {
                        "school": {
                            "name": "università degli studi di torino",
                            "type": "post-secondary institution",
                            "id": "2m80qGuctdRpDURmJRAzCw_0",
                            "location": {
                                "name": "torino, piedmont, italy",
                                "locality": "torino",
                                "region": "piedmont",
                                "country": "italy",
                                "continent": "europe"
                            },
                            "linkedin_url": "linkedin.com/school/universita-degli-studi-di-torino",
                            "facebook_url": "facebook.com/unito.it",
                            "twitter_url": "twitter.com/unito",
                            "linkedin_id": "13892",
                            "website": "unito.it",
                            "domain": "unito.it"
                        },
                        "end_date": "2008",
                        "start_date": "2006",
                        "gpa": None,
                        "degrees": [],
                        "majors": [],
                        "minors": [],
                        "summary": None
                    }
                ],
                "profiles": [
                    {
                        "network": "linkedin",
                        "id": "24188505",
                        "url": "linkedin.com/in/acremaschi",
                        "username": "acremaschi"
                    },
                    {
                        "network": "linkedin",
                        "id": None,
                        "url": "linkedin.com/in/andrea-cremaschi-b891258",
                        "username": "andrea-cremaschi-b891258"
                    },
                    {
                        "network": "github",
                        "id": None,
                        "url": "github.com/andreacremaschi",
                        "username": "andreacremaschi"
                    },
                    {
                        "network": "facebook",
                        "id": "1198114514",
                        "url": "facebook.com/andrea.cremaschi.33",
                        "username": "andrea.cremaschi.33"
                    },
                    {
                        "network": "twitter",
                        "id": None,
                        "url": "twitter.com/andreacremaschi",
                        "username": "andreacremaschi"
                    },
                    {
                        "network": "angellist",
                        "id": None,
                        "url": "angel.co/andrea-cremaschi",
                        "username": "andrea-cremaschi"
                    },
                    {
                        "network": "vimeo",
                        "id": None,
                        "url": "vimeo.com/user/4852624",
                        "username": "4852624"
                    },
                    {
                        "network": "gravatar",
                        "id": None,
                        "url": "gravatar.com/andreacremaschi",
                        "username": "andreacremaschi"
                    },
                    {
                        "network": "foursquare",
                        "id": None,
                        "url": "foursquare.com/user/47447123",
                        "username": "47447123"
                    },
                    {
                        "network": "pinterest",
                        "id": None,
                        "url": "pinterest.com/andreacremaschi",
                        "username": "andreacremaschi"
                    },
                    {
                        "network": "klout",
                        "id": None,
                        "url": "klout.com/andreacremaschi",
                        "username": "andreacremaschi"
                    }
                ],
                "certifications": [],
                "languages": [
                    {
                        "name": "english",
                        "proficiency": 4
                    },
                    {
                        "name": "german",
                        "proficiency": 1
                    }
                ]
            }
        }
    ]
]

selection = {}
# for value in datax[0]:
#     print(value['_source'][''])
# print(datax[0][0]['_source'])
# dict_json = { key: value for key, value in datax[0][0]['_source'].items() }
dict_json = dict(datax[0][0]['_source'].items())
selected_data = {
    "full_name": dict_json.get("full_name"),
    "job_title_role": dict_json.get("job_title_role"),
    "skills": dict_json.get("skills"),
    "experience": dict_json.get("experience")[-1] if dict_json.get("experience") else [] ,
    "job_title": dict_json.get("job_title")
}
print(selected_data)
json_object = json.dumps(selected_data, indent=4)

# for dat in datax[0]:
#     full_name = dat['_source']['full_name']
#     skills = dat['_source']['skills']
# if datax[0]:
    # selection["full_name"] = datax[0]["_source"]['full_name']
#     selection['job_title'] = datax[0]["_source"]['job_title']
#     selection['skills'] = datax[0]["_source"]['skills']
#     selection['education'] = datax[0]["_source"]['education']
#     selection['summary'] = datax[0]["_source"]['summary']

# print(selection)