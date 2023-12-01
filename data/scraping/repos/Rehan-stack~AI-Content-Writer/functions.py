import os
import openai
from django.conf import settings

# Load your API key from an environment variable or secret management service
openai.api_key = settings.OPENAI_API_KEYS





def generateBlogTopicIdeas(topic,audience,keywords):
    blog_topics = []
    response = openai.Completion.create(
  model="text-davinci-003",
  prompt="Generate 5 blog topic ideas on the given topic: {}\nAudience: {}\nkeywords: {} \n*".format(topic,audience,keywords),
  temperature=0.7,
  max_tokens=250,
  top_p=1,
  best_of =1,
  frequency_penalty=0,
  presence_penalty=0
)

    if 'choices' in response:
        if len(response['choices'])>0:
            res = response['choices'][0]['text']
        else:
            return []
    else:
        return []
    
    a_list = res.split('*')
    if len(a_list) > 0:
        for blog in a_list:
            blog_topics.append(blog)
    else:
        return []

    return blog_topics
    




def generateBlogSections(topic,audience,keywords):
    blog_sections = []
    response = openai.Completion.create(
  model="text-davinci-003",
  prompt="Generate 5 blog section titles for the provided blog topic, Audience, and keywords: {}\nAudience: {}\nkeywords: {} \n*".format(topic,audience,keywords),
  temperature=0.7,
  max_tokens=250,
  top_p=1,
  best_of =1,
  frequency_penalty=0,
  presence_penalty=0
)

    if 'choices' in response:
        if len(response['choices'])>0:
            res = response['choices'][0]['text']
        else:
            return []
    else:
        return []
    
    a_list = res.split('*')
    if len(a_list) > 0:
        for blog in a_list:
            blog_sections.append(blog)
    else:
        return []

    return blog_sections



def generateBlogSectionsDetails(blogTopic,sectionTopic,audience,keywords,profile):
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt="Generate detailed blog write up for the following blog section heading, given the blog title, audience and keywords.\nBlog Title: {}\nBlog Section Heading: {}\nAudience: {}\nKeywords: {}\n\n".format(blogTopic,sectionTopic,audience,keywords),
            temperature=0.7,
            max_tokens=2000,
            top_p=1,
            best_of =1,
            frequency_penalty=0,
            presence_penalty=0)

        if 'choices' in response:
            if len(response['choices'])>0:
                res =  response['choices'][0]['text']
                if not res == '':
                    cleanres = res.replace('\n','<br>')
                if profile.monthlyCount:
                    oldCount = int(profile.monthlyCount)
                else:
                    oldCount = 0
                
                oldCount += len(cleanres.split(' '))
                profile.monthlyCount = str(oldCount)
                profile.save()
                return cleanres
            else:
                return ''

            
        else:
             return [] 








def generateProductNames(p_desc,seedwords):
    product_names = []
    response = openai.Completion.create(
  model="text-davinci-003",
  prompt="Generate product names on the given topic.\nProduct description: {}\nSeed words: {}\n*".format(p_desc,seedwords),
  temperature=0.8,
  max_tokens=60,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)

    if 'choices' in response:
        if len(response['choices'])>0:
            res = response['choices'][0]['text']
        else:
            return []
    else:
        return []
    
    a_list = res.split('*')
    if len(a_list) > 0:
        for blog in a_list:
            product_names.append(blog)
    else:
        return []

    return product_names



#function for product ad generation


def product_ads(product_desc,keywords):
    product_ad = []

    response = openai.Completion.create(
  model="text-davinci-003",
  prompt="Write a creative ad for the following product to run on Facebook aimed at parents:\n\nProduct: {}\nkeywords: {}\n".format(product_desc,keywords),
  temperature=0.5,
  max_tokens=100,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)

    if 'choices' in response:
        if len(response['choices'])>0:
            res = response['choices'][0]['text']
        else:
            return []
    else:
        return []
    
    a_list = res.split('*')
    if len(a_list) > 0:
        for blog in a_list:
            product_ad.append(blog)
    else:
        return []

    return product_ad







#function for keywords extraction from text

def text_to_keywords(product_desc):
    keywords = []

    response = openai.Completion.create(
  model="text-davinci-003",
  prompt="Extract keywords from this text:{}\n*".format(product_desc),
  temperature=0.5,
  max_tokens=60,
  top_p=1,
  frequency_penalty=0.8,
  presence_penalty=0
)

    if 'choices' in response:
        if len(response['choices'])>0:
            res = response['choices'][0]['text']
        else:
            return []
    else:
        return []
    
    a_list = res.split('*')
    if len(a_list) > 0:
        for blog in a_list:
            keywords.append(blog)
    else:
        return []

    return keywords





#function for keyword extraction


def extract_contact(text):
    contact_info = []

    response = openai.Completion.create(
  model="text-davinci-003",
  prompt="Extract the all names, contact number, email, mailing address and dates from text:{} \n*".format(text),
  temperature=0,
  max_tokens=256,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)

    if 'choices' in response:
        if len(response['choices'])>0:
            res = response['choices'][0]['text']
        else:
            return []
    else:
        return []
    
    a_list = res.split('*')
    if len(a_list) > 0:
        for con in a_list:
            contact_info.append(con)
    else:
        return []

    return contact_info


#function for generating study notes

def study_notes(topic):
    topic_detail = []

    response = openai.Completion.create(
  model="text-davinci-003",
  prompt="Generate a study notes for the topic:{}\n*".format(topic),
  temperature=0.3,
  max_tokens=500,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)
    if 'choices' in response:
        if len(response['choices'])>0:
            res = response['choices'][0]['text']
        else:
            return []
    else:
        return []
    
    a_list = res.split('*')
    if len(a_list) > 0:
        for top_ic in a_list:
            topic_detail.append(top_ic)
    else:
        return []

    return topic_detail





#review generator for comapanies


def review_generator(company_type,company_name,keywords):
    reviews = []

    response = openai.Completion.create(
  model="text-davinci-003",
  prompt="Write a review based on these notes:\n\ncompnay type:{} \nName: {}\nkeywords: {}\n\n*".format(company_type,company_name,keywords),
  temperature=0.5,
  max_tokens=400,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)
    if 'choices' in response:
        if len(response['choices'])>0:
            res = response['choices'][0]['text']
        else:
            return []
    else:
        return []
    
    a_list = res.split('*')
    if len(a_list) > 0:
        for revi_ews in a_list:
            reviews.append(revi_ews)
    else:
        return []

    return reviews





#function for transforming text into easy to understand format



def easy_text(text):
    transformed_text = []

    response = openai.Completion.create(
  model="text-davinci-003",
  prompt="Transform this text into easy to understand format for a  students and convert difficult vocabulary into easy synonyms:{}\n".format(text),
  temperature=0.7,
  max_tokens=1000,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)

    if 'choices' in response:
        if len(response['choices'])>0:
            res = response['choices'][0]['text']
        else:
            return []
    else:
        return []
    
    a_list = res.split('*')
    if len(a_list) > 0:
        for tex_t in a_list:
            transformed_text.append(tex_t)
    else:
        return []

    return transformed_text