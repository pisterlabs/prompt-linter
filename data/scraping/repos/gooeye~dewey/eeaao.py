import requests
import openai

API_KEY = "06e387469e6b24ba4a28373d249c3f68"
openai.api_key = "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"

def search(query, count):
    base_url = "https://api.elsevier.com/content/search/scopus"
    params = {
        "query": query,
        "apikey": API_KEY,
        "count": count
    }
    response = requests.get(base_url, params=params)
    books = []

    if response.status_code == 200:
        books = []
        data = response.json()
        if "entry" in data["search-results"]:
            for item in data["search-results"]["entry"]:
                book = {}
                # Check if 'link' key exists and has at least 3 entries
                if "link" in item and len(item["link"]) >= 3:
                    book["url"] = item["link"][2]['@href']
                    book["title"] = item["dc:title"]
                    book["abstract"] = getAbstract(item["prism:doi"])
                    books.append(book)
        else:
            print("No search results found.")
    else:
        print("Error:", response.status_code)
        print("Response:", response.text)

    return books


def getAbstract(doi):
    base_url = "https://api.elsevier.com/content/article/doi/"
    url = base_url + doi
    # API parameters
    params = {
        "apikey": API_KEY,
        "httpAccept":"application/json"
    }

    # Making the API request
    response = requests.get(url, params=params)
    
    # Checking the response status
    if response.status_code == 200:
        data = response.json() 
        return data["full-text-retrieval-response"]["coredata"]["dc:description"]
    else:
        print("Error:", response.status_code)
        print("Response:", response.text)

def summarise_essay(essay):
        openai.api_key = "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"
        
        if len(essay)>1500:
            essay = essay[:1500]
        
            
        completion = openai.ChatCompletion.create( 
            model = "gpt-3.5-turbo",
            temperature = 0.8, 
            max_tokens = 2000, 
            messages = [
                {"role": "system", "content": "You are an intelligent summarisation system for a research essay."}, 
                {"role": "user", "content": "Briefly summarise this research essay"},
                {"role": "assistant", "content": "Q: Summarise the following essay: "+essay+"A: Summary"}, 
                {"role": "user", "content": "Content related to the essay: "+essay}
            ]
            )

        summarise_temp = completion.choices[0].message
        summary = summarise_temp['content']
        #print(summary)
        return summary
    

def relevance(query, para, essay, user_preference): #search_terms, input_text, query
    search_result = search(query, 10)
   
    return_list = list()

    counter = 0
    for b in search_result:
        
        if b['abstract']!=None and counter <3:
            counter += 1
            book = dict()
            url = b['url']
            title = b['title']
            abstract = b['abstract']
            
            openai.api_key = "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"

            #1 related
            if user_preference=='related':
                completion = openai.ChatCompletion.create( 
                model = "gpt-3.5-turbo",
                temperature = 0.8, 
                max_tokens = 2000, 
                messages = [
                    {"role": "system", "content": "You are an intelligent recommendation system for library resources for a research paper."}, 
                    {"role": "user", "content": "Give me in 3 sentences a brief reason why this article is relevant."},
                    {"role": "assistant", "content": "Q: Briefly, how relevant is the abstract of this article: "+abstract+" , to the selected"+ para + "This article is relevant because"}, 
                    {"role": "user", "content": "Content related to the following paragraph: "+para}
                ]
                )

                relevance_temp = completion.choices[0].message
                relevance1 = relevance_temp['content']
                book['title'] = title
                book['url'] = url
                book['abstract'] = abstract
                book['reinforcing'] = relevance1

            #4 Explanatory/Tangential - might not directly align with your main argument but still contribute to the overall theme
            if user_preference=='tangential':
                summary = summarise_essay(essay)
                completion = openai.ChatCompletion.create( 
                model = "gpt-3.5-turbo",
                temperature = 0.8, 
                max_tokens = 2000, 
                messages = [
                    {"role": "system", "content": "You do not apologise. You are an intelligent recommendation system for library resources for a research paper. You must say how the article provided might not be relevant to the paragraph but still relevant to the overall theme."}, 
                    {"role": "user", "content": "Give me in 3 sentences a brief reason why this article abstract: "+abstract+" might not be relevant to the following para: "+ para+", but is still relevant to the overall essay theme: "+summary},
                    {"role": "assistant", "content": "Q: Briefly, how different is the abstract of this article: "+abstract+" , to the selected"+ para +" and how similar is it to the "+ summary + "This article is different from the paragraph provided because...but in line with the overall theme because..."}, 
                    {"role": "user", "content": "Content related to the following paragraph: "+para}
                ]
                )

                relevance_temp = completion.choices[0].message
                relevance2 = relevance_temp['content']
                book['title'] = title
                book['url'] = url
                book['abstract'] = abstract
                book['tangential'] = relevance2

            #5 Counterpoint Suggestions - Opposing viewpoints
            if user_preference=='counterpoint':
                completion = openai.ChatCompletion.create( 
                model = "gpt-3.5-turbo",
                temperature = 0.8, 
                max_tokens = 2000, 
                messages = [
                    {"role": "system", "content": "Do not apologise. You are an intelligent recommendation system for library resources for a research paper that does not apologise."}, 
                    {"role": "user", "content": "Give me in 3 sentences a brief reason why this article opposes the following para: "+ para},
                    {"role": "assistant", "content": "Q: Briefly, how different is the abstract of this article: "+abstract+" , to the selected"+ para + "This article is not relevant because"}, 
                    {"role": "user", "content": "Content related to the following paragraph: "+para}
                ]
                )

                relevance_temp = completion.choices[0].message
                relevance3 = relevance_temp['content']
                book['title'] = title
                book['url'] = url
                book['abstract'] = abstract
                book['counterpoint'] = relevance3

            return_list.append(book)

        elif b['abstract']==None:
            print(b['url'])

    #print(return_list)
    return return_list


def generate_search_terms(input_text, user_preference, essay): #user_preference - related, tangential, counterpoint
    if user_preference=='related':
        conversation = [
            {"role": "system", "content": "You are an intelligent recommendation system for search terms."},
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": f"Q: Suggest top 3 relevant search terms that are related to '{input_text}'"},
            {"role": "user", "content": f"Provide search terms related to '{input_text}'"}
        ]
    elif user_preference=='tangential':
        summary = summarise_essay(essay)
        conversation = [
            {"role": "system", "content": "You are an intelligent recommendation system for search terms."},
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": f"Q: Suggest top 3 relevant search terms that are not directly related to '{input_text}' but are still relevant to the overall essay theme '{summary}'"},
            {"role": "user", "content": f"Provide search terms not directly related to '{input_text}' but are still related to the overall theme"}
        ]
    elif user_preference=='counterpoint':
        conversation = [
            {"role": "system", "content": "You are an intelligent recommendation system for search terms."},
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": f"Q: Suggest top 3 relevant search terms that contradict '{input_text}'"},
            {"role": "user", "content": f"Provide search terms that contradict '{input_text}'"}
        ]


    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0.8,
        max_tokens=2000,
        messages=conversation
    )

    assistant_reply = completion.choices[0].message['content']
    search_terms = assistant_reply.replace("Here are some search terms you can use:", "").strip()
    search_terms_list = search_terms.split('\n')
    search_terms_single_line = ' '.join(search_terms_list)
    
    return search_terms_single_line

#TEST CODE/using the functions
input_text = "wealth and poverty"
essay = """
    Poverty not only between countries, but within countries such as Singapore has also worsened with economic progress and prosperity. Presently, Singapore has one of the highest Gini coefficients and worst rates of poverty in the world (Donaldson et al., 2013, pp. 58-66). Inequality and poverty observed in our daily lives can be attributed, though not solely, to the practice of meritocracy in Singapore. The ways in which this is so, are elaborated on below.
Meritocracy in Singapore fails to provide equal opportunities for all as inherent inequalities experienced by the poor, that already disadvantages them in being able to compete fairly for equal opportunities, are not resolved.
Many Singaporeans of middle-class view having their parents or hiring their maid to take care of their children as an indispensable norm. This involves money and the outsourcing of care, a luxury that not all Singaporeans have. Furthermore, mothers of low-income families are often in a lose-lose situation with regards to childcare subsides policies that disadvantage them as they are made to choose between either keeping their wage jobs or neglecting the provision of physical and emotional care for their families and giving up their wage jobs but losing access to childcare subsidies (Teo, 2019). Due to unresolved inequalities in financial might and policies that disadvantage the poor in caring for their children and their families, they often have less opportunities to do so, which contributes to the unequalness in opportunities for social mobility as the lack of care inhibits one’s ability to optimally work hard to compete for opportunities for social mobility in the meritocratic system.
Furthermore, the higher-income and those in more well-regarded jobs have more flexibility in working hours compared to those in the lower-income bracket with less well-regarded jobs. Wages and working conditions for less well-regarded jobs of the poor are often also much less favourable. This is because such jobs are often outsourced and subject to competition due to the nature of such jobs being menial and requiring only simple skills, allowing such jobs to be easily replaced and hence placing greater downward pressure on wages. This means that the poor often have to compensate by obeying their employers’ well and are often at the mercy of their bosses. Even though the government, trade unions and the employers’ association have worked together to set wage and working conditions transparently, enforcement issues mean workers of such jobs remain vulnerable and need to rely on the kindness of their individual supervisors (Ng & Lee, 2018, pp. 308-327). The government has also implemented schemes such as the Workfare Skills Support Scheme in a bid to encourage low-wage workers to upskill and to attain more impactful employment outcomes (Workforce Singapore, n.d.). Hence, inequalities in wages, time-off and working conditions for low-wage jobs of the poor further disadvantages them in attaining opportunities for social mobility through hard work in Singapore’s meritocratic system. This also further cements the point alluded to above on poorer well-being as the effects of poorer pay and less flexibility of time also leads to poorer financial might and less time to care for the well-being of one’s family and oneself.
Education is known as one of the greatest social levellers in a meritocratic system. In Singapore, it is viewed very much as so. Many parents of middle-income households adopt the parenting style of concerted cultivation where their children are often at the centre of the family, with their organised activities and accomplishments as family projects. Such a method of upbringing is likened to large projects that requires having accumulated sufficient funds. This is a far-removed reality for the parents of lower-income households and even if they wanted to parent their children in this manner, they often do not have the monetary resources or time to do so (Teo, 2019). Children who live in poverty also tend to perform worse, are less likely to graduate and are more apt to be underemployed due to differences in brain health (Noble, 2017, pp. 44-49). Each decision parents make to send their children for tuition classes, hoping to give their child a head start, further worsens and perpetuates unequalness in the education system. Indeed, what is thought as one of the greatest social levellers in the meritocratic system, is not so. Hence, children of lower-income families are also disadvantaged in their participation in the very systems aimed at providing equal opportunities for social mobility. Instead of aiding low-income families in breaking out of the cycle of poverty, the perpetuation of such a cycle may instead be the result.
Further, families of lower-income households are also inherently disadvantaged due to the attainment of generational wealth by arbitrary luck. As alluded to above, differences in monetary resources significantly contribute to differences in opportunities for social mobility. With the large gulf in inheritance amounts individuals of higher- versus lower-income households attain via pure arbitrary luck and not by the choices individuals make, the difference in opportunities individuals are able to fairly attain is further worsened (Alstott, 2007, pp. 469-542). Although the Singapore government has attempted to combat this by implementing the policy of estate tax, it was abolished in 2008 as it served to disproportionately affect the lower- and upper-middle income instead of the rich and did not effectively achieve the goal of equalising wealth (Dentons Rodyk & Davidson LLP, 2022). Hence, the poor are still further disadvantaged with less monetary means to attain opportunities for social mobility. This is especially so where poverty is perpetuated across generations in a cycle as the lack in generational wealth passed down to the next generation serves to disadvantage them in their attainment of opportunities for social mobility.
What the said target audience should do to minimise poverty and how they can do so.
	One actionable is for the target audience to take concerted action to engage with the government on the issue of poverty. Many Singaporeans have a Not in My Backyard attitude (Tan & Loh, 2012) towards issues in society and since many of us are middle-class and are far removed from the experiences of the poor, we often neglect voicing out on the issue of poverty. One great way is for Singaporeans to make more effort to communicate with the government via communication platforms such as REACH (REACH, n.d.). REACH allows for the public to gain access to public consultations with government agencies and creates platforms for conversations surrounding a range of issues through various social media sites. The target audience should voice out on poverty and raise awareness of various experiences of poverty encountered in the daily lives of many low-income Singaporeans, which those in government, especially the higher-ups, may be blind to. The target audience need not attempt to come up with concrete ideas and policies the government can implement as they lack the expertise and should not be expected to do so. The role of doing so should be left to the policy makers. The target audience should instead focus on providing information on the unexpressed experiences of those in poverty. 
	Another actionable is for the target audience to take concerted effort in consciously avoiding the perpetuation of inequality and poverty in the decisions they make. For instance, parents can collectively make the conscious choice to avoid sending their children for tuition classes. This is undoubtedly challenging. Humans by nature act for their own interests and it is not in the interests of parents to send their children for relatively less tuition classes than other parents, as their own child(ren) would then lose out on head starts. Hence, it is important for parents to collectively agree to do so. Moreover, parents can start by cutting down on only excessive tuition classes that are unlikely to negatively affect their child’s performance. For instance, if their child is already scoring well for English language, parents can refrain from sending their child for English language tuition classes. An extreme way of regulating such unequalness is for the government to restrict parents from sending their children for tuition by law, such as that done in China (Ng & Sasges, 2021). The goal here is the avoidance of the mindset of trying to get ahead of others with means obtained through brute luck, in this case, financial might to afford tuition. The goal of this actionable is not for the eradication of tuition completely. Hopefully, Singapore will not have to resort to such a draconic policy as seen in China. 
	The third actionable would be for the target audience to bequeath their knowledge of poverty to the young and to similarly inspire them to do the same. I will seek to remind the audience that doing so is crucial for eradication of poverty in the long term as the young are indeed our future leaders in charge of regulating social structures and steering the direction of our future. Even if the target audience may not be closely acquainted with the young or interact much with the young in their day-to-day lives, whenever possible, the target audience can do so by taking the initiative to raise the issue of poverty when the opportunity arises, in their part to contribute to their communities, as role models for the younger generation. No matter how small an effort is, it is still of paramount importance. As alluded to earlier, only concerted effort can ultimately bring about drastic changes in social structures and bring us closer to eradicating poverty (Fromm, 1976, pp. 57-87).
Inspiring and getting the said target audience to make actionable changes.
	I will attempt to inspire the said target audience to make actionable changes by appealing to their emotions and humanity (SMU Newsroom, 2022) through inviting those in poverty to share their experiences and hardships, in person. In sharing the main messages, I shall place emphasis on the unravelling of meritocracy and the consequences of poverty. This is to highlight the severity of and the need to take timely action to combat poverty. 
Conclusion
	I will now conclude by addressing the expected outcomes of this campaign, how this campaign would be successful and why this campaign should be adopted. Immediate outcomes include the target audience being sufficiently proficient with the main messages and to in turn, educate the young and the target audience to be inspired to take concerted action. Longer-term outcomes include the young being educated on the main messages as well and being similarly inspired to take concerted action and the government coming up with more effective policies aimed at eradicating poverty, with the greater awareness of poverty experienced on the ground. One indication of the success of this campaign would be more Singaporeans having a greater awareness of daily experiences of those in poverty, in relation to meritocracy and Singaporeans having more conversations amongst themselves and with the government on the issue of poverty. I can determine so by following up with the target audience yearly, for some time, on actions they have taken after the campaign.
I dare not say that just this one campaign would be sufficient in eradicating poverty altogether as the perpetuation of the issue of poverty is dependent on many other issues (Sachs et al., 2001, pp. 70-75), which I would love to have addressed, but could not, due to the constraint of how wide the scope of this essay can be. Further, the eradication of poverty requires drastic changes and reforms over time. With that said, this campaign should still be adopted as targeting the lack of awareness of poverty amongst Singaporeans and promoting conversations both amongst citizens and between citizens and the government will allow for both concerted action (Fromm, 1976, pp. 57-87) to be taken by citizens and more effective policies to be enacted by the government to better work towards the eradication of poverty. The benefits of this campaign are substantial and will be a large step ahead towards our goal of eradicating poverty.	
"""
#1 related
# user_preference = 'related'
# search_terms = generate_search_terms(input_text, user_preference, essay)
# #2 tangential
# user_preference = 'tangential'
# search_terms = generate_search_terms(input_text, user_preference, essay)
# #3 counterpoint
user_preference = 'counterpoint'    
search_terms = generate_search_terms(input_text, user_preference, essay)

#print(search_terms)
print(relevance(search_terms,input_text, essay, user_preference))



# # Using flask to make an api
# # import necessary libraries and functions
# from flask import Flask, jsonify, request
  
# # creating a Flask app
# app = Flask(__name__)
  
# # on the terminal type: curl http://127.0.0.1:5000/
# # returns hello world when we use GET.
# # returns the data that we send when we use POST.
# @app.route('/get/<string:search_terms>/<string:input_text>/<string:essay>/<string:user_preference>', methods=['GET', 'POST']) #For a POST request, you can modify the data variable to return the data sent in the request.
# def home(search_terms, input_text, essay, user_preference):
#     if(request.method == 'GET'):
  
#         # data = relevance(search_terms,input_text, essay, user_preference)
#         # return jsonify({'data': data})
#         # return jsonify({'data': relevance(search_terms,input_text, essay, user_preference)})
#         data = relevance(search_terms, input_text, essay, user_preference)
#         return jsonify({'data': data})

# # driver function
# if __name__ == '__main__':
  
#     app.run(debug = True)
    
# #/home/Serene/mysite/flask_app.py #a Python file you wish to use to hold your Flask app
# #Serene.pythonanywhere.com #Domain name

# curl http://127.0.0.1:5000/get/generate_search_terms(input_text="wealth and poverty", user_preference='related', essay="""
#     Poverty not only between countries, but within countries such as Singapore has also worsened with economic progress and prosperity. Presently, Singapore has one of the highest Gini coefficients and worst rates of poverty in the world (Donaldson et al., 2013, pp. 58-66). Inequality and poverty observed in our daily lives can be attributed, though not solely, to the practice of meritocracy in Singapore. The ways in which this is so, are elaborated on below.
# Meritocracy in Singapore fails to provide equal opportunities for all as inherent inequalities experienced by the poor, that already disadvantages them in being able to compete fairly for equal opportunities, are not resolved.
# Many Singaporeans of middle-class view having their parents or hiring their maid to take care of their children as an indispensable norm. This involves money and the outsourcing of care, a luxury that not all Singaporeans have. Furthermore, mothers of low-income families are often in a lose-lose situation with regards to childcare subsides policies that disadvantage them as they are made to choose between either keeping their wage jobs or neglecting the provision of physical and emotional care for their families and giving up their wage jobs but losing access to childcare subsidies (Teo, 2019). Due to unresolved inequalities in financial might and policies that disadvantage the poor in caring for their children and their families, they often have less opportunities to do so, which contributes to the unequalness in opportunities for social mobility as the lack of care inhibits one’s ability to optimally work hard to compete for opportunities for social mobility in the meritocratic system.
# Furthermore, the higher-income and those in more well-regarded jobs have more flexibility in working hours compared to those in the lower-income bracket with less well-regarded jobs. Wages and working conditions for less well-regarded jobs of the poor are often also much less favourable. This is because such jobs are often outsourced and subject to competition due to the nature of such jobs being menial and requiring only simple skills, allowing such jobs to be easily replaced and hence placing greater downward pressure on wages. This means that the poor often have to compensate by obeying their employers’ well and are often at the mercy of their bosses. Even though the government, trade unions and the employers’ association have worked together to set wage and working conditions transparently, enforcement issues mean workers of such jobs remain vulnerable and need to rely on the kindness of their individual supervisors (Ng & Lee, 2018, pp. 308-327). The government has also implemented schemes such as the Workfare Skills Support Scheme in a bid to encourage low-wage workers to upskill and to attain more impactful employment outcomes (Workforce Singapore, n.d.). Hence, inequalities in wages, time-off and working conditions for low-wage jobs of the poor further disadvantages them in attaining opportunities for social mobility through hard work in Singapore’s meritocratic system. This also further cements the point alluded to above on poorer well-being as the effects of poorer pay and less flexibility of time also leads to poorer financial might and less time to care for the well-being of one’s family and oneself.
# Education is known as one of the greatest social levellers in a meritocratic system. In Singapore, it is viewed very much as so. Many parents of middle-income households adopt the parenting style of concerted cultivation where their children are often at the centre of the family, with their organised activities and accomplishments as family projects. Such a method of upbringing is likened to large projects that requires having accumulated sufficient funds. This is a far-removed reality for the parents of lower-income households and even if they wanted to parent their children in this manner, they often do not have the monetary resources or time to do so (Teo, 2019). Children who live in poverty also tend to perform worse, are less likely to graduate and are more apt to be underemployed due to differences in brain health (Noble, 2017, pp. 44-49). Each decision parents make to send their children for tuition classes, hoping to give their child a head start, further worsens and perpetuates unequalness in the education system. Indeed, what is thought as one of the greatest social levellers in the meritocratic system, is not so. Hence, children of lower-income families are also disadvantaged in their participation in the very systems aimed at providing equal opportunities for social mobility. Instead of aiding low-income families in breaking out of the cycle of poverty, the perpetuation of such a cycle may instead be the result.
# Further, families of lower-income households are also inherently disadvantaged due to the attainment of generational wealth by arbitrary luck. As alluded to above, differences in monetary resources significantly contribute to differences in opportunities for social mobility. With the large gulf in inheritance amounts individuals of higher- versus lower-income households attain via pure arbitrary luck and not by the choices individuals make, the difference in opportunities individuals are able to fairly attain is further worsened (Alstott, 2007, pp. 469-542). Although the Singapore government has attempted to combat this by implementing the policy of estate tax, it was abolished in 2008 as it served to disproportionately affect the lower- and upper-middle income instead of the rich and did not effectively achieve the goal of equalising wealth (Dentons Rodyk & Davidson LLP, 2022). Hence, the poor are still further disadvantaged with less monetary means to attain opportunities for social mobility. This is especially so where poverty is perpetuated across generations in a cycle as the lack in generational wealth passed down to the next generation serves to disadvantage them in their attainment of opportunities for social mobility.
# What the said target audience should do to minimise poverty and how they can do so.
# 	One actionable is for the target audience to take concerted action to engage with the government on the issue of poverty. Many Singaporeans have a Not in My Backyard attitude (Tan & Loh, 2012) towards issues in society and since many of us are middle-class and are far removed from the experiences of the poor, we often neglect voicing out on the issue of poverty. One great way is for Singaporeans to make more effort to communicate with the government via communication platforms such as REACH (REACH, n.d.). REACH allows for the public to gain access to public consultations with government agencies and creates platforms for conversations surrounding a range of issues through various social media sites. The target audience should voice out on poverty and raise awareness of various experiences of poverty encountered in the daily lives of many low-income Singaporeans, which those in government, especially the higher-ups, may be blind to. The target audience need not attempt to come up with concrete ideas and policies the government can implement as they lack the expertise and should not be expected to do so. The role of doing so should be left to the policy makers. The target audience should instead focus on providing information on the unexpressed experiences of those in poverty. 
# 	Another actionable is for the target audience to take concerted effort in consciously avoiding the perpetuation of inequality and poverty in the decisions they make. For instance, parents can collectively make the conscious choice to avoid sending their children for tuition classes. This is undoubtedly challenging. Humans by nature act for their own interests and it is not in the interests of parents to send their children for relatively less tuition classes than other parents, as their own child(ren) would then lose out on head starts. Hence, it is important for parents to collectively agree to do so. Moreover, parents can start by cutting down on only excessive tuition classes that are unlikely to negatively affect their child’s performance. For instance, if their child is already scoring well for English language, parents can refrain from sending their child for English language tuition classes. An extreme way of regulating such unequalness is for the government to restrict parents from sending their children for tuition by law, such as that done in China (Ng & Sasges, 2021). The goal here is the avoidance of the mindset of trying to get ahead of others with means obtained through brute luck, in this case, financial might to afford tuition. The goal of this actionable is not for the eradication of tuition completely. Hopefully, Singapore will not have to resort to such a draconic policy as seen in China. 
# 	The third actionable would be for the target audience to bequeath their knowledge of poverty to the young and to similarly inspire them to do the same. I will seek to remind the audience that doing so is crucial for eradication of poverty in the long term as the young are indeed our future leaders in charge of regulating social structures and steering the direction of our future. Even if the target audience may not be closely acquainted with the young or interact much with the young in their day-to-day lives, whenever possible, the target audience can do so by taking the initiative to raise the issue of poverty when the opportunity arises, in their part to contribute to their communities, as role models for the younger generation. No matter how small an effort is, it is still of paramount importance. As alluded to earlier, only concerted effort can ultimately bring about drastic changes in social structures and bring us closer to eradicating poverty (Fromm, 1976, pp. 57-87).
# Inspiring and getting the said target audience to make actionable changes.
# 	I will attempt to inspire the said target audience to make actionable changes by appealing to their emotions and humanity (SMU Newsroom, 2022) through inviting those in poverty to share their experiences and hardships, in person. In sharing the main messages, I shall place emphasis on the unravelling of meritocracy and the consequences of poverty. This is to highlight the severity of and the need to take timely action to combat poverty. 
# Conclusion
# 	I will now conclude by addressing the expected outcomes of this campaign, how this campaign would be successful and why this campaign should be adopted. Immediate outcomes include the target audience being sufficiently proficient with the main messages and to in turn, educate the young and the target audience to be inspired to take concerted action. Longer-term outcomes include the young being educated on the main messages as well and being similarly inspired to take concerted action and the government coming up with more effective policies aimed at eradicating poverty, with the greater awareness of poverty experienced on the ground. One indication of the success of this campaign would be more Singaporeans having a greater awareness of daily experiences of those in poverty, in relation to meritocracy and Singaporeans having more conversations amongst themselves and with the government on the issue of poverty. I can determine so by following up with the target audience yearly, for some time, on actions they have taken after the campaign.
# I dare not say that just this one campaign would be sufficient in eradicating poverty altogether as the perpetuation of the issue of poverty is dependent on many other issues (Sachs et al., 2001, pp. 70-75), which I would love to have addressed, but could not, due to the constraint of how wide the scope of this essay can be. Further, the eradication of poverty requires drastic changes and reforms over time. With that said, this campaign should still be adopted as targeting the lack of awareness of poverty amongst Singaporeans and promoting conversations both amongst citizens and between citizens and the government will allow for both concerted action (Fromm, 1976, pp. 57-87) to be taken by citizens and more effective policies to be enacted by the government to better work towards the eradication of poverty. The benefits of this campaign are substantial and will be a large step ahead towards our goal of eradicating poverty.	
# """)/"wealth and poverty"/"""
#     Poverty not only between countries, but within countries such as Singapore has also worsened with economic progress and prosperity. Presently, Singapore has one of the highest Gini coefficients and worst rates of poverty in the world (Donaldson et al., 2013, pp. 58-66). Inequality and poverty observed in our daily lives can be attributed, though not solely, to the practice of meritocracy in Singapore. The ways in which this is so, are elaborated on below.
# Meritocracy in Singapore fails to provide equal opportunities for all as inherent inequalities experienced by the poor, that already disadvantages them in being able to compete fairly for equal opportunities, are not resolved.
# Many Singaporeans of middle-class view having their parents or hiring their maid to take care of their children as an indispensable norm. This involves money and the outsourcing of care, a luxury that not all Singaporeans have. Furthermore, mothers of low-income families are often in a lose-lose situation with regards to childcare subsides policies that disadvantage them as they are made to choose between either keeping their wage jobs or neglecting the provision of physical and emotional care for their families and giving up their wage jobs but losing access to childcare subsidies (Teo, 2019). Due to unresolved inequalities in financial might and policies that disadvantage the poor in caring for their children and their families, they often have less opportunities to do so, which contributes to the unequalness in opportunities for social mobility as the lack of care inhibits one’s ability to optimally work hard to compete for opportunities for social mobility in the meritocratic system.
# Furthermore, the higher-income and those in more well-regarded jobs have more flexibility in working hours compared to those in the lower-income bracket with less well-regarded jobs. Wages and working conditions for less well-regarded jobs of the poor are often also much less favourable. This is because such jobs are often outsourced and subject to competition due to the nature of such jobs being menial and requiring only simple skills, allowing such jobs to be easily replaced and hence placing greater downward pressure on wages. This means that the poor often have to compensate by obeying their employers’ well and are often at the mercy of their bosses. Even though the government, trade unions and the employers’ association have worked together to set wage and working conditions transparently, enforcement issues mean workers of such jobs remain vulnerable and need to rely on the kindness of their individual supervisors (Ng & Lee, 2018, pp. 308-327). The government has also implemented schemes such as the Workfare Skills Support Scheme in a bid to encourage low-wage workers to upskill and to attain more impactful employment outcomes (Workforce Singapore, n.d.). Hence, inequalities in wages, time-off and working conditions for low-wage jobs of the poor further disadvantages them in attaining opportunities for social mobility through hard work in Singapore’s meritocratic system. This also further cements the point alluded to above on poorer well-being as the effects of poorer pay and less flexibility of time also leads to poorer financial might and less time to care for the well-being of one’s family and oneself.
# Education is known as one of the greatest social levellers in a meritocratic system. In Singapore, it is viewed very much as so. Many parents of middle-income households adopt the parenting style of concerted cultivation where their children are often at the centre of the family, with their organised activities and accomplishments as family projects. Such a method of upbringing is likened to large projects that requires having accumulated sufficient funds. This is a far-removed reality for the parents of lower-income households and even if they wanted to parent their children in this manner, they often do not have the monetary resources or time to do so (Teo, 2019). Children who live in poverty also tend to perform worse, are less likely to graduate and are more apt to be underemployed due to differences in brain health (Noble, 2017, pp. 44-49). Each decision parents make to send their children for tuition classes, hoping to give their child a head start, further worsens and perpetuates unequalness in the education system. Indeed, what is thought as one of the greatest social levellers in the meritocratic system, is not so. Hence, children of lower-income families are also disadvantaged in their participation in the very systems aimed at providing equal opportunities for social mobility. Instead of aiding low-income families in breaking out of the cycle of poverty, the perpetuation of such a cycle may instead be the result.
# Further, families of lower-income households are also inherently disadvantaged due to the attainment of generational wealth by arbitrary luck. As alluded to above, differences in monetary resources significantly contribute to differences in opportunities for social mobility. With the large gulf in inheritance amounts individuals of higher- versus lower-income households attain via pure arbitrary luck and not by the choices individuals make, the difference in opportunities individuals are able to fairly attain is further worsened (Alstott, 2007, pp. 469-542). Although the Singapore government has attempted to combat this by implementing the policy of estate tax, it was abolished in 2008 as it served to disproportionately affect the lower- and upper-middle income instead of the rich and did not effectively achieve the goal of equalising wealth (Dentons Rodyk & Davidson LLP, 2022). Hence, the poor are still further disadvantaged with less monetary means to attain opportunities for social mobility. This is especially so where poverty is perpetuated across generations in a cycle as the lack in generational wealth passed down to the next generation serves to disadvantage them in their attainment of opportunities for social mobility.
# What the said target audience should do to minimise poverty and how they can do so.
# 	One actionable is for the target audience to take concerted action to engage with the government on the issue of poverty. Many Singaporeans have a Not in My Backyard attitude (Tan & Loh, 2012) towards issues in society and since many of us are middle-class and are far removed from the experiences of the poor, we often neglect voicing out on the issue of poverty. One great way is for Singaporeans to make more effort to communicate with the government via communication platforms such as REACH (REACH, n.d.). REACH allows for the public to gain access to public consultations with government agencies and creates platforms for conversations surrounding a range of issues through various social media sites. The target audience should voice out on poverty and raise awareness of various experiences of poverty encountered in the daily lives of many low-income Singaporeans, which those in government, especially the higher-ups, may be blind to. The target audience need not attempt to come up with concrete ideas and policies the government can implement as they lack the expertise and should not be expected to do so. The role of doing so should be left to the policy makers. The target audience should instead focus on providing information on the unexpressed experiences of those in poverty. 
# 	Another actionable is for the target audience to take concerted effort in consciously avoiding the perpetuation of inequality and poverty in the decisions they make. For instance, parents can collectively make the conscious choice to avoid sending their children for tuition classes. This is undoubtedly challenging. Humans by nature act for their own interests and it is not in the interests of parents to send their children for relatively less tuition classes than other parents, as their own child(ren) would then lose out on head starts. Hence, it is important for parents to collectively agree to do so. Moreover, parents can start by cutting down on only excessive tuition classes that are unlikely to negatively affect their child’s performance. For instance, if their child is already scoring well for English language, parents can refrain from sending their child for English language tuition classes. An extreme way of regulating such unequalness is for the government to restrict parents from sending their children for tuition by law, such as that done in China (Ng & Sasges, 2021). The goal here is the avoidance of the mindset of trying to get ahead of others with means obtained through brute luck, in this case, financial might to afford tuition. The goal of this actionable is not for the eradication of tuition completely. Hopefully, Singapore will not have to resort to such a draconic policy as seen in China. 
# 	The third actionable would be for the target audience to bequeath their knowledge of poverty to the young and to similarly inspire them to do the same. I will seek to remind the audience that doing so is crucial for eradication of poverty in the long term as the young are indeed our future leaders in charge of regulating social structures and steering the direction of our future. Even if the target audience may not be closely acquainted with the young or interact much with the young in their day-to-day lives, whenever possible, the target audience can do so by taking the initiative to raise the issue of poverty when the opportunity arises, in their part to contribute to their communities, as role models for the younger generation. No matter how small an effort is, it is still of paramount importance. As alluded to earlier, only concerted effort can ultimately bring about drastic changes in social structures and bring us closer to eradicating poverty (Fromm, 1976, pp. 57-87).
# Inspiring and getting the said target audience to make actionable changes.
# 	I will attempt to inspire the said target audience to make actionable changes by appealing to their emotions and humanity (SMU Newsroom, 2022) through inviting those in poverty to share their experiences and hardships, in person. In sharing the main messages, I shall place emphasis on the unravelling of meritocracy and the consequences of poverty. This is to highlight the severity of and the need to take timely action to combat poverty. 
# Conclusion
# 	I will now conclude by addressing the expected outcomes of this campaign, how this campaign would be successful and why this campaign should be adopted. Immediate outcomes include the target audience being sufficiently proficient with the main messages and to in turn, educate the young and the target audience to be inspired to take concerted action. Longer-term outcomes include the young being educated on the main messages as well and being similarly inspired to take concerted action and the government coming up with more effective policies aimed at eradicating poverty, with the greater awareness of poverty experienced on the ground. One indication of the success of this campaign would be more Singaporeans having a greater awareness of daily experiences of those in poverty, in relation to meritocracy and Singaporeans having more conversations amongst themselves and with the government on the issue of poverty. I can determine so by following up with the target audience yearly, for some time, on actions they have taken after the campaign.
# I dare not say that just this one campaign would be sufficient in eradicating poverty altogether as the perpetuation of the issue of poverty is dependent on many other issues (Sachs et al., 2001, pp. 70-75), which I would love to have addressed, but could not, due to the constraint of how wide the scope of this essay can be. Further, the eradication of poverty requires drastic changes and reforms over time. With that said, this campaign should still be adopted as targeting the lack of awareness of poverty amongst Singaporeans and promoting conversations both amongst citizens and between citizens and the government will allow for both concerted action (Fromm, 1976, pp. 57-87) to be taken by citizens and more effective policies to be enacted by the government to better work towards the eradication of poverty. The benefits of this campaign are substantial and will be a large step ahead towards our goal of eradicating poverty.	
# """/'related'   

