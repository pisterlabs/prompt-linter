import streamlit as st
import json
import instagrapi
import openai
import statistics
import numpy as np
from ml_model import train_and_predict_knn

#initializing client and api key
apicl = instagrapi.Client()

openai.api_key = "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"

#streamlit

#st.markdown("<h1 style='text-align: center;'>FakeOut</h1>", unsafe_allow_html=True)
#st.markdown("<h2 style='text-align: center;'>Team Cookies, Smart India Hackathon</h2>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>FakeOut</h3>", unsafe_allow_html=True)

input = st.text_area("Enter instagram account to be scanned: ", height=20)

def caption_check(p):

    chatcompleter = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[ 

    {"role":"system","content":"You are a machine judging a piece of text from instagram captions on whether it belongs to a legitimate account or a bot/spam account. The likelihood of the text being legitimate is rated on a scale of 0 to 1. Respond only with the score. If a list of captions is provided, rate it overall and return only the numeric score."},

    {"role": "user","content": "On 3rd October , we wear pink"},

    {"role": "assistant", "content": "0.8"},

    {"role": "user","content": "Sometimes laying on the floor just hits different😳🤣"},

    {"role": "assistant", "content": "0.8"},

    {"role": "user","content": "bday dump #2"},

    {"role": "assistant", "content": "0.7"},

    {"role": "user","content": "Click here to watch my latest video 😍👇"},

    {"role": "assistant", "content": "0.1"},

    {"role": "user","content": "Don't click on my account"},

    {"role": "assistant", "content": "0.3"},

    {"role": "user","content": p}])

    return chatcompleter["choices"][0]["message"]["content"]
#bio coherence scoring

def bio_check(p):

    chatcompleter = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[ 

    {"role":"system","content":"You are a machine judging a piece of text from an instagram bio on whether it belongs to a legitimate account or a bot/spam account. The likelihood of the text being legitimate is rated on a scale of 0 to 1. Respond only with the score."},

    {"role": "user","content": ":3\nmy clothing thingy: @radiocure.in"},

    {"role": "assistant", "content": "0.8"},

    {"role": "user","content": "jia\n17"},

    {"role": "assistant", "content": "0.5"},

    {"role": "user","content": "Click here to watch my latest video 😍👇"},

    {"role": "assistant", "content": "0.1"},

    {"role": "user","content": p}])

    return chatcompleter["choices"][0]["message"]["content"]
#logging in

user='snuhaxs'
password='snuhaxs@123'
apicl.login(user, password)
#finding numeric id from username

if input:
    username=input
    st.text('so is the ' + username + ' account to be checked: ')
    choice=st.text_area("(respond with yes/no): ", height=20)
    answer=choice
    if choice=='yes':
        #RUN FROM THIS CELL IF CHECKING ACCOUNTS MULTIPLE TIMES
        targetid = apicl.user_id_from_username(username)
        #pulling data as json

        userinfo = apicl.user_info(targetid)
        userinfo = json.loads(userinfo.json())
        #assigning values:

        username =  userinfo["username"]

        full_name = userinfo["full_name"]

        is_private = userinfo["is_private"]

        profile_pic_url = userinfo["profile_pic_url"]

        is_verified = userinfo["is_verified"]

        media_count = userinfo["media_count"]

        follower_count = userinfo["follower_count"]

        following_count = userinfo["following_count"]

        biography = userinfo["biography"]

        external_url = userinfo["external_url"]

        is_business = userinfo["is_business"]

        public_email = userinfo["public_email"]

        contact_phone_number = userinfo["contact_phone_number"]
        detailslist = [username, full_name, is_private, profile_pic_url, is_verified, media_count, follower_count, following_count, biography, external_url, is_business, public_email, contact_phone_number]
        #post data:

        posts = apicl.user_medias(targetid)
        captions = []

        #collecting caption data

        for post in posts:
            captions.append(post.caption_text)

        #collecting like data

        likes_per_post = []

        for post in posts:
            likes_per_post.append(post.like_count)

        if len(posts) != 0:
            avg_likes = statistics.mean(likes_per_post)

        else:
            avg_likes = 0
        #likes to followers ratio
        if avg_likes:
            like_to_follower = avg_likes / follower_count

        else:
            like_to_follower = 0

        #followers to following ratio
        if following_count > 0:
            follower_to_following = follower_count / following_count

        else:
            follower_to_following = -1

        #non alphabet characters in username ratio

        alphcount = 0

        for i in username:
            if i.isalpha():
                alphcount += 1

        alpha_ratio = alphcount / len(username)
        #boolean values

        #checking if name is the same as bio

        if full_name.lower() == biography.lower():
            is_name_bio = 1

        else:
            is_name_bio = 0

        #checking for profile photo

        common_url_element = "t51.2885-19/44884218_345707102882519_2446069589734326272_n.jpg"

        if common_url_element in profile_pic_url:
            has_pfp = 0

        else:
            has_pfp = 1

        #checking for external url in bio

        if external_url:
            is_bio_url = 1

        else:
            is_bio_url = 0

        #checking for contact information

        if public_email:
            is_email = 1

        else:
            is_email = 0

        if contact_phone_number:
            is_phone = 1

        else:
            is_phone = 0

        #giving numeric values to booleans

        is_private = int(is_private)

        is_verified = int(is_verified)

        is_business = int(is_business)
        #final coherence test for bio
        if biography:
            bio_legit = float(bio_check(biography))

        else:
            bio_legit = -1
        #final coherence test for captions

        caption_list = ""

        for i in captions[0:15]:
            caption_list = caption_list + "'" + i + "'" + ", "

        if is_private == 0:
            caption_legit = float(caption_check(caption_list))  
            
        else:
            caption_legit = -1
        parameters = ["media count: ", "following count", "likes to followers: ", "followers to following: ", "ratio of alphabetical characters: ", "bio legitimacy: ", "caption legitimacy: ", "is name = bio: ", "is private: ", "has pfp: ", "is verified: ", "external url: ", "is business: ", "has email: ", "has phone: "]

        final_data = [media_count, following_count, like_to_follower, follower_to_following, alpha_ratio, bio_legit, caption_legit, is_name_bio, is_private, has_pfp, is_verified, is_bio_url, is_business, is_email, is_phone]
        
        st.text("You can now\n1. get user info\n2. bot detection\n3. account type\n4. end application")
        question=st.text_area("What would you like to do: ", height=20)
        if input:
            answer=question
            if answer=='get user info':
                for i in range(len(parameters)):
                    left_column, right_column = st.columns(2)
                    if i % 2 == 0:
                        with left_column:
                            st.text(parameters[i] + str(final_data[i]))
                    else:
                        with right_column:
                            st.text(parameters[i] + str(final_data[i]))

            if answer=='bot detection':
                checking_data=np.array([follower_count, following_count, len(biography), media_count, is_private, len(username)-alphcount, len(username)])
                train_and_predict_knn(checking_data)
                
            
            if answer=='account type':
                    if is_business:
                        st.text('Business account')
                    if is_private:
                        st.text('Private account')
                    if is_verified:
                        st.text('Verified account')
            
            if answer=='end application':
                st.text('Thank you for using FakeOut!')
    elif choice=='no':
        st.text('Thank you for using FakeOut!')
