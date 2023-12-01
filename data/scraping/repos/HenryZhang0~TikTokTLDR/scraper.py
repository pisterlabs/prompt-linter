import json
from TikTokApi import TikTokApi
from collections import Counter
import cohere
from cohere.classify import Example
from cohereClassifier import classify_hashtag
from hashtags import *
import asyncio

api = TikTokApi(custom_verify_fp="verify_7a8db9a47431e0ff5cbe0c6c565e015f")
def scrape(id):
    co = cohere.Client('tROv76zKglQGIbzsywOJ4b21v6UogXMwcl7qfSrU')
    user = api.user(username=id)
    # for liked_video in user.liked(username='public_likes'):
    #    print(liked_video)
    userdata = user.as_dict
    hashtags = []
    selfHashtags = []
    liked_users = []
    recent_like = []
    liked_sounds = []
    # list of common hastags to filter out
    keywords = ["fyp", "foryou", "xyzbca", "viral", "pov", "greenscreen", "stitch", "trending", "duet", "1", "2", "3", "4", "5",
                "6", "fy", "iyk", "ad"]
    classifyExamples = [Example("Trees", "Alt"), Example("asmr", "Straight"), Example("Diesel", "Alt"), Example("Donuts", "Alt"), Example("Watermelon", "Alt"), Example("Fruit", "Alt"), Example("Zoo", "Alt"), Example("swag", "Straight"), Example("techtok", "Alt"), Example("fitness", "Straight"), Example("prank", "Straight"), Example("whisper", "Straight"), Example("gay", "Straight"), Example("lgbt", "Straight"), Example("comedy", "Straight"),
                        Example("funny", "Straight"), Example("cemetery", "Alt"), Example("cute", "Straight"), Example("dance", "Straight"), Example("italian", "Alt"), Example("cars", "Alt"), Example("fun", "Straight"), Example("sealife", "Alt"), Example("doggo", "Alt"), Example("art", "Straight"), Example("painting", "Alt"), Example("love", "Straight"), Example("diy", "Alt"), Example("basketball", "Alt"), Example("backpack", "Alt"), Example("magic", "Alt"), Example("chef", "Alt"), Example("fashion", "Straight"), Example("swimming", "Alt"), Example("couple", "Straight"), Example("canada", "Straight"), Example("boys", "Straight"), Example("bts", "Straight"), Example("gaming", "Alt"), Example("valorant", "Alt"), Example("kpop", "Straight"), Example("toronto", "Straight"), Example("korean", "Straight"), Example("food", "Straight"), Example("mixed", "Straight"), Example("culture", "Straight"), Example("frat", "Straight")]
    # data
    data = {}
    likeduserprofile = {}
    userdata_info_full = user.info_full()
    full_stats = userdata_info_full["stats"]
    data['follower_count'] = full_stats["followerCount"]
    data['following_count'] = full_stats["followingCount"]
    data['likes_count'] = full_stats["heart"]
    data['videos_count'] = full_stats['videoCount']
    data['id'] = userdata["id"]
    data['username'] = userdata["nickname"]
    data['openFavorite'] = userdata['openFavorite']
    data['profilePicture'] = userdata['avatarLarger']
    liked_videos = list()
    liked_list = user.liked()

    # for video in user.videos(count = 100):
    # print(video.hashtags)
    # for hashtag in video.hashtags:
    # Exclude hashtags
    # if not any([a in hashtag.name for a in keywords]):
    # selfHashtags.append(hashtag.name)

    # for the example video
    hashtag_videos = {}
    sounds_videos = {}
    challenges = []
    liked_video_count = 0
    average_likes = 0
    average_views = 0
    average_duration = 0
    most_viewed_video = []
    hashtag_set = set()
    user_set = set()
    sound_set = set()
    #

    user_liked_videos = user.liked(username='public_likes', count=1000)
    user_liked_videos = list(user_liked_videos)
    for video in user_liked_videos:
        liked_video_count += 1

        parameters = {'hashtags': []}
        parameters['video_id'] = video.id
        try:
            parameters['video_sound'] = video.sound.title
        except:
            print('sound failure')
            # print(video.info()["video"]["playAddr"])
        
        # Averages for USER STATS
        
        most_viewed_video += [(video.id, video.as_dict["stats"]['playCount'])]

        
        average_likes += video.as_dict["stats"]['diggCount']
        average_views += video.as_dict["stats"]['playCount']
        average_duration += video.as_dict['video']['duration']
        
        # author
        parameters['video_author'] = video.author.username
        user_set.add(video.author.username)
        parameters['liked_profile_picture'] = video.author.as_dict['avatarLarger']
        likeduserprofile[video.author.username] = video.author.as_dict['avatarLarger']
        liked_users.append(video.author.username)

        #sounds
        try:
            sound_set.add(video.sound.title)
            if not any([a in video.sound.title for a in ["son original", "original sound", "sonido original"]]):
                liked_sounds.append(video.sound.title)
                # video sound example
                #print("KEEEEEK\n\n", video.sound.info()['playUrl'])
                if video.sound.title in sounds_videos:
                    sounds_videos[video.sound.title].append(video.sound)
                else:
                    sounds_videos[video.sound.title] = [video.sound]
                # 
        except:
            print('sound suck')

        for hashtag in video.hashtags:
            parameters['hashtags'].append(hashtag.name)
            hashtag_set.add(hashtag.name)

            # Exclude hashtags
            if not any([a in hashtag.name for a in keywords]):
                hashtags.append(hashtag.name)

            # video hashtag example
            if hashtag.name in hashtag_videos:
                hashtag_videos[hashtag.name].append(video)
            else:
                hashtag_videos[hashtag.name] = [video]
            #
        liked_videos.append(parameters)
    data['likedVideos'] = liked_videos


    most_viewed_video.sort(key = lambda x: x[1], reverse=True)
    most_viewed_video = most_viewed_video[:10]


    # find random videos to showcase
    likedVideoLen = len(data['likedVideos'])
    randomVideo = data['likedVideos']
    rewind = []
    randomVideo = randomVideo[:: round(likedVideoLen / 3)]
    for i in range(len(randomVideo)):
        rewind.append(get_video(randomVideo[i]['video_id']))
    data['rewind'] = rewind
    
    data['hashtag_count'] = len(hashtag_set)
    data['sound_count'] = len(sound_set)
    data['creator_count'] = len(user_set)

    
    c = Counter(hashtags)
    c_user = Counter(liked_users)
    c_sound = Counter(liked_sounds)
    c_self = Counter(selfHashtags)
    # print ('most common', c.most_common(10))
    data['self_hashtags'] = c_self.most_common(10)
    data['most_liked_users'] = list(
        map(lambda x: list(x) + [likeduserprofile[x[0]]], c_user.most_common(5)))
    #print(most_viewed_video)
   
   # MOST POPULAR LIKED VIDEOS
    #mvl = api.video(id=most_viewed_video[0][0]).as_dict
    #data['most_popular_liked_videos'] = [mvl["video"]["downloadAddr"], mvl["author"]]
    data['most_popular_liked_videos'] = get_video(most_viewed_video[0][0])
    # cohere_summary_str = ". ".join([api.video(id=x[0]).as_dict["desc"] for x in most_viewed_video])
    #print(cohere_summary_str)

   
   # MOST COMMON
    data['most_common_hashtags'] = c.most_common(10)
    #print(data['most_common_hashtags'])
    data['most_common_hashtags_videos'] = []
    for hse in data['most_common_hashtags']:
        data['most_common_hashtags_videos'] += [hashtag_videos[hse[0]][0].id]
        
    data['most_common_hashtags_video'] = hashtag_videos[data['most_common_hashtags']
        [0][0]][0].info()["video"]["downloadAddr"]
    
    # MOST COMMON SOUNDS
    data['most_liked_sounds'] = c_sound.most_common(10)
    data['most_liked_sounds_id'] = []
    for hse in data['most_liked_sounds']:
        data['most_liked_sounds_id'] += [sounds_videos[hse[0]][0].id]
    album_data = sounds_videos[data['most_liked_sounds'][0][0]][0].info()
    data['most_liked_sounds_album'] = {"most_liked_sounds_album":[album_data["playUrl"], album_data["coverLarge"], album_data["authorName"], album_data["album"]]}
    
    # USE FOR SEPARATE FUNCTION
    '''
    for x in data['most_liked_sounds']:
        thefofo = sounds_videos[x[0]][0].info()
        data['most_liked_sounds_video'] += [thefofo["playUrl"], thefofo["coverLarge"]] 
    '''
    
    ##data['most_liked_sounds_video'] = [sounds_videos[x[0]][0].info()["playUrl"] for x in data['most_liked_sounds']]
    #data['most_liked_sounds_video'] = sounds_videos[data['most_liked_sounds'][0][0]][0].info()["playUrl"]
    #print(data["most_liked_sounds_video"])
    #print("2222\n\n", data['most_common_sounds_video'])

    
    # USER DATA
    data['num_liked_videos'] = liked_video_count
    data['average_liked_views'] = average_views
    data['average_liked_likes'] = average_likes
    data['average_liked_duration'] = average_duration

    # Classification Model
    # classifications = co.classify(
    #     model='medium-20220217',
    #     taskDescription='Identify Users Hashtags as part of straight or alt tiktok',
    #     outputIndicator='Classify these hashtags',
    #     inputs=hashtags[:20],
    #     examples=[Example("Trees", "Alt"), Example("asmr", "Straight"), Example("Diesel", "Alt"), Example("Donuts", "Alt"), Example("Watermelon", "Alt"), Example("Fruit", "Alt"), Example("Zoo", "Alt"), Example("swag", "Straight"), Example("techtok", "Alt"), Example("fitness", "Straight"), Example("prank", "Straight"), Example("whisper", "Straight"), Example("gay", "Straight"), Example("lgbt", "Straight"), Example("comedy", "Straight"),
    #               Example("funny", "Straight"), Example("cemetery", "Alt"), Example("cute", "Straight"), Example("dance", "Straight"), Example("italian", "Alt"), Example("cars", "Alt"), Example("fun", "Straight"), Example("sealife", "Alt"), Example("doggo", "Alt"), Example("art", "Straight"), Example("painting", "Alt"), Example("love", "Straight"), Example("diy", "Alt"), Example("basketball", "Alt"), Example("backpack", "Alt"), Example("magic", "Alt"), Example("chef", "Alt"), Example("fashion", "Straight"), Example("swimming", "Alt"), Example("couple", "Straight"), Example("canada", "Straight"), Example("boys", "Straight"), Example("bts", "Straight"), Example("gaming", "Alt"), Example("valorant", "Alt"), Example("kpop", "Straight"), Example("toronto", "Straight"), Example("korean", "Straight"), Example("food", "Straight"), Example("mixed", "Straight"), Example("culture", "Straight"), Example("frat", "Straight")])

    # altScore = 0
    # straightScore = 0

    # Following code gives percentage weight for straightness!!!!!!!

    # for i in range(20):
    #     print(classifications.classifications[i].prediction)
    #     print(classifications.classifications[i].confidence.confidence)
    '''
    for i in classifications.classifications[0]["results"]:
            altScore += i.confidences[0].confidence
            straightScore += i.confidences[1].confidence
    '''
    #Generate scores based on hashtags 
    '''
    straight_score = classify_hashtag(hashtags, "straight", "alt", straight_examples, alt_examples, amount = 100)

    data['straight_score'] = straight_score
    # data['tiktokScoreResult'] = [tiktokScore, max(altScore, straightScore)/(altScore + straightScore)]

    cringe_score = classify_hashtag(hashtags, "cringe", "based", cringe_example, based_example, amount = 100)

    data['cringe_score'] = cringe_score

    education_score = classify_hashtag(hashtags, "educational", "entertainment", educational_example, entertainment_example, amount = 100)

    data['education_score'] = education_score
    ''' 
    # caching test users
    with open('test.json', 'w') as fp:
        json.dump(data, fp,  indent=4)
    return data

def get_audio(id):
    k = TikTokApi.sound(id=id)
    sound = k.info()
    url = sound['playUrl']
    cover = sound['coverLarge']
    authorName = sound['authorName']
    title = sound['title']
    print(url)
    return {"most_liked_sounds_album": [url, cover, authorName, title]}

def get_video(id):
    k = TikTokApi.video(id=id)
    url = k.as_dict["video"]["downloadAddr"]
    stats = k.as_dict["stats"]
    print(url)
    return {"address": url, "stats":stats}
# DATA SENDING TO FRONTEND
'''
{
    "id" : "",
    "username" : "",
    "picture" : "",
    "likeVideos" : [{
        "videoId" : "",
        "creatorId" : "",
        "hashtags" : [],
        "comments" : [{
            "comment" : ""
        }]
    }],
    
}
'''
