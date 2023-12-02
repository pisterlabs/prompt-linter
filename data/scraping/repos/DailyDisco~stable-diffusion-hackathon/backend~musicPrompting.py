# pip install openai - https://beta.openai.com/docs/api-reference/introduction
import openai
import os
from dotenv import load_dotenv
load_dotenv()

# openai.organization = "org-nbdjhImDsCphzU9Z3cDknm68"
openai.api_key = os.getenv("OPENAI_API_KEY")

def mubertPrompt(story_excerpt):
    mubert_tags_string = 'tribal,action,kids,neo-classic,run 130,pumped,jazz / funk,ethnic,dubtechno,reggae,acid jazz,liquidfunk,funk,witch house,tech house,underground,artists,mystical,disco,sensorium,r&b,agender,psychedelic trance / psytrance,peaceful,run 140,piano,run 160,setting,meditation,christmas,ambient,horror,cinematic,electro house,idm,bass,minimal,underscore,drums,glitchy,beautiful,technology,tribal house,country pop,jazz & funk,documentary,space,classical,valentines,chillstep,experimental,trap,new jack swing,drama,post-rock,tense,corporate,neutral,happy,analog,funky,spiritual,sberzvuk special,chill hop,dramatic,catchy,holidays,fitness 90,optimistic,orchestra,acid techno,energizing,romantic,minimal house,breaks,hyper pop,warm up,dreamy,dark,urban,microfunk,dub,nu disco,vogue,keys,hardcore,aggressive,indie,electro funk,beauty,relaxing,trance,pop,hiphop,soft,acoustic,chillrave / ethno-house,deep techno,angry,dance,fun,dubstep,tropical,latin pop,heroic,world music,inspirational,uplifting,atmosphere,art,epic,advertising,chillout,scary,spooky,slow ballad,saxophone,summer,erotic,jazzy,energy 100,kara mar,xmas,atmospheric,indie pop,hip-hop,yoga,reggaeton,lounge,travel,running,folk,chillrave & ethno-house,detective,darkambient,chill,fantasy,minimal techno,special,night,tropical house,downtempo,lullaby,meditative,upbeat,glitch hop,fitness,neurofunk,sexual,indie rock,future pop,jazz,cyberpunk,melancholic,happy hardcore,family / kids,synths,electric guitar,comedy,psychedelic trance & psytrance,edm,psychedelic rock,calm,zen,bells,podcast,melodic house,ethnic percussion,nature,heavy,bassline,indie dance,techno,drumnbass,synth pop,vaporwave,sad,8-bit,chillgressive,deep,orchestral,futuristic,hardtechno,nostalgic,big room,sci-fi,tutorial,joyful,pads,minimal 170,drill,ethnic 108,amusing,sleepy ambient,psychill,italo disco,lofi,house,acoustic guitar,bassline house,rock,k-pop,synthwave,deep house,electronica,gabber,nightlife,sport & fitness,road trip,celebration,electro,disco house,electronic'

    # Load tags
    mubert_tags = mubert_tags_string.split(',')
    # Sort by length
    mubert_tags.sort(key=len, reverse=False)
    few_tags = mubert_tags[100:]
    # Recompose string
    mubert_tags_string = ','.join(few_tags)


    ## GPT Prompting - The prompt should show the available tags, and then the story excerpt, then ask for a new tag
    music_tag_prompt= f"""
                        Example tags: {mubert_tags_string}
                        Story excerpt: {story_excerpt}
                        Classify the text from "Story excerpt" using the following tags from "Example tags":
                    """
    
    ## Send to OpenAI GPT-3 API
    response = openai.Completion.create(
        engine="davinci",
        prompt=music_tag_prompt,
        temperature=0.5,
        max_tokens=40,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.6,
        stop=["\n"]
    )
    print(response)
    ## Make sure response is under 20 tags and only matches options from mubert_tags_string
    music_tag = response['choices'][0]['text'].strip()

    ## Query MuBert API with these tags as the prompt
    ## TODO: What's the API endpoint?

    ## Save MuBert music response
    ## TODO: Save music
    # The name should reflect the tags given to MuBert
    music_mp3_path = f"music/{music_tag}.mp3"

    return music_mp3_path, music_tag
    print(music_mp3_path, music_tag)
    print(response['choices'])