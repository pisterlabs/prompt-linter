from time import perf_counter
from langchain import OpenAI
from gpt_index import GPTSimpleVectorIndex
from nltk import word_tokenize



def break_up_file(tokens, chunk_size, overlap_size):
    if len(tokens) <= chunk_size:
        yield tokens
    else:
        chunk = tokens[:chunk_size]
        yield chunk
        yield from break_up_file(tokens[chunk_size-overlap_size:], chunk_size, overlap_size)

def break_up_file_to_chunks(input_text, chunk_size=4096, overlap_size=100):
    """
    Break up a given meeting text to chunks
    params:
        input_text : GIven transcription of the meeting
        chunk_size : size of the chunk
        overlap_size : size of the overlap
    """
    tokens = word_tokenize(input_text)
    return list(break_up_file(tokens, chunk_size, overlap_size))


def generate_meeting_minutes(meeting_transcription,input_index="index.json"):
    """
    Gneerate minutes of the given meeting transcription by loading the index from index file trained from corpus
    params:
        meeting_transcription : str: transcription of the meeting
        input_index : file: trained index file
    """
    index = GPTSimpleVectorIndex.load_from_disk(input_index)
    chunks = break_up_file_to_chunks(meeting_transcription)
    response = []
    prompt_response = []
    for i, chunk in enumerate(chunks):
        query = "Below is the transcribed text of the meeting recording. Generate an appropriate minute for the meeting with the memebers of the meeting, agenda of the meeting, summary of the meeting, and actions items from the meeting.\n" + chunk[i]
        response = index.query(query, response_mode="default")
        prompt_response.append(response.response)
        # breakpoint()
    query = "Consolidate these meeting minutes: " + str(prompt_response)
    final_response = index.query(query, response_mode="default")
    return final_response.response



if __name__ == '__main__':
    
    meeting_text = """
                (PERSON7) Hi.
                (PERSON8) <unintelligible/> [PERSON9] <unintelligible/> some big <unintelligible/> news?
                (PERSON7) Yes. Probably.
                (PERSON9) Ah. 
                Sorry.
                (PERSON7) So, z- yeah.
                Great.
                (PERSON8) <unintelligible/> Great.
                So, basically, uh for today uh-
                <other_yawn/> 
                Uh, can you hear me, right?
                (PERSON4) Yes.
                (PERSON9) Yes.
                (PERSON7) Yes.
                (PERSON8) Okay.
                So, so we've had the call last week and I think one important-
                Well, first of all, uh [PERSON9] officially started working with us, so so let's welcome him, even though we already know him from previous calls.
                Uh and uh yeah.
                He wi- he will do mainly [ORGANIZATION5] at uhm-[PROJECT4] specific implementations and and uh he will probably help out with n-best list navigation mainly.
                And [PERSON7] uh has some plans with for him for [PROJECT2] too, but that's between [PERSON9] and [PERSON7], I guess.
                Uh, so.
                Uh re- related to uh call last week.
                First of all, I asked [PERSON10] whether we would be- are expected to have integration for the uh n-best list navigation.
                And so f- far, it seems that we should expect uh that uh we will be required to actually provide at least some prototype.
                So in practice, it seems that we will probably show our current prototype or maybe some modification of it to the u- u- user interfacting and they will try to implement into the final product.
                Uh I will-
                Well, this is mostly [PERSON11] related, so I will not go into more details about this, because I think it's still- we are still ahead of time with this one.
                Uh so, what else.
                Mhm.
                <other_yawn/>
                There was some s- some uh mention of the [PROJECT7] integration.
                And I'm not sure what's the current status.
                But I think the user interface team should have implemented it, right?
                Like, just in case someone has m- more uh detailed news about this?
                <unintelligible/>
                (PERSON4) S- sorry, about [PROJECT7]- [PROJECT7] integration, or about the experiment?
                (PERSON8) Uh pst- uh, no no no, I'm ta- I'm talking about the integration and and basically, [PERSON10] was-
                Well.
                [PERSON10] wanted it implemented by the end of last month, or maybe two months ago.
                But I I didn't notice any conversation about it <unintelligible/>.
                (PERSON4) Ah.
                (PERSON8) So I take it that it's been done.
                Because of what's supposed to be done.
                (PERSON4) Okay.
                I hope so.
                Yeah, we we just-
                (PERSON8) Yeah, like-
                This this was when our <unintelligible/>-
                (PERSON4) We just, we just-
                The-
                Suggested uh- or like we offered our help.
                (PERSON8) Mhm.
                (PERSON4) If they need like-
                So so uh yeah.
                We decided, or or they decided that only back translation will be implemented in this like pile of integration.
                (PERSON8) Mhm.
                (PERSON4) And we offered our help and nobody asked us for for for our help, so I hope everything went smoothly.
                (PERSON8) Mhm.
                Yeah, I I have the same opinion.
                Like I think did not tell us that we should do it.
                And and I think it was quite clear that, so.
                (PERSON4) Mhm.
                (PERSON8) Yeah.
                I think I think thus far as some integration is uh concerned, I think we're fine now, so.
                (PERSON7) Yeah.
                (PERSON8) Let's not worry about that.
                (PERSON7) Yeah, uh I agree, let's not worry about it at the moment.
                Uh at the same time-
                So this in Czech, we call this strategy of of the dead beetle.
                Uh so-
                (PERSON8) Mhm.
                (PERSON7) Uh <laugh/> let's play the dead beetle strategy on this.
                Because we assume that it has been properly picked up and they do what they uh needed with that.
                At the same time, uh at the back of our heads, we should uh mhm uh like have this little worry.
                That at any point, suddenly, they can jump on us and and uh realize oh, we forgot to pick this up and it was never uh actually degraded in any way.
                And you would have to record what you send to them.
                Uh so.
                Uh don't uh don't fully take it as delivered, until it has been picked up uh by them.
                But that's uh- let's not worry, I totally agree. 
                (PERSON8) Yeah, I think-
                Uh I'm I'm taking a note, then I think if I ask about it on the next call, it should be fine.
                Just uh again, like we we have lot of time, ah- ahead of us.
                So so if we uh catch it now, it should be completely fine.
                (PERSON7) Yeah.
                (PERSON8) Uhm.
                Yeah.
                [PERSON11] mentioned that the the n-best list navigation sh- will probably not be easy to integrate into [PROJECT7].
                So that's another thing I will probably ask, can I next time.
                Whether it's fine to deliver it as a separate prototype module or something.
                I'm I'm not actually up to date with the current uh implementation uh of the of the browser translator.
                S- so since I don't know the details, I'm not sure I can which way we we can do that, but again, like we we can discuss it with the on the on the next call. 
                With the others.
                Uhm, but it would be good to have uh-
                (PERSON11) I mean, can I ask?
                (PERSON8) So, a general idea-
                Mhm.
                (PERSON11) Uh, is is there any incentive to have this in [PROJECT7]?
                Because [PROJECT7] was a very specific, like-
                (PERSON8) N- no, no, no, no, no.
                The the mhm, my idea was that uh- let's let's have this single module that can have all this functionality.
                It it cannot even, it-
                Like technically, it would be alright if it was just uh separate option for the [PROJECT7] or something.
                Like it would- wouldn't be related to [PROJECT7] at all, but it would be at least in the single-
                (PERSON11) Yeah, yeah.
                (PERSON8) Instance, right?
                Yes, that that kind of thing, so so we have something that <unintelligible/> can look into.
                And like okay, now you can try [PROJECT7], now you can try the navigation.
                It, this is basically for the presentation purposes mainly.
                (PERSON11) Uh huh, I see.
                (PERSON8) Like-
                (PERSON11) Well, i- i- it's absolutely easy to like develop it within TNT and then just paste it into [PROJECT7].
                (PERSON8) Yeah, okay, yeah.
                This this should be fine as a as a initial solution, if if uh they do not want-
                Like, again.
                We we can show it to [PERSON10] and if he's fine with it, we're fine.
                And otherwise, we can we can think about something else.
                The- 
                I was not thinking about something more complex or anything more complex, so.
                <other_yawn/>
                So it's good to know that this at least this is doable and it probably should not be much work, right?
                Okay, that's great.
                That's good for now.
                Uh okay.
                So [PROJECT7] is uh advancing, right?
                Little bit?
                (PERSON11) Uh, yeah, I would like to ask [PERSON4]-
                (PERSON8) Mhm.
                (PERSON11) What what's the number of segments we actually want to to have.
                Like currently, we have two times what we had for 2B.
                Uhm-
                (PERSON4) Yes. 
                (PERSON11) I I don't think this enough to get any new significant findings, but you you-
                (PERSON4) Mm.
                Yeah.
                I actually haven't estimated what what's like the the best or or or the sufficient number of of segments. 
                Yeah.
                I can look at it.
                (PERSON11) Yeah.
                (PERSON4) Uhm-
                (PERSON11) Uhm-
                I <unintelligible/>-
                (PERSON4) But we have more like uh, we have more uh options.
                I mean uh, with the uh-
                We have additional e- empty systems uh.
                We definitely need more than twice.
                (PERSON11) Yeah, that that's that's-
                (PERSON4) Or or the double, yeah.
                (PERSON11) Yeah.
                Um, I would also like to get some more specific timeline, like what we have to do, like what conference are we aiming for?
                And uhm-
                (PERSON8) So so uh.
                Well, on on Wednesday, or maybe early Thursday, there's a [ORGANIZATION3], but I think that's probably out of question.
                (PERSON4) <laugh/>
                (PERSON11) <laugh/>
                Okay, we don't have any uh uh any-
                We didn't just-
                We even did look into the data, so yeah.
                (PERSON8) Okay, okay.
                So basically, the next one should be uh any-
                (PERSON4) <unintelligible/>
                (PERSON8) Yeah, any any CL and that's 21st of November, so I'm not sure how much you expect to spend on analyzing, but-
                (PERSON11) Well-
                (PERSON8) Yeah?
                (PERSON11) M- I I want to suggest-
                Like it's going to be them or us who's going to analyze it?
                Or just combinating of both?
                Because <unintelligible/>-
                (PERSON4) Mmm-
                (PERSON11) We could start analyzing it now and just get m- more precise data later.
                (PERSON4) O- my original intentions was that they should analyze the thing relate- or or the stuff related to u- user experience.
                Uh, so, if they can extract some information from from the from how much uh time did they did the user spend on on s- like the stimulis and wh- yeah.
                And we should analyze the other things, like related to the translation quality.
                (PERSON11) Okay.
                (PERSON4) Yeah.
                That was my original intentional idea.
                (PERSON7) Yeah, but the question is whether they will have anyone to work on this at all.
                Uhm.
                (PERSON4) Yeah.
                (PERSON11) I mean-
                (PERSON8) Uh so definity definitely ask them.
                Whether whether they plan to invest some time into the analysis, or like.
                At least so so we know whether we can expect some results from them, right?
                (PERSON4) So so we can still live- we can still live with the analysis as uh [PERSON11] did them in uh in our paper for uh [PROJECT8] so it's like-
                (PERSON8) Mhm.
                (PERSON11) Well, that that's the problem, because-
                If I started writing the paper now, then I would just write the same paper.
                I just need to cooperate with someone or yeah.
                (PERSON4) Uh wh- w- what's the like so so we we would have we will have new data, so-
                (PERSON7) Yeah.
                (PERSON4) Or the new results, so.
                (PERSON11) Y- yeah yeah, that's true, but I would like do the same analysis, or-
                (PERSON4) Hm.
                (PERSON11) Like yeah.
                I mean the the problem is more like asking the right questions and like answering the graph is easy.
                But just having to ask, like what's the correlation between the uh the estimation and the confidence and all that.
                So I I have time to write it, but I just need to consult with someone.
                (PERSON8) Okay, so so so-
                (PERSON4) Yeah okay, so so-
                Like, yeah I'm free l- l- like or willing to consult the things.
                (PERSON11) Okay.
                (PERSON4) So that's what I'm saying now, l- that the the like the kind of analysis, that you did for the [PROJECT8] paper-
                (PERSON11) Mhm.
                (PERSON4) It was okay.
                And I liked it.
                And so in this way, like-
                We should-
                (PERSON11) But the- there's no sense in writing the same two papers just with slightly different data.
                (PERSON4) Okay, so we can add some more analysis.
                (PERSON11) <laugh/>
                (PERSON8) But but the question is uh like what-
                (PERSON4) The- there there definitely there won't be like the same data, because we have uh additional uh, additional MT systems.
                So we can compare uh this stuff, that how <other_yawn/> you know-
                (PERSON7) Quality of the MT system effects, uh that affects the results.
                (PERSON4) The the-
                We have like three MT systems uh for Czech and uh the one is that uh-
                That is using the [PROJECT1] and and is trained on small data.
                And the other one is using [PROJECT1] and it's trained on big data.
                And the other one is trained on big data and but it's the student one and optimized one for the CPU and it's fast.
                So we can somehow compare this three systems. 
                (PERSON11) Yeah.
                (PERSON4) And yeah.
                So I think, like, there definitely there will be additional content.
                (PERSON11) Okay, so uh, I just make up a template from [ORGANIZATION7] and start bringing in and uh then ask you by email.
                (PERSON4) Yeah and, but we have to-
                Like, don't forget that we have to ask uh mhm Estonian, Estonians to-
                (PERSON11) Mhm.
                (PERSON4) Assess the translation or evaluate it.
                (PERSON11) Oh and we have to do that for Czech as well, right?
                (PERSON4) Yeah.
                We have to do it for Czech as well.
                (PERSON11) Okay.
                Okay.
                (PERSON8) S- so-
                (PERSON4) So I think this will be uh again uh quite a time demanding thing, so.
                Especially <unintelligible/>-
                (PERSON7) So at the-
                These days we are running uh like a campaign over a number of tasks, we call it the autumn annotation uh exercises, or tasks at [ORGANIZATION4].
                And uh this one w- uh was not listed there, uh because well like it didn't occur to me.
                Uh but uh we can probably ask the people who are now uh uh s- uh preparing the agreements with [PERSON5].
                I hope we all put also [PROJECT5] as as a uh third option.
                Uh source uh for money for for those uh uh short term contracts.
                And uh it should be uh relatively easy to uh uh get people uh to do this evaluation.
                Uh well, it's it's definitely easy to contact them and w- w- we're not sure how many of them will say yes to uh uh one more uh task.
                (PERSON11) Mhm.
                (PERSON7) But-
                (PERSON4) Yeah, I-
                Yeah, I uh-
                (PERSON7) Yes.
                (PERSON4) This is happen like uh uh, as it was during the like in the springtime.
                So I'm not afraid about us getting the people for evaluating the stuff for Czech.
                (PERSON7) Yeah.
                (PERSON4) But I'm again afraid of the other partners and here it is start to-
                (PERSON11) But I've hea- Estonians were like very cooperative.
                I-
                (PERSON4) Yeah, but-
                (PERSON7) I- if it's <unintelligible/> only, because he's overloaded.
                So uh it's-
                (PERSON11) Ah, yeah.
                (PERSON7) Make sure to have uh uh back-up person. 
                <laugh/>
                (PERSON11) Okay.
                (PERSON4) Yeah, but I don't know <unintelligible/>- 
                They are they are cooperative, but I don't know how many human resour- or how much human resources they have.
                So it can still like <laugh/>.
                Delay.
                Somehow. 
                (PERSON8) So w- we can definitely ask them, right?
                So to to get some estimate.
                But-
                (PERSON4) Yeah, okay.
                So we can now uh w- uh.
                Okay, n- now we can start uh like preparing them, f- that uh that the time when they have to uh, tor they should they should find some uh human resources to evaluate the translation.
                So that, when we when we are finished with the annotation, or with the experiment, they are ready to, to run the uh run the annotation, or the evaluation.
                (PERSON8) Mhm. 
                (PERSON11) Okay.
                Uh maybe uh, are y- are you able to to get the paper together even in like, in case they will not be able to deliver on time?
                Just to.
                (PERSON11) In this case, we would like probab- postpone the paper. 
                (PERSON8) Mhm.
                (PERSON11) Because I I don't think we are <unintelligible/> publish this-
                (PERSON4) Yeah, because-
                (PERSON11) And it was be better to have it like, not only for the Czech, but-
                (PERSON4) Yeah.
                (PERSON11) For everything.
                (PERSON8) Okay.
                (PERSON7) Well the [ORGANIZATION7] deadline seems doable, if uh the Estonians are very responsive. 
                (PERSON8) Mhm.
                (PERSON11) Bu- but we don't need [ORGANIZATION7] specifically.
                Or do we?
                (PERSON8) Well, mhm, uh probably not, but w- why not try it for <unintelligible/>
                (PERSON7) Yeah.
                (PERSON11) Yeah uh uh obviously, but-
                (PERSON7) Yeah.
                (PERSON11) Yeah.
                (PERSON7) So then if, well if we miss [ORGANIZATION7], then the next one would be the January [PROJECT8], right?
                <laugh/>
                So-
                (PERSON11) <laugh/>
                (PERSON8) Well-
                (PERSON7) I guess.
                (PERSON11) Ah ah I'm not sure whether they-
                (PERSON4) And they have to-
                (PERSON11) The same thing <unintelligible/>-
                (PERSON4) We have to decide, how many translations uh you know.
                Because uh like the uh for every stimuli, or stimulus, the uh like the translati- i- it's a process of like, uh.
                (PERSON11) Oh, you want to translate the whole process?
                (PERSON4) <unintelligible/> the input and-
                Yeah and.
                So we have to, we have to somehow filter out the translations that we want to be assessed.
                (PERSON8) Mhm. 
                (PERSON4) And-
                (PERSON11) Well-
                Uh uh in the last paper, I just evaluated the last one, like the confront one and not the process.
                (PERSON4) Okay.
                (PERSON11) But it would be e- e- interesting to look at the process, but just-
                Tis-
                There is ten times more data to annotate and it's not easy.
                (PERSON4) Yeah, yeah, yeah.
                I was thinking about the first one and the last one and or the-
                (PERSON11) Yeah, but but those are just-
                (PERSON7) First viable?
                (PERSON11) Yeah.
                (PERSON4) First viable, yeah.
                First viable.
                (PERSON11) <laugh/> 
                Okay, but I guess those are just the details we can discuss via email.
                (PERSON7) Mhm mhm.
                (PERSON4) Mhm.
                (PERSON8) Okay, so that's that.
                And uh lastly, [PERSON4] has some results for the uh paraphrase-based augmentation, right?
                Uh, uh-
                Ho- how how-
                (PERSON4) Yeah, everything is still running.
                (PERSON8) Mhm.
                (PERSON4) And uh.
                Uh.
                Yeah, so so I tried the experiments with additional monolingual data and like there's uh.
                Two like uh f- uh like uh very much data of like four hundred million sentence pairs.
                (PERSON8) Mhm.
                (PERSON4) Uh and like in this, in this setting mhm, the quality of the two systems will back translat- using back translation- the system using back translation and system using paraphrases.
                It's almost the same.
                So then I tried uh using uh like half of the monolingual data, quarter of the monolingual data and so like uh decreasing the amount of the monolingual data.
                Because uh like when using no monolingual data, uh we have seen that uh the the system that is using the paraphrases is better.
                (PERSON8) Mhm.
                (PERSON4) Uh than the system using just the back translation.
                So I just want to uh like explore where-
                (PERSON8) Uh so-
                (PERSON4) How it-
                (PERSON8) So-
                How it changes when I'm like uh decreasing the amount of the monolingual data.
                (PERSON8) Mhm.
                Uh so there's a difference between back translated data and monolingual data.
                Like because you are back t-
                (PERSON4) <unintelligible/>
                (PERSON8) You are back translating the the sentences from the parallel data, right?
                Also?
                (PERSON4) Like uh if you look uh, if you look to the to the document I like just listed all the data we are using in our experiment.
                (PERSON8) Mhm.
                (PERSON4) ORIG is the original CzEng.
                (PERSON8) Mhm.
                (PERSON4) So sixty million open <unintelligible/> English sentence pairs.
                (PERSON8) Mhm.
                (PERSON4) Then BT is the back translated version of CzEng. 
                Whe- when Czech, all the Czech uh sentences were back translated uh to English.
                (PERSON8) So-
                (PERSON4) Para-
                (PERSON8) Those were uh b- uh m- just uh the BT therefore gives you uh uh new synthetic uh source sentences, right?
                If I'm understanding correctly.
                (PERSON4) Yeah yeah yeah.
                It's like-
                (PERSON8) Okay, okay.
                (PERSON4) Yeah yeah, it's like a s- syntheti- synthetic version of CzEng. 
                (PERSON8) Mhm.
                (PERSON4) The other synthetic version of Cz- CzEng is PARA, where uh these uh back translations to English are made in the way that they are uh as diverse as possible.
                (PERSON8) Mhm.
                (PERSON4) From the original sources.
                (PERSON8) Mhm.
                (PERSON8) In CzEng or English sources.
                (PERSON7) S- sorry, that is that is confusing to me.
                Uh you are, y- uh so it's synthetic on the target side?
                Uh-
                (PERSON8) Uhm.
                On the source side.
                (PERSON7) On the source side.
                (PERSON4) Source side.
                (PERSON7) B- in uhm- normally, you're training uh uh a system which goes from English into Czech, or the other f- uh way around?
                (PERSON4) This is, this is like the setting for for the training the system from English to Czech.
                (PERSON7) From English into Czech.
                (PERSON4) Yes.
                (PERSON7) Uh so you are creating synthetic English, uh yep, okay.
                (PERSON4) Yes.
                (PERSON7) And uh you are translating.
                So the translations are as diverse from the English difference as possible, but still good enough.
                And how is that measured?
                How how do you uh?
                (PERSON4) Uh, this is the system that was designed by uh [PERSON3].
                (PERSON7) Okay, yep.
                (PERSON4) I don't know the the details about it, but-
                (PERSON7) Mhm.
                (PERSON4) Like uh [PERSON1] just generated-
                (PERSON7) Okay.
                (PERSON4) So-
                The-
                (PERSON7) Yeah.
                Mhm.
                (PERSON4) Data for this.
                So like, our assumption or or like the h- the way how the system was created suggests that the that the translation should be more like, diverse from the original sources.
                (PERSON7) Mhm.
                (PERSON4) I mean from the original English uh reference in <unintelligible/>.
                (PERSON8) Mhm.
                (PERSON4) And <unintelligible/> uh, there are just uh lot of Czech o- uh like authentic Czech uh sentences, back translated with that PT system or or uh just the like or normal back translation to English.
                (PERSON8) Mhm.
                (PERSON4) So <unintelligible/> the experiments were like the first experiment is using just the bilingual data.
                (PERSON8) Mhm.
                (PERSON4) Uh in that PT a- and in a way that we first pretrain on synthetic data and then fine tune on the original uh <unintelligible/>. Or the authentic <unintelligible/>.
                And uh the other exp- experiment is uh when we in in the pretraining phase, we extended the data with the those monolingual data.
                So there are like we are like instead of using just 60 million sentences, we are using 470 million sentences.
                (PERSON8) Mhm.
                (PERSON4) Or sentence pairs. 
                Yeah?
                And then uh I'm running the third experiment, is like series of experien.
                I'm running uh the same as in the second experiment.
                But lowering ma- uh lowering the amount of the uh monolingual data.
                Yeah?
                So, now 50 percent, 25 percent, 12.5 percent and 6 percent. 
                (PERSON8) Mhm.
                (PERSON4) Yeah, and the result, you can see the results in the table.
                I linked to you.
                And-
                (PERSON8) Mhm.
                (PERSON4) Some of the experience are still running, so or most of the experi- uh like those uh fraction experie- experiments I'm running I'm still running the pretraining phase.
                But uh from the pretraining phase, the uh the scores are very similar in the in both variants. 
                (PERSON8) Mhm.
                (PERSON4) But yeah.
                Let's wait for the fine tuning phase.
                What what will appear.
                (PERSON8) Okay.
                (PERSON7) And the main motivation is to have am Czech to English system, uh which is good for uh the outbound uh uh set up?
                Or?
                (PERSON4) Uhm.
                Not really.
                We we somehow div- <laugh/> diversed from this outbound translation motivation, because-
                (PERSON7) Mhm.
                (PERSON4) Like we started from the uh yeah, we started from the uh like [PROJECT9] translation, but [PERSON1], [PERSON1]'s branch of this experiment was about this data a- augmentation.
                So using paraphrases for data augmentation in standard machine translation.
                (PERSON7) Okay.
                (PERSON4) And uh so like when we were aiming for [PROJECT6] in the mid August, I was doing the m- [PROJECT9] branch and [PERSON1] was working on the this augmentation <unintelligible/>.
                (PERSON7) Mhm.
                (PERSON8) Hm.
                (PERSON4) And uh in the mid August, th- like this augmentastion uh uh branch uh appeared to be better and promising.
                More promising that the branch with [PROJECT9].
                (PERSON7) Mhm.
                (PERSON4) And since [PERSON1] was leaving, I also like uh started to work on this augmentation branch of the experiment.
                (PERSON7) Mhm.
                (PERSON4) So now I'm working on it uhm like alone, or like I'm discussing.
                I just discussed the stuff with [PERSON1].
                And we should probably the <other_yawn/> second half of of this week, we should uh have a call about this.
                But yeah.
                (PERSON7) So the motivation is-
                (PERSON4) It is somehow it' somehow diverged uh-
                (PERSON7) Mhm.
                (PERSON4) From the from the original [PROJECT9].
                (PERSON7) Yeah.
                So you're just trying to get a better performance uh by y- augmenting uh source uh side with paraphrases or back translation-
                (PERSON4) Yeah.
                (PERSON7) Or uh something else.
                Uh huh.
                (PERSON4) Yeah.
                Yeah.
                (PERSON8) Also-
                (PERSON4) Yeah with paraphrases.
                We are we're like we are looking s-
                So we're exploring whether those paraphrases or or uh creating back translations in a way that they are the diverse from the original sources.
                Whether it makes the system better.
                (PERSON8) Mhm.
                (PERSON7) Yeah.
                Uh there is uh one thing that uh is worth exploring here.
                And that is uh-
                It's also I think that the experiments that you are doing now are very similar uh in nature but obviously, you had the paraphrases and that's different.
                Uh from the [PERSON2] does with his concat uh back translation.
                (PERSON4) Yeah.
                (PERSON7) So you are doing uh mhm the uh like transfer uh.
                You pretrain on something and then switch once.
                Uh while [PERSON2] has these multiple switches uh uh on-
                (PERSON4) Yeah yeah yeah.
                (PERSON7) And when I was mentioning this to to some [ORGANIZATION2], uh they told me that what also works is to uh just label the input sentences with uh whether they are from the genuine or the back translated uh m- uh corpus.
                And then uh they saw some gains when mixing those in the standard way.
                So [PERSON2] set up uh has these camel uh uhm uh uh th- uh the camel shape.
                The waves.
                Uhm. 
                With the uh data labeled, uh you will mix it normally and you would see a steady growth.
                And hopefully it would grow uh higher.
                So I think that you could also for some of the promising set ups, you could also or try it with mixing the different styles of of data.
                And explicitly labelling that style.
                (PERSON4) Yeah.
                (PERSON7) Because then the system has the chance to uh like uh learn uh uh uh uh realize that some of the Czech uh v- or no English sentences, the target sentences, are uh uh uh more genuine, some are more translationese and benefit from th- this distinction.
                (PERSON4) Yeah, yeah, yeah.
                Okay.
                Thanks for the suggestion.
                But m- maybe I I don't want to diverge even more-
                (PERSON7) Yeah. 
                Yeah.
                Yeah.
                (PERSON4) From the original objected then.
                And uh as soon as we are finished with this experiment and and somehow ready uh have a pape- have have some results that are ready for paper, for for-
                (PERSON7) Mhm.
                (PERSON4) To be published, I want go back to the [PROJECT9] uh.
                (PERSON7) Yeah.
                (PERSON8) Mm.
                (PERSON4) Because otherwise-
                (PERSON7) That is needed.
                (PERSON4) <unintelligible/>
                (PERSON7) That is needed.
                (PERSON4) Yes, this is needed for the project, yeah.
                (PERSON7) So this is like the baseline. 
                We can s- see there's a baseline for the uh for the [PROJECT9] in a way.
                It's different single sources analyzings.
                (PERSON4) Yeah.
                (PERSON7) And technically, it's all [PROJECT4], right?
                Uh any-
                (PERSON4) Yeah, it's all [PROJECT4]. 
                (PERSON7) And you run it manually from some uh directory uh in or do you have some system to to launch these many experiments?
                (PERSON4) Uhm, not really.
                Yeah I'm running manually.
                (PERSON7) Yeah.
                Yeah, yeah.
                Uh b- because I've uh yeah.
                I need to get [PROJECT4] again up and running for myself. 
                (PERSON4) Okay.
                (PERSON7) Uh for uh some students of mine.
                So I'll probably ask uh uh uh you later if uh-
                Like I'll try-
                What is the current best uh uh set up, best compilation of [PROJECT4], how to get it easily running at [ORGANIZATION4].
                You you all now know, because you all all have gone through that.
                At-
                (PERSON4) Yeah yeah yeah.
                (PERSON8) Mm.
                (PERSON7) And-
                (PERSON4) Yeah, I'm actually using the Jin- some of the [PERSON1]'s compilation and uh yeah.
                I have to or we have to compile different [PROJECT4] for TLL 1 and 8.
                (PERSON7) Okay.
                (PERSON4) And different one for other TLLs and yeah.
                So.
                (PERSON7) So you use uh-
                You manually know which is which and you rely on [PERSON1]'s compilation?
                (PERSON4) Uh I have like a like a small screep, just deciding what-
                (PERSON7) What to run where. 
                (PERSON4) Version of of to run.
                Whether like-
                Depending on the machine.
                That it's running on.
                (PERSON7) Yeah.
                (PERSON8) Mm.
                D- definitely, you can ask [PERSON6], because he was dealing with this recently, so he might have some-
                (PERSON6) But I'm not sure that works.
                <laugh/>
                (PERSON8) Mhm.
                (PERSON7) Yeah, thank you.
                (PERSON8) S- so there's that.
                A- and also, like if if that's a- everything for the [PROJECT9], we can talk a little about the n-best list navigation.
                So [PERSON6], you you want to talk, or should I?
                (PERSON6) I can talk.
                (PERSON8) Mhm.
                (PERSON6) No problem.
                Uh, so.
                <other_yawn/>
                We're training module.
                Y- uh English-Czech module. 
                Using Fairseq, thanks to [PERSON11] and all that Fairseq has in implementation of the constraint decoding.
                So w- we will use that in our experiments.
                So uh, so far, we're still training and uh the validation shows that we have a a blue score of 27.
                So it's in it's okay.
                But we will still going to train for a couple of <unintelligible/>.
                And uh we're going to use the data sets uh provided by by [PERSON7].
                So yeah. 
                We have a particular one that is 50 sentences, 50 sentences, but many references.
                So this can be a good one.
                But uh uh I will still we will investigate the best one.
                And but now we're we are thinking about how we're going to select the constraints.
                So uh, we don't want to just select any word from the reference and use it as a constraint.
                So we are trying to think about uh ideas to select uh uh.
                I don't know.
                Uh informative words, maybe.
                I I uh we will have to think about that yet.
                But uh [PERSON8] told me about uh the use of word alignment.
                So uh we could try to try to check among the references, uh the words that uhm are different from the other references.
                Uh maybe I I I'm not sure yet. 
                But I will investigate.
                But I d-
                And uh what else?
                Let's say.
                About <unintelligible/>
                (PERSON7) Maybe maybe maybe-
                Sorry [PERSON6]- j-
                Maybe you could use TFIDF, uh for all the <laugh/>
                (PERSON6) Yeah.
                (PERSON7) All the variants and see if if anything sensible comes out of that.
                So first, look if if TFIDF offers uh some <unintelligible/> distinguish, essentially distinguish the uh references from one another.
                (PERSON8) Mhm.
                (PERSON6) Perfect.
                Yah.
                So I can-
                So the idea is that w- we will use these constraints and see uh how many constraints uh uh.
                It needs to to output the the reference we want and uh.
                <unintelligible/>
                Thinking in ways that we can uh <other_yawn/> test this constraints.
                But uh I think that we have some progress and-
                (PERSON7) Yeah.
                (PERSON8) Mhm.
                (PERSON6) To have to finetune things, but-
                (PERSON7) So w- what I find uh uh important is to decide how to compare the different uh uh approaches.
                Because uh you should always have fixed set of references, uh for a given uh comparison.
                And uh if you would do the like normal thing, or the most straightforward thing and leave one out, uh uh then you could easily end up with something which is not comparable.
                So I think that you would have to leave three out and then use these three uh uh left uh uh references as the source of the constraints.
                And then see how that behaves.
                But I I I um haven't thought about the set up carefully.
                But I just wanted to warn you that uh the set of references has to be fixed for the <unintelligible/> or any other scorce to be comparable. 
                (PERSON8) Mhm.
                (PERSON6) Okay.
                Yeah.
                (PERSON8) Mhm, so we still haven't discussed whether we will focus on the uh on the positive or negative constraints, right?
                Or [PERSON6] have you have you thought about it, which one would you-
                (PERSON6) I think the positive constraints would be-
                (PERSON8) Okay, so-
                (PERSON6) <unintelligible/>
                (PERSON8) The positive one-
                The positive one is telling the system which word needs to be outputed, right?
                (PERSON9) Yeah, perfect.
                (PERSON7) Yeah.
                And we get the positive ones from the references. 
                Because if the reference has it, then it's uh probably good word to to to use it.
                So-
                (PERSON8) Mhm.
                Because I-
                I've also just thought of an idea that be might for the negative ones.
                And in that case, we can compare uh when we have a a baseline output and it outputs some words from uh one of the references, which is not available in the other, we can start trying uh restricting the system and and pushing it to produce s- uh output similar to the other references.
                And whether it it is possible or not.
                But again, that's that's just-
                Just brief idea, so.
                Maybe something to think about.
                (PERSON7) So that uh so that uh like as soon as you see a hint of one of the references, you will use this particular reference to uh to-
                (PERSON6) Constrain?
                (PERSON7) To constrain it uh uh and to uh avoid the other ones?
                Or uh uh.
                Or to avoid this one?
                To avoid this one?
                (PERSON8) Well-
                So so obviously it would be nice to to identify uh any interesting words that that can be varied in the in the hypothesis.
                (PERSON7) Mhm.
                (PERSON8) Restrict the system and then try to produce the new hypothesis and this m- might give us some diversity.
                But the-
                In this case, the question is how to identify the interesting words right?
                Maybe we can just drop the stop words and and just focus on content words, and maybe restrict the con- content words, but.
                Again, like this is would be hard to estimate on the run without the, during the the mhm like when it's deployed to t- it would be hard to estimate whether the other hypotheses are relevant or mhm like qual- of of a decent quality, right?
                So yeah, like again, just may- maybe something to think about.
                I haven't though- thought this through either.
                yet.
                So so-
                (PERSON7) And and another word of warning.
                Even with the simpler uh idea that uh [PERSON6] is now exploring.
                (PERSON8) Mhm.
                (PERSON7) I think that we wi- we really have to do some manual evaluation, uh in addition to uh the blast course.
                (PERSON8) Mhm.
                (PERSON7) Uh because uh uhm well the uh the set of references is not uh uh big enough, it's not uh like uh solid evaluation.
                (PERSON6) Yeah, okay.
                (PERSON7) I think it's-
                Uh yeah, we really need to uh uh- 
                Once we have some some results, uh we need to uh select a small set of uh of the options and and evaluate those manually.
                In an annotation exercise.
                (PERSON8) So so so so the multi-referential data set that's how many sentences?
                50 or or 100?
                (PERSON6) There's one with 50 and another one there I'm planning to use has uh 1400 <unintelligible/> sentences.
                And each one has around 4300 references, so.
                (PERSON8) Mhm.
                (PERSON6) We have uh uh-
                Maybe these two data sets are going to work, I don't know.
                (PERSON7) How how many references do you have for the second one?
                Wh- what is the second one?
                (PERSON6) The second one is a multip- Czech ref- multiple Czech references in-
                It it has uh 1400 source sentences.
                (PERSON7) Yeah.
                (PERSON6) And uh it has has four thousand uh three hundred references.
                (PERSON7) Four thousand-
                (PERSON6) For each sentence, I think.
                (PERSON7) Four thousand?
                (PERSON6) Four. 
                Four.
                (PERSON7) Four.
                (PERSON6) Four thousand, sorry.
                (PERSON7) Yes.
                (PERSON6) Sorry, my pronunciation of numbers is terrible.
                (PERSON7) Mhm.
                (PERSON6) Okay.
                Four thousand, uh references.
                Around four thousand references for each sentence.
                As as I have noted-
                (PERSON4) <unintelligible/>
                (PERSON7) That's strange. 
                No, no, no-
                I d- I can't-
                (PERSON4) No no no-
                (PERSON6) No?
                (PERSON7) Four references.
                (PERSON4) I- i- it should be alto- altogether four thousand references. 
                (PERSON6) Oh man, yeah, I I I-
                This this notes are terrible, yeah, I'm sorry.
                (PERSON7) Yeah.
                (PERSON6) I will check that again, but uh the-
                Confirm that uh that the the the first one, that I uh has fif- uh the many Czech references-
                (PERSON7) Yes.
                (PERSON6) It has ifty sentences, but-
                (PERSON7) That's correct.
                (PERSON6) But-
                (PERSON7) Yeah.
                (PERSON6) One thousand sentence uh-
                One thousand references, right?
                (PERSON7) Easily.
                So there are many uh so-
                Uh there there are actually dozens of thousands uh of uh of of references.
                So this is this is a very small set in number of source sentences, but very diverse.
                And all the other ones are m- more normal.
                Uh so you have uh more sentences.
                And uh just uh a few handful of references there.
                (PERSON6) Yeah, it's the some of them, I'm sorry, yeah.
                The reference- the the second one is just the the some of the all all references-
                (PERSON7) Yeah. 
                Yeah yeah.
                (PERSON6) Okay, I'm sorry about that.
                (PERSON8) Actually, the the the uh data set with the 50 references per per sentence, that can be quite well used for measuring the diversity of the-
                (PERSON6) Yeah.
                (PERSON8) Mm.
                Uh our n-best list uh generation matters, right?
                Because we can do it in a way uh-
                We can uh basically do it a- as an-
                Well, okay.
                <laugh/>
                It it can be uh- treated as a uh information retrieval, right?
                W- we are hoping to produce uh several hypotheses that are in the in the list of similar references or provided references, right?
                (PERSON6) Yeah.
                Uh yeah.
                I was thinking that uh we can we can measure uhm how many constraints work.
                For example, let's uh think that we are the user and and we can think how many constraints we need to add to che- to to to reach the the reference w- we want.
                (PERSON7) Yeah.
                And this is uh also an something that we can try, right?
                (PERSON7) Mhm.
                (PERSON8) Mhm.
                (PERSON6) Uh okay, I get one one constraint and it already outputs the reference I want.
                The sentence I want.
                But how many will I need to to to provide the-
                (PERSON8) Mhm.
                (PERSON6) To reach the sentence.
                So I think that we can measure that also.
                Um I don't know.
                Uh I still have to think more about that.
                (PERSON7) Mhm.
                (PERSON8) Mhm.
                (PERSON6) Now I was focusing on training that and-
                (PERSON7) Mhm.
                Yeah.
                (PERSON6) Unfortunately it took too much time for for me to to to s- to start training.
                But uh now I think that I can gave more time to think about the experiments.
                (PERSON7) Yeah.
                Okay.
                Thank you.
                (PERSON8) [PERSON6]- I think I think that's that's fine for today.
                Uh as far as the task is concerned, we can discuss it several times during the week.
                Or possibly [PERSON7] wants to-
                But I don't think it it's this week, maybe next week, we can have a uh separate discussion on this.
                Right?
                So, I think if there are no other question, this should be it for this week, right?
                (PERSON7) Yeah, I was just curious if if [PERSON9] is happy, because uh it's uh his week like third day uh uh in practice, uh uh or maybe.
                So if if [PERSON9] is uh still on the call, if everything works as expected?
                (PERSON8) Uh.
                (PERSON7) Maybe he's muted, yeah.
                Uh too too strong.
                <laugh/>
                (PERSON8) Well.
                (PERSON6) <laugh/> 
                (PERSON8) He's he's still getting used to the environment.
                (PERSON7) Yeah, yeah.
                And uh and on uh for [PERSON6], I was curious, how is [LOCATION1] doing in in the COVID pandemic?
                Because here they're uh like making ever- uh all the restrictions harsher and harsher, so.
                (PERSON6) Yeah.
                (PERSON7) Uh how is [LOCATION1] doing?
                (PERSON6) We still have the the restrictions.
                But uh uh people people are are are ah are the big problem here, because they are still around and <unintelligible/> masks and stuff and stuff.
                But the the cases are decreasing, so we are-
                (PERSON8) Mhm.
                (PERSON7) Hm.
                (PERSON6) Happy about that, but it's uh slow decrease, because they are uh like 600 people dying every day, so <laugh/>.
                (PERSON7) Six hundred.
                Yeah, hm.
                Dying?
                And not not getting-
                (PERSON6) Dying, yeah.
                Uh yeah.
                I'm I'm checking the the your numbers as well.
                (PERSON7) Mhm.
                (PERSON6) But and.
                Uh they are increasing, right?
                But they're almost no deaths compared to to [LOCATION1].
                The [LOCATION1] is is is <laugh/>-
                (PERSON7) Yeah, but what is the population of [LOCATION1]?
                It's uh-
                (PERSON6) Yeah, it's like-
                (PERSON7) Quarter billion?
                No?
                (PERSON6) No, two billion I think.
                (PERSON7) Two billion, oh oh oh okay.
                <laugh/>
                (PERSON6) But yet, the- there are too many deaths-
                (PERSON7) Mhm.
                (PERSON6) And people don't seem to be worried, so.
                (PERSON7) Mhm.
                <laugh/>
                (PERSON6) <unintelligible/> anything.
                <laugh/>
                Uh yeah.
                There is, uhm-
                But that's fine <laugh/>
                I'm I'm-
                (PERSON7) As long as you keep yourself isolated.
                <laugh/>
                (PERSON6) <laugh/> 
                I'm trying to keep isolated.
                <unintelligible/>
                (PERSON8) Yeah, I I I I think we ar- also the [PERSON6]'s oc- currently working on the uh getting the visa, so.
                But-
                (PERSON6) Oh yeah.
                (PERSON8) But we can again that that's can be discussed in a separate-
                (PERSON7) Yeah.
                (PERSON8) Conversation.
                (PERSON6) Yeah.
                (PERSON7) Okay.
                (PERSON8) I think yeah, that's all for for today.
                So-
                (PERSON7) Yeah.
                Great.
                (PERSON8) See you next week.
                (PERSON7) Yeah.
                Great. 
                Thank you.
                (PERSON6) Thank you.
                Thank you.
                (PERSON8) Bye.
                (PERSON6) <unintelligible/> 
                Buh-bye. 
                (PERSON4) Bye.
                (PERSON9) Bye.
                Bye guys. 
                (PERSON6) Bye bye [PERSON9] <laugh/>.
                """
    
    start_time = perf_counter()
    minutes = generate_meeting_minutes(meeting_text,'index.json')
    print(minutes)
    print(f"Time taken: {perf_counter() - start_time:.2f} seconds")
