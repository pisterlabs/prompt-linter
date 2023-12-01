#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
[env]
# Conda Environment
# NO CONDA ENV

conda create --name ai_chatgpt_prompts python=3.9.13
conda info --envs
source activate ai_chatgpt_prompts
conda deactivate
# if needed to remove
conda env remove -n [NAME_OF_THE_CONDA_ENVIRONMENT]


# update conda 
conda update -n base -c defaults conda

[filename]
006_project_1_python_documentation_default_summarize_chatgpt_api.py

[path]
cd /Users/brunoflaven/Documents/03_git/BlogArticlesExamples/ai_chatgpt_prompts/project_1_python_documentation_chatgpt_api/

[run]
python 006_project_1_python_documentation_default_summarize_chatgpt_api.py

+ GREAT EXAMPLE FROM OPENAI.COM
https://platform.openai.com/examples




"""

import os
import openai

# personal configuration
import config_values.values_conf as conf

OPENAI_ORGANIZATION = conf.OPENAI_ORGANIZATION
OPENAI_API_KEY = conf.OPENAI_API_KEY

# quick and dirty
openai.organization = OPENAI_ORGANIZATION
# PAID ONE DO NOT DISPLAY
openai.api_key = OPENAI_API_KEY



PROMPT_INPUT = 'To a certain point, self-assessment can turn to self-harassment. I guess, most of us have experiment the “evil” todolist syndrom e.g the 10 things to eat, the 3 books to read, the 5 places to visit, the 7 python librairies to use… todolists that torment us by controlling what we do and highlighting what we haven\'t!\n\nA modern practice that denotes a very contemporary obsession with performance. This obsession constantly accentuates our tendency to treat ourselves like a machine or like a company where it is only a question of productivity, efficiency and effectiveness!\n\nHowever, these 3 notions are eminently relative because they are personal. So, we can legitimately wonder if productivity is not a myth, multitasking is not useless or even the best productivity hack is almost always to do less!\n\nBeyond these simple thoughts, let’s say that workflow’s optimization or the industrialization of certain tasks has limits especially for a job of P.O.\n\nAs a P.O, I always want to delegate tedious and boring chores to a computer for instance this time I made up my mind about these 2 purposes(again a todolist): \nUse voice assistant to automatize some day-to-day P.O tasks such as creating new ticket, lauching tests… etc so I can do more while expending less effort.\nComplete my Python self-training with new skills on how to log and how to test some personal applications that I wrote in Python to facilate debugging and improve overall quality.\nTo be totally transparent enough, I wanted also to explore the python ability to generate automatically a PPT for sprint review based on a Jira ticket list! But making the script took more time than manually make the google doc based on a template, so I also dropped this subject! I have left few links towards ressources about this topic below.\nWell, I must admit that I failed either the subjects were too complex for me as all these goes far beyond my ability to make development or either the results were insignificant in regard of what I was expecting especially for voice assistant!\nWith the voice assistant, I was also worry by the fact that I would have to talk with my computer like some kind of crazy dude… I didn’t really like the experience and neither did people around me 🙂\nSo I pivoted. So, instead of investing on means(hard skills) e.g mastering python, I reinvested on method by revisiting some useful Kanban concepts(soft skills) just to know how to handle my frustration and refine my objectives.\nAnyway, despite the fact that I did not get rid of some P.O chores, this brief reading was a real occasion to take some good hacks from Kanban. As I hate cumbersome and convoluted explanations here are below some of my essential but partial takeaways.\nI found great summaries in Keyvan Akbary Learning Notes but reading the books itself is always a must… See https: // github.com/keyvanakbary/learning-notes.\nUnfinished tasks vie for our attention, causing intrusive thoughts that ultimately impede productivity and increase the opportunity for error.\nA P.O must be tech literate, not tech fluent in order to make good trade-off decisions. Programming is far from being essential, the only asset is that you start sometime to think like a developer or a tek guy to provide better explanation for a possible product feature.\nThe key for improving quality is to reduce the quantity of work-in -progress. Shorter iterations will drive higher quality. So, mind the WIP and think lead time. I rename “lead time” as “human sizeable”. Keynes used to say “In the long run, we are all dead”, so the delay between from starting to done as to be reasonable for a human being especially in software development.\nFrequent releases build trust. The product is the message so reducing WIP shortens lead time and frequent releases build trust and confidence within the team and with external teams. Trust is event driven, frequent gestures or events enhance trust more than larger gestures made only occasionally. Small gestures often cost nothing but build more trust than large, expensive gestures bestowed occasionally. This “small gestures” are also strongly advised in Corporate Hacking and the smallest action must have the highest emotional impact on teams.\nWell anyway, I end up pitifully this exploration in Python and give away the code exactly where I stopped. It was urgent to stop what turned out evidently as an evil never-ending WIP.\nYou can grab the source for this post and some other resource on my GitHub account: Code for python_explorations_audio_log_unit_test\n1. Voice assistant or speak with Python\nWell, I know now that voice over computer is working and it is promising! Good for me. But, I am definitely reluctant to talk to my computer and become a noisy bird for colleagues and family so I have decided to remain silent. Either my accent sucks in English but many orders given such as launch a script or open a browser were seldom successful attempts! Instead of saving me time, I was pissed off and wasted time trying to get my personal voice assistant as obtuse as a mule to work.\nYou can grab the source for this post and some other resource on my GitHub account: Code for speech_to_text\n\n2. First steps with unit testing in Python\nNo doubt that Unit testing belongs to the developer’s teams. As a PO, you should have a eye on this KPI, unit testing coverage percentage. This coding is made during all the product life-cycle, if not you are creating technical debt, my friend!\n\nUnit testing makes it easy to fix the problems as the developers come to know which particular component of the system or software has the issues and the developers can fix that particular unit.\n\nYou can grab the source for this post and some other resource on my GitHub account: Code for unit_testing\n\n3. Using log in Python\n Well, like anyone who start to develop, you are usually using a print command to figure out where is the problem or bug. Print debugging is deceptively quick and simple. But it often requires multiple iterations of rerunning the program before you display the information you need to know to fix your bug.\n\nThat is the reason why, I have decided to discover log in Python even so it is like “swatting flies with a sledge hammer” in my case, more commonly said overkill.\n\nUsing debugger or set up logfiles for your script might seem slower than simply inserting a print() call, but it saves you time in the long run according to all developpers.\n\nYou can grab the source for this post and some other resource on my GitHub account: Code for using_log_python\n\nConclusion: \n\nOne for all, I certainly confuse several concepts such as productivity, automation or continuous improvement! All this with the childish desire to do the minimum! Well, I guess I am closer to a kaizen mindset where you improve on daily basis the way you work than the “throw-it-all-and -start-fresh” method. Never to late to learn. I will also think twice about this Peter Drucker’s sentence that I like: "There is surely nothing quite so useless as doing with great efficiency what should not be done at all."'

response = openai.Completion.create(
    model="text-davinci-003",
    prompt="Summarize this for a second-grade student:\n\n"+PROMPT_INPUT+"",
    temperature=0.7,
    max_tokens=64,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
)

print('\n --- RESULT')
# print(response)
print(response.choices[0].text)
