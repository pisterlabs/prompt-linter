import dotenv
import cohere
import cohere.classify
import os
import numpy as np

def analyze_sentiment(prompt: str):
    print(os.getenv("COHERE_KEY"))
    sentiment_decode = ['scary', 'furious', 'sad', 'joyful']

    # set up cohere client
    coClient = cohere.Client(f'{ os.getenv("COHERE_KEY") }')

    # splits paragraph into arr of sentences
    if prompt[-1] == '.':
        prompt = prompt[:-1]

    sentences = prompt.split('.')
    sentence_moods = []
    for index in range(len(sentences)):

        response = coClient.classify(
            # specifying the model type
            model = 'medium',

            # input to feed to cohere sentence mood classifier
            inputs = [f'{ sentences[index] }'],

            # samples to classify into desired moods [scary, furious, sad, joyful]
            examples=[
                cohere.classify.Example('It was horrible, there were ghosts everywhere!', 'scary'),
                cohere.classify.Example('The warfare was brutal, they wouldn\'t even spare the women nor the children', 'scary'),
                cohere.classify.Example('The officials were dumbfounded by the fact that there was a coalition army against them', 'scary'),
                cohere.classify.Example('The men and women were shocked by the horrifying discovery.', 'scary'),
                cohere.classify.Example('A ghost appeared in front of two men, shocking them to death', 'scary'),
                cohere.classify.Example('Jerry was frightned by the sound in his empty house.', 'scary'),
                cohere.classify.Example('The grim reaper visited people who were near death.', 'scary'),
                cohere.classify.Example('Out of the dark, wolves appeared approaching their prey.', 'scary'),
                cohere.classify.Example('The fire went out, leaving a family to suffer in the cold.', 'scary'),
                cohere.classify.Example('A dangerous serial killer entered the elevator.', 'scary'),
                cohere.classify.Example('Spiders crawled up the mans skin, poisining him slowly.', 'scary'),
                 
                cohere.classify.Example('The enemy faction began to invade our territory, the event was outrageous.', 'furious'),
                cohere.classify.Example('After the crimes committed, the state decided to conscript every able person.', 'furious'),
                cohere.classify.Example('The King was angered by the probability of facing complete annihilation by the enemy.', 'furious'),
                cohere.classify.Example('The person began punching and kicking in frustration over their undesired position.', 'furious'),
                cohere.classify.Example('This code has frustrated the developer for hours now because he could not figure out the issue.', 'furious'),
                cohere.classify.Example('Jerry was angry and jealous because he lost a competition.', 'furious'),
                cohere.classify.Example('The dog showed his teeth to the home invaders in order to protect his family.', 'furious'),
                cohere.classify.Example('The storm was strong and it devoured everything in its path.', 'furious'),
                cohere.classify.Example('The animals fought back to protect their home from the predators.', 'furious'),
                cohere.classify.Example('John threw his phone at the TV because he got fired from his job.', 'furious'),

                cohere.classify.Example('The entire infantry was battered, leaving a sole survivor.', 'sad'),
                cohere.classify.Example('The late army apologized as the families that were lost could not be recovered.', 'sad'),
                cohere.classify.Example('Everyone was saddened to see the old King go.', 'sad'),
                cohere.classify.Example('Today, they attended the grave of a long lost friend and couldn\'t help but cry.', 'sad'),
                cohere.classify.Example('Jerry lost his expensive phone by dropping it in the sewer.', 'sad'),
                cohere.classify.Example('He had not experienced more grief in his life than in that sad moment.', 'sad'),
                cohere.classify.Example('She cried because her mother died.', 'sad'),
                cohere.classify.Example('The lion starved after months of trying to survive by eating dried grass', 'sad'),
                cohere.classify.Example('The herbivore was cornered by the ugly little beasts', 'sad'),
                cohere.classify.Example('The animals and plants were hurt by the radioactive fallout.', 'sad'),


                cohere.classify.Example('At long last, the army had conquered the enemy state.', 'joyful'),
                cohere.classify.Example('The family was relieved to see that she returned home safely.', 'joyful'),
                cohere.classify.Example('They were overjoyed when the student made it to their desired university!', 'joyful'),
                cohere.classify.Example('They felt great as they finally receieved a break from working tirelessly.', 'joyful'),
                cohere.classify.Example('Robb grinned and looked up from the bundle in his arms.', 'joyful'),
                cohere.classify.Example('Excitement overwhelmed them when they receieved recognition for their work.', 'joyful'),
                cohere.classify.Example('Jerry saved the day, donating 50% of his savings to the poor.', 'joyful'),
                cohere.classify.Example('The rain disappeared, and it left a rainbow in its wake.', 'joyful'),
                cohere.classify.Example('She won the lottery, so she jumped with joy.', 'joyful'),
                cohere.classify.Example(' The earth was saved by all the police and firemen, present in society today.', 'joyful'),
                cohere.classify.Example('She recieved a passing grade on her exam, completing her final university course. ', 'joyful'),
                cohere.classify.Example('Ella smiles and accepts her new life as a goddess.', 'joyful'),

            ]
        )   

        # encoding mood as an probability distribution
        mood_encoding = [
            response.classifications[0].labels['scary'].confidence,
            response.classifications[0].labels['furious'].confidence,
            response.classifications[0].labels['sad'].confidence,
            response.classifications[0].labels['joyful'].confidence,
        ]

        # sentence paired with sentiment for each sentence 
        sentences[index] = sentences[index].strip()
        sentence_moods.append(int(np.argmax(mood_encoding)))

    # get most frequent element
    sentence_moods.sort()
    counter = 0
    overall_mood_index = sentence_moods[0]
     
    # gets the most frequent appearing mood and sets as overall mood
    for i in sentence_moods:
        curr_frequency = sentence_moods.count(i)
        if(curr_frequency> counter):
            counter = curr_frequency
            overall_mood_index = i
    
    return [sentences, sentiment_decode[int(overall_mood_index)]]


