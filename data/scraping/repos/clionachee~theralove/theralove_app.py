import cohere
from cohere.classify import Example

co = cohere.Client("{apiKey}")
response = co.classify(
    model="large",
    inputs=[
        "I am depressed, what therapy suits me?",
        "I am anxious, what therapy suits me the most?",
        "I have difficulty concentrating, what therapy suits me the most?",
        "I have difficulties in communicating, what therapy suits me the most?",
        "I just feel bad in general, what therapy suits me the most?",
        "I want to kill myself, what therapy suits me the most?",
        "I like music and yoga, what therapy suits me the most?",
        "I like healing alone, what therapy suits me the most?",
        "I want to talk to people, what therapy suits me the most?",
        "I have traumas from being bullied, what therapy suits me the most?",
        "I have family issues, what therapy suits me the most?",
        "I have a relationship problem, what therapy suits me the most?",
        "I am extroverted but now I don't want to talk to my friends, what therapy suits me the most?",
        "I suspect that I have Bipolar Disorder, what therapy suits me the most?",
    ],
    examples=[
        Example("Appetite change", "Dialectical behavior therapy"),
        Example("Average across symptoms", "Dialectical behavior therapy"),
        Example("Depressed mood", "Cognitive-behavioral therapy"),
        Example("Difficulty concentrating", "Cognitive-behavioral therapy"),
        Example("Loss of interest", "Cognitive-behavioral therapy"),
        Example("Low energy", "Cognitive-behavioral therapy"),
        Example("Low self-esteem", "Hypnotherapy"),
        Example(
            "Psychomotor agitation (Anxiously moving restlessly)",
            "Cognitive-behavioral therapy",
        ),
        Example("Sleep problems", "Cognitive-behavioral therapy"),
        Example(
            "Suicidal ideation ( I would like to kill or harm myself)",
            "Emergency Lifeline",
        ),
        Example("Violence ( I would harm myself or the others )", "Emergency Lifeline"),
        Example("Cognitive/emotional immaturity", "Cognitive-behavioral therapy"),
        Example("Difficulties in communicating", "Art therapy"),
        Example("Difficulties in communicating", "Music therapy"),
        Example(
            "Medical illness, substance use",
            "Physical healthcare, fitness, yoga as complimentary therapy",
        ),
        Example("Loneliness, bereavement", "Dialectical behavior therapy"),
        Example("Loneliness, bereavement", "Group therapy"),
        Example("Exposure to violence/abuse", "Emergency Lifeline"),
        Example("Exposure to violence/abuse", "Art therapy"),
        Example("Low income & poverty", "Group therapy"),
        Example("Low income & poverty", "Hypnotherapy"),
        Example("Low income & poverty", "Sponsored AI Mental Health Assistant"),
        Example("Loss of interest", "Sponsored AI Mental Health Assistant"),
        Example(
            "Anxious", "Physical healthcare, fitness, yoga as complimentary therapy"
        ),
        Example(
            "Auditory sensitivity (easily agitated by sound)",
            "Reduce noise in the environment",
        ),
        Example("Anxious", "Reduce noise in the environment"),
        Example("Calm when listening to music", "Music therapy"),
        Example("Lives in easy and feels uneasy to relax", "Chromotherapy"),
        Example("With VR Headset", "Chromotherapy"),
        Example(
            "Difficulty concentrating",
            "Physical healthcare, fitness, yoga as complimentary therapy",
        ),
        Example("Hope for alone therapy", "Chromotherapy"),
        Example("Hope for alone therapy", "Sponsored AI Mental Health Assistant"),
        Example("Hope for alone therapy", "Aromatherapy"),
        Example("Hope to relax without anxiety", "Aromatherapy"),
        Example("Social & gender inequalities", "Dialectical behavior therapy"),
        Example("Exposure to war or disaster", "Hypnotherapy"),
        Example("Exposure to war or disaster", "Dialectical behavior therapy"),
        Example("Exposure to war or disaster", "Emergency Lifeline"),
        Example("Severe depression", "Cognitive-behavioral therapy"),
        Example("Severe depression", "Emergency Lifeline"),
        Example("Neglect, family conflict", "Dialectical behavior therapy"),
        Example(
            "Want to talk to people but nobody talks to me",
            "Dialectical behavior therapy",
        ),
        Example("Want to talk to people but nobody talks to me", "Emergency Lifeline"),
        Example(
            "I like yoga but could not have time to exercise",
            "Physical healthcare, fitness, yoga as complimentary therapy",
        ),
        Example(
            "I want to talk to people for urgent needs", "Dialectical behavior therapy"
        ),
        Example("Trauma", "Dialectical behavior therapy"),
        Example("Trauma", "Hypnotherapy"),
        Example("Relationship problem", "Group therapy"),
        Example("Relationship problem", "Cognitive-behavioral therapy"),
        Example("Bipolar Disorder", "Dialectical behavior therapy"),
    ],
)
print("The confidence levels of the labels are: {}".format(response.classifications))
