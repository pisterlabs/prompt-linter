import json
import os
import cohere
import numpy as np

from dotenv import load_dotenv
load_dotenv()

Cohere_api_key = os.getenv("COHERE_API_KEY")
names = ["Anxiety", "Depression", "Schizophrenia", "Bipolar", "PTSD", "Eating", "Neurodevelopmental"]
descriptions = [
    "Anxiety disorders are characterised by excessive fear and worry and related behavioural disturbances. Symptoms are severe enough to result in significant distress or significant impairment in functioning. ",
    "During a depressive episode the person experiences depressed mood (feeling sad or irritable or empty) or a loss of pleasure or interest in activities for most of the day nearly every day for at least two weeks. Several other symptoms are also present which may include poor concentration or feelings of excessive guilt or low self-worth hopelessness about the future or thoughts about dying or suicide disrupted sleep or changes in appetite or weight and feeling especially tired or low in energy. ",
    "Schizophrenia is characterised by significant impairments in perception and changes in behaviour. Symptoms may include persistent delusions or hallucinations or disorganised thinking or highly disorganised behaviour or extreme agitation. People with schizophrenia may experience persistent difficulties with their cognitive functioning. Yet a range of effective treatment options exist including medication or psychoeducation or family interventions and psychosocial rehabilitation.",
    "People with bipolar disorder experience alternating depressive episodes with periods of manic symptoms.  During a depressive episode the person experiences depressed mood (feeling sad or irritable or empty) or a loss of pleasure or interest in activities for most of the day nearly every day.  Manic symptoms may include euphoria or irritability increased activity or energy and other symptoms such as increased talkativeness or racing thoughts or increased self-esteem or decreased need for sleep or distractibility and impulsive reckless behaviour."
    "PTSD may develop following exposure to an extremely threatening or horrific event or series of events.  Re-experiencing the traumatic event or events in the present (intrusive memories or flashbacks or nightmares); avoidance of thoughts and memories of the event or avoidance of activities or situations or people reminiscent of the event and  persistent perceptions of heightened current threat. A common cause of PTSD is war."
    "Disruptive behaviour and dissocial disorders are characterised by persistent behaviour problems such as persistently defiant or disobedient to behaviours that persistently violate the basic rights of others or major age-appropriate societal norms or rules or laws. Onset of disruptive and dissocial disorders is commonly though not always during childhood. Another common symptom is not eating or starving yourself because you don't like how you look."
    "Neurodevelopmental disorders include ADHD and ASD. ADHD is characterised by a persistent pattern of inattention and/or hyperactivity-impulsivity that has a direct negative impact on academic occupational or social functioning.  Disorders of intellectual development are characterised by significant limitations in intellectual functioning and adaptive behaviour which refers to difficulties with everyday conceptual or social and practical skills that are performed in daily life. Autism spectrum disorder (ASD) constitutes a diverse group of conditions characterised by some degree of difficulty with social communication and reciprocal social interaction as well as persistent restricted repetitive and inflexible patterns of behaviour interests or activities."
]


co = cohere.Client(Cohere_api_key)
response = co.embed(texts=descriptions)
embeddings = np.array(response.embeddings)

with open("embeddings.npy", "wb") as f:
    np.save(f, embeddings)

with open("descriptions.json", "w") as f:
    json.dump(descriptions, f)

with open("names.json", "w") as f:
    json.dump(names, f)