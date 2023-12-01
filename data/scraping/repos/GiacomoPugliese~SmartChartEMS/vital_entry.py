import openai
import re
openai.api_key = "#"

def extract_vitals(string, compare_to):
    string = string.replace('output:', 'output')
    string = string.replace('Output:', 'output')
    string = string.replace('string:', 'string')

    regex = r'("[^"]*"|\'[^\']*\'|[\w\\/,\\-\\.]+?):\s*("[^"]*"|\'[^\']*\'|[\w\s\\/,\\-\\.]+?)(?=\s{2}|;|$)'
    matches = re.findall(regex, string)
    result = []
    for name, value in matches:
        # Remove all commas from the value variable
        value = value.replace(',', '')
        if name.lower() == "blood_pressure":
            try:
                systolic, diastolic = value.strip().split("/")
            except:
                systolic = "110"
                diastolic = "70"
            if systolic.strip().lower() in compare_to.lower() or diastolic.strip().lower() in compare_to.lower():
                result.append('{}:{}'.format(name.strip('"\''), value.strip()))
        elif value.strip().lower() in compare_to.lower():
            result.append('{}:{}'.format(name.strip('"\''), value.strip()))
        if name.lower() == "output":
            break
    return ';'.join(result)

def analyze_special(vitals_string):
    # Call the OpenAI GPT-3 engine to analyze the vitals string
    print(vitals_string)
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt='''You are going to receive an input string that contains information about emergency medical technician interaction with a patient. You will receive information about one or multiple of the following:
            arrival_time, depart_time, burn, bleeding, traumas, ems_interventions, allergy, history, medications, pain_quality, pain_rating, and pain_radiation. Note that the degree of the burn and if the bleeding is 
            controlled/uncontrolled must be specified. Further, for all times please use a dot in the time stamp instead of a colon. 
            
            Input string: Ems arrived at  12.30am, left at 1am, and the patient had a fracture, a first degree burn, and controlled bleeding. Ems gave the patient a bandaid. The patient had a medical history of diabetes and takes insulin, and described his pain as crushing.
            Output string: arrival_time:12.30am; depart_time:1am; trauamas:fracture; bleeding:controlled bleeding; burn:1st degree burn; history:diabetes; medications:insulin; pain_quality:crushing
            
            Note that all of your patient information must be in the form of <category>:<description>;  where the category is a category explicitly mentioned in the list above. Do not list any categories not in the above list.
            Furthermore, note that all pieces of informtation must be terminated with a semi colon. You MUST include what degree burn it is when you are including information about burns. Here's another example input and output:
            
            Input string: The patient had a 2nd degree burn and a hip dislocation. Ems arrived at 2pm and left at 3.30pm, and gave the patient an ice pack as well as kept him warm. The patient is allergic to bees, and takes tylenol. He says the pain radiates to his back and rates his pain a 9/10.
            Output string: burn:2nd degree burn; traumas:hip dislocation; arrival_time:2pm; depart_time:3.30pm; ems_interventions:ice pack and kept patient warm; allergy:bees; medications:tylenol; pain_radiation:back; pain_rating:9/10
           
            Here's one last input and output:
            Input: Tha patient is allergic to peanut butter, has a medical history of cancer, and has pain that radiates to his shoulder and feels like a stabbing pain. He has uncontrolled bleeding. 
            Outpu: allergy:peanut butter; history:cancer; pain_radiation:shoulder; pain_quality:stabbing; bleeding:uncontrolled bleeding

            To reiterate, this is how you should process times:
            Input: Ems arrived at 4.30pm and departed at 5pm
            Output: arrival_time:4.30pm; depart_time:5pm
            Here is your input string: '''  + vitals_string,
        max_tokens=1000,
        n=1,
        stop=None,
        temperature=0.5,
    )
    output = response.choices[0].text
    print(output)
    return extract_vitals(output, vitals_string)

def analyze_vitals(vitals_string):
    #replace common transcription errors
    replaced = vitals_string.lower()
    replaced = replaced.replace(":", ".")
    replaced = replaced.replace(" and", "")
    replaced = replaced.replace("rails", "rales")
    replaced = replaced.replace('urinary care', 'urticaria')
    replaced = replaced.replace('erotic area:', 'urticaria')
    replaced = replaced.replace('purl', 'pearl')
    replaced = replaced.replace('hack inside hospital', 'Hackensack Hospital')
    replaced = replaced.replace('hack side hospital', 'Hackensack Hospital')
    replaced = replaced.replace('bilateral', 'bilaterally')
    replaced = replaced.replace('first', '1st')
    replaced = replaced.replace('second', '2nd')
    replaced = replaced.replace('third', '3rd')
    replaced = replaced.replace(' out of ', '/')
    replaced = replaced.replace('splunting', 'splinting')
    replaced = replaced.replace(' over ', '/')

    # Call the OpenAI GPT-3 engine to analyze the vitals string
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt='''For the following string of vitals, ONLY return a modified string of vitals that has the vitals in their isolated form. DO NOT INCLUDE vitals not 
            explicitly written in the input. Note that the avpu vital has to do with any text concerning patient consciousness or patient responsiveness. For lung sounds please only report one word. 
            
            For example, an input of:
            
            "The patient's name is Bob Smith, who is a 60 years old male and lives at 123 main Street. His chief complaint is chest pain. He describes this pain as crushing. 
            He is alert, and his gcs is 15. The patient has a pulse of 60 beats per minute, their pupils are pinpoint, respiratory rate of 15 breaths per second, 
            and a blood pressure of 160 over 80. Oxygen saturation is 95% and the temperature is 98.6 degrees Fahrenheit. Provider names 
            are Moe and Christine, and patient was transported to Hackensack Hospital." 
            
            Should have an output of:

            "name:Bob Smith; age:60; gender:male; address:123 Main Street; chief_complaint: headache; opqrst:crushing; avpu:alert; glasgow_coma_scale:15; pupils:pinpoint; pulse_rate:60; respiratory_rate:17; blood_pressure:160/80; pulse_ox:95; temperature:98.6; provider_names:Moe and Christine; receiving_facility:Hackensack Hospital;". 
            
            Include no units in your response. Furthermore, your response MAY ONLY use the following vitals, although you probably will not use all of them: 
            name, age, address, gender chief complaint, avpu, glasgow_coma_scale, pulse_rate, respiratory_rate, blood_pressure, pulse_ox, temperature, skin_condition, pupils, 
            breath_sounds, provider_names, and receiving_facility. If any vital given in the input doesn't math one of these vitals DO NOT INCLUDE. 
            Also, if there is no relevant input for any of the vitals, DO NOT INCLUDE THAT VITAL IN YOUR OUTPUT. DO NOT INCLUDE any text not in the form of <vital_name>:<vital>. 
            You have failed the task if there is any text that is not in the form of <vital_name>:<vital_input>, or if you present any information not explicitly contained in the input.

            For example, if my input is only:
            "The patient's name is Sally Smith, she is female they have a glasgow coma scale score of 12, and they have a pulse of 100. Their skin has urticaria."

            The output should ONLY contain (with no additional text):
            "name:Sally Smith; gender:female; glasgow_coma_scale:12; pulse_rate:100; skin_condition:urticaria"

            Here is one last example:
            Input: "Ems arrived at 12:30am. The patient is a man who has a blood pressure of 130 over 80, and responds to pain. They have a pulse ox of 98, and breath sounds of rales."
            Output: "gender:male; arrival_time:12.30am; blood_pressure:130/80; avpu:pain; pulse_ox:98; breath_sounds:rales"

            Here is the input string: '''  + replaced,
        max_tokens=1000,
        n=1,
        stop=None,
        temperature=0.5,
    )
    output = response.choices[0].text
    additional = ''
    #special cases: 
    if('allerg' in replaced or 'history' in replaced or 'last' in replaced or 'intake' in replaced or 'ate' in replaced
        or 'quality' in replaced or 'radiate' in replaced or 'onset' in replaced or 'ago' in replaced or 'rates the' in replaced
        or 'ems' in replaced or 'burn' in replaced or 'fracture' in replaced or 'sprain' in replaced or 'strain' in replaced
        or 'disloc' in replaced or 'bleed' in replaced or 'am' in replaced or 'pm' in replaced) or 'describe' in replaced:
        additional = analyze_special(replaced)

    return extract_vitals(output, replaced) + ";" + additional

string = "Ems arrived at 12:30am"
out = ''' 
'''

print(analyze_vitals(string), string)