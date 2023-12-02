import json

import pymongo
import openai



INDUSTRIES = [
    "qa",
    "customer success",
    "teaching",
    "student",
    "sales",
    "it",
    "engineering",
    "data science",
    "law",
    "finance",
    "research",
    "project manager",
    "marketing",
    "entrepreneur",
    "product management",
    "human resources",
    "analyst",
    "investments",
    "military"
]


messages = [ {"role": "system", "content": "please analyze the folowing json the user will provide and response with the city the user is most likly living in, please use no words other then the city name"} ]





excel_connection_string = "mongodb+srv://excel:Data2023@cluster0.nhaxlrp.mongodb.net/?retryWrites=true&w=majority"
pingo_connection_string = "mongodb+srv://barbot:barbot@barbot.ury12.mongodb.net/barbot?retryWrites=true&w=majority"
pingo_url = "https://pingoapp.net/api/contacts/createContacts"


def main():
    excel_client = pymongo.MongoClient(excel_connection_string, tls=True, tlsAllowInvalidCertificates=True)
    db = excel_client["excel"]
    profiles_collection = db["profile"]
    profiles = profiles_collection.find({"salesforceListed": True})
    for profile in profiles:
        number = profile.get("mobileNumber", None)
        location = profile.get("location", None)
        jobs = profile.get("jobs", None)
        print(number)
        print(location)
        _id = profile.get("_id")

        payload = {}
        
        payload["location"] = location
        # payload["jobs"] = jobs
        
        message = "User : \n" + json.dumps(payload)
        
        messages.append(
            {"role": "system", "content": "usually people live in israel if you have a doubt, prefer israel over other countries, for instance if someone seems to work at two jobs at the same time and one is resported abroad and the other is in israel he is most likly to be in isreal, also please prefer the city over the country, israel is not a city"},
        )

        messages.append(
            {"role": "system", "content": "sometimes tel aviv is writen with dashes or has the word yafo next to it, please ignore those words and just use tel aviv"},
        )

        messages.append(
            {"role": "system", "content": "please also never include the word israel in the name of the city, its just the country name"},
        )

        messages.append(
            {"role": "system", "content": "only reply with the name of the city, no other words"},
        )
        
        messages.append(
            {"role": "user", "content": message},            
        )
        

        openai.api_key = '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'

        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k", messages=messages
            # model="gpt-4", messages=messages
        )
        
        reply = chat.choices[0].message.content
        print(f"ChatGPT: {reply}")
        city = reply.lower()



        
        # todo decide what is the right state

        profiles_collection.update_one({"_id": _id}, {"$set": {"tags.city": city}})
        
    


if __name__ == '__main__':
    main()
