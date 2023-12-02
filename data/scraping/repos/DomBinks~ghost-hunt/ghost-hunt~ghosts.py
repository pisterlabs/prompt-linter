import math
import os
import openai
import random

ghost_types = [
    "Banshee",
    "Demon",
    "Deogen",
    "Goryo",
    "Hantu",
    "Jinn",
    "Mare",
    "Moroi",
    "Myling",
    "Obake",
    "Oni",
    "Onryo",
    "Phantom",
    "Poltergeist",
    "Raiju",
    "Revenant",
    "Shade",
    "Spirit",
    "Thaye",
    "Wraith",
    "Yokai",
    "Yurei"
]

first_names = [
    "Alex",
    "Amit",
    "Andrew",
    "Anthony",
    "Benjamin",
    "Billy",
    "Bradley",
    "Brenden",
    "Brian",
    "Carlos",
    "Charles",
    "Christopher",
    "Corey",
    "Daniel",
    "Dave",
    "David",
    "Donald",
    "Edward",
    "Eric",
    "Gary",
    "George",
    "Grant",
    "Gregory",
    "Harold",
    "Jack",
    "James",
    "Jan",
    "Jason",
    "Jay",
    "Jerry",
    "John",
    "Joseph",
    "Justin",
    "Keith",
    "Kenneth",
    "Kenny",
    "Kevin",
    "Kyle",
    "Larry",
    "Leslie",
    "Luke",
    "Mark",
    "Michael",
    "Paul",
    "Peter",
    "Raymond",
    "Richard",
    "Robert",
    "Ronald",
    "Russell",
    "Steven",
    "Thomas",
    "Ted",
    "Tim",
    "Walter",
    "William",
    "Ann",
    "April",
    "Barbara",
    "Becky",
    "Betty",
    "Carla",
    "Carol",
    "Catiana",
    "Cora",
    "Donna",
    "Doris",
    "Dorothy",
    "Edie",
    "Elizabeth",
    "Ella",
    "Ellen",
    "Emily",
    "Emma",
    "Eva",
    "Georgia",
    "Gloria",
    "Heather",
    "Helen",
    "Holly",
    "Jane",
    "Jennifer",
    "Jennise",
    "Jessica",
    "Jo",
    "Josefine",
    "Judy",
    "Julie",
    "Karen",
    "Kelly",
    "Kim",
    "Leslie",
    "Linda",
    "Lisa",
    "Livy",
    "Lori",
    "Lucy",
    "Marcia",
    "Margaret",
    "Maria",
    "Marie",
    "Mary",
    "Megan",
    "Michelle",
    "Nancy",
    "Nellie",
    "Patricia",
    "Robin",
    "Rose",
    "Ruth",
    "Sandra",
    "Sarah",
    "Shannon",
    "Sharne",
    "Shelly",
    "Sophie",
    "Stacey",
    "Susan",
    "Tricia"
]

last_names = [
    "Alexander",
    "Anderson",
    "Bailey",
    "Baker",
    "Barber",
    "Barton",
    "Bellfield",
    "Birch",
    "Bishop",
    "Bowen",
    "Brock",
    "Brooks",
    "Brown",
    "Buckley",
    "Carey",
    "Carter",
    "Clark",
    "Clarke",
    "Cordero",
    "Corrigan",
    "Davis",
    "Dexter",
    "Dixon",
    "Doe",
    "Douglas",
    "Dyer",
    "Elliott",
    "Emmett",
    "Everly",
    "Gacy",
    "Garcia",
    "Gayton",
    "Hans",
    "Harris",
    "Hill",
    "Holland",
    "Holmes",
    "Huntley",
    "Jackson",
    "Johnson",
    "Jones",
    "Kemper",
    "Knight",
    "Kraft",
    "Kray",
    "Lancaster",
    "Lavender",
    "Lee",
    "Lewis",
    "Manson",
    "Marsh",
    "Martin",
    "Martinez",
    "Maudsley",
    "Miller",
    "Mills",
    "Moore",
    "Myers",
    "Nilsen",
    "Norris",
    "Pettit",
    "Ramirez",
    "Rhoades",
    "Roberts",
    "Robinson",
    "Rook",
    "Roswell",
    "Schelin",
    "Shawcross",
    "Shipman",
    "Skinner",
    "Smith",
    "Stevens",
    "Straffen",
    "Sweeney",
    "Taylor",
    "Thomas",
    "Thompson",
    "Todd",
    "Walker",
    "Watts",
    "West",
    "White",
    "Williams",
    "Wilson",
    "Winter",
    "Wright",
    "Young"
]

occupations = [
    "accountant",
    "autioneer",
    "author",
    "attorney",
    "baker",
    "barber",
    "butcher",
    "carpenter",
    "chef",
    "farmer",
    "janitor",
    "landlord",
    "monk",
    "painter",
    "surgeon",
    "tailor",
    "waiter",
]

youngest_age = 18
oldest_age = 99

# Requires {name}, {last_name}, {age}, {occupation}
prompt = "Write ~100 word backstory about {name}, \
        who was {age} years old when they died, and was a {occupation}. \
        Now they are a vengeful ghost."

def getRandomFirstName():
    return first_names[random.randint(0,len(first_names)-1)]

def getRandomLastName():
    return last_names[random.randint(0,len(last_names)-1)]

def getLocation():
    return random.choice(['a','b','c'])

def getActivity():
    return math.floor((random.randint(1,10)+random.randint(1,10))/2)

def getRandomAge():
    return str(random.randint(youngest_age, oldest_age))

def getRandomOccupation():
    return occupations[random.randint(0,len(occupations)-1)]

def getType():
    return random.choice(ghost_types)

def getPortrait(first_name, last_name, age):
    pass
    try:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        response = openai.Image.create(
            prompt = f"{first_name} {last_name}, {age} years old, portrait in the style of Grant Wood"
        )
    except:
        return
        
def getBackstory(first_name, last_name, age, occupation):
    pass
    try:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt.format(
                name = first_name + " " + last_name,
                age = age,
                occupation = occupation
            ),
            temperature=0.6,
            stream=False,
            max_tokens=1024
        )

        return response.choices[0].text

    except:
        return