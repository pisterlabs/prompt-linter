import googlemaps
import cohere

# connecting to Google Maps and Cohere APIs
API_KEY = "AIzaSyBY4Qmdc6v9TWG3RAWLRRx4fZh8zu968Rg"
map_client = googlemaps.Client(key=API_KEY)
co = cohere.Client('Dbv19fLVO2RpG0HfwdaDIEQlHt2svAsebFq3WgG0', '2021-11-08')

# default src: https://www.google.com/maps/embed/v1/place?key=YOUR_API_KEY&PARAMETERS

# setting variables
location = (43.47322824035977, -80.53919130779408) # coordinates of Engineering 7 Building, UWaterloo
search_string = 'ice cream' # activity from user input
distance = 5000 # 5 km
business_list = []
max_top_search_results = 5
full_location_list = []


def generate_description(name, types, rating):
    '''
    Use Co:here to generate a description of the location of interest.
    '''
    prompt = f'''
Given a restaurant or location, keywords and its customer rating, this program will generate exciting location descriptions. Here are some examples:

Location: Dairy Queen
Keywords: ice cream, delicious, spacious, good customer-service
Rating: 4.5
Short Exciting Location Description: Tasty ice cream in a spacious restaurant with great customer service!
--
Location: Roy Thomson Hall
Keywords: music hub, amazing musicians, lively atmosphere
Rating: 4.8
Short Exciting Location Description: This world-reknown concert hall brings together audiences for an amazing musical experience featuring a diverse linup of artists each year.
--
Location: Sushi Inn Restaurant
Keywords: tasty, cheap, fast service
Rating: 4.2
Short Exciting Location Description: While you're in the area don't miss out on the best sushi in town at Sushi Inn!
--
Location: {name}
Keywords: {', '.join(types)}
Rating: {rating}
Short Exciting Location Description:'''

    response = co.generate(prompt, max_tokens=50, stop_sequences=['--'])
    description = response.generations[0].text
    #description.replace('\n', '')
    #description.replace('--', '')

    if '!' in description:
        exclamation_index = description.index('!')
        # remove all the characters of description after the exclamation index
        description = description[:exclamation_index+1]
        return description
    else:
        return description[:-3]

def generate_complete_location_recommendation(location_string):
    '''
    Takes the name of a location and returns a list of lists containing the name, address and description.
    '''
    response = map_client.places_nearby(location = location, keyword = location_string, radius = distance)
    business_list.extend(response.get('results'))
    one_location = []
    for i in range(0, max_top_search_results):
        one_location.append(business_list[i].get('name'))
        one_location.append(business_list[i].get('vicinity'))
        one_location.append(generate_description(business_list[i].get('name'), business_list[i].get('types'), business_list[i].get('rating')))
        one_location.append('https://www.google.com/maps/embed/v1/place?key='+str(API_KEY)+'&' + str(business_list[i].get('place_id')))
        full_location_list.append(one_location)
        one_location = []
    return full_location_list

# testing
'''
if __name__ == '__main__':
    print(generate_complete_location_recommendation("ice cream"))
'''