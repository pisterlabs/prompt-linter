import requests, datetime
import xmltodict
from geopy.geocoders import Nominatim
from idris.utils.openai import OpenAI
from idris.server import IdrisClient
from scipy import spatial

geoLoc = Nominatim(user_agent="GetLoc")

class IrishRail:

  route_map = {
    'Dublin Connolly - Maynooth': [
      'dublin connolly',
      'drumcondra',
      'broombridge',
      'pelletstown',
      'ashtown',
      'navan',
      'castleknock',
      'coolmine',
      'clonsilla',
      'leixlip',
      'maynooth'
    ],
    'Bray - Drogheda': [
      'bray',
      'dun laoghaire',
      'blackrock',
      'sydney parade',
      'lansdowne road',
      'grand canal dock',
      'pearse',
      'tara',
      'dublin connolly',
      'malahide',
      'donabate',
      'rush and lusk',
      'skerries',
      'balbriggan',
      'gormanston',
      'laytown',
      'drogheda'
    ],
    'Dublin Pearse - Dundalk': [
      'pearse',
      'tara',
      'dublin connolly',
      'donabate',
      'rush and lusk',
      'skerries',
      'balbriggan',
      'gormanston',
      'laytown',
      'drogheda',
      'dundalk'
    ],
    'Dublin Pearse - Drogheda': [
      'pearse',
      'tara',
      'dublin connolly',
      'malahide',
      'donabate',
      'rush and lusk',
      'skerries',
      'balbriggan',
      'gormanston',
      'laytown',
      'drogheda'
    ],
    'Balbriggan - Dublin Pearse': [
      'balbriggan',
      'skerries',
      'rush and lusk',
      'donabate',
      'malahide',
      'dublin connolly',
      'tara',
      'pearse'
    ],
    'Dublin Connolly - Balbriggan': [
      'dublin connolly',
      'malahide',
      'donabate',
      'rush and lusk',
      'skerries',
      'balbriggan'
    ],
    'Dublin Connolly - Drogheda': [
      'dublin connolly',
      'donabate',
      'rush and lusk',
      'skerries',
      'balbriggan',
      'laytown',
      'drogheda'
    ],
    'Dublin Connolly - Dundalk': [
      'dublin connolly',
      'malahide',
      'donabate',
      'rush and lusk',
      'skerries',
      'balbriggan',
      'gormanston',
      'laytown',
      'drogheda',
      'dundalk'
    ],
    'Drogheda - Dublin Connolly': [
      'drogheda',
      'laytown',
      'gormanston',
      'balbriggan',
      'skerries',
      'rush and lusk',
      'donabate',
      'malahide',
      'dublin connolly'
    ],
    'Drogheda - Dublin Pearse': [
      'drogheda',
      'balbriggan',
      'skerries',
      'dublin connolly',
      'tara',
      'pearse'
    ],
    'Dundalk - Dublin Connolly': [
      'dundalk',
      'drogheda',
      'laytown',
      'gormanston',
      'balbriggan',
      'skerries',
      'rush and lusk',
      'donabate',
      'malahide',
      'dublin connolly'
    ],
    'Dundalk - Dublin Pearse': [
      'dundalk',
      'drogheda',
      'laytown',
      'gormanston',
      'balbriggan',
      'skerries',
      'rush and lusk',
      'donabate',
      'malahide',
      'dublin connolly',
      'tara',
      'pearse'
    ],
    'Newry - Dublin Connolly': [
      'newry',
      'dundalk',
      'drogheda',
      'laytown',
      'balbriggan',
      'skerries',
      'rush and lusk',
      'donabate',
      'malahide',
      'dublin connolly'
    ],
  }

  def __init__(self):
    self.openai = OpenAI()
    self.server = IdrisClient()

  def get_all_stations(self):
    response = requests.get('https://api.irishrail.ie/realtime/realtime.asmx/getAllStationsXML')

    if response.ok:
      return xmltodict.parse(response.content)['ArrayOfObjStation']['objStation']

  def get_trains_to_station(self, station):
    current_location_arr = self.server.get_user_location().split(',')
    current_location = (float(current_location_arr[0]), float(current_location_arr[1]))

    print(current_location)

    all_stations = self.get_all_stations()

    stations_coords = [(station['StationLatitude'], station['StationLongitude']) for station in all_stations]

    tree = spatial.KDTree(stations_coords)
    closest_station_index = tree.query([current_location], k=1)[1][0]
    closest_station = all_stations[closest_station_index]

    destination_station = station.lower()

    response = requests.get(f'https://api.irishrail.ie/realtime/realtime.asmx/getStationDataByNameXML?StationDesc={closest_station["StationDesc"]}')
    if response.ok:
      stations = xmltodict.parse(response.content)['ArrayOfObjStationData']['objStationData']

    trains = []

    for train in stations:
      route_map_key = f"{train['Origin']} - {train['Destination']}"
      if route_map_key in IrishRail.route_map:
        route = IrishRail.route_map[route_map_key]
        if destination_station in route and route.index(closest_station['StationDesc'].lower()) < route.index(destination_station) and train['Status'] != 'En Route':
          trains.append(train)

    print(trains)
    return trains

  def get_next_train(self, query):
    misspelling_map = {
      'conley': 'dublin connolly',
      'connally': 'dublin connolly',
      'conway': 'dublin connolly',
      'connelly': 'dublin connolly',
      'connolly': 'dublin connolly',
    }

    station = query.split()[-1].lower()

    if station == 'dublin':
      station = 'dublin connolly'

    if station in misspelling_map:
      station = misspelling_map[station]

    trains = self.get_trains_to_station(station)

    data = ''

    if trains:
      for train in trains:
        data += f"- Departure: {train['Origintime']}, Destination: {station.title()}\n"
    else:
      data = 'No trains found.'
  
    prompt = f"Using the data, tell me when the next train I can catch is as well as how long until the departure. Fix the question first. If there are no trains, let me know.\nStructure the response like this:\nThe next train to [station] departs in [number] minutes.\nCurrent Time: {datetime.datetime.now().strftime('%H:%M')}\nData:\n{data}\n{query}? How long until departure?"

    print(prompt)

    completion = self.openai.create_completion(prompt)

    return completion.choices[0].text.strip()