import datetime
import json
import sys
import time
import openai
import requests
import os
if os.path.dirname(os.path.realpath(__file__)) not in sys.path:
    sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import keys
from Auxiliary.verbose_checkpoint import verbose
from amadeus import Client, ResponseError
import getParametersJson


# Function to check time difference between flights
def check_time_between_flights(itineraries, buffer):
    for itinerary in itineraries:
        segments = itinerary.get('segments', [])
        for i in range(len(segments) - 1):
            current_arrival_time = datetime.datetime.fromisoformat(segments[i]['arrival']['at'])
            next_departure_time = datetime.datetime.fromisoformat(segments[i + 1]['departure']['at'])
            time_difference = next_departure_time - current_arrival_time
            hours_difference = time_difference.total_seconds() / 3600

            if hours_difference > (2.5+buffer) or hours_difference < 1.7:
                return True

    return False

def check_number_of_stops(itineraries, numberOfStops):
    for itinerary in itineraries:
        segments = itinerary.get('segments', [])
        if numberOfStops == 2:
            if len(segments) >= (numberOfStops+1):
                return True
        elif len(segments) == (numberOfStops+1):
            return True

    return False

def find_closest_flight_offer(flight_offers, extraTimeframes):
    if extraTimeframes == {}:
        return flight_offers
    
    closest_offers = []
    for offer in flight_offers:
        for index1, itinerary in enumerate(offer['itineraries']):
            departure_time = itinerary['segments'][0]['departure']['at']
            arrival_time = itinerary['segments'][-1]['arrival']['at']

            departure_time = datetime.datetime.fromisoformat(departure_time).time()
            arrival_time = datetime.datetime.fromisoformat(arrival_time).time()

            time_diff = 0
            if "exactDepartureTime" in extraTimeframes[index1] and extraTimeframes[index1]["exactDepartureTime"] != "":
                exactDepartureTime = datetime.datetime.strptime(extraTimeframes[index1]["exactDepartureTime"], '%H:%M:%S').time()
                time_diff += abs((departure_time.hour + departure_time.minute) - (exactDepartureTime.hour + exactDepartureTime.minute))
            if "exactArrivalTime" in extraTimeframes[index1] and extraTimeframes[index1]["exactArrivalTime"] != "":
                exactArrivalTime = datetime.datetime.strptime(extraTimeframes[index1]["exactArrivalTime"], '%H:%M:%S').time()
                time_diff += abs((arrival_time.hour + arrival_time.minute) - (exactArrivalTime.hour + exactArrivalTime.minute))

        closest_offers.append({"offer": offer, "time_difference": time_diff})

    return closest_offers

amadeus = Client(
    client_id=keys.amadeus_client_id,
    client_secret=keys.amadeus_client_secret,
    hostname='production'
)

def get_access_token(api_key=keys.amadeus_client_id, api_secret=keys.amadeus_client_secret):
    auth_url = 'https://api.amadeus.com/v1/security/oauth2/token'
    response = requests.post(auth_url, data={
        'grant_type': 'client_credentials',
        'client_id': api_key,
        'client_secret': api_secret
    })
    print("Access key:\n", response.json())
    return response.json().get('access_token')

endpoint = 'https://api.amadeus.com/v2/shopping/flight-offers'

def get_price_offer(access_token, flight_offers):
    url = 'https://api.amadeus.com/v1/shopping/flight-offers/pricing'
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/vnd.amadeus+json'
    }
    payload = {
        'data': {
            'type': 'flight-offers-pricing',
            'flightOffers': flight_offers
        }
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        print('Flight Offers price information retrieved successfully!')
        return response.json()  # Return the JSON response
    else:
        print(f'Failed to retrieve data: {response.status_code} - {response.text}')
        return None
    
def get_flight_offers(access_token, search_params, verbose_checkpoint=None):
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/vnd.amadeus+json'
    }
    try:
        response = requests.post(endpoint, headers=headers, data=json.dumps(search_params))
        #response = requests.get(endpoint, headers=headers, params=search_params)
        responseJson = response.json()
        if "errors" in responseJson:
            print(f"Error with fetching flight offers: {responseJson}")
            verbose(f"Error with fetching flight offers: {responseJson}", verbose_checkpoint)
            combined_detail = '\n'.join(error['detail'] for error in responseJson["errors"])
            return {"status": "error", "details": combined_detail}
        else:
            print(responseJson)
            print("initial flight offers length: ", len(responseJson["data"]))
            verbose(f"initial flight offers:\n{responseJson}", verbose_checkpoint)
            verbose(("initial flight offers length: ", len(responseJson["data"])), verbose_checkpoint)
            return {"status": "ok", "details": responseJson}
    except ResponseError as error:
        print(error)
        return responseJson
    
def getFlightOffer(flightDetails, verbose_checkpoint=None):
    #error, search_params = extractSearchParameters(flightDetails, 250, verbose_checkpoint)
    #if error:
    #    return {"status": "error", "data": None}
    search_params, extraTimeframes = getParametersJson.extractSearchParameters(flightDetails, 250)

    try:
        print(search_params)
        print(extraTimeframes)
        verbose(search_params, verbose_checkpoint)
        verbose(extraTimeframes, verbose_checkpoint)
        access_token = get_access_token()

        iteration = 0
        flightsFound = False
        while iteration < 10 and not flightsFound:
            flightOffers = get_flight_offers(access_token, search_params, verbose_checkpoint)
            if flightOffers["status"] == "error":
                print("error 1")
                verbose("error 1", verbose_checkpoint)
                return {"status": "error", "data": flightOffers["details"]}
            elif flightOffers["status"] == "ok":
                flightOffers = flightOffers["details"]["data"]
            time.sleep(0.5)

            if len(flightOffers) <= 0:
                print("No flights found.. expanding time window by 2 hours")
                verbose("No flights found.. expanding time window by 2 hours", verbose_checkpoint)
                if iteration == 0:
                    for originDestination in search_params['originDestinations']:
                        originDestination["departureDateTimeRange"]["timeWindow"] = "4H"
                else:        
                    for originDestination in search_params['originDestinations']:
                        current_time_window = originDestination['departureDateTimeRange']['timeWindow']
                        new_time_window = int(current_time_window[:-1]) + 2
                        new_time_window = f"{new_time_window}H"
                        originDestination['departureDateTimeRange']['timeWindow'] = new_time_window
                iteration += 1
            else:
                flightsFound = True

    except ResponseError as error:
        print("error 4")
        verbose("error 4", verbose_checkpoint)
        print(error)
        verbose(error, verbose_checkpoint)
        time.sleep(0.5)
        return {"status": "error", "data": "Unknown error occured"}
    #print(flightOffers)

    if len(flightOffers) <= 0:
        print("no flights")
        verbose("no flights", verbose_checkpoint)
        return {"status": "ok", "data": None}
    
    cheapestFlightOffers = []
    #check which offers qualify
    for flightOffer in flightOffers:
        # Access 'itineraries' within each flight offer
        timeframesSatisfied = True
        for index, itinerary in enumerate(flightOffer['itineraries']):
            departure_time = itinerary['segments'][0]['departure']['at']
            arrival_time = itinerary['segments'][-1]['arrival']['at']

            departure_time = datetime.datetime.fromisoformat(departure_time).time()
            arrival_time = datetime.datetime.fromisoformat(arrival_time).time()

            if "earliestDepartureTime" in extraTimeframes[index] and extraTimeframes[index]["earliestDepartureTime"] != "":
                if departure_time < datetime.datetime.strptime(extraTimeframes[index]["earliestDepartureTime"], '%H:%M:%S').time():
                    timeframesSatisfied = False
                    break
            if "latestDepartureTime" in extraTimeframes[index] and extraTimeframes[index]["latestDepartureTime"] != "":
                if departure_time > datetime.datetime.strptime(extraTimeframes[index]["latestDepartureTime"], '%H:%M:%S').time():
                    timeframesSatisfied = False
                    break
            if "earliestArrivalTime" in extraTimeframes[index] and extraTimeframes[index]["earliestArrivalTime"] != "":
                if arrival_time < datetime.datetime.strptime(extraTimeframes[index]["earliestArrivalTime"], '%H:%M:%S').time():
                    timeframesSatisfied = False
                    break
            if "latestArrivalTime" in extraTimeframes[index] and extraTimeframes[index]["latestArrivalTime"] != "":
                if arrival_time > datetime.datetime.strptime(extraTimeframes[index]["latestArrivalTime"], '%H:%M:%S').time():
                    timeframesSatisfied = False
                    break

        if timeframesSatisfied:
            cheapestFlightOffers.append(flightOffer)
            print("satisfied all timeframes")

    if len(cheapestFlightOffers) <= 0:
        cheapestFlightOffers = flightOffers
        print("no flight offers satisfied all timeframes.. using not optimal flights..")
        verbose("no flight offers satisfied all timeframes.. using not optimal flights..", verbose_checkpoint)

    flightOffers = cheapestFlightOffers
    
    bestFlightOffersPerStopNumber = []
    for numberOfStops in range(0, 3):
        toAppend = {"numberOfStops": numberOfStops, "offers": []}
        #oldCheapestFlightOffers = cheapestFlightOffers
        cheapestFlightOffers = []
        #if "nonStopPreferred" in search_params["searchCriteria"]["flightFilters"]["connectionRestriction"]:
        #    if search_params["searchCriteria"]["flightFilters"]["connectionRestriction"]["nonStopPreferred"] == "true":
        if numberOfStops <= 2:
            for flightOffer in flightOffers:
                if not check_number_of_stops(flightOffer["itineraries"], numberOfStops):
                    print("satisfies number of stops")
                    cheapestFlightOffers.append(flightOffer)

        if len(cheapestFlightOffers) <= 0:
            #cheapestFlightOffers = oldCheapestFlightOffers
            #print("no flight offers satisfied all stop number conditions.. using not optimal flights..")
            #verbose("no flight offers satisfied all stop number conditions.. using not optimal flights..", verbose_checkpoint)
            print(f"no flight offers satisfied all stop number conditions - number of stops: {numberOfStops}")
            verbose(f"no flight offers satisfied all stop number conditions - number of stops: {numberOfStops}", verbose_checkpoint)
            bestFlightOffersPerStopNumber.append(toAppend)
            continue

        print("cheapestFlightOffers:\n", str(len(cheapestFlightOffers)))
        verbose(("cheapestFlightOffers:\n" + str(len(cheapestFlightOffers))), verbose_checkpoint)
        #try:
        cheapestFlightOffers = find_closest_flight_offer(cheapestFlightOffers, extraTimeframes)
        #print("----------------")
        #print([offer["time_difference"] for offer in cheapestFlightOffers])
        #print("----------------")
        cheapestFlightOffers = sorted(cheapestFlightOffers, key=lambda x: (x["time_difference"], float(x["offer"]["price"]["total"])))
        #print("----------------")
        #print([float(offer["offer"]["price"]["total"]) for offer in cheapestFlightOffers])
        #print("----------------")
        cheapestFlightOffers = [offer["offer"] for offer in cheapestFlightOffers][:6]

        print("length 1:", len(cheapestFlightOffers))
        print("get price offers for:\n", cheapestFlightOffers)
        verbose(f"get price offers for:\n{cheapestFlightOffers}", verbose_checkpoint)
        try:
            price_offers = get_price_offer(access_token, cheapestFlightOffers)["data"]["flightOffers"]
        except:
            return {"status": "error", "data": "Error with getting final price offer"}
        cheapestPriceOffers = find_closest_flight_offer(price_offers, extraTimeframes)
        cheapestPriceOffers = sorted(cheapestPriceOffers, key=lambda x: (x["time_difference"], float(x["offer"]["price"]["grandTotal"])))
        cheapestPriceOffers = [offer["offer"] for offer in cheapestPriceOffers]
        cheapestPriceOffers = cheapestPriceOffers[:3]
        toAppend["offers"] = cheapestPriceOffers
        if numberOfStops == 3:
            toAppend["numberOfStops"] = "unlimited"
        bestFlightOffersPerStopNumber.append(toAppend)
        #print(f"cheapest flight price offers:\n{cheapestPriceOffers}")
    
    #print("-----------")
    #print(bestFlightOffersPerStopNumber)
    #print("-----------")
    cheapestPriceOffers = []
    for numberOfStops, offers in enumerate(bestFlightOffersPerStopNumber):
        offersList = offers["offers"]
        if numberOfStops == 0:
            if len(offersList) > 0:
                print("added offers from flights with number of stops: 0")
                for i in range(0, min(2, len(offersList))):
                    cheapestPriceOffers.append(offersList[i])
        elif (3-len(cheapestPriceOffers)) > 0 and len(offersList) > 0:
                print(f"added offers from flights with number of stops: {numberOfStops}")
                for i in range(min(len(offersList), 3-len(cheapestPriceOffers))):
                    cheapestPriceOffers.append(offersList[i])

    print("forward\n", cheapestPriceOffers)
    cheapestPriceOffers = find_closest_flight_offer(cheapestPriceOffers, extraTimeframes)
    cheapestPriceOffers = sorted(cheapestPriceOffers, key=lambda x: (x["time_difference"]))
    print("----------------")
    print([offer["time_difference"] for offer in cheapestPriceOffers])
    print("----------------")
    all_zero = all(offer["time_difference"] == 0 for offer in cheapestPriceOffers)
    cheapestPriceOffers = [offer["offer"] for offer in cheapestPriceOffers]
    if all_zero:
        offersBySegment = []
        for offer_ in cheapestPriceOffers:
            all_segments = []
            for itinerary in offer_.get("itineraries", []):
                all_segments.extend(itinerary.get("segments", []))
            total_segments = len(all_segments)
            toAppend = {"offer": offer_, "numberOfSegments": total_segments}
            offersBySegment.append(toAppend)

        cheapestPriceOffers = sorted(offersBySegment, key=lambda x: float(x["numberOfSegments"]))
        cheapestPriceOffers = [offer["offer"] for offer in cheapestPriceOffers]
    print(f"final offers:\n{cheapestPriceOffers}")

    returnData = {"status": "ok", "data": {"offers": []}}
    for cheapest_price_offer in cheapestPriceOffers:
        flights = []
        for iterary in cheapest_price_offer["itineraries"]:
            for segment in iterary["segments"]:
                flights.append({"departure": segment["departure"], "arrival": segment["arrival"], "duration": segment["duration"], "flightNumber": segment["number"], "carrierCode": segment["carrierCode"]})

        includedCheckBagsOnly = False
        if "includedCheckedBagsOnly" in cheapest_price_offer["pricingOptions"]:
            includedCheckBagsOnly = cheapest_price_offer["pricingOptions"]["includedCheckedBagsOnly"]

        includedCheckedBags = None
        if "includedCheckedBags" in cheapest_price_offer["travelerPricings"][0]["fareDetailsBySegment"][0]:
            includedCheckedBags = cheapest_price_offer["travelerPricings"][0]["fareDetailsBySegment"][0]["includedCheckedBags"]
        returnData["data"]["offers"].append({"price": {"grandTotal": cheapest_price_offer["price"]["grandTotal"], "billingCurrency": cheapest_price_offer["price"]["billingCurrency"]}, "luggage": {"includedCheckBagsOnly": includedCheckBagsOnly, "includedCheckedBags": includedCheckedBags}, "flights": flights})
    
    return returnData