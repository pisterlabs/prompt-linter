from flask import Flask, redirect, url_for, request, render_template
import requests
import json
app = Flask(__name__)

@app.route('/reccomendations/<place>')
def reccomendations(place):
    api_key = "Your yelp api key"
    headers = {'Authorization' : 'Bearer {}'.format(api_key)}
    destination = place
    url = 'https://api.yelp.com/v3/businesses/search'

    params = {'term':'food', 'location':destination, 'limit':3, 'sort_by':'rating'}

    req = requests.get(url, params=params, headers=headers)

    parsed = json.loads(req.text)
    data = parsed['businesses'][0]['name'], parsed['businesses'][0]['rating'], parsed['businesses'][1]['name'], parsed['businesses'][1]['rating'], parsed['businesses'][2]['name'], parsed['businesses'][2]['rating']
    #send data to console log
    import openai
    openai.api_key = ("your openai api key")
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=("Give me a 3 day schedule for a trip to {} including {}, {} and {}".format(destination, data[0], data[2], data[4])),
        max_tokens=500,
        temperature=0,)
    itinerary = (str(response['choices'][0]['text']))
    return render_template('reccomendations.html', data=data, itinerary=itinerary)


# @app.route('/map/<place>/')
# def map(place):
#     return render_template('wheredoigo.html', place=place)

# @app.route('/schedule/<place>')
# def schedule(place):
#     api_key = "jvFKqmh_l2XRzDI1I9hXuGq0RntzZUhXvuk2lxs9gLFXPaC80uEDBlMyTsHoJYa6aOhDPQDEI8hNUy8COdJdB8gvFv8TIpTGy0teJ5q47SC6643BuQlXRB6_4BzMY3Yx"
#     headers = {'Authorization' : 'Bearer {}'.format(api_key)}
#     destination = place
#     url = 'https://api.yelp.com/v3/businesses/search'
#     params = {'term':'food', 'location':destination, 'limit':3, 'sort_by':'rating'}
#     req = requests.get(url, params=params, headers=headers)
#     parsed = json.loads(req.text)
#     data = parsed['businesses'][0]['name'], parsed['businesses'][0]['rating'], parsed['businesses'][1]['name'], parsed['businesses'][1]['rating'], parsed['businesses'][2]['name'], parsed['businesses'][2]['rating']    
    
#     import openai
#     openai.api_key = ("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")
#     response = openai.Completion.create(
#         model="text-davinci-003",
#         prompt=("Give me a 3 day schedule for a trip to {} including {}, {} and {}".format(destination, data[0], data[2], data[4])),
#         max_tokens=500,
#         temperature=0,)
#     itinerary = (str(response['choices'][0]['text']))
#     return render_template('schedule.html', itinerary=itinerary)


@app.route('/index',methods = ['POST', 'GET'])
def login():
   if request.method == 'POST':
      dest = request.form['dest']
      home = request.form['home']
      #send dest to other page
      return redirect(url_for('reccomendations',place = dest, place2 = home))


if __name__ == '__main__':
   app.run(debug = True)
