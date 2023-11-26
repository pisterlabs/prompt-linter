"""
    Write a targeted 1 short sentence long advertisement knowing the following information about the person:
    {gender}, {age} years old, who is currently feeling {emotion}.
    You should keep in mind that our target is a person taking a {flight_duration} flight, has {time_before_departure}
    left before departure, and flies with {airline_company} so keep it in mind to target the pricing accordingly.
    Capture their attention and emphasize how this {product} knowing that the meteo in the city the person is currently in is {weather}.
    Use this json file to decode the weather context but don't show anything in the ad: {json_context}.
    The output should exclude any personal information about the person and should adress the target personally,
    (speaking to him like a friend), and the him why he should be interested to the ad.
    NEVER USE WORD "neutral" in the ad.
    """