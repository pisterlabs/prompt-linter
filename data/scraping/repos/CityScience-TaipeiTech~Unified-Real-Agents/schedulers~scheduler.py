import openai, re

def parse_schedule(schedule):
    event_list = []

    pattern = re.compile(r'<\[([^@]*?)\s@([^f]*?)f/\s(\d+)\s-\s(\d+)\svia\s([^\]]*)\]>')

    for match in pattern.findall(schedule):
        event_list.append({
            'Activity': match[0].strip(),
            'Place': match[1].strip(),
            'Start time': match[2].strip(),
            'End time': match[3].strip(),
            'Transportation': match[4].strip(),
        })

    return event_list

def scheduler(person):
    textQuery = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=[
            {"role" : "system", "content" : "Follow all instructions."},
            {"role" : "system", "content" : "These are the buildings that you have available. You can not deviate from these. {'', 'greenhouse', 'terrace', 'university', 'yes', 'hotel', 'garage', 'warehouse', 'house', 'supermarket', 'public', 'kindergarten', 'industrial', 'college', 'roof', 'carport', 'static_railcar', 'bridge', 'dormitory', 'shed', 'residential', 'parking', 'train_station', 'grandstand', 'civic', 'retail', 'church', 'garages', 'temple', 'boathouse', 'detached', 'government', 'fire_station', 'construction', 'commercial', 'semidetached_house', 'hospital', 'sports_centre', 'kiosk', 'toilets', 'sports_hall', 'water_works', 'no', 'office', 'apartments', 'canopy', 'service', 'school'}"},
            {"role" : "system", "content" : """
            Describe what this person's schedule might look like.
            It should be based on events. 
            You must follow this format for each event <[event_name] @ [event_location] f/ [startTime] - [endTime] via [transportation]> for it to be parsed correctly. 
            You can add multiple events with a new line.
            Format starts and ends in military time without colons.
            IE: 7:20 AM is 720, and 2:00 PM is 1400 Be incredibly descriptive. Do not add anything not encapsulated in <>.

            Example:
            <[Wake up and freshen up, starting the day with a refreshing shower and grooming @ house f/ 530 - 600 via Walk]>
            <[Stretching and warm-up exercises before the jog, to prepare the body for the workout @ terrace f/ 600 - 610 via Walk]>
            <[Morning jog for exercise on the terrace, following doctor's advice to improve heart health @ terrace f/ 610 - 700 via Walk]>
            <[Cool down and relaxation exercises after the jog, to bring the heart rate back to normal @ terrace f/ 700 - 710 via Walk]>
            <[Morning Prayers at home in the prayer room, a daily ritual for inner peace and reflection @ house f/ 710 - 730 via Walk]>
            <[Reading the morning newspaper, staying updated with the world @ house f/ 730 - 740 via Walk]>
            <[Preparing breakfast in the kitchen, cooking a healthy and delicious meal @ house f/ 740 - 750 via Walk]>
            <[Breakfast at home in the dining room, a hearty meal of eggs, toast, and coffee to start the day right @ house f/ 750 - 810 via Walk]>
            <[Cleaning up after breakfast, maintaining a clean and tidy kitchen @ house f/ 810 - 820 via Walk]>
            <[Getting ready for work, dressing up in professional attire @ house f/ 820 - 830 via Walk]>
            <[Drive to Work in the city, the daily commute in the car to the bustling office @ commercial f/ 830 - 900 via Car]>
            <[Work at office in the personal cabin, a productive day of meetings, calls, and emails in the corporate world @ commercial f/ 900 - 1030 via Walk]>
            <[Coffee break at the office, a quick caffeine boost to keep the energy levels up @ commercial f/ 1030 - 1040 via Walk]>
            <[Team meeting at the office, discussing projects and strategies @ commercial f/ 1040 - 1130 via Walk]>
            <[Work at office, continuing the productive day @ commercial f/ 1130 - 1300 via Walk]>
            <[Lunchtime at the local Italian restaurant, a quick and tasty bite of pasta and salad @ retail f/ 1300 - 1400 via Walk]>
            <[Work at the office in the personal cabin, the afternoon hustle of reports, presentations, and client interactions @ commercial f/ 1400 - 1530 via Walk]>
            <[Afternoon coffee break at the office, another round of caffeine to maintain productivity @ commercial f/ 1530 - 1540 via Walk]>
            <[Work at the office, wrapping up the day's tasks @ commercial f/ 1540 - 1800 via Walk]>
            <[Drive Home in the car, the return journey through the city to the cozy haven @ house f/ 1800 - 1830 via Car]>
            <[Evening Prayers at home in the prayer room, a serene moment of spirituality and gratitude @ house f/ 1830 - 1900 via Walk]>
            <[Evening snack at home, a light and healthy snack to keep the hunger at bay @ house f/ 1900 - 1930 via Walk]>
            <[Dinner at home in the dining room, a delightful family mealtime with homemade dishes @ house f/ 1930 - 2030 via Walk]>
            <[Cleaning up after dinner, washing dishes and tidying up the dining area @ house f/ 2030 - 2040 via Walk]>
            <[Family time at home in the living room, bonding and making memories over board games and conversations @ house f/ 2040 - 2200 via Walk]>
            <[Reading time at home, indulging in a good book for personal growth @ house f/ 2200 - 2230 via Walk]>
            <[Night Prayers at home in the prayer room, a peaceful evening ritual for a good night's sleep @ house f/ 2230 - 2300 via Walk]>
            <[Night skincare routine at home, taking care of skin before bed @ house f/ 2300 - 2330 via Walk]>
            <[Meditation before sleep, calming the mind for a peaceful sleep @ house f/ 2330 - 2400 via Walk]>
            <[Bedtime at home in the bedroom, a well-deserved rest after a long day, reading a book before sleep @ house f/ 2400 - 530 via Walk]>
            
            Example 2

            <[Waking up and a few gentle stretches, easing into the day @ house f/ 900 - 915 via Stroll]>
<[Slow breakfast preparation in the kitchen, no rush, and no-fuss @ house f/ 915 - 945 via Stroll]>
<[Enjoying a leisurely breakfast, sipping coffee and taking your time @ house f/ 945 - 1015 via Stroll]>
<[Casual reading or watching TV, no specific agenda @ house f/ 1015 - 1130 via Stroll]>
<[Lunchtime, prepare a simple meal without any rush @ house f/ 1130 - 1200 via Stroll]>
<[Relaxing on the couch, maybe watching a favorite show @ house f/ 1200 - 1300 via Stroll]>
<[A short afternoon nap, a little siesta for relaxation @ house f/ 1300 - 1400 via Stroll]>
<[Slow stroll in the garden, take in the fresh air @ house f/ 1400 - 1430 via Stroll]>
<[Indulge in a hobby or creative project at your own pace @ house f/ 1430 - 1600 via Stroll]>
<[Dinner preparation, no rush, cook a comforting meal @ house f/ 1600 - 1700 via Stroll]>
<[Enjoying a relaxed dinner, savoring the flavors @ house f/ 1700 - 1800 via Stroll]>
<[Post-dinner relaxation, perhaps listening to music @ house f/ 1800 - 1900 via Stroll]>
<[Catching up on a favorite series or movie @ house f/ 1900 - 2100 via Stroll]>
<[A quiet evening, no agenda, just unwind @ house f/ 2100 - 2200 via Stroll]>
<[Late-night snack and a glass of wine @ house f/ 2200 - 2230 via Stroll]>
<[Reading or light journaling, no pressure, just thoughts @ house f/ 2230 - 2300 via Stroll]>
<[Getting ready for bed and a peaceful night's sleep @ house f/ 2300 - 900 via Stroll]>

Example 3

            <[Waking up and morning stretches, starting the day with a healthy dose of exercise @ house f/ 600 - 630 via Walk]>
<[Yoga and meditation for mental clarity and relaxation @ garden f/ 630 - 700 via Walk]>
<[Breakfast preparation in the kitchen, whipping up a nutritious smoothie @ house f/ 700 - 710 via Walk]>
<[Enjoying a delicious breakfast in the dining room, sipping on a fruit smoothie @ house f/ 710 - 730 via Walk]>
<[Clean up and tidy the kitchen, ensuring everything is in order @ house f/ 730 - 740 via Walk]>
<[Heading to the local library, spending time reading and exploring new books @ public library f/ 740 - 800 via Walk]>
<[Meeting a friend for a morning coffee, catching up and sharing stories @ cafe f/ 800 - 830 via Walk]>
<[Back to the library for more reading and research @ public library f/ 830 - 900 via Walk]>
<[Running errands, grocery shopping for the week's essentials @ supermarket f/ 900 - 930 via Walk]>
<[Returning home and unpacking groceries, organizing the kitchen @ house f/ 930 - 940 via Walk]>
<[Preparation for an online class, setting up the study area for learning @ house f/ 940 - 1000 via Walk]>
<[Participating in an online class, gaining new knowledge and skills @ house f/ 1000 - 1200 via Walk]>
<[Lunch break, a quick and healthy homemade meal @ house f/ 1200 - 1230 via Walk]>
<[Continuing the online class and absorbing information @ house f/ 1230 - 1400 via Walk]>
<[Taking a short break for some fresh air and a leisurely walk @ nearby park f/ 1400 - 1415 via Walk]>
<[Resuming the online class, focusing on the lesson at hand @ house f/ 1415 - 1500 via Walk]>
<[Completing the class and reviewing the material @ house f/ 1500 - 1530 via Walk]>
<[Unwinding with a cup of herbal tea and a good book, a moment of relaxation @ house f/ 1530 - 1600 via Walk]>
<[Engaging in a creative hobby, working on a painting project @ house f/ 1600 - 1700 via Walk]>
<[Dinner preparation in the kitchen, experimenting with a new recipe @ house f/ 1700 - 1745 via Walk]>
<[Enjoying a flavorful dinner in the dining room, savoring the culinary creation @ house f/ 1745 - 1830 via Walk]>
<[Cleaning up and tidying the kitchen after dinner @ house f/ 1830 - 1845 via Walk]>
<[Joining a virtual game night with friends, laughter and competition @ house f/ 1845 - 2100 via Walk]>
<[Wind down with a relaxing bath, soaking in tranquility @ house f/ 2100 - 2130 via Walk]>
<[Evening skincare routine, taking care of the skin's well-being @ house f/ 2130 - 2145 via Walk]>
<[Reflecting and journaling, capturing the day's thoughts and experiences @ house f/ 2145 - 2200 via Walk]>
<[Nighttime prayers for gratitude and peace @ house f/ 2200 - 2230 via Walk]>
<[Meditation and mindfulness practice, preparing for a restful night @ house f/ 2230 - 2300 via Walk]>
<[Heading to bed, ready for a rejuvenating night's sleep @ house f/ 2300 - 600 via Walk]>


            """},
            {"role": "user", "content": str(person) + "Make sure you factor in the demographic of the person; make their activities fit who they are. For example, if someone lives in a small town they might have a long drive, if they are unhealthy they might eat 4 meals, etc."}
        ],
    )
    
    data = textQuery["choices"][0]["message"]["content"]
    return parse_schedule(data)
