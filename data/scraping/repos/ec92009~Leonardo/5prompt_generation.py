import time
import openai
from dotenv import dotenv_values
import os


def oneAnswer(prompt):
    print(f'prompt = {prompt}')
    max_attempts = 5
    attempt = 0
    while attempt < max_attempts:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ]
            )
            retval = response["choices"][0]["message"]["content"]
            print(f'response = {retval}')
            return retval
        except openai.error.OpenAIError as e:
            print(f'Error: {e}')
            time.sleep(5)
            attempt += 1
            print(f'attempt = {attempt}')
            continue
    return ""


def get_subjects(N, theme, subjects):

    try:
        response = oneAnswer(
            f'a list of {N} subjects for stock images on the theme "{theme}", each subject and 10 associated keywords separated by comas. The lines of the list start with a line number followed by a period and a space. No introduction, no outtro.')
        # split response into lines
        print(f'response = {response}')

        new_subjects = response.splitlines()
        # remove the item number
        new_subjects = [subject.split(" ", 1)[1] for subject in new_subjects]

        # print(f'subject on the theme of {theme} = {new_subjects}')
        subjects.extend(new_subjects)
        return
    except openai.error.OpenAIError as e:
        print(f'Error: {e}')
        return
    except Exception as e:
        print(f'Error: {e}')
        return


def get_themes(N):
    response = oneAnswer(
        f"top {N} themes for stock photo requests, one theme per line")
    # split response into lines
    themes = response.splitlines()
    # remove the item number
    themes = [theme.split(" ", 1)[1] for theme in themes]

    print(f'-->themes = {themes}')

    subjects = []
    for theme in themes:
        get_subjects(N, theme, subjects)
        i = 0
        for subject in subjects:
            print(f'-->subject[{i}] = {subject}')
            i += 1

    return subjects


if __name__ == "__main__":
    config = dotenv_values(".env")
    openaiKey = config["OPENAI_KEY"]
    openai.api_key = openaiKey

    subjects = get_themes(10)
    print(f'-->subjects = {subjects}')


'''
['Beach Scenes - Sun, sand, ocean waves, palm trees, swimming, beach umbrellas, beach chairs, tropical drinks, boating, shore birds.', 'City Breaks - Skyscrapers, street signs, public transport, local attractions, shopping, dining, parks, historical landmarks, nightlife, street performers.', 'National Parks - Mountains, rivers and lakes, wildlife, hiking trails, campgrounds, scenic views, waterfalls, rock formations, forests, designated natural wonders.', 'Road Trips - Adventure, car rental, scenic highways, landscapes, gas stations, motels, pit-stops, route maps, vintage vehicles, driving safety.', 'Cultural Heritage - Museums, cathedrals, architecture, street scenes, art galleries, cultural events, religious landmarks, cuisine, folk traditions, local costumes.', 'Adventure Sports - Rafting, kayaking, hiking, biking, skiing, snowboarding, surfing, zip-lining, paragliding, extreme challenges.', 'Luxury Escapes - Resorts, mansions, spas, golf courses, private beaches, yacht clubs, gourmet dining, fine wines, indoor pools, lavish accommodations.', 'Family Fun - Amusement parks, zoos, theme parks, puppet shows, carnivals, picnics, theaters, aquariums, arcades, family-friendly attractions.', 'Backpacking - Hostels, budget travel, camping, backpacks, train journeys, cheap eats, alternative tours, selfie sticks, cultural immersion, travel hacks.', 'Wellness Retreats - Yoga, meditation, holistic treatments, healing springs, organic cuisine, detoxing, spiritual tourism, nature walks, remote destinations, mindful escapism.', 'Financial planning and analysis - budgeting, forecasting, data analysis, financial modeling, performance metrics, KPIs, business strategy, statistics, market research, financial statements', 'Entrepreneurship and startups - innovation, creativity, leadership, management, risk-taking, networking, collaboration, problem-solving, funding, growth', 'Banking and investments - savings, loans, mortgages, credit cards, financial planning, investment options, stock markets, portfolio management, risk management', 'Accounting and bookkeeping - tax preparation, auditing, compliance, payroll, invoicing, tax laws, financial reporting, cash flow analysis, record-keeping', 'Marketing and advertising - brand awareness, promotion, market research, social media, digital marketing, advertising campaigns, market trends, online marketing, customer loyalty', 'E-commerce and online business - online shopping, mobile commerce, e-payment systems, online security, e-commerce platforms, website design, online marketing, online marketplaces, customer experience', 'Human resources and talent management - recruitment, retention, talent development, labor laws, HR analytics, performance management, workforce diversity, employee engagement, employee benefits', 'Corporate social responsibility - sustainability, ethical business practices, environmental impact, social impact, community development, CSR programs, corporate governance, stakeholder engagement, triple bottom line', 'International business and globalization - cross-cultural communication, global trade, cultural diversity, international regulations, global supply chains, foreign investments, global markets, emerging economies, multinational corporations', 'Law and legal services - corporate law, contracts, intellectual property, labor laws, dispute resolution, litigation, regulatory compliance, legal research, legal services.', 'Fruit, healthy, ripe, juicy, refreshing, colorful, natural, organic, summer, tropical ', 'Coffee, latte, caffeine, aroma, brewed, barista, espresso, morning, cappuccino, beans ', 'Pizza, pepperoni, cheese, spicy, delicious, Italian, crusty, baked, toppings, tomato sauce ', 'Beer, cheers, cold, refreshing, frothy, brewery, pub, ale, amber, foam ', 'Grilling, barbecue, steak, meat, burger, smoke, outdoor, flame, charcoal, skewers ', 'Sushi, soy sauce, chopsticks, fresh, Japanese, rolls, salmon, wasabi, seafood, seaweed ', 'Chocolate, dessert, indulgence, sweet, cocoa, creamy, truffles, fondue, ganache, decadence ', 'Wine, vineyard, cellar, red, white, glass, bottle, cork, aged, aroma ', 'Salad, healthy eating, greens, fresh, veggies, dressing, bowls, rainbow, toppings, nutritious ', 'Cocktails, bartending, mixology, shaken, stirred, tropical, gin, rum, vodka, garnish', 'Yoga and Meditation - relaxation, mindfulness, stretching, balance, zen, peace, healthy lifestyle, concentration, harmony, inner peace', 'Diet and Nutrition - healthy eating, fruits and vegetables, meal planning, portion control, vitamins and minerals, energy boost, weight management, detoxification, antioxidants, hydration', 'Fitness and Exercise - cardio, strength training, endurance, bodybuilding, outdoor activities, workout equipment, gym, sports, weight lifting, self-discipline', 'Mental Health - emotional wellbeing, stress relief, anxiety management, therapy, self-care, coping skills, mindfulness, positivity, self-esteem, self-awareness', 'Spa and Massage - pampering, relaxation, rejuvenation, aromatherapy, hot stones, body scrubs, facials, foot massage, stress relief, healing', 'Nature and Outdoors - hiking, camping, biking, running, trekking, forest wellness, mountain climbing, fresh air, stress relief, adventure', 'Alternative Medicine - acupuncture, herbal remedies, aromatherapy, crystal healing, massage therapy, holistic health, naturopathy, meditation, Reiki, chakra balancing', 'Medical Science and Technology - medical equipment, blood donation, vaccination, telehealth, medical research, genome sequencing, pharmaceuticals, medical insurance, medical staff, healthy habits', 'Sleep Hygiene - restful sleep, insomnia, sleep disorders, sleep environment, bedtime routine, sleep quality, sleep aids, stress management, relaxation techniques, sleep patterns', 'Senior Health and Wellness - healthy aging, retirement planning, assisted living, long-term care, dementia, elder care, geriatric care, exercise for seniors, nutrition for seniors, mental health for seniors', 'Books - knowledge, study, education, textbooks, library, research, reading, literature, information, intellect.', 'Classroom - teaching, students, school, chalkboard, learning, lectures, lesson, study group, education, seminar.', 'Technology - online learning, e-books, digital devices, e-learning, computers, distance learning, remote classes, wifi, internet, apps.', 'Science lab - experiments, laboratory, research, scientific method, hypothesis, data analysis, equipment, microscope, chemical reactions, lab coat.', 'College campus - higher education, university life, student activities, socializing, campus tours, college community, extracurricular, sports, clubs, student diversity.', 'Creative arts - music, writing, performing arts, art supplies, creativity, concerts, plays, musical instruments, dance, theater.', 'Tutoring - one-on-one teaching, academic support, homework help, private lessons, exam preparation, coaching, educational guidance, personalized learning, mentorship.', 'Online education - web-based courses, virtual learning, live webinars, educational videos, e-tutoring, remote work, online seminars, video conferencing, online degrees.', "Children's education - early childhood learning, kindergarten, homeschooling, family learning, children's books, learning through play, child growth, daycare, afterschool programs.", 'Graduation - commencement ceremony, caps and gowns, alumni, university programs, graduation speech, diploma, collegiate achievement, higher education, celebratory decor, life after graduation.', 'Mobile Devices - smartphone, tablet, apps, device, screen, touch, communication, mobility, Internet, connection', 'Artificial Intelligence - AI, robot, cyborg, machine learning, automation, neural network, data analytics, computer vision, chatbot, algorithm', 'Cybersecurity - hacker, cyber attack, encryption, phishing, firewall, data breach, password, vulnerability, antivirus, cybercrime', 'Smart Home - home automation, internet of things, IoT, smart speaker, Alexa, Google Home, connected devices, voice recognition, security, comfort', 'Virtual Reality - VR, headset, simulation, gaming, experience, immersion, 360 degree, interactivity, entertainment, education', 'Cloud Computing - cloud storage, server, data center, SaaS, PaaS, IaaS, AWS, Google Cloud, Microsoft Azure, hybrid cloud', 'E-commerce - online shopping, website, customer, shopping cart, payment, delivery, marketplace, mobile commerce, social commerce, product', 'Blockchain - cryptocurrency, bitcoin, ethereum, token, transaction, ledger, decentralization, smart contract, mining, digital asset', 'Medical Technology - health care, telemedicine, medical device, patient, diagnosis, treatment, biotechnology, digital health, pharmaceutical, life sciences', 'Renewable Energy - solar power, wind turbines, hydroelectricity, bioenergy, geothermal, clean energy, carbon footprint, sustainability, green technology, energy efficient.', 'Family dinner - Food, table setting, togetherness, conversation, bonding, sharing, laughter, joy, happiness, home.', 'Grandparents and grandchildren - Love, affection, care, playtime, storytelling, wisdom, memories, generations, family ties, laughter.', 'Family vacation - Travel, adventure, fun, relaxation, exploration, experience, memories, love, bonding, togetherness.', 'Couple in love - Romance, intimacy, affection, joy, companionship, trust, love, togetherness, happiness, relationship.', 'New parents with baby - Love, care, family, joy, happiness, bonding, parenting, newborn, sleepless nights, cooing.', "Siblings - Love-hate, camaraderie, bonding, playtime, sharing, laughter, togetherness, siblings' rivalry, memories, friendship.", 'Pet and family - Unconditional love, fun, bonding, responsibility, care, companionship, pet therapy, happiness, playtime, togetherness.', 'Family gardening - Nature, plants, growth, responsibility, bonding, teamwork, fresh produce, togetherness, hobby, relaxation.', 'Multi-generational family - Family ties, bonding, love, diversity, care, tradition, togetherness, memories, support, joy.', 'Family reading time - Literacy, love for books, relaxation, bonding, storytelling, imagination, education, family values, togetherness, fun.', 'Forests - Trees, Leaves, Woods, Wild, Greenery, Ecosystem, Oxygen, Oxygenation, Natural, Scenic', 'Oceans - Waves, Coral reefs, Marine life, Seashells, Sand, Seagulls, Tides, Beaches, Water, Blue', 'Mountains - Peaks, Hikes, Trails, Cliffs, Snowcapped, Panorama, Landscape, Misty, Alpine, Valleys', 'Rivers - Rapids, Banks, Flowing, Stones, Calm, Waterfalls, Serene, Nature, Reflections, Clear', 'Wildlife - Animals, Habitats, Protection, Sanctuaries, Jungle, Wild, Safari, Preservation, Bird Watching, Eco-tourism', 'Flowers - Blossoms, Petals, Gardens, Blooming, Botany, Pollen, Fragrant, Nature, Daisy, Sunflowers', 'Climate change- Melting, Carbon Footprint, Ways to reduce, Saving Resources, Eco-Friendly Practices, Renewable Energy, Global warming, Solar Energy, Effects, Solutions', 'National parks - Scenic, Explore, Adventure, Trails, Serene, Biodiversity, Camping, Hiking, Landscapes, Views', 'Rural Landscapes - Countryside, Farm, Agriculture, Harvest, Rural Life, Rural paths, Serenity, Nature, Trees, Sunset', 'Waterfalls - Cascades, Natural Beauty, Flowing, Streams, Adventure, River, Pools, Forests, Scenic, Mystical', 'Running - marathon, jogger, shoes, race, fitness, exercise, outdoors, road, health, determination', 'Yoga - mat, meditation, balance, relaxation, flexibility, wellness, namaste, stretch, pose, tranquility', 'Cycling - road bike, cycling shorts, helmet, speed, endurance, pavement, exercise, outdoors, cycling shoes, health', 'Swimming - pool, goggles, swim cap, freestyle, butterfly stroke, backstroke, breaststroke, swimmer, lanes, water', 'Weightlifting - barbell, dumbbell, strength, fitness, gym, powerlifting, muscle, reps, plates, workout', 'Soccer - cleats, ball, field, team, goal, kick, dribble, referee, net, sportsmanship', 'Basketball - court, hoop, ball, jump, dribble, layup, three-pointer, net, team, defense', 'Hiking - trail, trekking poles, backpack, mountain, nature, woods, rock, adventure, view, outdoors', 'Tennis - racket, ball, court, serve, backhand, forehand, net, singles, doubles, match point', 'Gymnastics - leotard, balance beam, floor, vault, gymnast, strength, flexibility, tumbling, competition, high bar', 'Fireworks display - sparkle, explosion, night sky, celebration, colorful, holiday, party, festive, joy, entertainment', 'Christmas decorations - ornaments, lights, tree, snow, candy canes, wreath, gift, festive, merry, jolly', 'Thanksgiving feast - turkey, cranberry sauce, stuffing, pumpkin pie, gravy, yams, mashed potatoes, pecan pie, family, gathering', 'Easter egg hunt - bunny, basket, Pastels, treats, flowers, spring, festive, hunt, colorful, eggs', 'Birthday party - balloons, cake, candles, hats, presents, party favors, singing, confetti, celebrations, joy', "New Year's Eve bash - champagne, countdown, noise makers, party hats, sparklers, ball drop, confetti, festive, fun, New Year", 'Hanukkah celebrations - menorah, dreidel, candles, Latkes, sufganiyot, gelt, blessings, family, holiday, lighting', 'Ramadan Iftar dinner - dates, prayer, mosques, lanterns, family, gathering, food, traditions, celebrations, fasting', 'Diwali celebrations - candles, lights, rangoli, sweets, fireworks, blessings, family, colors, culture, festival', 'Chinese New Year festivities - dragon, lantern parade, fireworks, red envelopes, family, blessings, food, tradition, culture, celebration']

'''

'''
-->subject[0] = Beach Scenes - Sun, sand, ocean waves, palm trees, swimming, beach umbrellas, beach chairs, tropical drinks, boating, shore birds.
-->subject[1] = City Breaks - Skyscrapers, street signs, public transport, local attractions, shopping, dining, parks, historical landmarks, nightlife, street performers.
-->subject[2] = National Parks - Mountains, rivers and lakes, wildlife, hiking trails, campgrounds, scenic views, waterfalls, rock formations, forests, designated natural wonders.
-->subject[3] = Road Trips - Adventure, car rental, scenic highways, landscapes, gas stations, motels, pit-stops, route maps, vintage vehicles, driving safety.
-->subject[4] = Cultural Heritage - Museums, cathedrals, architecture, street scenes, art galleries, cultural events, religious landmarks, cuisine, folk traditions, local costumes.
-->subject[5] = Adventure Sports - Rafting, kayaking, hiking, biking, skiing, snowboarding, surfing, zip-lining, paragliding, extreme challenges.
-->subject[6] = Luxury Escapes - Resorts, mansions, spas, golf courses, private beaches, yacht clubs, gourmet dining, fine wines, indoor pools, lavish accommodations.
-->subject[7] = Family Fun - Amusement parks, zoos, theme parks, puppet shows, carnivals, picnics, theaters, aquariums, arcades, family-friendly attractions.
-->subject[8] = Backpacking - Hostels, budget travel, camping, backpacks, train journeys, cheap eats, alternative tours, selfie sticks, cultural immersion, travel hacks.
-->subject[9] = Wellness Retreats - Yoga, meditation, holistic treatments, healing springs, organic cuisine, detoxing, spiritual tourism, nature walks, remote destinations, mindful escapism.
-->subject[10] = Financial planning and analysis - budgeting, forecasting, data analysis, financial modeling, performance metrics, KPIs, business strategy, statistics, market research, financial statements
-->subject[11] = Entrepreneurship and startups - innovation, creativity, leadership, management, risk-taking, networking, collaboration, problem-solving, funding, growth
-->subject[12] = Banking and investments - savings, loans, mortgages, credit cards, financial planning, investment options, stock markets, portfolio management, risk management
-->subject[13] = Accounting and bookkeeping - tax preparation, auditing, compliance, payroll, invoicing, tax laws, financial reporting, cash flow analysis, record-keeping
-->subject[14] = Marketing and advertising - brand awareness, promotion, market research, social media, digital marketing, advertising campaigns, market trends, online marketing, customer loyalty
-->subject[15] = E-commerce and online business - online shopping, mobile commerce, e-payment systems, online security, e-commerce platforms, website design, online marketing, online marketplaces, customer experience
-->subject[16] = Human resources and talent management - recruitment, retention, talent development, labor laws, HR analytics, performance management, workforce diversity, employee engagement, employee benefits
-->subject[17] = Corporate social responsibility - sustainability, ethical business practices, environmental impact, social impact, community development, CSR programs, corporate governance, stakeholder engagement, triple bottom line
-->subject[18] = International business and globalization - cross-cultural communication, global trade, cultural diversity, international regulations, global supply chains, foreign investments, global markets, emerging economies, multinational corporations
-->subject[19] = Law and legal services - corporate law, contracts, intellectual property, labor laws, dispute resolution, litigation, regulatory compliance, legal research, legal services.
-->subject[20] = Fruit, healthy, ripe, juicy, refreshing, colorful, natural, organic, summer, tropical 
-->subject[21] = Coffee, latte, caffeine, aroma, brewed, barista, espresso, morning, cappuccino, beans 
-->subject[22] = Pizza, pepperoni, cheese, spicy, delicious, Italian, crusty, baked, toppings, tomato sauce 
-->subject[23] = Beer, cheers, cold, refreshing, frothy, brewery, pub, ale, amber, foam 
-->subject[24] = Grilling, barbecue, steak, meat, burger, smoke, outdoor, flame, charcoal, skewers 
-->subject[25] = Sushi, soy sauce, chopsticks, fresh, Japanese, rolls, salmon, wasabi, seafood, seaweed 
-->subject[26] = Chocolate, dessert, indulgence, sweet, cocoa, creamy, truffles, fondue, ganache, decadence 
-->subject[27] = Wine, vineyard, cellar, red, white, glass, bottle, cork, aged, aroma 
-->subject[28] = Salad, healthy eating, greens, fresh, veggies, dressing, bowls, rainbow, toppings, nutritious 
-->subject[29] = Cocktails, bartending, mixology, shaken, stirred, tropical, gin, rum, vodka, garnish
-->subject[30] = Yoga and Meditation - relaxation, mindfulness, stretching, balance, zen, peace, healthy lifestyle, concentration, harmony, inner peace
-->subject[31] = Diet and Nutrition - healthy eating, fruits and vegetables, meal planning, portion control, vitamins and minerals, energy boost, weight management, detoxification, antioxidants, hydration
-->subject[32] = Fitness and Exercise - cardio, strength training, endurance, bodybuilding, outdoor activities, workout equipment, gym, sports, weight lifting, self-discipline
-->subject[33] = Mental Health - emotional wellbeing, stress relief, anxiety management, therapy, self-care, coping skills, mindfulness, positivity, self-esteem, self-awareness
-->subject[34] = Spa and Massage - pampering, relaxation, rejuvenation, aromatherapy, hot stones, body scrubs, facials, foot massage, stress relief, healing
-->subject[35] = Nature and Outdoors - hiking, camping, biking, running, trekking, forest wellness, mountain climbing, fresh air, stress relief, adventure
-->subject[36] = Alternative Medicine - acupuncture, herbal remedies, aromatherapy, crystal healing, massage therapy, holistic health, naturopathy, meditation, Reiki, chakra balancing
-->subject[37] = Medical Science and Technology - medical equipment, blood donation, vaccination, telehealth, medical research, genome sequencing, pharmaceuticals, medical insurance, medical staff, healthy habits
-->subject[38] = Sleep Hygiene - restful sleep, insomnia, sleep disorders, sleep environment, bedtime routine, sleep quality, sleep aids, stress management, relaxation techniques, sleep patterns
-->subject[39] = Senior Health and Wellness - healthy aging, retirement planning, assisted living, long-term care, dementia, elder care, geriatric care, exercise for seniors, nutrition for seniors, mental health for seniors
-->subject[40] = Books - knowledge, study, education, textbooks, library, research, reading, literature, information, intellect.
-->subject[41] = Classroom - teaching, students, school, chalkboard, learning, lectures, lesson, study group, education, seminar.
-->subject[42] = Technology - online learning, e-books, digital devices, e-learning, computers, distance learning, remote classes, wifi, internet, apps.
-->subject[43] = Science lab - experiments, laboratory, research, scientific method, hypothesis, data analysis, equipment, microscope, chemical reactions, lab coat.
-->subject[44] = College campus - higher education, university life, student activities, socializing, campus tours, college community, extracurricular, sports, clubs, student diversity.
-->subject[45] = Creative arts - music, writing, performing arts, art supplies, creativity, concerts, plays, musical instruments, dance, theater.
-->subject[46] = Tutoring - one-on-one teaching, academic support, homework help, private lessons, exam preparation, coaching, educational guidance, personalized learning, mentorship.
-->subject[47] = Online education - web-based courses, virtual learning, live webinars, educational videos, e-tutoring, remote work, online seminars, video conferencing, online degrees.
-->subject[48] = Children's education - early childhood learning, kindergarten, homeschooling, family learning, children's books, learning through play, child growth, daycare, afterschool programs.
-->subject[49] = Graduation - commencement ceremony, caps and gowns, alumni, university programs, graduation speech, diploma, collegiate achievement, higher education, celebratory decor, life after graduation.
-->subject[50] = Mobile Devices - smartphone, tablet, apps, device, screen, touch, communication, mobility, Internet, connection
-->subject[51] = Artificial Intelligence - AI, robot, cyborg, machine learning, automation, neural network, data analytics, computer vision, chatbot, algorithm
-->subject[52] = Cybersecurity - hacker, cyber attack, encryption, phishing, firewall, data breach, password, vulnerability, antivirus, cybercrime
-->subject[53] = Smart Home - home automation, internet of things, IoT, smart speaker, Alexa, Google Home, connected devices, voice recognition, security, comfort
-->subject[54] = Virtual Reality - VR, headset, simulation, gaming, experience, immersion, 360 degree, interactivity, entertainment, education
-->subject[55] = Cloud Computing - cloud storage, server, data center, SaaS, PaaS, IaaS, AWS, Google Cloud, Microsoft Azure, hybrid cloud
-->subject[56] = E-commerce - online shopping, website, customer, shopping cart, payment, delivery, marketplace, mobile commerce, social commerce, product
-->subject[57] = Blockchain - cryptocurrency, bitcoin, ethereum, token, transaction, ledger, decentralization, smart contract, mining, digital asset
-->subject[58] = Medical Technology - health care, telemedicine, medical device, patient, diagnosis, treatment, biotechnology, digital health, pharmaceutical, life sciences
-->subject[59] = Renewable Energy - solar power, wind turbines, hydroelectricity, bioenergy, geothermal, clean energy, carbon footprint, sustainability, green technology, energy efficient.
-->subject[60] = Family dinner - Food, table setting, togetherness, conversation, bonding, sharing, laughter, joy, happiness, home.
-->subject[61] = Grandparents and grandchildren - Love, affection, care, playtime, storytelling, wisdom, memories, generations, family ties, laughter.
-->subject[62] = Family vacation - Travel, adventure, fun, relaxation, exploration, experience, memories, love, bonding, togetherness.
-->subject[63] = Couple in love - Romance, intimacy, affection, joy, companionship, trust, love, togetherness, happiness, relationship.
-->subject[64] = New parents with baby - Love, care, family, joy, happiness, bonding, parenting, newborn, sleepless nights, cooing.
-->subject[65] = Siblings - Love-hate, camaraderie, bonding, playtime, sharing, laughter, togetherness, siblings' rivalry, memories, friendship.
-->subject[66] = Pet and family - Unconditional love, fun, bonding, responsibility, care, companionship, pet therapy, happiness, playtime, togetherness.
-->subject[67] = Family gardening - Nature, plants, growth, responsibility, bonding, teamwork, fresh produce, togetherness, hobby, relaxation.
-->subject[68] = Multi-generational family - Family ties, bonding, love, diversity, care, tradition, togetherness, memories, support, joy.
-->subject[69] = Family reading time - Literacy, love for books, relaxation, bonding, storytelling, imagination, education, family values, togetherness, fun.
-->subject[70] = Forests - Trees, Leaves, Woods, Wild, Greenery, Ecosystem, Oxygen, Oxygenation, Natural, Scenic
-->subject[71] = Oceans - Waves, Coral reefs, Marine life, Seashells, Sand, Seagulls, Tides, Beaches, Water, Blue
-->subject[72] = Mountains - Peaks, Hikes, Trails, Cliffs, Snowcapped, Panorama, Landscape, Misty, Alpine, Valleys
-->subject[73] = Rivers - Rapids, Banks, Flowing, Stones, Calm, Waterfalls, Serene, Nature, Reflections, Clear
-->subject[74] = Wildlife - Animals, Habitats, Protection, Sanctuaries, Jungle, Wild, Safari, Preservation, Bird Watching, Eco-tourism
-->subject[75] = Flowers - Blossoms, Petals, Gardens, Blooming, Botany, Pollen, Fragrant, Nature, Daisy, Sunflowers
-->subject[76] = Climate change- Melting, Carbon Footprint, Ways to reduce, Saving Resources, Eco-Friendly Practices, Renewable Energy, Global warming, Solar Energy, Effects, Solutions
-->subject[77] = National parks - Scenic, Explore, Adventure, Trails, Serene, Biodiversity, Camping, Hiking, Landscapes, Views
-->subject[78] = Rural Landscapes - Countryside, Farm, Agriculture, Harvest, Rural Life, Rural paths, Serenity, Nature, Trees, Sunset
-->subject[79] = Waterfalls - Cascades, Natural Beauty, Flowing, Streams, Adventure, River, Pools, Forests, Scenic, Mystical
-->subject[80] = Running - marathon, jogger, shoes, race, fitness, exercise, outdoors, road, health, determination
-->subject[81] = Yoga - mat, meditation, balance, relaxation, flexibility, wellness, namaste, stretch, pose, tranquility
-->subject[82] = Cycling - road bike, cycling shorts, helmet, speed, endurance, pavement, exercise, outdoors, cycling shoes, health
-->subject[83] = Swimming - pool, goggles, swim cap, freestyle, butterfly stroke, backstroke, breaststroke, swimmer, lanes, water
-->subject[84] = Weightlifting - barbell, dumbbell, strength, fitness, gym, powerlifting, muscle, reps, plates, workout
-->subject[85] = Soccer - cleats, ball, field, team, goal, kick, dribble, referee, net, sportsmanship
-->subject[86] = Basketball - court, hoop, ball, jump, dribble, layup, three-pointer, net, team, defense
-->subject[87] = Hiking - trail, trekking poles, backpack, mountain, nature, woods, rock, adventure, view, outdoors
-->subject[88] = Tennis - racket, ball, court, serve, backhand, forehand, net, singles, doubles, match point
-->subject[89] = Gymnastics - leotard, balance beam, floor, vault, gymnast, strength, flexibility, tumbling, competition, high bar
-->subject[90] = Fireworks display - sparkle, explosion, night sky, celebration, colorful, holiday, party, festive, joy, entertainment
-->subject[91] = Christmas decorations - ornaments, lights, tree, snow, candy canes, wreath, gift, festive, merry, jolly
-->subject[92] = Thanksgiving feast - turkey, cranberry sauce, stuffing, pumpkin pie, gravy, yams, mashed potatoes, pecan pie, family, gathering
-->subject[93] = Easter egg hunt - bunny, basket, Pastels, treats, flowers, spring, festive, hunt, colorful, eggs
-->subject[94] = Birthday party - balloons, cake, candles, hats, presents, party favors, singing, confetti, celebrations, joy
-->subject[95] = New Year's Eve bash - champagne, countdown, noise makers, party hats, sparklers, ball drop, confetti, festive, fun, New Year
-->subject[96] = Hanukkah celebrations - menorah, dreidel, candles, Latkes, sufganiyot, gelt, blessings, family, holiday, lighting
-->subject[97] = Ramadan Iftar dinner - dates, prayer, mosques, lanterns, family, gathering, food, traditions, celebrations, fasting
-->subject[98] = Diwali celebrations - candles, lights, rangoli, sweets, fireworks, blessings, family, colors, culture, festival
-->subject[99] = Chinese New Year festivities - dragon, lantern parade, fireworks, red envelopes, family, blessings, food, tradition, culture, celebration
'''

'''

-->subject[0] = Hospitals and medical facilities - Doctor, nurse, patient, emergency, surgery, equipment, medicine, diagnosis, treatment, care 
-->subject[1] = Medical research and development - Laboratory, experiment, scientist, microscope, testing, analysis, discovery, innovation, breakthrough, trial 
-->subject[2] = Healthcare professionals - Health, wellness, specialist, therapist, counselor, care provider, support, professionalism, expertise, compassion 
-->subject[3] = Mental health and therapy - Depression, anxiety, counseling, therapy, treatment, self-care, mindfulness, mental health, emotional support, wellness 
-->subject[4] = Senior care and assisted living - Elderly, aging, care, assisted, senior living, health services, nursing, companionship, caregivers, lifestyle 
-->subject[5] = Medical technology and innovation - Artificial intelligence, virtual reality, robotics, medical devices, telemedicine, software, big data, healthcare IT, digital health, wearable 
-->subject[6] = Dental care and services - Teeth, dentist, oral health, hygiene, cavity, braces, toothbrush, floss, mouthwash, gum care 
-->subject[7] = Alternative medicine and healing - Acupuncture, herbal medicine, yoga, meditation, massage, aromatherapy, natural remedies, spiritual healing, wellness practices, holistic therapy 
-->subject[8] = Public health and preventive care - Vaccinations, disease prevention, health education, nutrition, hygiene, healthy living, community health, environmental health, safety, wellness 
-->subject[9] = Medical education and training - Medical school, residency, continuing education, training programs, medical textbooks, professional development, simulation, skills assessment, healthcare administration, medical ethics.
-->subject[10] = Delicious desserts - chocolate, cake, creamy, sweet, indulgent, fruit, tasty, whipped cream, icing, decadent
-->subject[11] = Fruits and vegetables - colorful, fresh, organic, healthy, vitamins, nutrients, salad, juice, market, harvest
-->subject[12] = Fast food - burger, fries, soda, pizza, fried chicken, fast, drive-thru, convenience, unhealthy, guilty pleasure
-->subject[13] = Drinks and cocktails - martini, margarita, beer, wine, cocktail, ice, glass, refreshing, bar, happy hour
-->subject[14] = BBQ and grill - steak, ribs, corn, burgers, hot dogs, grill marks, smoke, BBQ sauce, charcoal, summertime
-->subject[15] = International cuisine - sushi, tacos, pasta, curry, stir fry, falafel, dumplings, paella, naan, kimchi
-->subject[16] = Breakfast spread - pancakes, waffles, bacon, eggs, toast, yogurt, cereal, milk, syrup, morning
-->subject[17] = Comfort food - mac and cheese, meatloaf, mashed potatoes, casseroles, pot pies, gravy, warmth, heartwarming, cozy, homey
-->subject[18] = Baked goods - bread, croissants, muffins, bagels, cinnamon rolls, cookies, pastries, bakery, oven, aroma
-->subject[19] = Seafood - shrimp, lobster, crab, clams, mussels, fish, oysters, seafood platter, fried, grilled.
-->subject[20] = Beach Vacation - Sand, Sun, Ocean, Relaxation, Waves, Seashells, Boardwalk, Umbrella, Palms, Swimming
-->subject[21] = Adventure Travel - Mountains, Camping, Hiking, Exploration, Wildlife, Rafting, Climbing, Backpacking, Scenery, Trekking
-->subject[22] = City Tour - Architecture, Skyline, Culture, Museum, Landmarks, Traffic, Skyscrapers, Pedestrians, Nightlife, Urban
-->subject[23] = Cruise Holidays - Deck, Port, Ship, Ocean View, Dining, Aqua Park, Entertainment, Excursion, Relaxation, Sea
-->subject[24] = Road Trip - Car, Map, Highway, Landscape, Scenic Route, Stopover, Pit Stop, Adventure, Journey, Freedom
-->subject[25] = Tropical Island - Rainforest, Waterfall, Coral Reef, Exotic Animals, Beach, Clear Waters, Sandbar, Diving, Snorkeling, Sunsets
-->subject[26] = Train Journey - Railway, Luggage, Locomotive, Station, Passengers, Scenery, Sleeper Car, Dining, Sightseeing, Adventure
-->subject[27] = Skiing and Snowboarding - Snow, Slopes, Ski Lift, Goggles, Helmet, Frozen Lakes, Snowshoeing, Après-Ski, Alpine, Winter Sports
-->subject[28] = Family Vacation - Amusement Park, Playground, Resort, Kids, Family, Waterpark, Pool, Fun, Relaxation, Activities
-->subject[29] = Cultural Holidays - Heritage, Religion, Art, Food, Architecture, Music, Tradition, People, Festivals, Passport.
-->subject[30] = Futuristic technology: Artificial Intelligence, Robotics, Augmented Reality, Virtual Reality, Cybersecurity, Machine Learning, Big Data, Quantum Computing, Automation, Internet of Things.
-->subject[31] = Digital communication: Social Media, Video Conferencing, Messaging, Email, Voice Assistant, Online Meeting, Telecommunication, Information Transfer, Digital Networking, Voice Call.
-->subject[32] = E-commerce: Online shopping, Electronic payment, Digital Marketing, Virtual Shopping, Cryptocurrency, E-wallet, Online Marketplace, Online Retailer, Online Banking, Cashless Transaction.
-->subject[33] = Cloud Computing: Cloud Storage, Cloud Security, Cloud Deployment, Cloud Database, Cloud Backup, Cloud Software, Content Delivery Network, Cloud Infrastructure, Multi-Cloud, Hybrid Cloud.
-->subject[34] = Mobile Technology: Smartphone, Mobile Devices, Tablet, Mobile App, Wearable Technology, Mobile Commerce, Mobile Gaming, Mobile Payment, Mobile GPS, Mobile Security.
-->subject[35] = 3D Printing: Rapid Prototyping, Additive Manufacturing, Stereo lithography, Digital Fabrication, 3D Modeling, 3D Scanner, Industrial Design, 3D Scanning, 3D Rendering, 3D Animation.
-->subject[36] = Renewable energy: Solar Power, Wind Power, Hydro Power, Geothermal Energy, Bioenergy, Tidal Energy, Wave Energy, Electric Vehicle, Electric Grid, Energy Storage.
-->subject[37] = Green technology: Sustainable Architecture, Green Building, Eco-Friendly Products, Energy Efficiency, Sustainable Energy, Recycling, Waste Management, Green Energy, Carbon Footprint, Sustainable Transportation.
-->subject[38] = Cybersecurity: Data Security, Network Security, Encryption, Antivirus, Cybercrime, Firewalls, Secure Communication, Digital Certificate, Biometric Security, Malware Protection.
-->subject[39] = Blockchain: Cryptocurrency, Smart Contract, Blockchain Security, Digital Asset, Decentralized Network, Bitcoin, Public Ledger, Digital Identity, Distributed Ledger, Immutable Record.
-->subject[40] = Forests – trees, leaves, wildlife, hiking, camping, greenery, nature trails, woodlands, pine cones, ferns
-->subject[41] = Oceans – sea creatures, beach, waves, surfing, coral reefs, sand, seashells, marine life, sailing, sunsets
-->subject[42] = Mountains – peaks, valleys, rock formations, snow, glaciers, mountaineering, wildflowers, alpine lakes, hiking trails, scenic vistas
-->subject[43] = Rivers – waterfalls, rapids, kayaking, fishing, canoeing, scenic landscapes, river banks, bridges, wildlife, reflections
-->subject[44] = National Parks – camping, scenic drives, hiking, wildlife, natural wonders, landmarks, waterfalls, lakes, mountain views, forests
-->subject[45] = Deserts – sand dunes, rock formations, cactus, rattlesnakes, sunsets, hiking, sandstorms, mirages, oases, road trips
-->subject[46] = Caves – stalactites, stalagmites, underground lakes, glow worms, darkness, exploration, spelunking, cave diving, rock formations, subterranean rivers
-->subject[47] = Wildlife – animals, birds, insects, marine life, safari, national parks, zoos, habitats, migration, ecosystems
-->subject[48] = Flowers – gardens, floral arrangements, wildflowers, plant life, botany, bee pollination, nature photography, color palettes, floral backgrounds
-->subject[49] = Landscapes – sunsets, sunrises, mountains, rivers, beaches, forests, plains, natural wonders, city parks, countryside.
-->subject[50] = Gym equipment: Dumbbell, treadmill, elliptical machine, resistance band, exercise ball, weight bench, kettlebell, spin bike, yoga mat, foam roller.
-->subject[51] = Running: Jogging, marathon, road race, trail running, sneakers, finish line, warm-up, sprint, stretching, recovery.
-->subject[52] = Yoga: Stretching, meditation, relaxation, balance, flexibility, strength, sunrise, sunset, outdoor, studio.
-->subject[53] = Swimming: Pool, diving, freestyle, breaststroke, backstroke, butterfly, goggles, swim cap, diving board, lane ropes.
-->subject[54] = Cycling: Tour de France, mountain biking, BMX, road cycling, cycling shoes, helmet, cycling computer, cycling jersey, cycling shorts, speed.
-->subject[55] = Team sports: Soccer, basketball, baseball, volleyball, football, hockey, team spirit, practice, uniforms, game time.
-->subject[56] = Strength training: Bodybuilding, powerlifting, squatting, lifting weights, barbell, weight plates, gym gloves, gym bag, motivaton, core strength.
-->subject[57] = Outdoor fitness: Hiking, trekking, camping, backpacking, rock climbing, kayaking, canoeing, nature, adventure, fresh air.
-->subject[58] = Dance fitness: Zumba, salsa, belly dancing, hip hop, ballroom dancing, jazzercise, aerobic dance, music, movement, fun.
-->subject[59] = Martial arts: Karate, Judo, Tae Kwon Do, MMA, self-defense, discipline, respect, sparring, kicks, punches.
-->subject[60] = School Supplies - pencils, notebooks, rulers, textbooks, backpacks, highlighters, erasers, paperclips, calculators, binders
-->subject[61] = Classroom Environment - desks, chairs, whiteboards, posters, maps, chalkboards, projectors, carpets, books, windows
-->subject[62] = Science Lab - test tubes, beakers, microscopes, lab coats, goggles, pipettes, Bunsen burners, chemicals, petri dishes, lab gloves
-->subject[63] = Digital Learning - laptops, tablets, smartphones, headphones, computer screens, keyboards, mouses, online courses, webinars, e-books
-->subject[64] = Student Life - group projects, extracurricular activities, sports, clubs, events, friendships, communication, social media, diversity, inclusivity
-->subject[65] = Creative Arts - paintbrushes, canvases, musical instruments, costumes, dance, theater, poetry, literature, photography, design
-->subject[66] = Teacher and Pupil - mentoring, guidance, encouragement, feedback, discussion, inspiration, role modeling, support, cooperation, respect
-->subject[67] = Early Childhood Education - blocks, puzzles, toys, crafts, games, finger painting, coloring books, sandboxes, swings, educational play
-->subject[68] = University Life - lectures, seminars, libraries, research, internships, academic writing, exams, career guidance, job fairs, campus landscapes
-->subject[69] = Distance Education - video conference, online classes, telecommuting, E-learning, distance learning, virtual classrooms, remote education, telecommuting, online degrees, instructional design
-->subject[70] = Family dinner - togetherness, conversation, dining table, food, love, bonding, relationship, smiles, laughter, sharing.
-->subject[71] = Mother and child - affection, care, nurturing, love, childhood, parent, protection, embrace, happiness, trust.
-->subject[72] = Father and child - support, guidance, mentorship, playfulness, fatherhood, activity, adventure, bonding, emotions, connection.
-->subject[73] = Siblings - rivalry, friendship, siblinghood, laughter, sharing, memories, fights, love-hate, bonding, communication.
-->subject[74] = Family vacation - travel, adventure, photography, sightseeing, relaxation, road trip, hotels, family time, fun, memories.
-->subject[75] = Grandparents and grandchildren - wisdom, stories, love, bonding, mentoring, care, attention, timelessness, relationship, memories.
-->subject[76] = Family celebration - party, togetherness, happiness, event, tradition, cake, gifts, decorations, joy, family reunion.
-->subject[77] = Pet and family - loyalty, care, companionship, love, pet parenting, family bonding, responsibility, activity, fun, friendship.
-->subject[78] = Couple in love - romance, trust, intimacy, commitment, passion, relationship goals, chemistry, togetherness, sharing, communication.
-->subject[79] = Parents and teenage child - guidance, adolescence, understanding, trust, communication, emotional support, care, nurturing, respect, bonding.

'''
