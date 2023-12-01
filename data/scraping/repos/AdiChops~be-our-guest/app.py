from flask import Flask, session, request, jsonify
from flask_session import Session
from flask_socketio import SocketIO
import cohere
from datetime import timedelta
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("COHERE_API_KEY")
co = cohere.Client(api_key)

convId = 0

######-------Menu-------######
starters = f"""C6STARTERS 2
C
Hummus
7.99
Creamy dip made from blended chickpeas, tahini, lemon juice, garlic, & olive oil
Baba Ganoush
7.99
Smoky and flavourful crafted from roasted eggplants, tahini, lemon juice & garlic
Falafel (8pc)
7.99
Crispy deep fried chickpeas mixed with our special blend of spices
Greek Bruschetta
8.99
Toasted baguette bread topped with fresh tomatoes, garlic, basil & feta cheese
Grilled Mushroom
Sizzled & flavourful mushrooms with garlic, rosemary & butter
8.99
Poutine
9.99
Crispy fries topped with mozzarella cheese and rich gravy
-add chicken $4.99
Greek Fries
9.99
Crispy fries topped with feta cheese, lemon sauce & our special blend of spices
-add chicken $4.99
Chicken Fingers (6р)
12.99
Breaded and fried chicken strips served alongside crispy French fries
Beef Samosa (8pc)
Fried pastries filled with perfectly seasoned spiced ground beef
12.99
Veggie Samosa
10.99
Fried pastries filled with carrots, onions, beans, garlic and potatoes
Breaded Shrimp
13.99
Golden crispy deep fried shrimp mixed with our special blend of spices
Chicken Wings
15.99
Classic non breaded wings seasoned and deep fried to tender perfection
Classic Nachos
15.99
Tortilla chips topped with a blend of cheddar and mozzarella cheeses, zesty jalapeños, olives, tomatoes, and onions
-add beef chilli or chicken $4.99
SALADS
Mann
Fattoush Salad
9.99
Topped with lettuce, tomatoes, onions, cucumbers, toasted pita chips, and tangy dressing
Greek Salad
10.99
Crisp lettuce, ripe tomatoes, cucumbers, red onions, green peppers, kalamata olives, and tangy feta cheese.
Caesar Salad
Lettuce, croutons, grated parmesan cheese, and a caesar dressing
8.99
All prices exclude taxes & gratuity
"""
mains = f"""C6MAINS & WRAPS IS
NO
Falafel Dinner
19.99
Deep fried ground chickpeas balls mixed with our blend of spices served with house salad and vour choice of fries or rice and roasted potatoes
Chicken Souvlaki Dinner
22.99
Our famous chicken souvlaki plate served with garlic bread, tzatziki sauce, with greek salad and fries or rice with roasted potatoes
Chicken Schnitzel Dinner
22.99
1 -pound of our mouth watering chicken schnitzel breast served with salad and fries or rice with roasted potatoes
Grilled Shrimp Dinner
24.99
Two skewers of shrimp mixed with our special blend of spices served with greek salad and fries or rice with roasted potatoes
Lamb Souvlaki Dinner
26.99
Two skewers of our famous Lamb
Souvlaki, served with garlic bread, tzatziki sauce, with greek salad and fries or rice with roasted potatoes
Chef's Mixed Platter
89.99
Classic's kitchen mix of 2 chicken souvlaki skewers, 2 lamb souvlaki skewers, 2 shrimp skewers, 1 pound of chicken schnitzel, wings, veggie samosa, falafel, fries and rice (serves 3-4 people)
WRAPS & BURGERS
Veggie Falafel Wrap
8.99
Deep fried ground chickpeas balls topped with tomatoes, lettuce onions & fries
Chicken Souvlaki Wrap
10.99
Classic's favourite wrap, topped with tzitizki, onions, tomatoes, fries & lemon sauce
Lamb Souvlaki Wrap
12.99
One skewer of our famous lamb souvlaki topped with tzatziki, tomatoes and onions and grilled to perfection
Beef Burger
13.99
Juicy patty made from the finest ground beef, perfectly grilled to perfection served with fresh lettuce, tomatoes and mayo.
Served with fries
Cheese Burger
14.99
Juicy patty made from the finest ground beef, perfectly grilled to perfection served with cheddar cheese fresh lettuce, tomatoes and mayo. Served with fries
Chicken Burger
13.99
Tender chicken fillet, expertly seasoned and fried to perfection, served with crisp lettuce, tomatoes & mayo. Served with fries
Veggie Burger
12.99
Plant-based patty crafted from a medley of wholesome vegetables and seasonings accompanied by fresh lettuce, tomatoes & mayo. Served with fries
"""

shisha = f"""C6 SHISHA a
• • •
REGULAR
Orange Mint
Gum Mint
Lemon Mint
Double Apple
• •
PREMIUM
Blueberry Mint
Kiwi
24.00
24.00
24.00
24.00
24.00
24.00
Comes with regular hookah and regular coals
Love 66
30.00
Passion fruit, honeydew melon, watermelon, and mint.
Lady Killer
30.00
Mix of honeydew melon, mango, berries and menthol
Blue Dragon
30.00
A little sweet & slightly velvety with a mix of berries
Candy Lemonade
30.00
Mix of strawberry, cherry & lemonade
Mojito
30.00
Refreshing Cuban Mojito flavour
Sex On The Beach
30.00
Mix of Orange, Cranberry, Lime & Mint
Comes with premium German hookah & coconut coals bowl
EXTRAS
Ice Hose
5.00
Coconut Coals Bowl
Orange Head
3.00
6.00
Pineapple Head
20.00
"""

######-------Menu-End-------######


app = Flask(__name__)

app.config['SESSION_TYPE'] = 'filesystem'

app.config['SESSION_PERMANENT'] = True
app.config['SESSION_USE_SIGNER'] = True

app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)

app.config['SECRET_KEY'] = 'put-our-service-to-the-test'

Session(app)
socketio = SocketIO(app)

@app.route('/ready', methods=['GET'])
def ready():
    socketio.emit('receive_requests', {})
    return "success"

@app.route('/sales', methods=['POST'])
def sales():
    user_data = request.get_json()

    global starters
    global mains
    global shisha
    global convId

    message = user_data.get('message', "")    

    # Retrieve or create a new conversation_id
    if session.get('convId') is None:
        session['convId'] = f"conversation{convId}"
        convId += 1
    conversation_id = session['convId']

    preamble = f"""Your name is Dwight, you are the Host of a popular restaurant, your job is to recommend different items of varying price points to customers \
                Based on the Starters: {starters}, Mains: {mains}, and Shisha: {shisha} \
                and get the customer's input on what they would like to order \
                recommend 3 items from Mains. Do not mention the prices while taking their orders unless asked. \
                In the end, ask what would they like to order \
                allow them to make any substitutions or modifications to the order and record the changes \
                ask if there is anyone else on the table who needs to order \
                confirm the order before moving to the next person \
                confirm the entire table's order in the end \
                Example:(  \
                Welcome to our restaurant! My name is Dwight and I will be your host tonight! \
                We have a diverse menu to choose from. For starters, we offer items like Caesar Salad, Garlic Bread, and Spicy Chicken Wings. \
                Our main courses include options such as Beef Steak, Grilled Salmon, and Vegetable Stir-Fry. 
                We also have a variety of Shisha flavors for you to enjoy, including Apple Mint, Blueberry, and Watermelon.\
                For mains, I'd like to recommend the following three options: \
                Beef Steak - a succulent, perfectly-cooked steak served with a side of mashed potatoes and seasonal vegetables. \
                Grilled Salmon - a fresh salmon fillet seasoned with herbs and grilled to perfection, accompanied by rice pilaf and steamed asparagus. \
                Vegetable Stir-Fry - a medley of seasonal vegetables sautéed in a light garlic sauce, perfect for our vegetarian guests. \
                May I take your order now? Feel free to make any substitutions or modifications. \
                Is there anyone else at the table who would like to place an order? \
                [After taking orders from everyone] \
                Great! Just to confirm the entire table's order: [Recap the orders]. \
                Would that be all for you and your party this evening?)
                Example: (Hello! Welcome to our restaurant. My name is Dwight and I'll be your host this evening. \nWe have a great selection of starters, mains, and shisha for you to choose from. To get started, would you like to hear about our starters?) \
                    """

    # Generate a response with the current chat history
    response = co.chat(
        message,
        temperature=0.4,
        prompt_truncation="AUTO",
        preamble_override=preamble,
        stream=False,
        conversation_id=conversation_id  # Include conversation_id
    )
    answer = response.text

    return jsonify({"response": answer, "conversation_id": conversation_id})

if __name__ == "__main__":
    socketio.run(app, debug=True)