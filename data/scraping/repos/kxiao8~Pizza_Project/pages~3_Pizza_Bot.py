import openai
import streamlit as st

st.set_page_config(page_title="Pizza Chatbot", page_icon = "🍕", layout="centered",
                   initial_sidebar_state="auto", menu_items=None)

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="feedback_api_key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"

st.title("🍕 Pizza Bot")

"""
---
In this example, I use the 
ChatGPT Prompt Engineering for Developers course's source code
to create a chatbot that can take orders for a pizza restaurant.

---
## Menu
#### Pizzas

| Item             | Large | Med.  | Small |
|------------------|-------|-------|-------|
| Pepperoni Pizza  | 12.95 | 10.00 |  7.00 |
| Cheese Pizza     | 10.95 |  9.25 |  6.50 |
| Eggplant Pizza   | 11.95 |  9.75 |  6.75 |

#### Sides

| Item         | Reg.  | Small |
|--------------|-------|-------|
| Fries        |  4.50 |  3.50 |
| Greek Salad  |  7.25 |       |

#### Toppings

| Item            | Reg.  | 
|-----------------|-------|
| Extra Cheese    |  2.00 |   
| Mushrooms       |  1.50 |       
| Sausage         |  3.00 |       
| Canadian Bacon  |  3.50 |       
| AI Sauce        |  1.50 |       
| Peppers         |  1.00 |       

#### Drinks

| Item          | Large | Med.  | Small |
|---------------|-------|-------|-------|
| Coke          |  3.00 |  2.00 |  1.00 |
| Sprite        |  3.00 |  2.00 |  1.00 |
| Bottled Water |  5.00 |       |       |
---

"""
inital_context = [ {'role':'system', 'content':"""
    You are OrderBot, designed specifically to assist customers in placing their pizza orders.
    Your interactions should be based on the following:
    1. Greet the customer with warmth and friendliness.
    2. Show all prices with the \$ prefix. It's essential to always display dollar amounts this way.
    3. Guided Order Collection: Present the menu and collect their order. Only items on the menu are available.
    4. Once the order is collected, provide a summary. Ask if they'd like to add anything more.
    5. Determine if the order is for pickup or delivery. If for delivery, collect the address.
    6. Collect payment details.
    7. Show a final summary of the customer's order.
    8. Maintain a short, friendly, and clear conversational style throughout.
    9. Be detailed: clarify all options, extras, and sizes to uniquely identify items from the menu.
    10. Be careful: don't allow the customer to order items that are not on the menu!
                    
    Store Location: 123 Main Street, Anytown, USA.

    Menu:

    Pizzas:
        - Pepperoni: \$12.95 (Large), \$10.00 (Med.), \$7.00 (Small)
        - Cheese: \$10.95 (Large), \$9.25 (Med.), \$6.50 (Small)
        - Eggplant: \$11.95 (Large), \$9.75 (Med.), \$6.75 (Small)
    Sides:
        - Fries: \$4.50 (Regular), \$3.50 (Small)
        - Greek Salad: \$7.25
    Toppings:
        - Extra Cheese: \$2.00
        - Mushrooms: \$1.50
        - Sausage: \$3.00
        - Canadian Bacon: \$3.50
        - AI Sauce: \$1.50
        - Peppers: \$1.00
    Drinks:
        - Coke: \$3.00 (Large), \$2.00 (Med.), \$1.00 (Small)
        - Sprite: \$3.00 (Large), \$2.00 (Med.), \$1.00 (Small)
        - Bottled Water: \$5.00

    """} ]  # accumulate messages

if "messages" not in st.session_state:
    st.session_state.messages = inital_context

if "response" not in st.session_state:
    st.session_state["response"] = None

messages = st.session_state.messages
for msg in messages[1:]: # skip the initial setting message
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="OrderBot is ready to take your order!"):
    messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()
    else:
        openai.api_key = openai_api_key
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
    st.session_state["response"] = response.choices[0].message.content
    with st.chat_message("assistant"):
        messages.append({"role": "assistant", "content": st.session_state["response"]})
        st.write(st.session_state["response"])