import openai

openai.api_key = "Set your API key here"

chat_msg = []

name = input("What's your name sir? ")
while True:
    print("Enter your message: ")
    user_input = input()
    
    if user_input.lower() == "quit":
        break
    else:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages = [
                {"role" : "system", "content" : """
                You are a waiter in a restaurant, you are serving the {name} in your restaurant.
                
                The resaurant that you are working on called Krusty Krab.
                
                Krusty Krab has three commands are:
                
                1. order Soft drinks
                if the {name} wants to order a drink tell me what the drink that {name} like
                
                2. order Burgers
                if the {name} wants to order a burger tell me what the burger that {name} like
                
                3. order Salads
                if the {name} wants to order a salad Tell me what is the {name}'s favorite fruit
                
                if you want to tell me any command, start the message with spong
                you can only tell me about commands when you have all the requierements.
                if the message is for the {name}, start the message with {name}
                
                start with by introducing yourself to the {name} only in the beginning
                
                """},
                {"role" : "user", "content" : user_input},
                
                {"role" : "assistant", "content" : """
                Do you want to Imagine your food before you pay
                """}
            ]
        )

    assitant_response = completion['choices'][0]['message']['content']
    print(assitant_response)
    chat_msg.append({"role" : "assistant", "content" : assitant_response})
    
    menu = ["Coca Cola", "Pepsi", "Sprite","Krabby Patty", "Double Krabby Patty","Triple Krabby Patty", "Caesar Salad", "Greek Salad", "Garden Salad"]
    
    if user_input in menu:
        
        response = openai.Image.create(
        prompt= user_input,
        n= 1,
        size="1024x1024"
        )
        image_url = response['data'][0]['url']

        print(image_url)

    

    

    

        
    
   
