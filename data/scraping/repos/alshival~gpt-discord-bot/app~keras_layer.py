from app.config import *

max_sequence_length = None
word_to_index = None
model = None
label_to_index = None 

async def train_keras():
    global max_sequence_length
    global word_to_index
    global label_to_index
    global model
    ############################################
    # Training Data
    ############################################
    
    # When adding new features that require task assignment in the Keras model,
    # make sure to update the list of labels accordingly in conjunction with the changes made in app/keras_layer.py.
    
    sample_data = {
        """
        organize this code for me with comments.
        
        **`config.py`**
        ```
        import openai
        db_name = 'app/data.db' # Where you wish to store the bot data. 
        
        
        openai.api_key = os.environ.get("OPENAI_API_KEY") # Set up the OpenAI API. The key is stored as an environment variable for security reasons. 
        
        google_api_key = os.environ.get("google_api_key") # Set up the Google Youtube Data API key. For youtube searching and playback.
        
        epochs = 25 # Number of training epochs for the keras layer.
        
        prompt_table_cache_size = 200 # Number of prompts stored in the local SQLite database. The table is truncated when the bot starts up.
        
        #############################################
        # KERAS LAYER - Task Assignment
        #############################################
        
        # Define the labels for task assignment in the Keras model
        keras_labels = ['other', 'reminder', 'youtube']
        
        # Note for developers:
        # When adding new features that require task assignment in the Keras model,
        # make sure to update this list of labels accordingly in conjunction with the changes made in app/keras_layer.py.
        ```
        """: 'other',
        'Remind me to pick up the kids in 45 minutes': 'reminder',
        'Remind me to turn in my homework at midnight': 'reminder',
        "What's your favorite color?": 'other',
        'Remind me to call mom at 3pm.': 'reminder',
        'Can you send me the report?': 'other',
        'Buy tickets for the concert': 'other',
        'Remind me to pick up some milk later.': 'reminder',
        'Remind us to study for the final exam next week.': 'reminder',
        'remind me to fix my essay in an hour.': 'reminder',
        'remind me to throw my shoes away in three days.': 'reminder',
        'About how many atoms are there in the universe?': 'other',
        'Remind me to feed the dog at 6pm.': 'reminder',
        "Let's play a game.": 'other',
        'Can you turn on the TV?': 'other',
        'Remind me to check my email after dinner.': 'reminder',
        "What's the weather like today?": 'other',
        'Remind me to take my medicine at 8am.': 'reminder',
        "Let's order pizza for dinner.": 'other',
        'Remind me to water the plants tomorrow morning.': 'reminder',
        'How many planets are there in our solar system?': 'other',
        "Remind me to book a doctor's appointment next Monday.": 'reminder',
        'Can you find a good recipe for spaghetti bolognese?': 'other',
        'Remind me to charge my phone.': 'reminder',
        'Who won the basketball game last night?': 'other',
        'Remind me to finish my online course this weekend.': 'reminder',
        'Can you tell me a joke?': 'other',
        'Remind me to call the plumber tomorrow.': 'reminder',
        "What's the capital of Australia?": 'other',
        "Remind me to renew my driver's license next month.": 'reminder',
        'Remind me to buy groceries on the way home.': 'reminder',
        'What time is it?': 'other',
        'Remind me to pick up my dry cleaning this afternoon.': 'reminder',
        'Who won the Oscar for Best Picture last year?': 'other',
        'Remind me to check the oven in 30 minutes.': 'reminder',
        'Can you recommend a good book?': 'other',
        'Remind me to schedule a team meeting for next Tuesday.': 'reminder',
        "What's the score of the baseball game?": 'other',
        'Remind me to fill up the car with gas tomorrow.': 'reminder',
        'Can you find the fastest route to the airport?': 'other',
        'Remind me to pay the electric bill by the end of the week.': 'reminder',
        'Who is the president of the United States?': 'other',
        'Remind me to update my resume this weekend.': 'reminder',
        'Can you play my favorite song?': 'other',
        'Remind me to check in for my flight 24 hours before departure.': 'reminder',
        'What are the ingredients in a Caesar salad?': 'other',
        "Remind me to bring my umbrella if it's going to rain tomorrow.": 'reminder',
        'How do you make a margarita?': 'other',
        'In a short answer, tell me how to write pi/2 as an infinite sum.': 'other',
        'Play Spirit in the Sky.': 'youtube',
        'I want to listen to Jay-Z': 'youtube',
        'Can you find a video explaining how quantum computers work?': 'youtube',
        'Play the phantom of the opera.': 'youtube',
        'play a song from the guardians of the galaxy soundtrack.': 'youtube',
        'can you find a video about the mathematics of neural networks?': 'youtube'}

    messages = list(sample_data.keys())
    labels = list(sample_data.values())
    
    ############################################
    # Import labeled prompts
    ############################################
    #---------------------------------------------
    # Check if `data.db` has labeled_prompts table
    conn = sqlite3.connect(db_name)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    # Execute the query to retrieve the table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    tables = [table[0] for table in tables]
    if 'labeled_prompts' in tables:
        cursor.execute("select prompt,label from labeled_prompts")
        rows = cursor.fetchall()
        if rows is not None:
            dict_rows = [dict(row) for row in rows]
            messages += [row['prompt'] for row in dict_rows]
            labels += [row['label'] for row in dict_rows]
        
    conn.close()
    
    ############################################
    #  Preprocessing
    ############################################
    vocab = set(' '.join(messages).lower().split())
    vocab_size = len(vocab)
    word_to_index = {word: index for index, word in enumerate(vocab)}
    max_sequence_length = max(len(message.split()) for message in messages)
    #---------------------------------------------
    # Convert sentences to numerical sequences
    X = np.zeros((len(messages), max_sequence_length))
    for i, message in enumerate(messages):
        words = message.lower().split()
        for j, word in enumerate(words):
            X[i, j] = word_to_index[word]
      #---------------------------------------------
    # Convert labels to numerical values
    label_to_index = dict(zip(keras_labels,range(len(keras_labels))))#{'reminder': 0, 'other': 1, 'youtube': 2}
    y = np.array([label_to_index[label] for label in labels])

    # Define the model
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=50, input_length=max_sequence_length))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(len(keras_labels), activation='softmax'))  # assuming you have 3 classes

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) 


    # Train the model
    model.fit(X, y, epochs=epochs, batch_size=1, verbose=1)

############################################
#  Classification Function
############################################
async def classify_prompt(input_string):
    global max_sequence_length
    global word_to_index
    global model

    # Preprocess the input string
    new_sequence = np.zeros((1, max_sequence_length))
    words = input_string.lower().split()
    for j, word in enumerate(words):
        if word in word_to_index:
            new_sequence[0, j] = word_to_index[word]

    # Make prediction
    prediction = model.predict(new_sequence)
    predicted_index = np.argmax(prediction)  # Change here, get the index of max value
    index_to_label = {v: k for k, v in label_to_index.items()}
    predicted_label = index_to_label[predicted_index]

    return predicted_label

