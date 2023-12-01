from app.config import *
import openai
import aioduckdb
import jsonlines

# Create database 
async def create_connection():
    return await aioduckdb.connect('app/data.db')

async def create_fefe_mode_table():
    db = await create_connection()
    cursor = await db.cursor()
    await cursor.execute("""
CREATE TABLE IF NOT EXISTS fefe_mode (mode VARCHAR);
INSERT INTO fefe_mode VALUES ('when_called');
    """)
    await db.commit()
    await db.close()

async def change_fefe_mode(interaction,new_mode):
    db = await create_connection()
    cursor = await db.cursor()
    query = "UPDATE fefe_mode SET mode = ?"
    await db.execute(query, (new_mode,))
    await db.close()
    await interaction.response.send_message(f"Response mode set: {new_mode}")

async def get_fefe_mode():
    db = await create_connection()
    cursor = await db.cursor()
    await cursor.execute("SELECT mode FROM fefe_mode LIMIT 1")
    mode = await cursor.fetchall()
    mode = mode[0][0]
    await db.close()
    return mode

async def create_fefe_model_table():
    db = await create_connection()
    cursor = await db.cursor()
    await cursor.execute("""
CREATE TABLE IF NOT EXISTS fefe_model (model VARCHAR);
INSERT INTO fefe_model VALUES ('gpt-3.5-turbo');
    """)
    await db.commit()
    await db.close()
async def change_fefe_model(interaction,new_model):
    db = await create_connection()
    cursor = await db.cursor()
    query = "UPDATE fefe_model SET model = ?"
    await db.execute(query, (new_model,))
    await db.close()
    await interaction.response.send_message(f"Fefe's openAi model set: {new_model}")
async def get_fefe_model():
    db = await create_connection()
    cursor = await db.cursor()
    await cursor.execute("SELECT model FROM fefe_model LIMIT 1")
    model = await cursor.fetchall()
    model = model[0][0]
    await db.close()
    return model

async def create_chat_history_table():
    db = await create_connection()
    cursor = await db.cursor()
    await cursor.execute("""
CREATE SEQUENCE IF NOT EXISTS seq_chat_history_id START 1;
CREATE TABLE IF NOT EXISTS chat_history
    (id INTEGER PRIMARY KEY DEFAULT nextval('seq_chat_history_id'),
    jsonl TEXT NOT NULL,
    channel_id TEXT,
    channel_name TEXT,
    source TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
    """)
    await db.commit()
    await db.close()

# clear chat history
async def clear_chat_history_db():
    # Create a connection pool
    db = await create_connection()
    cursor = await db.cursor()
    await cursor.execute("DROP TABLE IF EXISTS chat_history")
    await db.commit()
    await db.close()

    await create_chat_history_table()
    
async def clear_listening():
    db = await create_connection()
    cursor = await db.cursor()
    await cursor.execute("""
DELETE FROM chat_history 
WHERE id NOT IN (
    SELECT id 
    FROM chat_history 
    WHERE source = 'listening' 
    ORDER BY timestamp DESC 
    LIMIT 4)
and source = 'listening'
    """)
    await db.commit()
    await db.close()

# Store listening
async def store_listening(bot,message):
    ctx = await bot.get_context(message)
    db = await create_connection()
    await store_prompt(db,json.dumps({
        'role':'user',
        'content':f"'{ctx.author.mention}': {message.content}"}),
                               message.channel.id,
                               message.channel.name,
                               'listening')
    await db.close()
    
# Function to store a prompt
async def store_prompt(db_conn,jsonl,channel_id,channel_name,source):
    cursor = await db_conn.cursor()
    await cursor.execute("""
INSERT INTO chat_history (jsonl,channel_id,channel_name,source) VALUES (?,?,?,?) 
""",(jsonl,channel_id,channel_name,source))
    await db_conn.commit()

async def create_memories():
    db = await create_connection()
    cursor = await db.cursor()
    await cursor.execute("""
CREATE TABLE IF NOT EXISTS memories
    (jsonl TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
    """)
    await db.commit()
    await db.close()

# clear memories
async def clear_memory_db():
    # Create a connection pool
    db = await create_connection()
    cursor = await db.cursor()
    await cursor.execute("DROP TABLE IF EXISTS memories")
    await db.commit()
    await db.close()

    await create_memories()


# Function to store a memory
async def store_memory(db,jsonl):
    cursor = await db.cursor()
    await cursor.execute("""
INSERT INTO memories (jsonl) VALUES (?) 
""",(jsonl,))
    await db.commit()
    
# Function to fetch the last few prompts. Used to provide chat history to openAi.
async def fetch_prompts(db,channel_id,limit,source=None):
    cursor = await db.cursor()
    # Fetch the last few rows from the table for the given channel_id
    await cursor.execute("""
    select jsonl
    from (
        SELECT jsonl,timestamp FROM chat_history
        WHERE channel_id in (?,'bot')
        ORDER BY timestamp DESC
        LIMIT ?
    ) AS subquery
    order by timestamp asc
    """, (channel_id, limit))
    
    # Fetch and load the json data from the selected rows
    rows = await cursor.fetchall()
    prompts = []
    for row in rows:
        json_data = json.loads(row[0])
        prompts.append(json_data)
    
    return prompts

# Function to fetch the last few prompts. Used to provide chat history to openAi.
async def fetch_dataviz_prompts(db,channel_id,limit,source=None):
    cursor = await db.cursor()
    # Fetch the last few rows from the table for the given channel_id
    await cursor.execute("""
    select jsonl
    from (
        SELECT jsonl,timestamp FROM chat_history
        WHERE channel_id = ?
        AND source in ('DATALL-E','interpreter')
        ORDER BY timestamp DESC
        LIMIT ?
    ) AS subquery
    order by timestamp asc
    """, (channel_id, limit))
    
    # Fetch and load the json data from the selected rows
    rows = await cursor.fetchall()
    prompts = []
    for row in rows:
        json_data = json.loads(row[0])
        prompts.append(json_data)
    
    return prompts

# Function to create reminder table
async def create_reminders():
    db = await create_connection()
    cursor = await db.cursor()
    await cursor.execute("""
CREATE TABLE IF NOT EXISTS reminders
    (username TEXT NOT NULL,
    time TIMESTAMP NOT NULL,
    note TEXT,
    channel_id TEXT,
    channel_name TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
    """)
    await db.commit()
    await db.close()
    
# Function to store a reminder
async def store_reminder(db,username,reminder_time,note,channel_id,channel_name):
    cursor = await db.cursor()
    await cursor.execute("""
INSERT INTO reminders (username,time,note,channel_id,channel_name) VALUES (?,?,?,?,?) 
""",(username,reminder_time,note,channel_id,channel_name))
    await db.commit()

# 'delete_reminder' function deletes a reminder from the 'reminders' table using the reminder_id.
async def delete_reminder(username,reminder_time,note,timestamp):
    conn = await create_connection()
    cursor = await conn.cursor()

    await cursor.execute('DELETE FROM reminders WHERE username = ? AND time = ? AND note = ? AND timestamp = ?', (username, reminder_time,note,timestamp))
    await conn.commit()
    await conn.close()
    
# 'fetch_due_reminders' function fetches all reminders that are due to be sent.
# It selects all reminders from the 'reminders' table where the reminder_time is less than or equal to the current timestamp.
async def fetch_due_reminders():
    conn = await create_connection()
    cursor = await conn.cursor()

    await cursor.execute('SELECT time, note,channel_id FROM reminders WHERE time <= CURRENT_TIMESTAMP')
    reminders = await cursor.fetchall()

    await conn.close()

    return reminders

# Clears out a user's reminders
async def clear_user_reminders(interaction):
    """Clears all reminders of the invoking user"""
    conn = await create_connection()
    cursor = await conn.cursor()
    # Delete records from the reminders table where username is ctx.author.name
    await cursor.execute("DELETE FROM reminders WHERE username=?", (str(interaction.user.name),))
    await conn.commit()
    await conn.close()
    embed=discord.Embed(
                #title="Sample Embed", 
                #url="https://realdrewdata.medium.com/", 
                description=f"Clear reminders.", 
                color=discord.Color.blue())
            
    embed.set_author(name=f"{interaction.user.name}",
                             icon_url=interaction.user.avatar)
    await interaction.response.send_message('All your reminders have been cleared.',embed=embed)

async def send_reminders(bot):
    conn = await create_connection()
    cursor = await conn.cursor()
    # Send out the reminders
    await cursor.execute('''
    SELECT 
        username, 
        time, 
        note, 
        channel_id, 
        channel_name,
        timestamp
    FROM reminders 
    WHERE time is not null''')
    
    reminders = await cursor.fetchall()
    
    for reminder in reminders: 
        username, reminder_time, note, channel_id, channel_name,timestamp = reminder
        
        if datetime.now() >= reminder_time:
            channel = bot.get_channel(int(channel_id))
            if channel is not None:
                await channel.send(note)
            else:
                print(f"Channel with ID {channel_id} not found.")

            await cursor.execute('DELETE FROM reminders WHERE username = ? AND timestamp = ?',
                                 (username, timestamp,))

    # Commit the changes here
    await conn.commit()

# Get list of channels
async def list_channels(bot):
    channel_names = []
    for guild in bot.guilds:
        for channel in guild.channels:
            channel_names.append(channel.name)
    return channel_names

# Get first text channel
async def get_first_text_channel(bot):
    for guild in bot.guilds:
        for channel in guild.channels:
            if isinstance(channel, discord.TextChannel):  # If you want text channels only
                return channel
    return None

############################################
# Useful Functions
############################################
# Code used to strip commentary from GPT response.
def extract_code(response_text):
    pattern = pattern = r"```(?:[a-z]*\s*)?(.*?)```\s*"
    match = re.search(pattern, response_text, re.DOTALL)
    if match:
        extracted_code = match.group(1) # Get the content between the tags\n",
    elif 'import' in response_text:
        extracted_code = response_text
    else:
        extracted_code = response_text
        print("No code found.")
    return extracted_code
    
async def gather_files_to_send(author_name):
    reg_string = "^[^.][a-zA-Z0-9_]+\.[a-zA-Z0-9_]+$"
    user_dir = f"app/downloads/{author_name}/"
    files = os.listdir(user_dir)
    files_to_send = [user_dir + x for x in files if re.search(reg_string,x)]
    files_to_send = [x for x in files_to_send if file_size_ok(x)==True]
    files_to_send = [discord.File(x) for x in files_to_send]
    return files_to_send

# Function to send response in chunks. Used to adhere to discord's 2000 character limit.
async def send_results(ctx, output, files_to_send=[],embed=None):
    chunk_size = 2000  # Maximum length of each chunk
    
    response = f'''
{output}
'''
    
    chunks = [response[i:i+chunk_size] for i in range(0, len(response), chunk_size)]
    
    if len(chunks) == 1:
        if embed:
            await ctx.send(chunks[0],files = files_to_send,embed=embed)
        else:
            await ctx.send(chunks[0],files = files_to_send)
    else:
        for chunk in chunks:
            if chunk != chunks[len(chunks)-1]:
                await ctx.send(chunk)
            else:
                if embed:
                    await ctx.send(chunk,files = files_to_send)
                else: 
                    await ctx.send(chunk,files = files_to_send,embed=embed)

# Function to clear the downloads folder
async def delete_music_downloads(bot):
    def delete_everything(directory):
        for file_name in os.listdir(directory):
            file_path = os.path.join(directory, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                delete_everything(file_path)
                os.rmdir(file_path)
    
    directory = 'app/downloads'
    if os.path.exists(directory) and os.path.isdir(directory):
        delete_everything(directory)

# Check tokens
def check_tokens(jsonl, model,completion_limit):
    enc = tiktoken.encoding_for_model(model)
    messages_string = json.dumps(jsonl)
    tokens = len(enc.encode(messages_string))

    if model == 'gpt-3.5-turbo':
        token_limit = 4096
    if model == 'gpt-4':
        token_limit = 8000
    
    while tokens > token_limit - completion_limit:
        # Remove the first two messages from the JSON list
        jsonl = jsonl[2:]
        
        # Update the messages string and token count
        messages_string = json.dumps(jsonl)
        tokens = len(enc.encode(messages_string))
    
    return jsonl

# Used to abide by Discord's 2000 character limit.

async def send_followups(interaction, output, files=[],embed=None):
    chunk_size = 2000  # Maximum length of each chunk
    
    response = f'''
{output}
'''
    
    chunks = [response[i:i+chunk_size] for i in range(0, len(response), chunk_size)]
    
    if len(chunks) == 1:
        if embed:
            await interaction.followup.send(chunks[0],files = files,embed=embed)
        else:
            await interaction.followup.send(chunks[0],files = files)
    else:
        for chunk in chunks:
            if chunk != chunks[len(chunks)-1]:
                await interaction.followup.send(chunk)
            else:
                if embed:
                    await interaction.followup.send(chunk,files = files,embed=embed)
                else: 
                    await interaction.followup.send(chunk,files = files)
            
# Used to abide by Discord's 2000 character limit.
async def send_chunks(ctx, text):
    chunk_size = 2000  # Maximum length of each chunk

    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    for chunk in chunks:
        await ctx.send(chunk)

# check file size
def file_size_ok(file_path):
    # Get file size in bytes
    file_size = os.path.getsize(file_path)
    # Convert to megabytes
    file_size_mb = file_size / (1024 * 1024)
    # Check if file size is less than 25MB
    if file_size_mb < 25:
        return True
    else:
        return False    
            
# Generate an image
async def generate_image(text):
    response = openai.Image.create(
      prompt=text,
      n=1,
      size="1024x1024"
    )
    image_url = response['data'][0]['url']
    return image_url

# Create a user's directory
async def create_user_dir(author_name):
    dir_name = f'app/downloads/{author_name}/'
    # Check if the directory exists
    if not os.path.exists(dir_name):
        # If the directory does not exist, create it
        os.makedirs(dir_name)
    return dir_name
    
# Delete files from a user's directory.
async def delete_files(author_name):
    directory = f"app/downloads/{author_name}/"

    # Check if the directory exists
    if os.path.exists(directory):
        # Delete all files in the directory
        shutil.rmtree(directory)
        print(f"All files in {directory} have been deleted.")
    else:
        print("The directory does not exist.")
