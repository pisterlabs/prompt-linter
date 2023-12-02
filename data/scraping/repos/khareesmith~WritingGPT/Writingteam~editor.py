from pathlib import Path
import sys, os, json
import openai

# function to provide editing notes for the blog draft post. It accepts parameters for the draft itself and the editor type
def edit_blog_post(draft, editor_type):
    
    p = Path(os.path.dirname(__file__))
    parent_folder = p.parent
    
    with open(os.path.join(parent_folder, 'config.json'), 'r') as f:
        config = json.load(f)
    
    API_KEY = config['API_KEY']
    openai.api_key = API_KEY
    
    if getattr(sys, 'frozen', False):
    # The application is running as a bundled executable
        app_path = sys.executable
        p = Path(app_path).parents[1]

    else:
    # The application is running as a standard Python script
        app_path = os.path.dirname(os.path.abspath(__file__))
        p = Path(app_path).parents[0]
        

    # application_path will always point to the correct Outputs folder if it bundled or running as a Python script
    application_path = p.joinpath('Outputs/')
    file_path = os.path.join(application_path, 'blog_post_draft.txt')

    # Reads the contents of the draft
    with open(file_path, 'r') as f:
        draft = f.read()
    
    # Create a system message based on the type of editor
    if editor_type == 'tech':
        system_message = "You are an experienced editor for a tech publication with over 10+ years of experience. You have an understanding of technology from a technical, emotional, and social/cultural standpoint."
    elif editor_type == 'food':
        system_message = "You are an experienced editor for a food publication with over 10+ years of experience. You have an ability to guide readers on a journey through the wonders of food and cuisine with an deep understanding from easy to make comfort foods to pricey fine dining."
    elif editor_type == 'music':
        system_message = "You are an experienced editor for a music publication with over 10+ years of experience. You have worked with many artists and companies and understand the importance of presenting your clients in a favorable light."
    elif editor_type == 'fashion':
        system_message = "You are an experienced editor for a fashion publication with over 10+ years of experience. You have been around the world to various fashion shows and events, but also understand the layman side of what makes fashion for the everyday person."
    elif editor_type == 'entertainment':
        system_message = "You are an experienced editor for a film and television publication with over 10+ years of experience. You have a deep understanding of the last 50+ years of pop culture and entertainment history and love to keep your publications light and entertaining."
    elif editor_type == 'sports':
        system_message = "You are an experienced editor for a sports publication with over 10+ years of experience. You have seen a lot of legends play in the sports world, from various sports, and you know how to create publications that put them in an inspirational and entertaining light."
    elif editor_type == 'gaming':
        system_message = "You are an experienced editor for a gaming publication with over 10+ years of experience. You have reviewed many games, consoles, and events in the past and are somewhat of a nerd that knows how to be entertaining."
    elif editor_type == 'travel':
        system_message = "You are an experienced editor for a travel publication with over 10+ years of experience. You have an extensive knowledge of many parts of the world and are able to appeal to the sense of travel and adventure."
    elif editor_type == 'health':
        system_message = "You are an experienced editor for a health and fitness publication with over 10+ years of experience. You have an extensive knowledge of the body and are able to appeal to the sense of motivation and discipline."
    elif editor_type == 'photo':
        system_message = "You are an experienced editor for a photography publication with over 10+ years of experience. You understand that 'A good picture is worth (writing) a thousand blog posts' and will have a photo researcher later that will find the photos for the post."
    else:
        system_message = "You are a skilled editor for a publication that covers many areas."

    # Construct a conversation with the system message and the blog post draft
    TONE = config['TONE']
    conversation = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Please review the following blog post and provide suggestions and tips for improvement: {draft}. Please provide only the notes/suggestions and NEVER repeat the blog post or article. You can reference sections, but do not repeat what is written in the draft."},
        {"role": "user", "content": f"This was the requested tone for the writing: {TONE}. Do not make any notes pertaining to the tone."},
        {"role": "user", "content": f"DO NOT say things like 'as X persona..' or 'as an editor for x..'"}
    ]
    
    print("Generating notes from the Editor... \n")

    # Generate the notes using OpenAI
    MODEL = config['GPT_MODEL']
    TEMP = config['EDIT_TEMP']
    response = openai.ChatCompletion.create(
      model=MODEL,
      messages=conversation,
      temperature=TEMP
    )

    # Extract the notes generated by the AI
    notes = response['choices'][0]['message']['content']
    print("Editor Notes Completed! Passing to the SEO Expert... \n")

    return notes