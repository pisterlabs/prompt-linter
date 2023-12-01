import random
from datetime import datetime

import openai

from app.database import artgenstyle_sql, artgenurl_sql, db, artgen_sql


def select_random_elements(genres_list=None):
    if genres_list:
        all_genres = [record.genre_name for record in
                      artgen_sql.query.filter(artgen_sql.genre_name.in_(genres_list)).all()]
    else:
        all_genres = [record.genre_name for record in artgen_sql.query.all()]

    if not all_genres:
        raise ValueError("No genres available in the database.")

    genre_name = random.choice(all_genres)
    record = artgen_sql.query.filter_by(genre_name=genre_name).first()

    # Fetch all art styles along with their corresponding gen_style values
    all_styles = [(style.art_style, style.gen_style) for style in artgenstyle_sql.query.all()]
    if not all_styles:
        raise ValueError("No styles available in the database.")

    art_style, gen_style = random.choice(all_styles)

    columns = [
        record.place_1, record.place_2, record.place_3, record.place_4, record.place_5,
        record.role_1, record.role_2, record.role_3, record.role_4, record.role_5,
        record.item_1, record.item_2, record.item_3, record.item_4, record.item_5,
        record.symbol_1, record.symbol_2, record.symbol_3, record.symbol_4, record.symbol_5,
        record.event_1, record.event_2, record.event_3, record.event_4, record.event_5
    ]

    random_attribute = random.choice([col for col in columns if col])

    return genre_name, art_style, random_attribute, gen_style


def generate_dalle_prompt(genre_name, art_style, random_attribute):
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system",
             "content": "You are a helpful creative assistant. You will be provided with randomized attributes relating to music genres and artistic styles. Help the user craft the most optimal and most detailed possible DALL-E prompt. Under no circumstances will you return anything besides the prompt."},
            {"role": "user",
             "content": f"Craft this into a topically specific DALL-E prompt that uses comprehensive descriptions that embody the characteristics of: {art_style}. You should ensure you visually incorporate the {genre_name} music genre, and ensure {random_attribute} is the visual focal point. Do nothing but send back the prompt."}
        ],
        temperature=1,
        max_tokens=500,
    )
    original_prompt = response.choices[0].message.content
    print(original_prompt)
    additional_string = f"Avoid using text as a major element. Emphasize the use of '{art_style}.' Maintain the thematic element of {genre_name}."
    final_prompt = original_prompt + additional_string
    return final_prompt


def generate_images_dalle(prompt, style, quality='standard'):
    image_data = []

    for i in range(3):
        generation_response = openai.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality=quality,
            style=style,
            n=1,
        )
        generated_image_url = generation_response.data[0].url

        generated_revised_prompt = generation_response.data[0].revised_prompt

        image_data.append({
            "url": generated_image_url,
            "revised_prompt": generated_revised_prompt
        })

    return image_data


def generate_and_save_images(playlist_id, refresh=False, genre_list=None, quality='standard'):
    current_time = datetime.utcnow()

    if refresh:
        last_entry = artgenurl_sql.query.filter_by(playlist_id=playlist_id).order_by(
            artgenurl_sql.timestamp.desc()).first()
        if not last_entry:
            raise ValueError("No previous entry found for the given playlist_id to refresh.")

        genre_name = last_entry.genre_name
        art_style = last_entry.art_style
        random_attribute = last_entry.random_attribute
        gen_style = artgenstyle_sql.query.filter_by(art_style=art_style).first().gen_style if art_style else "vivid"
    else:
        genre_name, art_style, random_attribute, gen_style = select_random_elements(genre_list)

    prompt = generate_dalle_prompt(genre_name, art_style, random_attribute)
    image_data = generate_images_dalle(prompt, gen_style, quality=quality)

    image_urls = []
    revised_prompts = []

    for data in image_data:
        url = data['url']
        revised_prompt = data['revised_prompt']
        image_urls.append(url)
        revised_prompts.append(revised_prompt)

        # Store the generated or refreshed data
        new_artgenurl_record = artgenurl_sql(
            url=url,
            genre_name=genre_name,
            art_style=art_style,
            random_attribute=random_attribute,
            prompt=revised_prompt,  # Store the revised prompt instead of the original
            playlist_id=playlist_id,
            timestamp=current_time
        )

        db.session.merge(new_artgenurl_record)

    db.session.commit()

    return image_urls, revised_prompts
