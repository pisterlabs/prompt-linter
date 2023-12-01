from AutoEditor.audio_builder.audio_transcriber import Audio
import openai
import nltk


def split_text_gpt(text, max_tokens_per_part):
    # Divide the text into sentences
    sentences = nltk.sent_tokenize(text)

    # Group sentences into parts based on the token limit
    parts = []
    current_part = ""
    for sentence in sentences:
        # If adding the next sentence doesn't exceed the token limit, add it to the current part
        if len((current_part + " " + sentence).split()) <= max_tokens_per_part:
            current_part += " " + sentence
        else:
            # Otherwise, save the current part and start a new one
            parts.append(current_part.strip())
            current_part = sentence

    # Add the last part if it's not empty
    if current_part.strip():
        parts.append(current_part.strip())

    return parts


def generate_keywords_gpt(part):
    sentences = nltk.sent_tokenize(part)

    keywords = []
    for sentence in sentences:
        if len(sentence.split()) <= 4096:  # The sentence is within GPT-4's token limit
            prompt = f"Résumez le texte suivant en un ensemble de mots-clés: {sentence}"
            response = openai.Completion.create(engine="text-davinci-003", prompt=prompt, temperature=0.5,
                                                max_tokens=200)

            # We assume that the GPT-4 response text contains the keywords separated by commas
            sentence_keywords = response.choices[0].text.strip().split(',')
            for keyword in sentence_keywords[:2]:  # Get the first two keywords
                keywords.append(keyword)
                if len(keywords) == 3:  # We have reached the limit of 3 keywords
                    break

        if len(keywords) == 3:
            break

    return keywords


def Helper(options, logger, open_ai_key):
    nltk.download('punkt')
    openai.api_key = open_ai_key

    # Lire le contenu du fichier du script vidéo
    with open(options["script_path"], "r") as script_file:
        video_script = script_file.read()

    """
    Transcription audio
    """
    audioReader = Audio(options["audio_path"])

    logger.write('Début de la transcription audio, veuillez patienter...', 'info')

    # Commencer la transcription
    audioReader.convert_m4a_to_mp3()
    audioReader.transcribe()

    logger.write('Transcription audio terminée.', 'info')

    # Lire les transcriptions de l'audio
    transcriptions = audioReader.get_transcriptions

    # Calculer la vitesse de lecture moyenne en fonction des transcriptions
    total_duration = sum(transcription["end_time"] - transcription["start_time"] for transcription in transcriptions)
    average_reading_speed = len(video_script.split()) / total_duration

    # Nombre maximum de mots dans chaque partie en fonction de la vitesse de lecture moyenne et de la limite de 20 secondes
    max_words_per_part = int(average_reading_speed * 10)

    # Utiliser GPT pour diviser le texte en parties
    parts = split_text_gpt(video_script, max_words_per_part)

    # Générer les mots clés pour chaque partie et calculer les temps de début et de fin de chaque partie
    parts_keywords_times = []
    total_parts = len(parts)
    part_transcriptions = []
    transcription_index = 0

    for i, part in enumerate(parts):
        # Générer les mots clés pour chaque partie
        keywords = generate_keywords_gpt(part)  # Générer les mots clés

        part_transcriptions = []
        part_word_count = 0
        while transcription_index < len(transcriptions) and part_word_count < len(part.split()):
            part_transcriptions.append(transcriptions[transcription_index])
            part_word_count += len(transcriptions[transcription_index]['word'].split())
            transcription_index += 1

        # Calculer le temps de début et de fin de la partie
        start_time = part_transcriptions[0]['start_time']
        end_time = part_transcriptions[-1]['end_time'] + part_transcriptions[-1]['pause_after']

        # Calculate progress and emit progress update
        progress = int((i + 1) / total_parts * 100)

        # Stocker les temps de début et de fin de la partie
        part_dict = {"part": part, "keywords": keywords, "start_time": start_time, "end_time": end_time}
        parts_keywords_times.append(part_dict)

        # Réinitialiser les transcriptions de la partie pour la partie suivante
        part_transcriptions = []

    return parts_keywords_times


