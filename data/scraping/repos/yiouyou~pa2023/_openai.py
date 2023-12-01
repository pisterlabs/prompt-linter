def audio_transcribe(_audio):
    import openai
    _audio_file = open(_audio, "rb")
    _transcribe = openai.Audio.transcribe("whisper-1", _audio_file)
    return _transcribe["text"]

