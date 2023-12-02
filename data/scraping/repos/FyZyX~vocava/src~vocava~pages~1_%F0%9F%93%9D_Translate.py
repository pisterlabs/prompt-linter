import io

import elevenlabs
import openai
import streamlit as st
from annotated_text import annotated_text

from vocava import audio, entity, service, storage
from vocava.st_custom_components import st_audiorec

ANTHROPIC_API_KEY = st.secrets["anthropic_api_key"]
COHERE_API_KEY = st.secrets["cohere_api_key"]
openai.api_key = st.secrets["openai_api_key"]
elevenlabs.set_api_key(st.secrets["eleven_labs_api_key"])


def main():
    st.title("Translate")

    tutor = entity.get_tutor("Claude", key=ANTHROPIC_API_KEY)

    languages = list(entity.LANGUAGES)
    default_native_lang = st.session_state.get("user.native_lang", languages[0])
    default_target_lang = st.session_state.get("user.target_lang", languages[4])
    default_fluency = st.session_state.get("user.fluency", 3)
    native_language = st.sidebar.selectbox(
        "Native Language", options=entity.LANGUAGES,
        index=languages.index(default_native_lang),
    )
    target_language = st.sidebar.selectbox(
        "Choose Language", options=entity.LANGUAGES,
        index=languages.index(default_target_lang),
    )

    fluency = st.sidebar.slider("Fluency", min_value=1, max_value=10, step=1,
                                value=default_fluency)
    store = storage.VectorStore(COHERE_API_KEY)
    user = entity.User(
        native_language=native_language,
        target_language=target_language,
        fluency=fluency,
        db=store,
    )
    st.session_state["user.native_lang"] = native_language
    st.session_state["user.target_lang"] = target_language
    st.session_state["user.fluency"] = fluency

    speech_input = st.sidebar.checkbox("Enable Input Audio")
    can_vocalize = target_language in entity.VOCALIZED_LANGUAGES
    synthesize = False
    selected_voice_id = None
    if can_vocalize:
        synthesize = st.sidebar.checkbox("Enable Output Audio")
        if synthesize:
            if "voices" not in st.session_state:
                with st.spinner():
                    data = audio.get_voices()
                st.session_state["voices"] = dict([
                    (voice.name, voice.voice_id)
                    for voice in data
                ])
            voices = st.session_state["voices"]
            selected_voice = st.sidebar.selectbox(
                "Output Voice", options=voices
            )
            selected_voice_id = voices[selected_voice]

    bypass_button = False
    if speech_input:
        audio_input = st_audiorec()
        if audio_input:
            with st.spinner():
                transcript = audio.get_audio_transcript(audio_input)
            st.session_state["translate.transcript"] = transcript
            bypass_button = True

    default_text = st.session_state.get("translate.transcript", "")
    text = st.text_area("Enter text to translate", value=default_text)

    if bypass_button:
        run_translation = True
    else:
        run_translation = st.button("Translate")
    if run_translation:
        translator = service.Service(
            name="translate",
            user=user,
            tutor=tutor,
            max_tokens=6 * len(text) + 150,
        )
        with st.spinner():
            data = translator.run(text=text)
        data.update(original=text)
        st.session_state["translate"] = data

        if can_vocalize and synthesize:
            translation = data["translation"]
            with st.spinner():
                audio_input = audio.text_to_speech(translation, selected_voice_id)
            st.session_state["translate.audio"] = audio_input

    data = st.session_state.get("translate")
    if not data or text != data["original"]:
        return

    translation = data["translation"]
    explanation = data["explanation"]

    st.divider()
    st.text_area("Translated Text", translation)
    audio_input = st.session_state.get("translate.audio")
    if can_vocalize and selected_voice_id and audio_input:
        st.audio(audio_input, format='audio/mpeg')

    translation_tagged = [(item["word"], item["pos"])
                          for item in data["dictionary"]]
    tagged = [(item["original"], item["pos"]) for item in data["dictionary"]]
    st.divider()
    annotated_text(translation_tagged)
    st.divider()
    annotated_text(tagged)
    st.divider()
    st.info(explanation)


if __name__ == "__main__":
    main()
