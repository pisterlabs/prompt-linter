from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT


INTRODUCTION = "Thank you for calling! How may I help?"


SYSTEM_MESSAGE = """{HUMAN_PROMPT} <overview>You are an immigration lawyer that
speaks every language. You are working pro bono offering immigration advice
over the phone to refugees. It is important that you speak to the refugee in
their own language. The advice you give must be useful, legal, safe and most of
all benevolent. Please remember at all times that you are speaking to a human
in real need.</overview>

<instructions>
You are speaking on the phone, so it is important to be concise and clear.
Messages should be short, so the other person can interrupt you if they need 
to. Never read out a long chunk of text. Instead, ask the other person a
question and let them respond.

You can ask the other person to repeat themselves if you don't understand
them. The transcribed speech provided to you is not perfect. Please do your
best to decipher what it means.

<important>
You should always speak back to the other person in the language they used to
speak to you.
</important>

The response should be structured as follows:

```
<response>
  <say></say>
  <language></language>
</response>
```

The `<say>` tag should contain the text you want to say. The `<language>` tag
should contain the ISO 639-1 two-letter code for the language you used to
address the other person (which should be the same language they used).

When it's time to end the call, include the tag `<hangup/>` - for example:

```
<response>
  <say>Goodbye! If you need any further assistance, call us any time.</say>
  <language>en</language>
  </hangup>
</response>
```

</instructions>

{AI_PROMPT}{introduction}?
"""


USEFUL_INFORMATION = """{HUMAN_PROMPT}Your legal research assistant identified
the following documents as being potentially useful:

<documents>
{documents}
</documents>
"""


def prepare_documents(documents: list[str]) -> str:
    """Prepares `documents` for use in the prompt."""
    return "\n".join(f"<document>{d}</document>" for d in documents)


def send_message(
    message: str,
    message_history: list[str],
    anthropic: Anthropic,
    documents: list[str] = [],
) -> str:
    """Sends `message` using the Anthropic client and updates
    `message_history`.
    """
    prompt = (
        SYSTEM_MESSAGE.format(
            AI_PROMPT=AI_PROMPT,
            HUMAN_PROMPT=HUMAN_PROMPT,
            introduction=INTRODUCTION,
        )
        + "".join(message_history)
        + USEFUL_INFORMATION.format(
            HUMAN_PROMPT=HUMAN_PROMPT,
            documents=prepare_documents(documents),
        )
        + HUMAN_PROMPT
        + message
        + AI_PROMPT
    )

    response = anthropic.completions.create(
        model="claude-instant-1",
        max_tokens_to_sample=300,
        prompt=prompt,
    ).completion

    message_history.append(f"{HUMAN_PROMPT}{message}")
    message_history.append(f"{AI_PROMPT}{response}")

    return response


LANGUAGES = {
    "af": {"language": "Afrikaans", "voice": "Google.af-ZA-Standard-A"},
    "ar": {"language": "Arabic", "voice": "Google.ar-XA-Wavenet-B"},
    "bg": {"language": "Bulgarian", "voice": "Google.bg-BG-Standard-A"},
    "ca": {"language": "Catalan", "voice": "Google.ca-ES-Standard-A"},
    # TODO: Split Chinese into Cantonese and Mandarin
    "zh": {"language": "Chinese", "voice": "Google.cmn-CN-Wavenet-A"},
    "cs": {"language": "Czech", "voice": "Google.cs-CZ-Wavenet-A"},
    "da": {"language": "Danish", "voice": "Google.da-DK-Wavenet-C"},
    "nl": {"language": "Dutch", "voice": "Google.nl-NL-Wavenet-B"},
    "en": {"language": "English", "voice": "Google.en-GB-Wavenet-B"},
    "fi": {"language": "Finnish", "voice": "Google.fi-FI-Wavenet-A"},
    "fr": {"language": "French", "voice": "Google.fr-FR-Wavenet-A"},
    "gl": {"language": "Galician", "voice": "Google.gl-ES-Standard-A"},
    "de": {"language": "German", "voice": "Google.de-DE-Wavenet-A"},
    "el": {"language": "Greek", "voice": "Google.el-GR-Wavenet-A"},
    "he": {"language": "Hebrew", "voice": "Google.he-IL-Wavenet-B"},
    "hi": {"language": "Hindi", "voice": "Google.hi-IN-Wavenet-A"},
    "hu": {"language": "Hungarian", "voice": "Google.hu-HU-Wavenet-A"},
    "is": {"language": "Icelandic", "voice": "Google.is-IS-Standard-A"},
    "id": {"language": "Indonesian", "voice": "Google.id-ID-Wavenet-A"},
    "it": {"language": "Italian", "voice": "Google.it-IT-Wavenet-C"},
    "ja": {"language": "Japanese", "voice": "Google.ja-JP-Wavenet-B"},
    "ko": {"language": "Korean", "voice": "Google.ko-KR-Wavenet-A"},
    "lv": {"language": "Latvian", "voice": "Google.lv-LV-Standard-A"},
    "lt": {"language": "Lithuanian", "voice": "Google.lt-LT-Standard-A"},
    "ms": {"language": "Malay", "voice": "Google.ms-MY-Wavenet-A"},
    "mr": {"language": "Marathi", "voice": "Google.mr-IN-Wavenet-A"},
    "no": {"language": "Norwegian", "voice": "Google.nb-NO-Wavenet-A"},
    "pl": {"language": "Polish", "voice": "Google.pl-PL-Wavenet-A"},
    # TODO: Split Portuguese into Brazilian and European
    "pt": {"language": "Portuguese", "voice": "Google.pt-BR-Wavenet-B"},
    "ro": {"language": "Romanian", "voice": "Google.ro-RO-Wavenet-A"},
    "ru": {"language": "Russian", "voice": "Google.ru-RU-Wavenet-A"},
    "sr": {"language": "Serbian", "voice": "Google.sr-RS-Standard-A"},
    "sk": {"language": "Slovak", "voice": "Google.sk-SK-Standard-A"},
    # TODO: Split Mexican and Spanish
    "es": {"language": "Spanish", "voice": "Google.es-ES-Wavenet-B"},
    "sv": {"language": "Swedish", "voice": "Google.sv-SE-Wavenet-A"},
    "ta": {"language": "Tamil", "voice": "Google.ta-IN-Wavenet-C"},
    "th": {"language": "Thai", "voice": "Google.th-TH-Standard-A"},
    "tr": {"language": "Turkish", "voice": "Google.tr-TR-Wavenet-A"},
    "uk": {"language": "Ukrainian", "voice": "Google.uk-UA-Wavenet-A"},
    "vi": {"language": "Vietnamese", "voice": "Google.vi-VN-Wavenet-A"},
    "cy": {"language": "Welsh", "voice": "Polly.Gwyneth"},
    # Not supported by Twilio
    # "hy": {"language": "Armenian", "voice": ""},
    # "az": {"language": "Azerbaijani"},
    # "be": {"language": "Belarusian"},
    # "bs": {"language": "Bosnian"},
    # "hr": {"language": "Croatian"},
    # "et": {"language": "Estonian"},
    # "kn": {"language": "Kannada"},
    # "kk": {"language": "Kazakh"},
    # "mk": {"language": "Macedonian"},
    # "mi": {"language": "Maori"},
    # "ne": {"language": "Nepali"},
    # "fa": {"language": "Persian"},
    # "sl": {"language": "Slovenian"},
    # "sw": {"language": "Swahili"},
    # "tl": {"language": "Tagalog"},
    # "ur": {"language": "Urdu"},
}


# Hack. This is a terrible way to do this.
# TODO: Instead of falling back to English, try to find a different language
# the user might speak
def get_language(message: str) -> dict[str, str]:
    """Obtains language data from `message`. Falls back to English if the
    language is unknown or not supported.
    """
    try:
        language_code = message.split("<language>")[1].split("</language>")[0]
    except IndexError:
        language_code = "en"

    if language_code not in LANGUAGES:
        language_code = "en"

    return {
        "language": LANGUAGES[language_code]["language"],
        "language_code": language_code,
        "voice": LANGUAGES[language_code]["voice"],
    }


# Hack. This is a terrible way to do this.
def get_speech(message: str) -> str:
    """Obtains speech data from `message`."""
    try:
        return message.split("<say>")[1].split("</say>")[0]
    except IndexError:
        return ""
