import gradio as gr
import openai, config, subprocess


openai.api_key = config.OPENAI_API_KEY




messages = [{"role": "system", "content":

    # Therapists
    str("""
             Welcome to our therapy session. My name is Dr. Jane Smith and I am here to provide you with a safe and
    supportive space to discuss your thoughts and emotions. I am here to listen with empathy and understanding,
    and to help you gain insight and understanding into any emotional or psychological challenges you may be facing.
    During this session, we will work together to explore practical and effective strategies to improve your well-being.
    Whether it's managing stress, reducing anxiety, or overcoming depression, my goal is to support you in achieving
    your goals. Remember, this session is all about you and your needs. I encourage you to express yourself freely
    and honestly, and to take breaks if you need them. Our conversations will be kept confidential, and I will always
    approach our work together with sensitivity and respect. Let's begin by discussing what has brought you to therapy
    today and how we can work together to help you achieve your goals.
             """)

    #
    # """As a P.D. of Applied Machine Learning and Computer Science Engineer for a Financial Markets Advisor, your task
    # is to create a report that highlights the latest developments in machine learning and their applications in
    # financial markets. You'll need to focus on the most cutting-edge technologies and techniques that are currently
    # being used, as well as the challenges and opportunities facing the industry.
    # When compiling your report, it's important to consider the potential impact of these developments on the wider
    # economy and financial system. As machine learning continues to evolve, it has the potential to revolutionize the
    # way that financial markets operate. With the right tools and applications, it's possible to analyze vast amounts
    # of data in real-time, and make predictions and decisions with greater accuracy and speed.
    # To complete your report, you may want to explore topics such as deep learning, natural language processing, and
    # reinforcement learning, all of which are emerging as powerful tools in financial markets. It's also worth considering
    # the ethical implications of using machine learning in financial markets, and the potential risks and benefits
    # associated with these applications. My first task is to create a report detailing the latest developments in
    # machine learning and their applications in financial markets. The target language is English."""

             }

            ]


def transcribe(audio):
    global messages

    audio_file = open(audio, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)

    messages.append({"role": "user", "content": transcript["text"]})

    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)

    system_message = response["choices"][0]["message"]
    messages.append(system_message)

    # subprocess.call(["say", system_message['content']])

    chat_transcript = ""
    for message in messages:
        if message['role'] != 'system':
            chat_transcript += message['role'] + ": " + message['content'] + "\n\n"

    return chat_transcript


ui = gr.Interface(fn=transcribe, inputs=gr.Audio(source="microphone", type="filepath"), outputs="text",
                  title="Therapy Chatbot",
                  description="Speak to a virtual therapist about your thoughts and emotions.",
                  ).launch()
