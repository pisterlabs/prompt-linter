import PySimpleGUI as Sg
import openai
import logging.config
import time


def create_layout(religions):
    layout = [
        [Sg.Text('Parameters Setting', font=('Helvetica', 20))],
        [Sg.Text('Nice Mode:'), Sg.Checkbox('ON', key="-niceModeOn-")],
        [Sg.Text('Discussion between:'), Sg.Checkbox('human & 1 robot', key="-humanRobot-"),
         Sg.Checkbox('2 robots', key="-twoRobots-")],
        [Sg.Text('Religion:'), Sg.Text('robot 1'), Sg.Combo(religions, key="-religion1-"), Sg.Text('robot 2'),
         Sg.Combo(religions, key="-religion2-")],
        [Sg.Text('Discussion:', font=('Helvetica', 35))],
        [Sg.Output(key="-print-", size=(180, 35))],
        [Sg.Button("Start"), Sg.Button("Clear")]
    ]
    return layout


def gpt3_call(prompt_text):
    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt_text,
        temperature=0.7,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0.4,
        presence_penalty=0.4,
        stop=["Human:", "\n"],
    )
    story = response['choices'][0]['text']
    if str(story) == "" or not str(story):
        story = "I don't know, can you elaborate?"
    return prompt_text + str(story)


def run(config, intro=None):
    logger = logging.getLogger(__name__)
    logger.info('Start logging:')
    religions = list(config['religions'].keys())
    layout = create_layout(religions)
    window = Sg.Window('Deus Ex Machina Demo', layout=layout, scaling=2)
    while True:
        event, values = window.read()
        if event in (Sg.WIN_CLOSED, 'Quit'):
            break
        if event == 'Clear':
            window["-print-"].update("")
        if event == 'Start' and values["-humanRobot-"]:
            window.Element('-religion2-').Update(visible=False)
            window.Element('-twoRobots-').Update(visible=False)
            if intro is None:
                intro = config["religions"][values["-religion1-"]]
            window["-print-"].update(intro)
            question = Sg.popup_get_text("Your input: ")
            restart_sequence = "Human: "
            start_sequence = values["-religion1-"]
            prompt_text = f'{intro}\n{restart_sequence}: {question}\n{start_sequence}:'
            intro = gpt3_call(prompt_text)
            window["-print-"].update(intro)
        if event == 'Start' and values["-twoRobots-"]:
            window.Element('-humanRobot-').Update(visible=False)
            restart_sequence = values["-religion1-"]
            start_sequence = values["-religion2-"]
            question = Sg.popup_get_text("Subject: ")
            intro = "The {} religion is {}. \n" \
                    "The {} religion is {}.\n" \
                    "The following is a conversation between {} and {} about {}."\
                .format(values["-religion1-"],
                        config["religions"][values["-religion1-"]],
                        values["-religion2-"],
                        config["religions"][values["-religion2-"]],
                        values["-religion2-"],
                        values["-religion1-"],
                        question)
            #prompt_text = f'{intro}\n{start_sequence}: '
            prompt_text = "{}\n{}: ".format(intro, start_sequence)
            intro = gpt3_call(prompt_text)
            window["-print-"].update(intro)
            turn = True
            for _ in range(12):
                if turn:
                    prompt_text = intro + "\n" + restart_sequence + ": "
                    turn = False
                else:
                    prompt_text = intro + "\n" + start_sequence + ": "
                    turn = True
                intro = gpt3_call(prompt_text)
                window["-print-"].update(intro)
                window.Refresh()
