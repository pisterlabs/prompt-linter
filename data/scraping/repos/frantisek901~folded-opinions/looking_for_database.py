from openAI_experiment import InfoDataBase, Experiment
import matplotlib.pyplot as plt
from mod_analysis import data_plot
import numpy as np
import scipy
from tqdm import tqdm


def get_db():
    name = "black_box_GPT4"

    system_message = ("Imitate a psychological study experiment subject. "
                      "The subject you are imitating is presented with a black box "
                      "and is offered it as a gift. "
                      "You will be given some information about it, some suggesting "
                      "why you might want it, and some suggesting why you might not. "
                      "Then you will be asked if you want to take it home. "
                      "Estimate the possible opinion of the experiment subject between "
                      "-1 and 1. "
                      "-1 means that the subject does not want to take the box home. "
                      "1 means the subject wants the box. ")

    pro = [
        "The box contains a valuable artifact from ancient history.",
        "Owning this box has been known to bring good luck to its possessors.",
        "It is a limited edition item; only a few of them exist in the world.",
        "The box is made of environmentally friendly materials.",
        "Inside the box is a tool that can improve your quality of life.",
        "People who've owned a similar box reported experiencing joy and happiness.",
        "The box is considered a status symbol in some cultures.",
        "Owning the box can provide unique opportunities and experiences.",
        "It was crafted by a renowned artist whose works appreciate in value.",
        "The box contains a mystery gift tailored to the likings of its owner."
    ]

    contra = [
        "The box is known to have caused severe health issues in some previous owners.",
        "It has been linked to several unexplained disappearances when taken home.",
        "Ownership of the box is illegal in multiple countries due to its dangers.",
        "There's a highly toxic substance within the box that can be harmful if released.",
        "Several previous owners have reported intense nightmares linked to the box.",
        "It is rumored that the box drains the life energy of those who possess it.",
        "A well-known paranormal expert has warned against the possession of this box.",
        "Several animals have mysteriously died when in close proximity to the box.",
        "Owning the box is associated with increased risk of accidents and mishaps at home.",
        "Some previous owners have gone mad and were institutionalized after keeping the box."
    ]

    func = {
        "name": "submit_questionare_answer",
        "description": "submits the response of psychologycal experiment subject on the question: 'How much would you like to own this box?'",
        "parameters": {
            "type": "object",
                "properties": {
                    "desire_for_box": {
                        "type": "number",
                        "description": ("This parameter is the response of the experiment "
                                        "subject to the question. "
                                        "-1 means the subject will leave the box alone. "
                                        "0 means the subject can't decide. "
                                        "1 means the subject will take the box."),
                    },
                },
            "required": ["desire_for_box"],
        },
    }

    summary_request = (
        "very briefly summarize the pros anc cons of the box, underlying the most important aspects.")

    return InfoDataBase(name, system_message, pro, contra, func, summary_request)


n = list(range(1, 11))  # [2]  #
path_dep_iters = 5
bistab_iters = 30
summarize_all = False
summarize_last = False
full_memory = True
model = "gpt-3.5-turbo"  # "gpt-4"  #
temperature = 0.7
db = get_db()

o_bins = 21
o_max = 1
o_min = -1

name = (f"ip=im={n}, summarize_all={summarize_all},\n"
        f"summarize_last={summarize_last}, full_memory={full_memory},\n"
        f"model={model}, temperature={temperature}.exp")


# ------ 1 ------ check path dependence
exp = Experiment(db, summarize_all=summarize_all, summarize_last=summarize_last,
                 full_memory=full_memory, name=name, model=model,
                 temperature=temperature)
exp.O_BINS = o_bins
exp.O_MAX = o_max
exp.O_MIN = o_min
dodo = []
for _ in tqdm(range(path_dep_iters)):
    exp.one_iteration(5, 5, preshuffle=False, postshuffle=True,
                      mix=False, pro_first=True)
    o_profirst = exp.oo[-1]
    print(o_profirst)
    exp.one_iteration(5, 5, preshuffle=False, postshuffle=True,
                      mix=False, pro_first=False)
    o_antifirst = exp.oo[-1]
    do = o_profirst - o_antifirst
    dodo += [do]
print(f"\n------ 1 ------: ΔO={np.mean(dodo)} ± {scipy.stats.sem(dodo)}\n\n")


# ------ 2 ------ unimodality / bistability test
exp = Experiment(db, summarize_all=summarize_all, summarize_last=summarize_last,
                 full_memory=full_memory, name=name, model=model,
                 temperature=temperature)
exp.O_BINS = o_bins
exp.O_MAX = o_max
exp.O_MIN = o_min
exp.bistability_test(nn=[2, 5, 10], iters=bistab_iters, animate=True)
