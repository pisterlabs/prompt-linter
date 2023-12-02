from gensim.models import CoherenceModel
from gensim.topic_coherence import segmentation
from configs import LDAConfig, LSIConfig, GenerationConfig
from lda_model import LDAModel
import numpy as np
from evaluation.metrics import TopicCoherence
from lsi_model import LSIModel
from run_generation import generate_unconditional_text
from topical_generation import generate_lsi_text, generate_lda_text, ctrl_text, pplm_text


def eval_topic_coherence(model, config, generation_config, prompt_file, out_file, topic_index=0):
    num_prompt_words = 4
    text_length = 10

    topic_coherence = TopicCoherence(config)
    coherences = []
    with open(prompt_file) as fr:
        for i, line in enumerate(fr):
            if len(coherences) > 200:
                break
            prompt_text = " ".join(line.split()[:num_prompt_words])
            if model == "gpt2":
                text = generate_unconditional_text(prompt_text=prompt_text,
                                                   generation_config=generation_config)
            if model == "lsi":
                text, _, _ = generate_lsi_text(prompt_text=prompt_text,
                                         selected_topic_index=topic_index,
                                         lsi_config=config,
                                         generation_config=generation_config)
            elif model == "lda":
                text, _, _ = generate_lda_text(prompt_text=prompt_text,
                                         selected_topic_index=topic_index,
                                         lda_config=config,
                                         generation_config=generation_config
                                         )
            elif model == "ctrl":
                text = ctrl_text(prompt_text=prompt_text,
                                 topic="Opinion",
                                 generation_config=generation_config)

            elif model == "pplm":
                text = pplm_text(prompt_text=prompt_text,
                          topic="military",
                          generation_config=generation_config)

            if len(text.split()) > text_length:
                print(text)
                coherence = topic_coherence.get_coherence(text)
                coherences.append(coherence)

    with open(out_file, 'w') as file_writer:
        file_writer.write(str(coherences) + "\n")
        file_writer.write("mean coherence: " + str(np.mean(coherences)) + "\n")

    #print(coherences)
    print("mean coherence: ", np.mean(coherences), np.std(coherences))


if __name__ == "__main__":
    lda_config_file = "/home/rohola/codes/topical_language_generation/configs/alexa_lda_config.json"
    lda_config = LDAConfig.from_json_file(lda_config_file)

    # lsi_config_file = "/home/rohola/codes/topical_language_generation/configs/alexa_lsi_config.json"
    # lsi_config = LSIConfig.from_json_file(lsi_config_file)

    #prompt_file = "/media/rohola/data/sample_texts/films/film_reviews.txt"

    #generation_config_file = "/home/rohola/codes/topical_language_generation/configs/ctrl_generation_config.json"
    # generation_config_file = "/home/rohola/codes/topical_language_generation/configs/generation_config.json"
    # generation_config = GenerationConfig.from_json_file(generation_config_file)
    # prompt_file = "/media/data2/rohola_data/film_reviews.txt"
    # out_file = "/home/rohola/codes/topical_language_generation/results/topic_coherence/topic_coherence_gpt_lda_sparsemax.txt"

    ##Unconditional GPT2
    generation_config_file = "/home/rohola/codes/topical_language_generation/configs/generation_config.json"
    generation_config = GenerationConfig.from_json_file(generation_config_file)
    prompt_file = "/media/data2/rohola_data/film_reviews.txt"
    out_file = "/home/rohola/codes/topical_language_generation/results/topic_coherence/topic_coherence_gpt2_sparsemax.txt"

    eval_topic_coherence(model="pplm",
                         config=lda_config,
                         generation_config=generation_config,
                         prompt_file=prompt_file,
                         out_file=out_file,
                         topic_index=1)
