import os
import numpy as np
import sys
from pathlib import Path
sys.path.extend(['/om/user/ehoseini/sent_sampling', '/om/user/ehoseini/sent_sampling'])
from utils.data_utils import SENTENCE_CONFIG
from utils.data_utils import load_obj, SAVE_DIR, UD_PARENT, RESULTS_DIR, LEX_PATH_SET, save_obj,ANALYZE_DIR
from utils import extract_pool
from utils.optim_utils import optim_pool, low_dim_project
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import seaborn
from tqdm import tqdm
from matplotlib.pyplot import GridSpec
import pandas as pd
from pathlib import Path
import torch
from utils import make_shorthand
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
import openai
import base64
import pandas as pd
if __name__ == '__main__':
    # these are the fixed set
    ds_random_sentence=pd.read_csv(os.path.join(ANALYZE_DIR,'ds_parametric','sent,G=best_performing_pereira_1-D=ud_sentencez_ds_random_100_edited_selected_textNoPeriod_final.csv'))
    ds_min_sentence=pd.read_csv(os.path.join(ANALYZE_DIR,'ds_parametric','sent,G=best_performing_pereira_1-D=ud_sentencez_ds_min_100_edited_selected_textNoPeriod_final.csv'))
    ds_max_sentence=pd.read_csv(os.path.join(ANALYZE_DIR,'ds_parametric','sent,G=best_performing_pereira_1-D=ud_sentencez_ds_max_100_edited_selected_textNoPeriod_final.csv'))

    ds_dict={'ds_random':ds_random_sentence,'ds_min':ds_min_sentence,'ds_max':ds_max_sentence}
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.organization = "org-kiIqN9X8obEBpYxCuLS0l3qs"
    for key, val in ds_dict.items():
        sentences=[x[1] for x in val.values]
        for kk in range(10,len(sentences)):
            prompt=sentences[kk]
            #skipp1=np.logical_and(kk==76, key=='ds_random')
            skipp1=prompt=='This morning, Pope Francis met Mark Zuckerberg and his wife at Casa Santa Marta'
            skips2=prompt=='It was huge and scared the crap out of me'
            skips3=prompt=='He absorbed her while she failed to absorb him'
            skipp4=prompt=='This has not yet penetrated the thinking of the Western World'
            skipps=np.any([skipp1,skips2,skips3,skipp4])
            for index in range(3):
                image_file = Path(ANALYZE_DIR, 'ds_parametric', f'dalle_{key}',
                              f"{key}_sent_{kk}_render_{index}_{prompt}.png")
            # if image file exist ignore
                if image_file.exists() or skipps:
                    print(f"{key}, Skipping {prompt}")
                    continue
                    # print which sentence is skipped
                else:
                    print(f"{key}, imaging {prompt}")
                    response = openai.Image.create(prompt=prompt,n=3,size="1024x1024",response_format="b64_json")
                    for index, image_dict in enumerate(response["data"]):
                        image_data = base64.b64decode(image_dict["b64_json"])
                    # new image
                        new_image_file = Path(ANALYZE_DIR, 'ds_parametric', f'dalle_{key}',
                                      f"{key}_sent_{kk}_render_{index}_{prompt}.png")
                # make sure the directory exists
                        new_image_file.parent.mkdir(parents=True, exist_ok=True)
                        with open(new_image_file.__str__(), mode="wb") as png:
                            png.write(image_data)
