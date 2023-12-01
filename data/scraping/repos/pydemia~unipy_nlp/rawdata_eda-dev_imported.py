#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'scripts'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# # EDA & Preprocessing
#
# * Loading the data
# * Formatting & Organizing
# * Preprocessing
#   - Replacing & Substituting(on loading)
#   - Sentencing
#   - Spell Correction
#   - Tokenizing: comparision (bpemb, cohesion, pos-tag)
#%% [markdown]
# ## Setting Environments

#%%
import os
import re
import sys
import json
import random
import warnings
import itertools as it
import unidecode
from unicodedata import normalize
from glob import glob
from pprint import pprint
from collections import OrderedDict
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

fpath = '../data'

# warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

mpl.rcParams['font.family'] = 'NanumGothic'
mpl.rcParams['font.serif'] = 'NanumMyeongjo'
mpl.rcParams['font.sans-serif'] = 'NanumGothic'
mpl.rcParams['font.monospace'] = 'NanumGothicCoding'

[f.name for f in fm.fontManager.ttflist if 'nanum' in f.name.lower()]

font_dict = {
    path.split('/')[-1][:-4]: path
    for path in fm.get_fontconfig_fonts()
    if 'nanum' in path.lower().split('/')[-1]
}
font_path = font_dict['NanumBarunGothic']

#%% [markdown]
# ## Loading the data

#%%
glob('../*')


#%%
flist = glob(f'{fpath}/saveasnew/*.xlsx')
fname_list = [re.findall(r'.*rawdata_(.+)_saveasnew.xlsx', s)[0] for s in flist]
fdict = {n: f for n, f in zip(fname_list, flist)}
fdict


#%%
def read_xlsx(filename):
    print(f"Loading '{filename}'...")
    res = pd.read_excel(
        filename,
        sheet_name=None,
        header=None, names=['content'],
        dtype=str,
        encoding='utf-8',
        # encoding='ascii',
        # encoding='ISO-8859-1',
    )
    return res


#%%
def read_xlsx_usymp(filename):
    print(f"Loading '{filename}'...")
    tmp = pd.read_excel(
        '../data/saveasnew/rawdata_usymphony_saveasnew.xlsx',
        sheet_name='Clientes',
        # header=None, names=['content'],
        dtype=str,
        encoding='utf-8',
        # encoding='ascii',
        # encoding='ISO-8859-1',
        # na_values='nan',
        # keep_default_na=True,
    ).replace('nan', np.NaN)

    # tmp = full_df['Clientes']
    tmp.columns = [
        c.replace('Unnamed: ', 'un_')
        if 'Unnamed: ' in c
        else c
        for c in tmp.columns
    ]

    tmp['title_agg'] = tmp.iloc[:, :4].apply(
        lambda x: x.dropna().max(),
        axis=1,
    )
    tmp['reply_agg'] = (
        tmp.loc[:, tmp.columns[tmp.columns.str.contains('un_')][3:]]
        .apply(lambda x: x.dropna().max(), axis=1)
    )

    tmp['title_yn'] = tmp['조회수'].notnull()

    tmp['title'] = tmp.loc[tmp['title_yn'] == True, ['title_agg']]
    tmp['body'] = tmp.loc[tmp['title_yn'].shift(1) == True, ['title_agg']]
    tmp['body'] = tmp['body'].shift(-1)

    idx_srs = tmp['title'].dropna().index.to_series()

    idx_start = idx_srs + 2

    idx_end =idx_srs.shift(-1)
    idx_end.iloc[-1] = tmp.index[-1]

    idx_range_df = (
        pd.DataFrame(
            [idx_start, idx_end],
            index=['start', 'end']
        )
        .T
    ).astype(np.int)

    tmp['reply_idx'] = idx_range_df.apply(
        lambda x: list(range(x['start'], x['end'])),
        axis=1,
    )

    def collect_reply(df, reply_idx):
        if reply_idx not in ([], np.NaN):
            res = df.loc[reply_idx, 'reply_agg'].dropna().tolist()
        else:
            res = []
        return res

    tmp['reply'] = (
        tmp['reply_idx']
        .apply(lambda x: collect_reply(tmp, x))
        .apply(lambda x: '\n'.join(x))
    )
#     tmp['reply_joined'] = (
#         tmp['reply']
#         .apply(lambda x: '\n'.join(x))
#     )
    tmp = tmp[['title', 'body', 'reply']].dropna().reset_index(drop=True)
    # tmp['content'] = tmp.values.sum(axis=1)
    tmp['content'] = (
        pd.Series(
            tmp[tmp.columns.drop('reply')]
            .fillna('')
            .values
            .tolist()
        )
        .str.join('\n')
    )

    return OrderedDict({'Clientes': tmp})


#%%
ddict = {
    n: read_xlsx(f)
    if 'rawdata_usymphony_saveasnew.xlsx' not in f
    else read_xlsx_usymp(f)
    for n, f in fdict.items()
}


#%%
ddict.keys()

#%% [markdown]
# ## Formatting & Organizing

#%%
def refine_content(df):

    def repl(mat, ri=re.compile(r'([\.\,\'\"]+\s*)')):
        return ri.sub(r'', ''.join(mat.group(0, 2)))
        # return ri.sub(r'', mat.group(0, 2))
        # return ri.sub(r'', mat.group(0))

    def replace_str(s):
        s = re.sub(r'[(re\:)(FW\:)]+', r'', s, flags=re.IGNORECASE)
        s = re.sub(r'40(.*)41', r'(\1)', s)
        s = re.sub(r'39(.*)', r'`\1', s)
        s = re.sub(r'\t|\a', r' ', s)
        s = re.sub(r'\r\n|\r', r'\n', s)
        s = re.sub(r'http\S+', '', s)
        s = re.sub(r'\[답변\]|\[열린소리\]|\[분실물\D*\]', r'', s)
        s = re.sub(r'\(\?\)', r' ', s)
        s = re.sub(r'[▷◇△▲▽▼★]', r' ', s)
        s = re.sub(r'[<]*[-=]{2,}[>]*', r' ', s)
        s = re.sub(r'물류/서비스사업', r'물류서비스사업', s)
        s = re.sub(r'^(.+)([\s]*[(From)(Sent)]\: .+Subject\:)(.+)', repl, s, flags=re.IGNORECASE)
        s = normalize('NFKD', s)
        return s

    def conditional_replace(row):
        if len(row) > 0:
            res = [replace_str(s) for s in row]
        else:
            res = row
        return res

    col_list = df.columns[~df.columns.str.contains('_refined')]
    print(f'{col_list.tolist()}')
    for col in col_list:
        if isinstance(df[col][0], str):
            df[f'{col}_refined'] = (
                df[col]
                .str.replace(r'[(re\:)(FW\:)]+', r'', flags=re.IGNORECASE)
                .str.replace(r'40(.*)41', r'(\1)')
                .str.replace(r'39(.*)', r'`\1')
                .str.replace(r'\t|\a', r' ')
                .str.replace(r'\r\n|\r', r'\n')
                .str.replace(r'http\S+', '')
                .str.replace(r'\[답변\]|\[열린소리\]|\[분실물\D*\]', r'')
                .str.replace(r'\(\?\)', r' ')
                .str.replace(r'[▷◇△▲▽▼]', r' ')
                .str.replace(r'[<]*[-=]{2,}[>]*', r' ')
                .str.replace(r'물류/서비스사업', r'물류서비스사업')
                .str.replace(r'^(.+)([\s]*[(From)(Sent)]\: .+Subject\:)(.+)', repl, flags=re.IGNORECASE)
                .apply(lambda s: normalize('NFKD', s))
            )
        elif isinstance(df[col][0], list):
            df[f'{col}_refined'] = df[col].apply(conditional_replace)

    return df


#%%
def convert_excel_to_dict(key, xlsx_loaded):

    if isinstance(xlsx_loaded, OrderedDict):
        for sheet in xlsx_loaded:
            print(f'{key}: {sheet}')
            xlsx_loaded[sheet] = refine_content(xlsx_loaded[sheet])

    else:
        print(f'{key}')
        xlsx_loaded = refine_content(xlsx_loaded)

    return xlsx_loaded


#%%
rdict = {k: convert_excel_to_dict(k, d) for k, d in ddict.items()}

#%% [markdown]
# ### Dump to `json`
#%% [markdown]
# #### Nested Objects

#%%
_tmp_dict = {
    table_nm: {
        sheet_nm: sheet_contents.T.to_dict(orient='records')[0]
    }
    for table_nm, table_contents in rdict.items()
    for sheet_nm, sheet_contents in table_contents.items()
}


#%%
for table_nm, table_contents in _tmp_dict.items():
    for sheet_nm, sheet_contents in table_contents.items():
        print(type(sheet_contents), len(sheet_contents))


#%%
with open('../data/rawdata_cpred.json', 'w', encoding='utf8') as jfile:
    json.dump(_tmp_dict, jfile, ensure_ascii=False)


#%%
with open('../data/rawdata_cpred.json', 'r', encoding='utf8') as jfile:
    _test_dict = json.loads(jfile.read())


#%%
for t_idx, (table_nm, table_contents) in enumerate(_test_dict.items()):
    for s_idx, (sheet_nm, sheet_contents) in enumerate(table_contents.items()):
        print(type(sheet_contents), len(sheet_contents))
        print(t_idx, s_idx)
    if max(t_idx, s_idx) > 2:
        break

#%% [markdown]
# #### Flat Objects

#%%
_tmp_gen = (
    {
        'table_nm': table_nm,
        'sheet_nm': sheet_nm,
        'contents': contents,
    }
    for table_nm, table_contents in rdict.items()
    for sheet_nm, sheet_contents in table_contents.items()
    for contents in sheet_contents.T.to_dict(orient='records')[0].values()
)
# _tmp_dict = {idx: value_dict for idx, value_dict in enumerate(_tmp_gen)}
_tmp_list = list(_tmp_gen)


#%%
for t_idx, (table_nm, table_contents) in enumerate(_test_dict.items()):
    for s_idx, (sheet_nm, sheet_contents) in enumerate(table_contents.items()):
        print(type(sheet_contents), len(sheet_contents))
        print(t_idx, s_idx)
    if max(t_idx, s_idx) > 2:
        break

#%% [markdown]
# #### Dump it

#%%
with open('../data/rawdata_cpred_flatted.json', 'w', encoding='utf8') as jfile:
    json.dump(_tmp_dict, jfile, ensure_ascii=False)


#%%
with open('../data/rawdata_cpred_flatted.json', 'w', encoding='utf8') as jfile:
    json.dump(_tmp_dict, jfile, ensure_ascii=False)


#%%
with open('../data/rawdata_cpred_flatted.json', 'r', encoding='utf8') as jfile:
    _test_dict = json.loads(jfile.read())


#%%
str(_test_dict)[:500]


#%%
aa = list(_test_dict.values())


#%%
with open('../data/rawdata_cpred_flatted.json', 'w', encoding='utf8') as jfile:
    for row in _test_dict.values():
        jfile.write(str(row) + '\n')


#%%
aa[:10]


#%%
for t_idx, (table_nm, table_contents) in enumerate(_test_dict.items()):
    for s_idx, (sheet_nm, sheet_contents) in enumerate(table_contents.items()):
        print(type(sheet_contents), len(sheet_contents))
        print(t_idx, s_idx)
    if max(t_idx, s_idx) > 2:
        break

#%% [markdown]
# ## Preprocessing
#%% [markdown]
# ### Glimsping

#%%
klist = list(rdict.keys())


#%%
[(k, len(v)) for k, v in rdict.items()]


#%%
def rm_suffix(s):
    return s.replace('_refined', '')

ldict = {
    key: {
        '18년' if sheet == 'Sheet1' else sheet
        : (
            df.loc[:, df.columns.str.contains('_refined')]
            .rename(rm_suffix, axis=1)
            .to_dict(orient='list')
        )
        for sheet, df in xlf.items()
    }
    for key, xlf in rdict.items()
}

#%%
# def concat_nested_list(nested_list):
#     return sum(nested_list, [])

content_only = {
    key: sum(
        [sum(list(dic.values()), []) for sheet, dic in xlf.items()],
        [],
    )
    for key, xlf in ldict.items()
}

full_content = sum(list(content_only.values()), [])
fcont = sum([s.split('\n') for s in full_content], [])
fcont = list(
    set(filter(lambda x: x not in (r'', r'\n'), fcont))
)


#%%
print(
    len(list(filter(lambda x: isinstance(x, str), full_content))),
    len(full_content),
    len(fcont),
)

#%% [markdown]
# ### Replacing & Substituting
#%% [markdown]
# ```
# [답변], [열린소리], [분실물\D*],
# ♬, <>, (), [], {}, *, \', \", ·, \.+, \!+, \?+, à, ->,
# \D/\D
# ```
#
# ```py
# re.sub(r'[_·à♬\<\>\(\)\[\]\{\}\*\'\"\.\!\?\-\+], r'')
# ```
#%% [markdown]
# * Normal Form Selection

#%%
jcont = '\n'.join(fcont)

NF_TYP = 'NFC'  # {'NFC', 'NFKC', 'NFD', 'NFKD'}
jcont = normalize(NF_TYP, jcont)


#%%
len(jcont)

#%% [markdown]
# * Recursive Substitutor

#%%
def loop_substitutor(joined_str, pattern_list):

    if len(pattern_list) > 0:
        pattern, target = pattern_list.pop(0)
        joined_str = re.sub(pattern, target, joined_str)
        return loop_substitutor(joined_str, pattern_list)
    else:
        return joined_str


#%%
ptn_a_s = re.compile(
    r'[\_\·\♬\<\>\(\)\[\]\{\}\*\'\"\-\+\~\|⓪①②③④⑤⑥⑦⑧⑨⑩]+' # \à\è
)
ptn_a_t = r' '
ptn_b_s = re.compile(r'[\^]{2,}')
ptn_b_t = r'\^\^\n'
ptn_c_s = re.compile(r'([^\s]+[음임함됨요(니다)])[\W]{1,}')
ptn_c_t = r'\1\n'
ptn_d_s = re.compile(r'[\.{2,}\!\?]+[\s]+|[\s]{3,}')
ptn_d_t = r'\n'
ptn_e_s = re.compile(r'[\s]{2,}|[(nbsp)]+')
ptn_e_t = r' '
ptn_f_s = re.compile(r'( [\s\D]+/[\s\w]+/sk[;]*)')
ptn_f_t = r''
ptn_g_s = re.compile(r'[(subject:)|(re:)]+')
ptn_g_t = r''

ptn_list = [
    (ptn_a_s, ptn_a_t),
    (ptn_b_s, ptn_b_t),
    (ptn_c_s, ptn_c_t),
    (ptn_d_s, ptn_d_t),
    (ptn_e_s, ptn_e_t),
    (ptn_f_s, ptn_f_t),
    (ptn_g_s, ptn_g_t),
]

replaced = loop_substitutor(jcont, ptn_list)
sentenced = replaced.lower().split('\n')

ptn_z = re.compile(
    r'^((\d{4} [\d]{1,2}:\d{2} [ap]{1}m[\s]{0,1}to:){0,1} .+)'
)
ptn_z_t = r''
ptn_y = re.compile(
    r'([\s\D]+/[\s\w]+/sk[;]*)'
)
ptn_y_t = r''

sentenced = [re.sub(ptn_z, ptn_z_t, s) for s in sentenced]
sentenced = [re.sub(ptn_y, ptn_y_t, s) for s in sentenced]

# sentenced = list(set(sentenced))
sentenced = (
    pd.Series(sentenced)
    .drop_duplicates()
    .loc[lambda x: (10 < x.str.len()) & (x.str.len() <= 500)]
    .tolist()
)


#%%
len(sentenced)

#%% [markdown]
# ### Spelling Correction

#%%
SPELLCHECK_OK = False
if SPELLCHECK_OK:

    from spellchecker import SpellChecker
    spell = SpellChecker(language='kr')

    # find those words that may be misspelled
    misspelled = spell.unknown(tmp.split(' '))

    for word in misspelled:
        print(
            # Get the one `most likely` answer
            '%s: %s, %s' % (
                word,
                spell.correction(word),
                spell.candidates(word),
            ),
              # Get a list of `likely` options
            sep='\n',
        )

#%% [markdown]
# ### Tokenizing

#%%
import gensim

#%% [markdown]
# #### NER
#%% [markdown]
# #### POS Tagging
#%% [markdown]
# ```sh
# sudo apt-get install curl
# bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)
# ```
#
# ```sh
# source activate <>
# git clone https://bitbucket.org/eunjeon/mecab-python-0.996.git
# python setup.py build
# python setup.py install
# ```
#%% [markdown]
# ##### User-Defined DIctionary

#%%
udf_token = pd.DataFrame(
    [
        ['워라밸', 'T'],
        ['dt', 'F'],
        ['d/t', 'F'],
        ['dt총괄', 'T'],
        ['dt 총괄', 'T'],
        ['총괄', 'T'],
        ['digital transformation', 'T'],
        ['deep change', 'F'],
        ['deepchange', 'F'],
        ['happy hour', 'F'],
        ['해피 아워', 'F'],
        ['해피아워', 'F'],
        ['dt 총괄', 'T'],
        ['comm', 'T'],
        ['comm.', 'T'],
        ['mbwa', 'F'],
        ['캔미팅', 'T'],
        ['can meeting', 'T'],
        ['fu', 'T'],
        ['f/u', 'T'],
        ['모니터링', 'T'],
        ['dw', 'F'],
        ['d/w', 'F'],
        ['vwbe', 'F'],
        ['supex', 'F'],
        ['수펙스', 'F'],
        ['tm', 'T'],
        ['top', 'T'],
        ['탑', 'T'],
        ['its', 'F'],
        ['bottom up', 'T'],
        ['top down', 'T'],
        ['의사결정', 'T'],
        ['의사 결정', 'T'],
        ['self design', 'T'],
        ['self-design', 'T'],
        ['딜리버리', 'F'],
        ['delivery', 'F'],
        ['pt', 'F'],
        ['장표', 'F'],
        ['kpi', 'F'],
        ['hr', 'T'],
        ['h/r', 'T'],
        ['기업문화', 'F'],
        ['하이닉스', 'F'],
        ['이노베이션', 'T'],
        ['skt', 'F'],
        ['bm', 'T'],
        ['pm', 'T'],
        ['프로젝트', 'F'],
        ['pjt', 'F'],
        ['rm', 'T'],
        ['r/m', 'T'],
        ['culture', 'F'],
        ['cs', 'F'],
        ['c/s', 'F'],
        ['culture survey', 'F'],
        ['컬처 서베이', 'F'],
        ['컬쳐 서베이', 'F'],
        ['idp', 'F'],
        ['역량개발', 'T'],
        ['스탭', 'T'],
        ['스텝', 'T'],
        ['경영지원', 'T'],
        ['skcc', 'F'],
        ['sk cc', 'F'],
        ['sk cnc', 'F'],
        ['sk c&c', 'F'],
        ['ski', 'F'],
        ['이노베이션', 'T'],
        ['하이닉스', 'F'],
        ['텔레콤', 'T'],
        ['skh', 'F'],
        ['플래닛', 'T'],
        ['skp', 'F'],
        ['skc', 'F'],
        ['홀딩스', 'F'],
        ['sk 주식회사', 'F'],
        ['sk주식회사', 'F'],
        ['sk홀딩스', 'F'],
        ['sk주식회사 c&c', 'F'],
        ['sk 주식회사 c&c', 'F'],
        ['sk주식회사 cc', 'F'],
        ['sk주식회사cc', 'F'],
        ['self design', 'T'],
        ['selfdesign', 'T'],
        ['self-design', 'T'],
        ['경영협의회', 'F'],
        ['경영 협의회', 'F'],
        ['사업대표', 'F'],
        ['현장경영', 'T'],
        ['gtm', 'T'],
        ['vdi', 'F'],
        ['cloud-z', 'F'],
        ['cloudz', 'F'],
        ['cloud z', 'F'],
        ['cloud-edge', 'F'],
        ['cloudedge', 'F'],
        ['cloud edge', 'F'],
        ['클라우드', 'F'],
        ['사내 시스템', 'T'],
        ['사내시스템', 'T'],
        ['단기 성과', 'F'],
        ['단기성과', 'F'],
        ['watson', 'T'],
        ['왓슨', 'T'],
        ['유심포니', 'F'],
        ['선거운동', 'T'],
        ['연봉체계', 'F'],
        ['포괄 임금제', 'F'],
        ['포괄임금제', 'F'],
        ['장기 투자', 'F'],
        ['장기투자', 'F'],
        ['구성원 의사', 'F'],
        ['스마트워크', 'F'],
        ['스마트 워크', 'F'],
        ['smartwork', 'F'],
        ['여유시간', 'T'],
        ['여유 시간', 'T'],
        ['우리 회사', 'F'],
        ['우리회사', 'F'],
        ['외부 사이트', 'F'],
        ['외부사이트', 'F'],
        ['전사공지', 'F'],
        ['전사 공지', 'F'],
        ['점심시간', 'T'],
        ['점심 시간', 'T'],
        ['제조사업', 'T'],
        ['전략사업', 'T'],
        ['금융사업', 'T'],
        ['its사업', 'T'],
        ['its 사업', 'T'],
        ['물류서비스사업', 'T'],
        ['물류/서비스사업', 'T'],
        ['업무체계', 'F'],
        ['업무 체계', 'F'],
    ],
    columns=['word', 'last_yn'],
).drop_duplicates(subset='word')


#%%
udf_token['0'] = udf_token['word']
udf_token['1'] = 0
udf_token['2'] = 0
udf_token['3'] = 0
udf_token['4'] = 'NNG'
udf_token['5'] = '*'
udf_token['6'] = udf_token['last_yn']
udf_token['7'] = udf_token['word']
udf_token['8'] = '*'
udf_token['9'] = '*'
udf_token['10'] = '*'
udf_token['11'] = '*'
udf_token['11'] = '*'


#%%
udf_token.columns.str.isnumeric()


#%%
udf_token_mecab = udf_token.loc[:, udf_token.columns.str.isnumeric()]


#%%
"""
표층형 (표현형태)	0	0	0	품사 태그	의미 부류	종성 유무	읽기	타입	첫번째 품사	마지막 품사
서울	          0	0	0	  NNG	지명	T	서울	*	*	*	*
불태워졌	     0	0	0	 VV+EM+VX+EP	*	T	불태워졌	inflected	VV	EP	*	불태우/VV/+어/EC/+지/VX/+었/EP/
해수욕장      	0	0	0	 NNG	 	T	해수욕장	Compound	*	*	해수/NNG/+욕/NNG/+장/NNG/*
"""


#%%
# udf_token_mecab.to_csv('/tmp/mecab-ko-dic-2.1.1-20180720/user-dic/nnp.csv', header=False, index=False)
udf_token_mecab.to_csv('~/Downloads/mecab-ko-dic-2.1.1-20180720/user-dic/nnp.csv', header=False, index=False)


#%%
# !/tmp/mecab-ko-dic-2.1.1-20180720/tools/add-userdic.sh
# !cd /tmp/mecab-ko-dic-2.1.1-20180720;make install
get_ipython().system('~/Downloads/mecab-ko-dic-2.1.1-20180720/tools/add-userdic.sh')
get_ipython().system('cd ~/Downloads/mecab-ko-dic-2.1.1-20180720;sudo make install')


#%%
aa = sorted(sentenced, key=len, reverse=True)
tt = '\n'.join(aa)


#%%
from konlpy.tag import Kkma, Mecab, Okt

# tagger = Kkma()
tagger = Mecab()


#%%
tag_list = [
    '체언 접두사', '명사', '한자', '외국어',
    '수사', '구분자',
    '동사',
    '부정 지정사', '긍정 지정사',
]
tagset_wanted = [
    tag
    for tag, desc in tagger.tagset.items()
    for key in tag_list
    if key in desc
]

#%% [markdown]
# * nouned

#%%
nouned = [
    tagger.nouns(s)
    for s in sentenced
]

#%% [markdown]
# * morphed

#%%
morphed = [
    tagger.morphs(s)
    for s in sentenced
]

#%% [markdown]
# * morphed, filtered

#%%
def get_wanted_morphs(s, wanted_tags):
    res_pos = tagger.pos(s)

    res = list(
        filter(
            lambda x: (x[1] in wanted_tags) and (len(x[0]) > 1),
            res_pos,
        )
    )
    return [morph[0] for morph in res]


#%%
morphed_filtered = [
    get_wanted_morphs(s, tagset_wanted)
    for s in sentenced
]

#%% [markdown]
# #### Stemming & Lemmatization
#%% [markdown]
# #### `BPE`

#%%
import sentencepiece as spm


#%%
def utf8len(s):
    return len(s.encode('utf-8'))

tmp = list(filter(lambda x: utf8len(x) > 512, sentenced))


#%%
spm_source = (
    # sentenced
    [' '.join(s) for s in morphed_filtered]
    # [''.join(s) for s in morphed]
)
spm_source_joined = '\n'.join(
    spm_source
)

new_file = f'{fpath}/full_sentence.txt'
with open(new_file, 'w') as file:
    file.write(spm_source_joined)


#%%
SPM_VOCAB_SIZE = 50000
SPM_MODEL_TYPE = 'word'  # {unigram (default), bpe, char, word}
SPM_MODEL_NAME = f'happy_spm_{SPM_MODEL_TYPE}_{SPM_VOCAB_SIZE}'


cmd_train = ' '.join(
    [
        # 'spm_train',
        f'--input={new_file}',
        f'--model_prefix={SPM_MODEL_NAME}',
        '' if SPM_MODEL_TYPE == 'word' else f'--vocab_size={SPM_VOCAB_SIZE}',
        f'--character_coverage=0.9995',
        # '--seed_sentencepiece_size=10000',
        # f'--pieces_size={SPM_VOCAB_SIZE}',
        f'--model_type={SPM_MODEL_TYPE}',
        f'--input_sentence_size={len(sentenced)}',
        # f'--max_sentencepiece_length={max(map(len, sentenced))}',
        f'--max_sentencepiece_length={512}',
    ],
)

# cmd_encode = ' '.join(
#     [
#         'spm_encode',
#         f'--model={SPM_MODEL_NAME}.model,
#         f'--output_format=piece',
#         '<',
#         f'newsdata_concatted.txt',
#         '>',
#         'newsdata_bpe_200000.piece
# spm_encode --model=news_spm_200000.model --output_format=id < newsdata_concatted.txt > newsdata_bpe_200000.id
# spm_export_vocab --model=news_spm_200000.model --output=news_spm_200000.vocab
# """
cmd_train


#%%
random.seed(1)
np.random.seed(1)

spm.SentencePieceTrainer.Train(cmd_train)

sp = spm.SentencePieceProcessor()
sp.Load(f'{SPM_MODEL_NAME}.model')
# sp.EncodeAsPieces
# sp.EncodeAsIds
# sp.DecodePieces
# sp.NBestEncodeAsPieces

spmed = [
    sp.EncodeAsPieces(l) for l in spm_source
]
spmed_ids = [
    sp.EncodeAsIds(l) for l in spm_source
]
spmed_unspaced = [
    list(
        filter(
            lambda x: len(x) > 1,
            (t.replace('▁', '') for t in l)
        )
    )
    for l in spmed
]


#%%
sentenced[:5]


#%%
spmed_unspaced[:5]

#%% [markdown]
# self design
# self-design
#%% [markdown]
# ## BOW

#%%
import json
from gensim.corpora import Dictionary as corpora_dict


#%%
save_list = [
    ('nouned', nouned),
    ('morphed', morphed),
    ('morphed_filtered', morphed_filtered),
    ('spmed', spmed),
    ('spmed_unspaced', spmed_unspaced),
]

for savename, obj in save_list:
    with open(f'data/{savename}.json', 'w', encoding='utf-8') as jfile:
        # converted_json = json.dumps(obj)
        converted_json = obj
        json.dump(converted_json, jfile, ensure_ascii=False)


#%%
tokenized = spmed_unspaced  # {nouned, morphed, morphed_filtered, spmed, spmed_unspaced}

# Create Dictionary
cdict = corpora_dict(tokenized)
cdict.filter_extremes(no_below=30, no_above=.5, keep_n=100000)

max(cdict.keys())


#%%
tokenized[:10]


#%%
bow_corpus[:10]


#%%
# Create Corpus: Term Document Frequency

bow_corpus_idx = [cdict.doc2idx(doc) for doc in tokenized]
bow_corpus_raw = [cdict.doc2bow(doc) for doc in tokenized]
bow_corpus_raw[0]


#%%
bow_doc_tmp = bow_corpus_raw[10]
for i in range(len(bow_doc_tmp)):
    print(
        "Word {} (\"{}\") appears {} time.".format(
            bow_doc_tmp[i][0],
            cdict[bow_doc_tmp[i][0]],
            bow_doc_tmp[i][1]
        )
    )

#%% [markdown]
# ## TF-IDF

#%%
from gensim import corpora, models


#%%
tfidf = models.TfidfModel(bow_corpus_raw)
corpus_tfidf = tfidf[bow_corpus_raw]

bow_corpus_tfidf = corpus_tfidf.corpus

#%% [markdown]
# ## Topic Modeling (via `LDA`)
#%% [markdown]
# ### LDA Only

#%%
LDA_TOPIC_NUM = 10
LDA_MODEL_NAME = f'happy_lda_{LDA_TOPIC_NUM}topic'


#%%
lda_model_raw = gensim.models.LdaMulticore(
    bow_corpus_raw,
    num_topics=LDA_TOPIC_NUM,
    id2word=cdict,
    passes=2,
    workers=8,
    # decay=.5, # {.5, 1.}
    per_word_topics=False,
    # per_word_topics=True,
    # minimum_probability=.1,
    # minimum_phi_value=.01,
)
lda_doc_raw = lda_model_raw[bow_corpus_raw]
lda_model_raw.save(f'{LDA_MODEL_NAME}_raw.model')


#%%
lda_model_raw.show_topics(LDA_TOPIC_NUM, 5)

#%% [markdown]
# ### LDA with TF-IDF (better)

#%%
lda_model_tfidf = gensim.models.LdaMulticore(
    bow_corpus_tfidf,
    num_topics=LDA_TOPIC_NUM,
    id2word=cdict,
    passes=2,
    workers=8,
    # decay=.5, # {.5, 1.}
    per_word_topics=True,
)
lda_doc_tfidf = lda_model_tfidf[bow_corpus_tfidf]
lda_model_tfidf.save(f'{LDA_MODEL_NAME}_tfidf.model')


#%%
lda_model_tfidf.show_topics(LDA_TOPIC_NUM, 5)


#%%
bow_corpus = bow_corpus_tfidf
lda_model = lda_model_tfidf

#%% [markdown]
# ### HDP (Hierarchical Dirichlet Process)
#%% [markdown]
# Teh, Y. W., Jordan, M. I., Beal, M. J., & Blei, D. M. (2005). Sharing clusters among related groups: Hierarchical Dirichlet processes. In Advances in neural information processing systems (pp. 1385-1392).

#%%
hdp_model = gensim.models.hdpmodel.HdpModel(
    bow_corpus,
    id2word=cdict,
    # max_chunks=,
    # kappa=,
    # tau=,
    K=5,  # Second level truncation level
    T=3,  # Top level truncation level
    alpha=10.,  # Second level concentration
    gamma=1.,  # First level concentration
    eta=.1,  # The topic Dirichlet
    # scale=,  # Weights information from the mini-chunk of corpus to calculate rhot.
    random_state=1,
)


#%%
hdp_model.show_topics(LDA_TOPIC_NUM, 15)


#%%
hdp_model.show_topics()


#%%
hdp_model.print_topics()


#%%
hdp_model.suggested_lda_model()

gamma, _ = hdp_model.inference(bow_corpus[:2])#%% [markdown]
# ### Select Method
bow_corpus = bow_corpus_tfidf
lda_model = lda_model_tfidf#%% [markdown]
# ### Get the Optimal N with cohesion (via LDA)

#%%
from gensim.models import CoherenceModel

#%% [markdown]
# ##### Mallet

#%%
def compute_coherence_values(
        dictionary, corpus, id2word, texts,
        num_topic_list=[5, 10],
        lda_typ='default',  # {'default', 'mallet'}
        random_seed=1,
        ):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    model_list = []
    coherence_list = []

    if random_seed:
        random.seed(random_seed)
        np.random.seed(random_seed)
    if lda_typ == 'default':
        for num_topics in num_topic_list:
            model = gensim.models.LdaMulticore(
                corpus,
                num_topics=num_topics,
                id2word=id2word,
                passes=2,
                workers=8,
                eta='symmetric',
                decay=.8, # {.5, 1.}
                per_word_topics=False,
                offset=1.,
                iterations=30,
                gamma_threshold=.001, # 0.001,
                minimum_probability=.05,  # .01,
                minimum_phi_value=.01,
                random_state=1,
            )
            coherence_model = CoherenceModel(
                model=model,
                texts=texts,
                dictionary=id2word,
                coherence='c_v',
            )

            model_list += [model]
            coherence_list += [coherence_model.get_coherence()]

    elif lda_typ == 'hdp':
        for num_topics in num_topic_list:
            model = gensim.models.HdpModel(
                corpus,
                id2word=id2word,
                T=3,
                # alpha=,
                K=num_topics,
                # gamma=,
                # decay=.5, # {.5, 1.}
                # per_word_topics=True,
                # minimum_probability=.1,
                # minimum_phi_value=.01,
                random_state=1,
            )
            coherence_model = CoherenceModel(
                model=model,
                texts=texts,
                dictionary=id2word,
                coherence='c_v',
            )

            model_list += [model]
            coherence_list += [coherence_model.get_coherence()]

    elif lda_typ == 'mallet':
        # Download File: http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip
        mallet_url = 'http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip'
        mallet_filename = mallet_url.split('/')[-1]
        mallet_unzipped_dirname = mallet_filename.split('.zip')[0]
        mallet_path = f'{mallet_unzipped_dirname}/bin/mallet'

        import zipfile
        import urllib

        if not os.path.exists(mallet_path):
            # download the url contents in binary format
            urllib.urlretrieve (mallet_url, mallet_filename)

            # open method to open a file on your system and write the contents
            with zipfile.ZipFile(mallet_filename, "r") as zip_ref:
                zip_ref.extractall(mallet_unzipped_dirname)

        for num_topics in num_topic_list:
            model = gensim.models.wrappers.LdaMallet(
                mallet_path,
                corpus=corpus,
                num_topics=num_topics,
                id2word=id2word,
            )
            coherence_model = CoherenceModel(
                model=model,
                texts=texts,
                dictionary=id2word,
                coherence='c_v',
            )

            model_list += [model]
            # coherence_list += [coherence_model.get_coherence()]

    return model_list, coherence_list


#%%
# Print the coherence scores
def pick_best_n_topics(dictionary, corpus, texts, lda_typ='default'):

    model_list, coherence_values = compute_coherence_values(
        dictionary=dictionary,
        corpus=corpus,
        id2word=dictionary,
        texts=texts,
        num_topic_list=[5, 7, 10, 12, 15, 17, 20],
        lda_typ=lda_typ,
        #  start=2, limit=40, step=6,
    )

    paired = zip(model_list, coherence_values)
    ordered = sorted(paired, key=lambda x: x[1], reverse=True)
    best_model = ordered[0][0]

    model_topicnum_list = []
    for i, (m, cv) in enumerate(zip(model_list, coherence_values)):
        topic_num = m.num_topics
        coh_value = round(cv, 4)
        print(
            f'[{i}] Num Topics ({topic_num:2})' +
            f' has Coherence Value of {coh_value}'
        )
        model_topicnum_list += [(topic_num, m)]

    model_dict = dict(model_topicnum_list)
    print(f'Best N topics: {best_model.num_topics}')

    return best_model, model_list, model_dict, coherence_values

#%% [markdown]
# #### Load the Model

#%%
MODEL_SAVED_OK = True

if not MODEL_SAVED_OK:

    lda_model, model_list, model_dict, coherence_values = pick_best_n_topics(
        dictionary=cdict,
        corpus=bow_corpus,
        texts=tokenized,
        lda_typ='default',  # 'default',
    )

    for _topic_num, _model in model_dict.items():

        # LDA_TOPIC_NUM = _model.num_topics
        LDA_TOPIC_NUM = _topic_num
        LDA_MODEL_NAME = f'lda_{LDA_TOPIC_NUM}_topics_model.ldamodel'

        print(f'{LDA_TOPIC_NUM:2}: {LDA_MODEL_NAME}')

        _filename = f'data/{LDA_MODEL_NAME}'
        _model.save(_filename)

else:
    n_list = [5, 7, 10, 12, 15, 17, 20]
    model_dict = {}
    model_list = []
    for _topic_num in n_list:

        LDA_TOPIC_NUM = _topic_num
        LDA_MODEL_NAME = f'lda_{LDA_TOPIC_NUM}_topics_model.ldamodel'

        print(f'{LDA_TOPIC_NUM:2}: {LDA_MODEL_NAME}')

        _filename = f'data/{LDA_MODEL_NAME}'

        _model = gensim.models.LdaMulticore.load(_filename)

        model_dict.setdefault(_topic_num, _model)
        model_list += [_model]


#%%
lda_model = model_dict[7] # 15 is the best
# lda_model = model_list[2] # 15 is the best

LDA_TOPIC_NUM = lda_model.num_topics
LDA_MODEL_NAME = f'happy_lda_{LDA_TOPIC_NUM}topic'

#%% [markdown]
# ### Visualization (via `T-SNE`)

#%%
import pyLDAvis
import pyLDAvis.gensim as gensimvis


#%%
prepared_data = gensimvis.prepare(
    topic_model=lda_model,
    corpus=bow_corpus,
    dictionary=cdict,
    doc_topic_dist=None,
    R=30,
    lambda_step=0.2,
    mds='tsne',
    # mds=<function js_PCoA>,
    n_jobs=-1,
    plot_opts={'xlab': 'PC1', 'ylab': 'PC2'},
    sort_topics=True,
)

LDA_HTML = f'data/lda_vis_result_{LDA_TOPIC_NUM}_topics.html'
LDA_JSON = f'data/lda_vis_result_{LDA_TOPIC_NUM}_topics.json'

pyLDAvis.save_html(prepared_data, LDA_HTML)
pyLDAvis.save_json(prepared_data, LDA_JSON)


#%%
pyLDAvis.display(prepared_data, local=False)

#%% [markdown]
# ## Term Relevance
#%% [markdown]
# $$ distinctiveness(w) = \sum P(t \vert w) log\frac{P(t \vert w)}{P(w)} $$
# $$ saliency(w) = P(w) \times distinctiveness(w) $$
#
# <div align="right">(Chuang, J., 2012. Termite: Visualization techniques for assessing textual topic models)</div>

#%%
def get_saliency(tinfo_df):
    """Calculate Saliency for terms within a topic.
    
    Parameters
    ----------
    tinfo: pandas.DataFrame
        `pyLDAvis.gensim.prepare`.to_dict()['tinfo'] containing
        ['Category', 'Freq', 'Term', 'Total', 'loglift', 'logprob']

    """

    saliency = tinfo_df['Freq'] / tinfo_df['Total']

    return saliency

#%% [markdown]
# $$ relevance(t,w) = \lambda \cdot P(w \vert t) + (1 - \lambda) \cdot \frac{P(w \vert t)}{P(w)} $$
#
# <div align="center"> Recommended $\lambda = 0.6$ </div>
#
# <div align="right">(Sievert, C., 2014. LDAvis: A method for visualizing and interpreting topics)</div>

#%%
def get_relevance(tinfo_df, l=.6):
    """Calculate Relevances with a given lambda value.
    
    Parameters
    ----------
    tinfo: pandas.DataFrame
        `pyLDAvis.gensim.prepare`.to_dict()['tinfo'] containing
        ['Category', 'Freq', 'Term', 'Total', 'loglift', 'logprob']

    l: float
        lambda_ratio between {0-1}. default is .6 (recommended from its paper)

    """

    relevance = l * tinfo_df['logprob'] + (1 - l) * tinfo_df['loglift']

    return relevance


#%%
def groupby_top_n(
        dataframe,
        group_by=None,
        order_by=None,
        ascending=False,
        n=5,
        ):

    res_df = (
        dataframe
        .groupby(group_by)
        [dataframe.columns.drop(group_by)]
        .apply(
            lambda x: x.sort_values(order_by, ascending=ascending).head(n)
        )
    )
    return res_df


#%%
def get_terminfo_table(
        lda_model,
        corpus: list=None,
        dictionary: gensim.corpora.dictionary.Dictionary=None,
        doc_topic_dists=None,
        use_gensim_prepared=True,
        top_n=10,
        r_normalized=False,
        random_seed=1,
        ):

    if random_seed:
        random.seed(random_seed)
        np.random.seed(random_seed)
    if use_gensim_prepared:

        _prepared = gensimvis.prepare(
            topic_model=lda_model,
            corpus=corpus,
            dictionary=dictionary,
            doc_topic_dist=None,
            R=len(dictionary),
            # lambda_step=0.2,
            mds='tsne',
            # mds=<function js_PCoA>,
            n_jobs=-1,
            plot_opts={'xlab': 'PC1', 'ylab': 'PC2'},
            sort_topics=True,
        )
        tinfo_df = pd.DataFrame(_prepared.to_dict()['tinfo'])

        tinfo_df['topic_term_dists'] = np.exp(tinfo_df['logprob'])
        tinfo_df['term_proportion'] = (
            np.exp(tinfo_df['logprob']) / np.exp(tinfo_df['loglift'])
        )
        tinfo_df['saliency'] = get_saliency(tinfo_df)
        tinfo_df['relevance'] = get_relevance(tinfo_df)

        tinfo_df['term_prob'] = np.exp(tinfo_df['logprob'])
        tinfo_df['term_r_prob'] = np.exp(tinfo_df['relevance'])
        tinfo_df['term_r_adj_prob'] = (
            tinfo_df
            .groupby(['Category'])
            ['term_r_prob']
            .apply(lambda x: x / x.sum())
        )

        if r_normalized:
            r_colname = 'term_r_adj_prob'
        else:
            r_colname = 'term_r_prob'

        relevance_score_df = (
            tinfo_df[tinfo_df['Category'] != 'Default']
            .groupby(['Category', 'Term'])
            [[r_colname]]
            .sum()
            .reset_index()
        )

#         corpus_dict_df = pd.DataFrame(
#             # It is possible
#             # because the keys of this dictionary generated from range(int).
#             # Usually the dictionary is iterable but not ordered.
#             list(dictionary.values()),
#             # [dictionary[i] for i, _ in enumerate(dictionary)],
#             columns=['Term'],
#         )
#         corpus_dict_df['term_id'] = corpus_dict_df.index
        corpus_dict_df = pd.DataFrame(
            list(dictionary.items()),
            columns=['term_id', 'Term'],
        )
        corpus_dict_df.set_index('term_id', drop=False, inplace=True)

        r_score_df = pd.merge(
            relevance_score_df,
            corpus_dict_df,
            on=['Term'],
            how='left',
        )
        r_score_df['category_num'] = (
            r_score_df['Category']
            .str
            .replace('Topic', '')
            .astype(int) - 1
        ).astype('category')
        r_score_df.set_index(['category_num', 'term_id'], inplace=True)
        ixs = pd.IndexSlice

        topic_list = r_score_df.index.levels[0]
        equal_prob = 1. / len(topic_list)
        empty_bow_case_list = list(
            zip(topic_list, [equal_prob] * len(topic_list))
        )


        def get_bow_score(
                bow_chunk,
                score_df=r_score_df,
                colname=r_colname,
                ):

            bow_chunk_arr = np.array(bow_chunk)
            word_id_arr = bow_chunk_arr[:, 0]
            word_cnt_arr = bow_chunk_arr[:, 1]

            # normed_word_cnt_arr = (word_cnt_arr / word_cnt_arr.sum()) * 10
            clipped_word_cnt_arr = np.clip(word_cnt_arr, 0, 3)

            score_series = (score_df.loc[ixs[:, word_id_arr], :]
                .groupby(level=0)
                [colname]
                .apply(lambda x: x @ clipped_word_cnt_arr)
            )
            score_list = list(score_series.iteritems())
            # normed_score_series = score_series / score_series.sum()
            # score_list = list(normed_score_series.iteritems())

            return score_list

        bow_score_list = [
            get_bow_score(bow_chunk)
            if bow_chunk not in (None, [])
            else empty_bow_case_list
            for bow_chunk in corpus
        ]

        relevant_terms_df = groupby_top_n(
            tinfo_df,
            group_by=['Category'],
            order_by=['relevance'],
            ascending=False,
            n=top_n,
        )
        relevant_terms_df['rank'] = (
            relevant_terms_df
            .groupby(['Category'])
            ['relevance']
            # .rank(method='max')
            .rank(method='max', ascending=False)
            .astype(int)
        )

    else:

        vis_attr_dict = gensimvis._extract_data(
            topic_model=ldamodel,
            corpus=corpus,
            dictionary=dictionary,
            doc_topic_dists=None,
        )
        topic_term_dists = _df_with_names(
            vis_attr_dict['topic_term_dists'],
            'topic', 'term',
        )
        doc_topic_dists = _df_with_names(
            vis_attr_dict['doc_topic_dists'],
            'doc', 'topic',
        )
        term_frequency = _series_with_name(
            vis_attr_dict['term_frequency'],
            'term_frequency',
        )
        doc_lengths = _series_with_name(
            vis_attr_dict['doc_lengths'],
            'doc_length',
        )
        vocab = _series_with_name(
            vis_attr_dict['vocab'],
            'vocab',
        )

        ## Topic
        topic_freq = (doc_topic_dists.T * doc_lengths).T.sum()  # doc_lengths @ doc_topic_dists
        topic_proportion = (topic_freq / topic_freq.sum())

        ## reorder all data based on new ordering of topics
        # topic_proportion = (topic_freq / topic_freq.sum()).sort_values(ascending=False)
        # topic_order = topic_proportion.index
        # topic_freq = topic_freq[topic_order]
        # topic_term_dists = topic_term_dists.iloc[topic_order]
        # doc_topic_dists = doc_topic_dists[topic_order]

        # token counts for each term-topic combination
        term_topic_freq = (topic_term_dists.T * topic_freq).T
        term_frequency = np.sum(term_topic_freq, axis=0)

        ## Term
        term_proportion = term_frequency / term_frequency.sum()

        # compute the distinctiveness and saliency of the terms
        topic_given_term = topic_term_dists / topic_term_dists.sum()
        kernel = (topic_given_term * np.log((topic_given_term.T / topic_proportion).T))
        distinctiveness = kernel.sum()
        saliency = term_proportion * distinctiveness

        default_tinfo_df = pd.DataFrame(
            {
                'saliency': saliency,
                'term': vocab,
                'freq': term_frequency,
                'total': term_frequency,
                'category': 'default',
                'logprob': np.arange(len(vocab), 0, -1),
                'loglift': np.arange(len(vocab), 0, -1),
            }
        )

        log_lift = np.log(topic_term_dists / term_proportion)
        log_prob = log_ttd = np.log(topic_term_dists)

    return tinfo_df, relevant_terms_df, r_score_df, bow_score_list


#%%
(total_terms_df, top_relevant_terms_df,
 r_adj_score_df, bow_score_list) = get_terminfo_table(
    lda_model,
    corpus=bow_corpus,
    dictionary=cdict,
    doc_topic_dists=None,
    use_gensim_prepared=True,
    top_n=30,
)


#%%
top_relevant_terms_df.head()


#%%
top_relevant_terms_df.to_csv(
    f'data/{LDA_TOPIC_NUM}topic_top_relevant_terms_df.csv',
    index=True,
    header=True,
    encoding='utf-8',
)

#%% [markdown]
# ## Representitive Text
#%% [markdown]
# ### Get Dominant Topics
# Get dominant topics & its contribution scores from each documents.

#%%
def format_topics_sentences(
        ldamodel=None,
        corpus=None,
        docs=None,
        bow_r_score_list=None,
        top_r_terms_df=None,
        random_seed=1,
        ):

    res_df = pd.DataFrame(
        columns=[
            'dominant_topic',
            'contribution',
            'topic_keywords',
            'documents',
            'lda_prob',
        ]
    )

    if random_seed:
        random.seed(random_seed)
        np.random.seed(random_seed)
    r_colname = top_r_terms_df.columns.drop(['Category', 'Term'])[0]

    if top_r_terms_df is not None:
        top_sorted_words = groupby_top_n(
            top_r_terms_df.reset_index(),
            group_by=['category_num'],
            order_by=[r_colname],
            ascending=False,
            n=10,
        )
        top_word_str = (
            top_sorted_words
            .groupby(level=0)
            ['Term']
            .apply(lambda x: ', '.join(x.tolist()))
        )


    def normalize_prob(prob_row):
        total_prob = sum([prob for topic, prob in prob_row])
        normed_prob_row = [
            (topic, prob / total_prob) for topic, prob in prob_row
        ]
        return normed_prob_row

    def sort_prob(prob_row, bow_r_score_list=bow_r_score_list):
        return sorted(prob_row, key=lambda x: x[1], reverse=True)

    def get_dominant_prob(prob_row):
        return pd.Series(prob_row[0])

    def get_topic_keywords(
            dom_topic_num,
            lda_model=lda_model,
            top_r_terms_df=top_r_terms_df,
            ):
        if top_r_terms_df is not None:
            return top_word_str[int(dom_topic_num)]
        else:
            return ', '.join(
                np.array(lda_model.show_topic(int(dom_topic_num)))[:, 0]
            )

    res_df['documents'] = docs

    if bow_r_score_list is not None:
        # bow_score_series = pd.Series(bow_score_list).apply(normalize_prob)
        bow_score_series = pd.Series(bow_score_list)
    else:
        bow_score_series = pd.Series(ldamodel[corpus])

    res_df['lda_prob'] = bow_score_series.apply(sort_prob)
    res_df[['dominant_topic', 'contribution']] = (
        res_df['lda_prob']
        .apply(get_dominant_prob)
    )
    res_df['dominant_topic'] = res_df['dominant_topic'].astype(int).astype('category')
    res_df['topic_keywords'] = (
        res_df['dominant_topic']
        .apply(get_topic_keywords)
    )
    res_df['lda_prob'] = res_df['lda_prob'].apply(dict)
    res_df.index.name = 'doc_num'
    res_df.reset_index(inplace=True)

    return res_df


#%%
dominant_topic_kwd_df = format_topics_sentences(
    ldamodel=lda_model,
    corpus=bow_corpus,
    docs=sentenced,
    bow_r_score_list=bow_score_list,
    top_r_terms_df=r_adj_score_df,
)


#%%
dominant_topic_kwd_df.to_csv(
    f'dominant_{LDA_TOPIC_NUM}topic_kwd_df.csv',
    index=False,
    header=True,
    encoding='utf-8',
)

# Show
dominant_topic_kwd_df.head(10)

#%% [markdown]
# ### Topic Frequency

#%%
lda_freq = (
    dominant_topic_kwd_df
    .groupby('dominant_topic')
    ['doc_num']
    .count()
    .reset_index()
)
lda_freq.to_csv(
    f'data/lda_frequencies_{LDA_TOPIC_NUM}topic.csv',
    header=True,
    index=False,
    encoding='utf-8',
)

ordered_lda_freq = lda_freq.sort_values('doc_num', ascending=False)
ordered_lda_freq

#%% [markdown]
# ### Get Representitive documents (top 5)

#%%
def clip_document_len(topic_kwd_df, len_range=(100, 300)):
    len_min, len_max = len_range
    len_series = topic_kwd_df['documents'].apply(len)
    # mask = np.where((len_series >= 100) & (len_series < 300))
    res = topic_kwd_df.loc[
        (len_series >= len_min) & (len_series < len_max),
        :
    ]
    return res# topic_kwd_df.loc[mask, :]


#%%
representitive_sentences_df = (
    pd.Series(
        list(filter(lambda x: 100 <= len(x) < 300, sentenced)),
        name='documents',
    )
    .to_frame()
)
representitive_sentences_df.to_csv(
    'data/representitive_sentences_df.csv',
    index=False,
    header=True,
    encoding='utf-8',
)

representitive_short_sentences_df = (
    pd.Series(
        list(filter(lambda x: 5 <= len(x) < 20, sentenced)),
        name='documents',
    )
    .to_frame()
)
representitive_short_sentences_df.to_csv(
    'data/representitive_short_sentences_df.csv',
    index=False,
    header=True,
    encoding='utf-8',
)

#%% [markdown]
# * repr_tokenized

#%%
def get_repr_tokenized(repr_sentenced, dictionary=None):

    def get_wanted_morphs(s, wanted_tags):
        res_pos = tagger.pos(s)

        res = list(
            filter(
                lambda x: (x[1] in wanted_tags) and (len(x[0]) > 1),
                res_pos,
            )
        )
        return [morph[0] for morph in res]


    morphed_filtered = [
        get_wanted_morphs(s, tagset_wanted)
        for s in repr_sentenced
    ]

    spm_source = (
        # sentenced
        [' '.join(s) for s in morphed_filtered]
        # [''.join(s) for s in morphed]
    )
    spm_source_joined = '\n'.join(
        spm_source
    )

    SPM_VOCAB_SIZE = 50000
    SPM_MODEL_TYPE = 'word'  # {unigram (default), bpe, char, word}
    SPM_MODEL_NAME = f'happy_spm_{SPM_MODEL_TYPE}_{SPM_VOCAB_SIZE}'

    random.seed(1)
    np.random.seed(1)

    sp = spm.SentencePieceProcessor()
    sp.Load(f'{SPM_MODEL_NAME}.model')
    # sp.EncodeAsPieces
    # sp.EncodeAsIds
    # sp.DecodePieces
    # sp.NBestEncodeAsPieces

    spmed = [
        sp.EncodeAsPieces(l) for l in spm_source
    ]
    spmed_ids = [
        sp.EncodeAsIds(l) for l in spm_source
    ]
    spmed_unspaced = [
        list(
            filter(
                lambda x: len(x) > 1,
                (t.replace('▁', '') for t in l)
            )
        )
        for l in spmed
    ]
    bow_corpus_idx = [dictionary.doc2idx(doc) for doc in tokenized]
    bow_corpus_raw = [dictionary.doc2bow(doc) for doc in tokenized]

    return spmed_unspaced, bow_corpus_raw, bow_corpus_idx


#%%
(tokenized_repr,
 bow_corpus_raw_repr,
 bow_corpus_idx_repr) = get_repr_tokenized(
    representitive_short_sentences_df,
    dictionary=cdict,
)

#%% [markdown]
# * Get Representitive docs

#%%
TOP_DOC_NUM = 5

# topic_kwd_df = dominant_topic_kwd_df
topic_kwd_df = clip_document_len(
    dominant_topic_kwd_df,
    len_range=(5, 50),
)
sentenced

top_docs_df = groupby_top_n(
    topic_kwd_df,
    group_by=['dominant_topic'],
    order_by=['contribution'],
    ascending=False,
    n=TOP_DOC_NUM,
)
top_docs_df.to_csv(
    f'data/dominant_{LDA_TOPIC_NUM}_topics_kwd_sorted_top_{TOP_DOC_NUM}_df.csv',
    header=True,
    index=False,
    encoding='utf-8',
)


#%%
grouped_dict = dict(list(top_docs_df.groupby(level=0)))
reordered = pd.concat(
    [grouped_dict[i] for i in ordered_lda_freq.head(50)['dominant_topic']]
)
reordered
# reordered.reset_index().loc[:, ['documents']]


#%%
len(top_docs_df.iloc[14, 3])


#%%
print(f'Topic N: {LDA_TOPIC_NUM}')
top_docs_df

# Get topic weights and dominant topics ------------
from sklearn.manifold import TSNE
from bokeh.plotting import figure, output_file, show
from bokeh.models import Label
from bokeh.io import output_notebooklda_model.id2word[1]def get_topic_term_prob(lda_model):
    topic_term_freqs = lda_model.state.get_lambda()
    topic_term_prob = topic_term_freqs / topic_term_freqs.sum(axis=1)[:, None]
    return topic_term_prob# Get topic weights
topic_weights = []
for i, row_list in enumerate(lda_model.id2word):
    topic_weights.append([w for i, w in row_list[0]])

# Array of topic weights
arr = pd.DataFrame(topic_weights).fillna(0).values

# Keep the well separated points (optional)
arr = arr[np.amax(arr, axis=1) > 0.35]

# Dominant topic number in each doc
topic_num = np.argmax(arr, axis=1)

# tSNE Dimension Reduction
tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
tsne_lda = tsne_model.fit_transform(arr)

# Plot the Topic Clusters using Bokeh
output_notebook()
n_topics = 4
mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])
plot = figure(title="t-SNE Clustering of {} LDA Topics".format(n_topics),
              plot_width=900, plot_height=700)
plot.scatter(x=tsne_lda[:,0], y=tsne_lda[:,1], color=mycolors[topic_num])
show(plot)#%% [markdown]
# ## Hierarchical LDA
#%% [markdown]
# H
#
# Blei, D. M., Griffiths, T. L., & Jordan, M. I. (2010). The nested chinese restaurant process and bayesian nonparametric inference of topic hierarchies. Journal of the ACM (JACM), 57(2), 7.

#%%
from hlda import sampler as hlda_sampler

HierarchicalLDA = hlda_sampler.HierarchicalLDA


#%%
cdict_vocab_list = list(cdict.items())
vocab = list(cdict.values())
bow_corpus_ids = bow_corpus_idx


#%%
print(
    len(vocab),
    len(bow_corpus),
    len(bow_corpus[0]),
    len(bow_corpus_ids[1]),
)

n_samples = 500       # no of iterations for the sampler
alpha = 10.0          # smoothing over level distributions
gamma = 1.0           # CRP smoothing parameter; number of imaginary customers at next, as yet unused table
eta = 0.1             # smoothing over topic-word distributions
num_levels = 3        # the number of levels in the tree
display_topics = 50   # the number of iterations between printing a brief summary of the topics so far
n_words = 5           # the number of most probable words to print for each topic after model estimation
with_weights = False  # whether to print the words with the weights
#%%
hlda_model = HierarchicalLDA(
    corpus=bow_corpus_ids,
    vocab=vocab,
    alpha=10.,
    gamma=1.,
    eta=.1,
    seed=1,
    num_levels=3,
    verbose=True,
)


#%%
hlda_model.estimate(
    num_samples=4,
    display_topics=5,
    n_words=5,
    with_weights=False,
)

#%% [markdown]
# ### hLDA Visualization

#%%
from ipywidgets import widgets
from IPython.core.display import HTML, display


#%%
colour_map = {
    0: 'blue',
    1: 'red',
    2: 'green'
}

def show_doc(d=0):

    node = hlda_model.document_leaves[d]
    path = []
    while node is not None:
        path.append(node)
        node = node.parent
    path.reverse()

    n_words = 10
    with_weights = False
    for n in range(len(path)):
        node = path[n]
        colour = colour_map[n]
        msg = 'Level %d Topic %d: ' % (node.level, node.node_id)
        msg += node.get_top_words(n_words, with_weights)
        output = '<h%d><span style="color:%s">%s</span></h3>' % (n+1, colour, msg)
        display(HTML(output))

    display(HTML('<hr/><h5>Processed Document</h5>'))

    doc = bow_corpus_ids[d]
    output = ''
    for n in range(len(doc)):
        w = doc[n]
        l = hlda_model.levels[d][n]
        colour = colour_map[l]
        output += '<span style="color:%s">%s</span> ' % (colour, w)
    display(HTML(output))


#%%
widgets.interact(show_doc, d=(0, len(bow_corpus_ids)-1))


#%%
import pickle
import gzip

def save_zipped_pickle(obj, filename, protocol=-1):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol)

def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object


#%%
save_zipped_pickle(hlda_model, 'data/hlda_result.pkl')

#%% [markdown]
# ## Vectorization (via `Word2Vec`)

#%%
from gensim.models import Word2Vec
from gensim.test.utils import common_texts


#%%
W2V_MODEL_NAME = f'happy_w2v.bin'

trained_ok = False
w2v_bak = W2V_MODEL_NAME

if not trained_ok or not os.path.isfile(w2v_bak):
    embed = Word2Vec(tokenized, size=70,
                     window=4, min_count=10,
                     negative=16,
                     workers=8, iter=50, sg=1)
    embed.save(w2v_bak)
else:
    embed = Word2Vec.load(w2v_bak)

#%% [markdown]
# ### Similar Words

#%%
pprint(embed.wv.most_similar(
    positive="행복",
    topn=20,
))


#%%
pprint(embed.wv.most_similar(
    # positive=['성과', '연계'],
    positive=['업무', '필요', '평가', '개선'],
    topn=20,
))


#%%
pprint(embed.wv.most_similar(
    # positive=['성과', '연계'],
    positive=['변화', '기회', '고민', '관점'],
    topn=20,
))


#%%
with open(f'{fpath}/saveasnew/survey_keyword_saveasnew.csv') as f:
    survey_keywords = f.read()


#%%
survey_keywords_token = [
    re.sub(
        r'^.*\"(.+)"$',
        r'\1',
        s,
    ).lower().split(', ')
    for s in survey_keywords.split('\n')
]


#%%
def get_similar(words, topn=2):
    try:
        res = embed.wv.most_similar(
            # positive=['성과', '연계'],
            positive=words,
            topn=topn,
        )
        return res
    except KeyError as err:
        return None


#%%
survey_answer_token = [
    get_similar(s, topn=15)
    for s in survey_keywords_token
]


#%%
survey_answer_token[:5]


#%%
embed.wv.most_similar(
    # positive=['성과', '연계'],
    positive=['변화', '기회', '고민', '관점'],
    topn=20,
)


#%%
embed.wv.most_similar(
    positive=['성과', '연계'],
    # positive=['변화', '기회', '고민', '관점'],
    topn=20,
)


#%%
embed.wv.most_similar(
    positive=['comm', '안내', '설명'],
    # positive=['변화', '기회', '고민', '관점'],
    topn=20,
)

#%% [markdown]
# ### Visualization (via `Tensorboard`)

#%%
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector


#%%
def create_embeddings(gensim_model=None, model_folder=None):
    weights = gensim_model.wv.vectors
    idx2words = gensim_model.wv.index2word

    vocab_size = weights.shape[0]
    embedding_dim = weights.shape[1]

    with open(os.path.join("metadata.tsv"), 'w') as f:
        f.writelines("\n".join(idx2words))

    tf.reset_default_graph()

    W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]), trainable=False, name="W")
    embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
    embedding_init = W.assign(embedding_placeholder)

    writer = tf.summary.FileWriter(model_folder, graph=tf.get_default_graph())
    saver = tf.train.Saver()
    # Format: tensorflow/contrib/tensorboard/plugins/projector/projector_config.proto
    config = projector.ProjectorConfig()

    # You can add multiple embeddings. Here we add only one.
    embedding = config.embeddings.add()
    embedding.tensor_name = W.name
    embedding.metadata_path = os.path.join(model_folder, "metadata.tsv")
    # Saves a configuration file that TensorBoard will read during startup.
    projector.visualize_embeddings(writer, config)

    with tf.Session() as sess:
        sess.run(embedding_init, feed_dict={embedding_placeholder: weights})
        save_path = saver.save(sess, os.path.join(model_folder, "tf-model.cpkt"))

    return save_path


#%%
# os.rmdir('./tsboard')
# os.makedirs('./tsboard/embeddings', exist_ok=True)
create_embeddings(gensim_model=embed, model_folder='./')

#%% [markdown]
# ## WordCloud

#%%
from wordcloud import WordCloud


#%%
get_ipython().magic('pinfo WordCloud')


#%%
font_path = font_dict['NanumBarunGothic']


#%%
wcloud = WordCloud(
    font_path=font_path,
    background_color=None,
    # background_color='rgba(255, 255, 255, 0)',
    colormap='tab10',
    mode='RGBA',
)
wcloud.generate(' '.join(sum(tokenized, [])))

plt.figure(figsize=(12, 12))
plt.imshow(wcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

#%% [markdown]
# ## Sentence Coloring
f = [f.name for f in fm.fontManager.ttflist]
sorted(f)
#%%
# Sentence Coloring of N Sentences
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import matplotlib as mpl
# import matplotlib.font_manager as fm
# font_path = 'C:/Windows/Fonts/EBS훈민정음R.ttf'
# fm.OSXInstalledFonts()
# font_path = '/Library/Fonts/NanumBarunpenRegular.otf'
# fontprop = fm.FontProperties(fname=font_path, size=18)
from matplotlib import rc
rc('font', family='NanumGothic')

mpl.rcParams['axes.unicode_minus'] = False


def sentences_chart(lda_model=lda_model, corpus=bow_corpus, start = 0, end = 13):
    corp = bow_corpus[start:end]
    mycolors = [color for name, color in mcolors.TABLEAU_COLORS.items()]

    fig, axes = plt.subplots(end-start, 1, figsize=(20, (end-start)*0.95), dpi=160)
    axes[0].axis('off')
    for i, ax in enumerate(axes):
        if i > 0:
            corp_cur = corp[i-1]
            # topic_percs, wordid_topics, wordid_phivalues = lda_model[corp_cur]
            print(lda_model[corp_cur])
            topic_percs, wordid_topics = lda_model[corp_cur]
            word_dominanttopic = [(lda_model.id2word[wd], topic[0]) for wd, topic in wordid_topics]
            ax.text(0.001, 0.5, "Doc " + str(i-1) + ": ", verticalalignment='center',
                    fontsize=16, color='black', transform=ax.transAxes, fontweight=1000)

            # Draw Rectange
            topic_percs_sorted = sorted(topic_percs, key=lambda x: (x[1]), reverse=True)
            ax.add_patch(Rectangle((0.0, 0.01), 0.99, 0.90, fill=None, alpha=1,
                                   color=mycolors[topic_percs_sorted[0][0]], linewidth=2))

            word_pos = 0.06
            for j, (word, topics) in enumerate(word_dominanttopic):
                if j < 14:
                    ax.text(word_pos, 0.5, word,
                            horizontalalignment='left',
                            verticalalignment='center',
                            fontsize=16, color=mycolors[topics],
                            transform=ax.transAxes, fontweight=1000)
                    word_pos += .009 * len(word)  # to move the word for the next iter
                    ax.axis('off')
            ax.text(word_pos, 0.5, '. . .',
                    horizontalalignment='left',
                    verticalalignment='center',
                    fontsize=16, color='black',
                    transform=ax.transAxes)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.suptitle('Sentence Topic Coloring for Documents: ' + str(start) + ' to ' + str(end-2), fontsize=22, y=0.95, fontweight=700)
    plt.tight_layout()
    plt.show()

    # return fig

sentences_chart()

#%% [markdown]
# Done.
