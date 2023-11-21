#!/usr/bin/env python3

from pathlib import Path

DEFAULT_PDF_WHAT = 'the content of a PDF file'
DEFAULT_URL_WHAT = 'the content of a webpage'
DEFAULT_HTML_WHAT = 'the text content of a html file'
DEFAULT_GENERAL_WHAT = 'the content of a document'

MODEL_TO_CONTEXT_LENGTH_MAPPING = {
    'gpt-3.5-turbo': 4096,
    'text-davinci-003': 4096,
    'gpt-4': 8192,
    'gpt-4-32k': 32768,
}

DEFAULT_MODEL = 'gpt-3.5-turbo'


_REPO_ROOT_DIR = Path(__file__).parent.parent
TEMPLATE_DIR = _REPO_ROOT_DIR / 'templates'

_README_PATH = _REPO_ROOT_DIR / 'README.md'
_README_COMMANDS = ['urlprompt', 'pdfprompt', 'ytprompt', 'textprompt', 'htmlprompt']
_README_TEMPLATE = TEMPLATE_DIR / 'README.jinja.md'
_COMMAND_USAGE_TEMPLATE = TEMPLATE_DIR / 'readme-command-usage.jinja.md'

LOW_QUALITY_PAGE_CONTENT_PUNC_WHITESPACE_PCT_THRESHOLD = 0.15

TESSERACT_OCR_DEFAULT_LANG = 'chi_sim'

tesseract_langs = """
afr
amh
ara
asm
aze
aze_cyrl
bel
ben
bod
bos
bre
bul
cat
ceb
ces
chi_sim
chi_sim_vert
chi_tra
chi_tra_vert
chr
cos
cym
dan
deu
div
dzo
ell
eng
enm
epo
equ
est
eus
fao
fas
fil
fin
fra
frk
frm
fry
gla
gle
glg
grc
guj
hat
heb
hin
hrv
hun
hye
iku
ind
isl
ita
ita_old
jav
jpn
jpn_vert
kan
kat
kat_old
kaz
khm
kir
kmr
kor
kor_vert
lao
lat
lav
lit
ltz
mal
mar
mkd
mlt
mon
mri
msa
mya
nep
nld
nor
oci
ori
osd
pan
pol
por
pus
que
ron
rus
san
script/Arabic
script/Armenian
script/Bengali
script/Canadian_Aboriginal
script/Cherokee
script/Cyrillic
script/Devanagari
script/Ethiopic
script/Fraktur
script/Georgian
script/Greek
script/Gujarati
script/Gurmukhi
script/HanS
script/HanS_vert
script/HanT
script/HanT_vert
script/Hangul
script/Hangul_vert
script/Hebrew
script/Japanese
script/Japanese_vert
script/Kannada
script/Khmer
script/Lao
script/Latin
script/Malayalam
script/Myanmar
script/Oriya
script/Sinhala
script/Syriac
script/Tamil
script/Telugu
script/Thaana
script/Thai
script/Tibetan
script/Vietnamese
sin
slk
slv
snd
snum
spa
spa_old
sqi
srp
srp_latn
sun
swa
swe
syr
tam
tat
tel
tgk
tha
tir
ton
tur
uig
ukr
urd
uzb
uzb_cyrl
vie
yid
yor
""".strip().splitlines()

_MACOS_CONDA_ENV_EG_TESSDATA_PREFIX = (
    '/usr/local/Caskroom/miniconda/base/envs/eg/share/tessdata/'
)
