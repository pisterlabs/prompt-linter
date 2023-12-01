import re

import openai
import pysradb
import requests
from Bio import Entrez, Medline, SeqIO
from bs4 import BeautifulSoup

pmc_list = []

Entrez.email = 'aye10032@gmail.com'
openai.api_key_path = '../API.txt'


def get_articles():
    global pmc_list

    search_term = ('(Sun, Luyang[Full Author Name]) AND '
                   '('
                   '(Auburn University[Affiliation]) OR '
                   '(Baylor College of Medicine[Affiliation]) OR '
                   '(Qingdao Institute of Bioenergy and Bioprocess Technology[Affiliation])'
                   ') AND '
                   '(\"2013-01-01\"[Publication Date] : \"3000\"[Publication Date])')

    handle = Entrez.esearch(db='PMC', term=search_term)
    record = Entrez.read(handle)
    handle.close()

    pmc_list = record['IdList']


def analyse_article():
    with open('../output.txt', 'a', encoding='utf-8') as _file:
        index = 1
        for pmc in pmc_list:
            handle = Entrez.efetch(db='PMC', id=pmc, rettype='medline', retmode='text')
            article = Medline.read(handle)
            handle.close()

            _file.write(f'{index}、' + article['TI'] + '\n')
            _file.write(f'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc}\n')
            _file.write(f'PMC:{pmc}\n')
            _file.write(article['AID'][0] + '\n')

            seq_id = get_seq_id(pmc)
            if not seq_id == '':
                count = analyse_seq(seq_id)
                _file.write(f'本文中的数据来源:{seq_id}，包含{count}个序列\n')

            if 'AB' in article:
                species = analyse_abstract(article['AB'])

                print(species)
                if len(species) == 0:
                    _file.write('本文中提到了0个物种\n')
                else:
                    _file.write(f'本文中提到了{len(species)}个物种，分别是: ' + ','.join(map(str, species)) + '\n\n')
            else:
                _file.write('本文中提到了0个物种\n')

            index += 1

    print('done')


def get_seq_id(pmc):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    url = f'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc}/'
    print(url)

    response = requests.get(url, headers=headers)
    html_content = response.content
    soup = BeautifulSoup(html_content, 'html.parser')

    try:
        data_block = (soup.find('div', id='mc')
                      .find('div', class_='jig-ncbiinpagenav')
                      .find('div', id='ass-data')
                      .find('dd', id='data-avl-stmnt'))

        _url = data_block.find('a')['href']
        matches = re.findall(r'[A-Z]{3,4}\d{6,7}$', _url)
        if matches:
            return matches[-1]
    except AttributeError:
        return ''

    return ''


def analyse_seq(seq_id):
    if seq_id.startswith('GES'):
        return get_geo(seq_id)
    # elif seq_id.startswith('RJNA'):
    #     return get_bioproject(seq_id)
    # elif seq_id.startswith('LBML'):
    #     return get_genbank(seq_id)


def get_geo(gse_id):
    handle = Entrez.esearch(db='geo', term=f'{gse_id}[ACCN]')
    record = Entrez.read(handle)
    handle.close()

    gds_list = record['IdList']

    seq_id = []

    for gds in gds_list:
        print(gds)
        handle = Entrez.esummary(db='gds', id=gds)
        record = Entrez.read(handle)
        handle.close()

        sequences = record[0]['Samples']

        for seq in sequences:
            if not seq['Accession'] in seq_id:
                seq_id.append(seq['Accession'])

    return len(seq_id)


def get_bioproject(rj_id):
    handle = Entrez.esearch(db='sra', term=f'{rj_id[-4:]}[BioProject]')
    summary = Entrez.read(handle)
    handle.close()

    seq_id = summary['IdList']

    sra_db = pysradb.SRAweb()
    results = sra_db.search_sra(seq_id[0])


def get_genbank(uid):
    handle = Entrez.esearch(db='nucleotide', term='LBML01000000')
    record = Entrez.read(handle, "genbank")

    handle = Entrez.efetch(db='nucleotide', id=record['IdList'][0], rettype='gb', retmode='text')
    seq = SeqIO.read(handle, 'genbank')
    handle.close()


def analyse_abstract(abstract):
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {'role': 'assistant',
             'content': f"下面我有一段生物领域论文的摘要，请问当中提到了哪些生物学上的分类为物种的名称？{abstract}。"
                        f"你的回答必须是一个python能够直接转为list的字符串，即形如'[]'的结构。其中是物种名称，若没有提到，则返回一个空的列表。"},
        ]
    )

    result = response['choices'][0]['message']['content']
    print(result)
    list_str = re.findall(r'\[.*?\]', result)[-1]
    return eval(list_str)


if __name__ == '__main__':
    with open('../output.txt', 'a', encoding='utf-8') as file:
        file.write('过去10年内发表文章:\n\n')

    get_articles()
    analyse_article()
