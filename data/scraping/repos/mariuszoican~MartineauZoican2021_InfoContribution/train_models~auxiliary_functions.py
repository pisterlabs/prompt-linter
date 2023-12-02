import pandas as pd
import numpy as np
import math
import os
import nltk
import Powerset as ps
from nltk.tokenize import word_tokenize
from gensim import corpora, models, similarities
import os
import time
import sys
import warnings
import random
from gensim.models import CoherenceModel
warnings.filterwarnings("ignore", category=UserWarning)


# functions to organize data before building matrix
# -------------------------------------------------
def construct_indu_index_mapping(df):
    """
    Construct a dictionary with
    key: industry code
    value: indexes of all reports in the dataframe
    """
    industries_to_index = {}
    industries = df["ggroup"].dropna().astype(int).unique()
    industries = industries.tolist()
    quarters = (df["year"].astype("str") + " q" + df["quarter"].astype("str")).unique()
    for i in range(df.shape[0]):
        row = df.iloc[i, :]
        if math.isnan(row["ggroup"]):
            continue
        industries_to_index[int(row["ggroup"])] = industries_to_index.get(int(row["ggroup"]), set())
        industries_to_index[int(row["ggroup"])].add(i)
    return industries_to_index

def construct_ticker_index_mapping(df):
    """
    Construct a dictionary with
    key: Ticker
    value: indexes of all reports in the dataframe
    """
    ticker_to_index = {}
    tickers = df["Ticker"].dropna().astype(str).unique()
    tickers = tickers.tolist()
    quarters = (df["year"].astype("str") + " q" + df["quarter"].astype("str")).unique()
    for i in range(df.shape[0]):
        row = df.iloc[i, :]
        ticker_to_index[row["Ticker"]] = ticker_to_index.get(row["Ticker"], set())
        ticker_to_index[row["Ticker"]].add(i)
    return ticker_to_index

def construct_quar_index_mapping(df):
    """
    Construct a dictionary with
    key: quarter
    value: indexes of all reports in the dataframe
    """
    quarters = (df["year"].astype("str") + " q" + df["quarter"].astype("str")).unique()
    quarter_to_index = {}
    for i in range(df.shape[0]):
        row = df.iloc[i, :]
        quarter = row["year"].astype("str") + " q" + row["quarter"].astype("str")
        quarter_to_index[quarter] = quarter_to_index.get(quarter, set())
        quarter_to_index[quarter].add(i)
    return quarter_to_index
def construct_analyst_index_mapping(df, all_files_dcns):
    """
    Construct a dictionary with
    key: analyst
    value: indexes of all reports in the dataframe with the given DCNs(unique identification code for the reports)
    """
    analyst_to_index = {}
    for i, (_, dcn) in enumerate(all_files_dcns):
        analyst = max(df[df["DCN"] == dcn]["Analyst"])
        if not analyst is np.nan:
            analyst_to_index[analyst] = analyst_to_index.get(analyst, []) + [i]
    return analyst_to_index

def get_all_companies(df, indexes):
    """
    Return the set of companies in the dataframe with the given indexes
    """
    raw_companies = df.iloc[list(indexes), 4].unique()
    all_companies = set()
    for item in raw_companies:
        l = item.split(",")
        for company in l:
            all_companies.add(company.strip(" ").strip("^L19"))
    return all_companies
def get_company_files_training(target_dcns):
    """
    Return a list of tuples that contains file paths and DCNs of all reports with the target DCNs
    """
    # directory = r".\PDFParsing\parsed_files"
    directory = r"../../PDFParsing/new_all_parsed"
    files = []
    temp = os.path.join(directory)
    list_files = os.listdir(temp)
    for item in list_files:
        l = item.split("-")
        dcn = l[-1].rstrip(".txt").replace("(1)","")
        while dcn and not dcn[-1].isdigit():
            dcn = dcn[:-1]
        while dcn and not dcn[0].isdigit():
            dcn = dcn[1:]
        if dcn:
            dcn = int(dcn)
        else:
            continue
        if dcn in target_dcns:
            files.append((os.path.join(temp, item), dcn))
    return files
def get_company_files(target_dcns):
    """
    Return a list of tuples that contains file paths and DCNs of all reports with the target DCNs
    """
    # directory = r".\PDFParsing\parsed_files"
    directory = r"../PDFParsing/new_all_parsed"
    files = []
    temp = os.path.join(directory)
    list_files = os.listdir(temp)
    for item in list_files:
        l = item.split("-")
        dcn = l[-1].rstrip(".txt").replace("(1)","")
        while dcn and not dcn[-1].isdigit():
            dcn = dcn[:-1]
        while dcn and not dcn[0].isdigit():
            dcn = dcn[1:]
        if dcn:
            dcn = int(dcn)
        else:
            continue
        if dcn in target_dcns:
            files.append((os.path.join(temp, item), dcn))
    return files

def construct_corpus(all_files_dcns):
    words = []
    did = 0
    for fname, _ in all_files_dcns:
        f = open(fname, 'r')
        result = f.readlines()
        tokens = []
        for i, line in enumerate(result):
            if "redistribut reproduct prohibit without written permiss copyright cfra document intend provid person invest advic take account specif invest" in line \
                    or "redistribut reproduct prohibit without prior written permiss copyright cfra" in line \
                    or "object financi situat particular need specif person may receiv report investor seek independ financi advic regard suitabl and/or appropri make" in line \
                    or "invest implement invest strategi discuss document understand statement regard futur prospect may realiz investor note incom" in line \
                    or "invest may fluctuat valu invest may rise fall accordingli investor may receiv back less origin invest investor seek advic concern impact" in line \
                    or "invest may person tax posit tax advisor pleas note public date document may contain specif inform longer current use make" in line \
                    or "invest decis unless otherwis indic intent updat document" in line:
                continue

            if "mm" not in line and len(word_tokenize(line)) > 2:
                tokens.extend(word_tokenize(line))
        #         tokens = word_tokenize(result)
        tokens = list(filter(("--").__ne__, tokens))
        tokens = list(filter(("fy").__ne__, tokens))
        tokens = list(filter(("could").__ne__, tokens))
        tokens = list(filter(("would").__ne__, tokens))
        tokens = list(filter(("like").__ne__, tokens))
        tokens = list(filter(("see").__ne__, tokens))
        tokens = list(filter(("also").__ne__, tokens))
        tokens = list(filter(("one").__ne__, tokens))
        tokens = list(filter(("vs").__ne__, tokens))
        tokens = list(filter(("may").__ne__, tokens))
        tokens = list(filter(("herein").__ne__, tokens))
        tokens = list(filter(("mr").__ne__, tokens))
        tokens = list(filter(("plc").__ne__, tokens))
        tokens = list(filter(("use").__ne__, tokens))
        tokens = list(filter(("cfra").__ne__, tokens))
        tokens = list(filter(("et").__ne__, tokens))
        tokens = list(filter(("am").__ne__, tokens))
        tokens = list(filter(("pm").__ne__, tokens))
        tokens = list(filter(("compani").__ne__, tokens))
        tokens = list(filter(("otherwis").__ne__, tokens))
        tokens = list(filter(("year").__ne__, tokens))
        tokens = list(filter(("analys").__ne__, tokens))
        tokens = list(filter(("research").__ne__, tokens))
        tokens = list(filter(("analyst").__ne__, tokens))
        tokens = list(filter(("believ").__ne__, tokens))
        tokens = list(filter(("report").__ne__, tokens))
        tokens = list(filter(("cowen").__ne__, tokens))
        tokens = list(filter(("llc").__ne__, tokens))
        tokens = list(filter(("y/i").__ne__, tokens))
        tokens = list(filter(("estim").__ne__, tokens))
        tokens = list(filter(("total").__ne__, tokens))
        tokens = list(filter(("price").__ne__, tokens))
        tokens = list(filter(("new").__ne__, tokens))
        tokens = list(filter(("ttm").__ne__, tokens))
        tokens = list(filter(("page").__ne__, tokens))
        tokens = list(filter(("disclosur").__ne__, tokens))
        tokens = list(filter(("and/or").__ne__, tokens))
        tokens = list(filter(("barclay").__ne__, tokens))
        tokens = list(filter(("deutsch").__ne__, tokens))
        tokens = list(filter(("without").__ne__, tokens))
        tokens = list(filter(("provid").__ne__, tokens))
        tokens = list(filter(("written").__ne__, tokens))
        tokens = list(filter(("overal").__ne__, tokens))
        tokens = list(filter(("unit").__ne__, tokens))
        tokens = list(filter(("lower").__ne__, tokens))
        tokens = list(filter(("higher").__ne__, tokens))
        # new roll: 25/June
        tokens = list(filter(("morningstar").__ne__, tokens))
        tokens = list(filter(("bofa").__ne__, tokens))
        tokens = list(filter(("sm").__ne__, tokens))
        tokens = list(filter(("ep").__ne__, tokens))
        tokens = list(filter(("guidanc").__ne__, tokens))
        tokens = list(filter(("com").__ne__, tokens))
        tokens = list(filter(("inc").__ne__, tokens))
        tokens = list(filter(("analysi").__ne__, tokens))
        tokens = list(filter(("includ").__ne__, tokens))
        tokens = list(filter(("subject").__ne__, tokens))
        tokens = list(filter(("time").__ne__, tokens))
        tokens = list(filter(("still").__ne__, tokens))
        tokens = list(filter(("think").__ne__, tokens))
        tokens = list(filter(("come").__ne__, tokens))
        tokens = list(filter(("take").__ne__, tokens))
        tokens = list(filter(("much").__ne__, tokens))
        tokens = list(filter(("even").__ne__, tokens))
        tokens = list(filter(("first").__ne__, tokens))
        tokens = list(filter(("make").__ne__, tokens))
        tokens = list(filter(("busi").__ne__, tokens))
        tokens = list(filter(("versu").__ne__, tokens))
        tokens = list(filter(("parti").__ne__, tokens))
        tokens = list(filter(("opinion").__ne__, tokens))
        tokens = list(filter(("yoy").__ne__, tokens))
        tokens = list(filter(("net").__ne__, tokens))
        tokens = list(filter(("million").__ne__, tokens))
        tokens = list(filter(("given").__ne__, tokens))
        tokens = list(filter(("note").__ne__, tokens))
        tokens = list(filter(("morgan").__ne__, tokens))
        tokens = list(filter(("stanley").__ne__, tokens))
        tokens = list(filter(("sg").__ne__, tokens))
        tokens = list(filter(("month").__ne__, tokens))
        tokens = list(filter(("recent").__ne__, tokens))
        tokens = list(filter(("look").__ne__, tokens))
        tokens = list(filter(("current").__ne__, tokens))
        tokens = list(filter(("remain").__ne__, tokens))
        tokens = list(filter(("view").__ne__, tokens))
        tokens = list(filter(("po").__ne__, tokens))
        tokens = list(filter(("iqmethod").__ne__, tokens))
        tokens = list(filter(("declin").__ne__, tokens))
        tokens = list(filter(("increas").__ne__, tokens))
        tokens = list(filter(("sfg").__ne__, tokens))
        tokens = list(filter(("averag").__ne__, tokens))
        tokens = list(filter(("base").__ne__, tokens))
        tokens = list(filter(("reflect").__ne__, tokens))
        tokens = list(filter(("ffo").__ne__, tokens))
        if "bankdatesecur" in tokens:
            continue
        words.append(tokens)
        did += 1

    dictionary_LDA = corpora.Dictionary(words)
    corpus = [dictionary_LDA.doc2bow(list_of_tokens) for list_of_tokens in words]
    return words, dictionary_LDA, corpus


# normalize rows in a two-dimensional matrix
def normalize_rows(x: np.ndarray):  # function to normalize rows in a two-dimensional materix
    return x / np.linalg.norm(x, ord=2, axis=1, keepdims=True)
# get informational diversity measure
def diversity(loading_matrix):
    ld_matrix_norm = normalize_rows(loading_matrix)  # normalize all row vectors
    cosine_matrix = np.dot(ld_matrix_norm, ld_matrix_norm.T)  # compute dot products across normalized rows
    avg_similarity = cosine_matrix[np.triu_indices(np.shape(cosine_matrix)[1], k=1)].mean()

    if np.shape(loading_matrix)[0] == 1:
        return 0
    else:
        return 1 - avg_similarity
# randomly draw shapley values
def coalition_sample(lm, smple):
    # number of analysts
    no_analysts = np.shape(lm)[0]

    # get a random number between 1 and 2^(no analysts)

    # draw random numbers (decimal)
    #list_samples = [random.randrange(1, 2 ** no_analysts) for x in range(0, smple)]

    list_samples=[]
    no_samples=0
    while (no_samples<smple):
        x=random.randrange(1, 2 ** no_analysts)
        if not(x in list_samples):
            list_samples.append(x)
            no_samples+=1


    # list_samples = np.random.choice(range(1, 2 ** no_analysts), size=smple, replace=False)
    # convert random sample to binary (corresponding to rows in the power set)
    list_samples_bin = [[int(x) for x in list(format(y, "0%ib" % no_analysts))] for y in list_samples]

    shapley_sample = [lm[np.flatnonzero(x)] for x in list_samples_bin]

    return shapley_sample, [[index for index, value in enumerate(lst) if value == 1] for lst in list_samples_bin]

# shapley values function
def shapley_values(loading_matrix):
    loading_matrix = normalize_rows(loading_matrix)

    no_analysts = np.shape(np.dot(loading_matrix, loading_matrix.T))[1]  # number of analysts
    list_analysts = [x for x in range(no_analysts)]
    data = pd.DataFrame(columns={'Analyst', 'InfoContribution'})

    for k in range(no_analysts):
        list_minusone = [x for x in list_analysts if x != k]  # list without the analyst
        all_sets = [x for x in ps.powerset(list_minusone) if x]

        shapley_value = []

        for coalition in all_sets:
            other_coal = loading_matrix[coalition, :].sum(axis=0)
            other_coal = other_coal / np.linalg.norm(other_coal, ord=2, axis=0, keepdims=True)

            contribution = 1 - np.dot(other_coal, loading_matrix[k, :])

            shapley_value.append(contribution)

            # print(coalition, np.dot(other_coal,loading_matrix[k,:]), contribution)

        # print(np.array(shapley_value).mean())
        data = data.append({'Analyst': k, 'InfoContribution': np.array(shapley_value).mean()}, ignore_index=True)

    return data
# shapley values function with random draw
def shapley_values_draw(loading_matrix, no_draws):
    loading_matrix = normalize_rows(loading_matrix)

    no_analysts = np.shape(np.dot(loading_matrix, loading_matrix.T))[1]  # number of analysts
    list_analysts = [x for x in range(no_analysts)]
    data = pd.DataFrame(columns={'Analyst', 'InfoContribution'})

    for k in range(no_analysts):
        print (k)
        loading_others = np.delete(loading_matrix, k, 0)
        all_sets = coalition_sample(np.delete(loading_matrix, k, 0), no_draws)[1]

        shapley_value = []

        for coalition in all_sets:
            other_coal = loading_others[coalition, :].sum(axis=0)
            other_coal = other_coal / np.linalg.norm(other_coal, ord=2, axis=0, keepdims=True)

            contribution = 1 - np.dot(other_coal, loading_matrix[k, :])

            shapley_value.append(contribution)

            # print(coalition, np.dot(other_coal,loading_matrix[k,:]), contribution)

        # print(np.array(shapley_value).mean())
        data = data.append({'Analyst': k, 'InfoContribution': np.array(shapley_value).mean()}, ignore_index=True)

    return data

# compute Shapley values and Information Diversity)
def get_shapley(df, industry, quarter,lda_model,num_topics):
    LDA_Objects = get_factor_matrix(df, industry, quarter,lda_model,num_topics)

    loading_matrices = LDA_Objects[0]

    max_analyst_to_sample = 16  # compute full Shapley for <= x analysts, 16 is the 80% quantile by industry-quarter.
    print(max_analyst_to_sample)

    list_of_dataframes = []
    for i in range(len(loading_matrices)):

        temp = loading_matrices[i]  # get a particular stock

        if len(temp[2])==0: # deal with empty matrix exceptions
            continue

        print(temp[0])
        if [;len(temp[2]) <= max_analyst_to_sample:
            sval = shapley_values(temp[1])
        else:
            sval = shapley_values_draw(temp[1], 2 ** max_analyst_to_sample - 1)

        sval['Analyst'] = sval['Analyst'].apply(lambda x: list(temp[2].keys())[int(x)])
        sval['InfoDiversity'] = diversity(temp[1])
        sval['Ticker'] = temp[0]
        sv'' \
          'al['Industry'] = LDA_Objects[1]
        sval['Quarter'] = LDA_Objects[2]

        list_of_dataframes.append(sval)


    data_industry_quarter = list_of_dataframes[0].append(list_of_dataframes[1:], ignore_index=True)
    columns = ['Ticker', 'Quarter', 'Industry', 'InfoDiversity', 'Analyst', 'InfoContribution']
    data_industry_quarter = data_industry_quarter[columns]

    return data_industry_quarter

# get industry-level factors
def get_factor_industry(df, industry, quarter,lda_model,num_topics):
    # dictionary: {Key=Industry Code, Value=Index of Report in Metadata}
    industries_to_index = construct_indu_index_mapping(df)

    # dictionary: {Key = Quarter 'YYYY qQ', Value = Index of Report in Metadata}
    quarter_to_index = construct_quar_index_mapping(df)

    # select all report indices (rows in metadata) for the industry-quarter
    indexes = industries_to_index[industry].intersection(quarter_to_index[quarter])
    # DCN is the unique identification code for the reports

    # subset_companies = ["AAL.OQ", 'ALK.N', 'FDX.N', "DAL.N", "UAL.OQ"]
    dcns = set(df.iloc[list(indexes), :]["DCN"])
    all_files_dcns= get_company_files(dcns)
    # dictionary: {Key=Analyst Name, Value = Index of Report in Metadata}

    words, dictionary_LDA, corpus=construct_corpus(all_files_dcns)
    dcns = set(df.iloc[list(indexes), :]["DCN"])
    dcn_company = get_company_files(dcns)
    # print (dcn_company)
    analyst_to_index = construct_analyst_index_mapping(df, dcn_company)
    matrix = []
    row = [0] * num_topics
    all_words=[]
    for i in range(len(all_files_dcns)):
        all_words.extend(words[i])

    temp_bow=lda_model.id2word.doc2bow(all_words)
    topics = lda_model[temp_bow]
    for index, dist in topics:
        row[index] = dist
    matrix.append(row)
    matrix = np.array(matrix)

    return matrix.mean(axis=0)

def get_factor_ticker(df, ticker, quarter,lda_model,num_topics):
    # dictionary: {Key=Industry Code, Value=Index of Report in Metadata}
    ticker_to_index = construct_ticker_index_mapping(df)

    # dictionary: {Key = Quarter 'YYYY qQ', Value = Index of Report in Metadata}
    quarter_to_index = construct_quar_index_mapping(df)

    # select all report indices (rows in metadata) for the industry-quarter
    indexes = ticker_to_index[ticker].intersection(quarter_to_index[quarter])
    # DCN is the unique identification code for the reports

    # subset_companies = ["AAL.OQ", 'ALK.N', 'FDX.N', "DAL.N", "UAL.OQ"]
    dcns = set(df.iloc[list(indexes), :]["DCN"])
    all_files_dcns= get_company_files(dcns)
    # dictionary: {Key=Analyst Name, Value = Index of Report in Metadata}

    words, dictionary_LDA, corpus=construct_corpus(all_files_dcns)
    dcns = set(df.iloc[list(indexes), :]["DCN"])
    dcn_company = get_company_files(dcns)
    # print (dcn_company)
    analyst_to_index = construct_analyst_index_mapping(df, dcn_company)
    matrix = []
    row = [0] * num_topics
    all_words=[]
    for i in range(len(all_files_dcns)):
        all_words.extend(words[i])

    temp_bow=lda_model.id2word.doc2bow(all_words)
    topics = lda_model[temp_bow]
    for index, dist in topics:
        row[index] = dist
    matrix.append(row)
    matrix = np.array(matrix)

    return matrix.mean(axis=0)

# get a factor loading matrix for each stock + analyst names
def get_factor_matrix(df, industry, quarter,lda_model,num_topics):
    # dictionary: {Key=Industry Code, Value=Index of Report in Metadata}
    industries_to_index = construct_indu_index_mapping(df)

    # dictionary: {Key = Quarter 'YYYY qQ', Value = Index of Report in Metadata}
    quarter_to_index = construct_quar_index_mapping(df)

    # select all report indices (rows in metadata) for the industry-quarter
    indexes = industries_to_index[industry].intersection(quarter_to_index[quarter])
    # set of all company names for the industry-quarter
    all_companies = df.iloc[list(indexes), :].groupby('TICKER')["DCN"].count().reset_index()['TICKER'].tolist()
    # DCN is the unique identification code for the reports

    # subset_companies = ["AAL.OQ", 'ALK.N', 'FDX.N', "DAL.N", "UAL.OQ"]
    dcns = set(df.iloc[list(indexes), :]["DCN"])
    all_files_dcns= get_company_files(dcns)
    # dictionary: {Key=Analyst Name, Value = Index of Report in Metadata}
    analyst_to_index = construct_analyst_index_mapping(df, all_files_dcns)



    loading_matrices = []
    for companies in all_companies:
        # print(companies)
        dcns = set(df.iloc[list(indexes), :][df.TICKER == companies]["DCN"])
        dcn_company = get_company_files(dcns)
        # print (dcn_company)
        words, dictionary_LDA, corpus = construct_corpus(dcn_company)
        analyst_to_index = construct_analyst_index_mapping(df, dcn_company)
        matrix = []
        for analyst, anal_indexes in analyst_to_index.items():
            row = [0] * num_topics
            all_words = []
            for i in anal_indexes:
                all_words.extend(words[i])
            temp_bow = lda_model.id2word.doc2bow(all_words)
            topics = lda_model[temp_bow]
            for index, dist in topics:
                row[index] = dist
            matrix.append(row)
        matrix = np.array(matrix)
        loading_matrices.append((companies, matrix, analyst_to_index))

    return [loading_matrices, industry, quarter]
