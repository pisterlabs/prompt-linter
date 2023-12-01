

# imports
from langchain import PromptTemplate
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS 
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA, LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.docstore.document import Document
import time 
import os

# for AWS
ENV_DIR_CODE = os.environ.get('DIR_CODE')
ENV_DIR_PUBMED = os.environ.get('DIR_PUBMED')

# local imports
dir_code = "/home/javaprog/Code/PythonWorkspace/"
if ENV_DIR_CODE:
    dir_code = ENV_DIR_CODE
import sys
sys.path.insert(0, dir_code + 'MachineLearningPython/DccKP/GPT/')
import dcc_gpt_lib
import dcc_langchain_lib

# constants
DIR_DATA = "/home/javaprog/Data/"
DIR_DOCS = DIR_DATA + "ML/Llama2Test/Genetics/Docs"
DIR_VECTOR_STORE = DIR_DATA + "ML/Llama2Test/Genetics/VectorStore"
FILE_MODEL = DIR_DATA + "ML/Llama2Test/Model/llama-2-7b-chat.ggmlv3.q8_0.bin"
FILE_MODEL = DIR_DATA + "ML/Llama2Test/Model/llama-2-13b-chat.ggmlv3.q8_0.bin"
PROMPT = """Use the following piece of information to anser the user's question.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing elase.
Helpful answer:
"""
GPT_PROMPT = """
Below are the abstracts from different research papers on gene {gene}. 
Please read through the abstracts and as a genetics researcher write a 100 word summary that synthesizes the key findings of the papers on the biology of gene {gene}
{abstracts}
"""

GPT_PROMPT = """
Below are the abstracts from different research papers on genetics. 
Please read through the abstracts and as a genetics researcher write a 100 word summary that synthesizes the key biology findings of the papers
{abstracts}
"""

GPT_PROMPT = "{text_prompt}"

# methods
def set_prompt(prompt, log=False):
    '''
    returns the prompt to use
    '''
    result_prompt = PromptTemplate(template=prompt, input_variables=['gene', 'abstracts'])

    return result_prompt

def load_llm(file_model, log=False):
    if log:
        print("loading model: {}".format(file_model))

    llm = CTransformers(
        model=file_model,
        model_type = "llama",
        max_new_tokens = 512,
        # temperature = 0.1
        temperature = 0.5
    )

    if log:
        print("loaded model from: {}".format(file_model))

    return llm

def get_qa_chain(llm, prompt, db=None, log=False):
    '''
    get the langchain
    '''
    qa_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type="stuff",
        # retriever = db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )

    return qa_chain

def get_qa_bot(dir_db, file_model, prompt, log=False):
    # embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
    # db = FAISS.load_local(dir_db, embeddings)

    llm = load_llm(file_model=file_model, log=log)

    prompt_qa = set_prompt(prompt=prompt, log=log)

    chain_qa = get_qa_chain(llm=llm, prompt=prompt_qa, log=log)

    return chain_qa



def get_inference(gene, text_input, log=False):
    '''
    do the llm inference
    '''
    if log:
        print("doing llm inference using gene: {} and abstracts: \n{}\n".format(gene, text_input))

    # Define prompt
    # prompt_template = """Write a concise summary of the following:
    # "{text}"
    # CONCISE SUMMARY:"""
    # prompt = GPT_PROMPT
    prompt = dcc_langchain_lib.TEMPLATE_PROMPT_EMPTY
    prompt_template = PromptTemplate.from_template(prompt)

    # build the instruction
    text_instruction = dcc_langchain_lib.get_biology_summary_instruction(gene=gene, text=text_input)

    # build the text prompt
    text_prompt = dcc_langchain_lib.get_prompt_no_tags(instruction=text_instruction, system_prompt=dcc_langchain_lib.ROLE_BIOLOGIST_IN_GENETICS)

    # log
    if log:
        print("Using prompt text: \n{}\n".format(text_prompt))

    # Define LLM chain
    llm = load_llm(file_model=file_model, log=log)
    # llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
    llm_chain = LLMChain(llm=llm, prompt=prompt_template)

    # Define StuffDocumentsChain
    stuff_chain = StuffDocumentsChain(
        llm_chain=llm_chain, document_variable_name="text_prompt"
    )

    # loader = text_to_docs(abstracts)
    # docs = loader.load()
    docs = Document(page_content=text_prompt)
    print("\n docs: \n{}".format(docs))
    result = stuff_chain.run([docs])

    # return
    return result

def get_inference_gene_abstracts(gene, abstracts, chain_qa, log=False):
    '''
    do the llm inference
    '''
    if log:
        print("doing llm inference using gene: {} and abstcats: \n{}".format(gene, abstracts))
    result = chain_qa({'gene': gene, 'abstracts': abstracts})

    # if log:
    #     print("got result: {}".format(result))

    # return
    return result


# main
if __name__ == "__main__":
    # initialize
    # dir_db = DIR_VECTOR_STORE
    file_model = FILE_MODEL
    prompt = PROMPT
    num_abstracts_per_summary = 5
    max_per_level = 50
    id_run = 7
    num_level = 0

    # get the connection
    conn = dcc_gpt_lib.get_connection()

    # get the name and prompt of the run
    _, name_run, prompt_run = dcc_gpt_lib.get_db_run_data(conn=conn, id_run=id_run)
    print("got run: {} with prompt: \n'{}'\n".format(name_run, prompt_run))

    # get the list of searches
    list_searches = dcc_gpt_lib.get_db_list_ready_searches(conn=conn, num_searches=1)
    # get the best abstracts for gene

    for search in list_searches:
        id_search = search.get('id')
        id_top_level_abstract = -1
        gene = search.get('gene')

        # log
        print("\nprocessing search: {} for gene: {} for run id: {} of name: {}".format(id_search, gene, id_run, name_run))
        # time.sleep(5)

        # get all the abstracts for the document level and run
        list_abstracts = dcc_gpt_lib.get_list_abstracts(conn=conn, id_search=id_search, id_run=id_run, num_level=num_level, num_abstracts=max_per_level, log=True)

        for i in range(0, len(list_abstracts), num_abstracts_per_summary):
            list_sub = list_abstracts[i : i + num_abstracts_per_summary] 

            # for the sub list
            str_abstracts = ""
            for item in list_sub:
                abstract = item.get('abstract')
                word_count = len(abstract.split())
                print("using abstract with count: {} and content: \n{}".format(word_count, abstract))
                str_abstracts = str_abstracts + "\n" + abstract

            # log
            # print("using abstract count: {} for gpt query for level: {} and search: {}".format(len(list_sub), num_level, id_search))

            # build the prompt
            str_prompt = prompt_run.format(gene, gene, str_abstracts)


#             str_abstracts = """
# Below are the abstracts from different research papers on gene CCDC171. 
# Please read through the abstracts and as a genetics researcher write a 100 word summary that synthesizes the key findings of the papers on the biology of gene CCDC171
# MicroRNAs (miRNAs) play an important role in posttranscriptional regulation by binding to target sites in the 3'UTR of protein-coding genes. Genetic variation within target sites may potentially disrupt the binding activity of miRNAs, thereby impacting this regulation. In the current study, we investigated whether any established BMI-associated genetic variants potentially function by altering a miRNA target site. The genomic positions of all predicted miRNA target site seed regions were identified, and these positions were queried in the T2D Knowledge Portal for variants that associated with BMI in the GIANT UK Biobank. This in silico analysis identified ten target site variants that associated with BMI with a P value ≤ 5\u2009×\u200910. These ten variants mapped to nine genes, FAIM2, CCDC171, ADPGK, ZNF654, MLXIP, NT5C2, SHISA4, SLC25A22, and CTNNB1. In vitro functional analyses showed that five of these target site variants, rs7132908 (FAIM2), rs4963153 (SLC25A22), rs9460 (ADPGK), rs11191548 (NT5C2), and rs3008747 (CCDC171), disrupted the binding activity of miRNAs to their target in an allele-specific manner. In conclusion, our study suggests that some established variants for BMI may function by altering miRNA binding to a 3'UTR target site.\nWomen with epithelial ovarian cancer (EOC) are usually treated with platinum/taxane therapy after cytoreductive surgery but there is considerable inter-individual variation in response. To identify germline single-nucleotide polymorphisms (SNPs) that contribute to variations in individual responses to chemotherapy, we carried out a multi-phase genome-wide association study (GWAS) in 1,244 women diagnosed with serous EOC who were treated with the same first-line chemotherapy, carboplatin and paclitaxel. We identified two SNPs (rs7874043 and rs72700653) in TTC39B (best P=7x10-5, HR=1.90, for rs7874043) associated with progression-free survival (PFS). Functional analyses show that both SNPs lie in a putative regulatory element (PRE) that physically interacts with the promoters of PSIP1, CCDC171 and an alternative promoter of TTC39B. The C allele of rs7874043 is associated with poor PFS and showed increased binding of the Sp1 transcription factor, which is critical for chromatin interactions with PSIP1. Silencing of PSIP1 significantly impaired DNA damage-induced Rad51 nuclear foci and reduced cell viability in ovarian cancer lines. PSIP1 (PC4 and SFRS1 Interacting Protein 1) is known to protect cells from stress-induced apoptosis, and high expression is associated with poor PFS in EOC patients. We therefore suggest that the minor allele of rs7874043 confers poor PFS by increasing PSIP1 expression.\nChronic kidney disease (CKD) is an important public health problem in the world. The aim of our research was to identify novel potential serum biomarkers of renal injury. ELISA assay showed that cytokines and chemokines IL-1β, IL-2, IL-4, IL-5, IL-6, IL-7, IL-8, IL-9, IL-10, IL-12 (p70), IL-13, IL-15, IL-17, Eotaxin, FGFb, G-CSF, GM-CSF, IP-10, MCP-1, MIP-1α, MIP-1β, PDGF-1bb, RANTES, TNF-α and VEGF were significantly higher (R > 0.6,  value < 0.05) in the serum of patients with CKD compared to healthy subjects, and they were positively correlated with well-established markers (urea and creatinine). The multiple reaction monitoring (MRM) quantification method revealed that levels of HSP90B2, AAT, IGSF22, CUL5, PKCE, APOA4, APOE, APOA1, CCDC171, CCDC43, VIL1, Antigen KI-67, NKRF, APPBP2, CAPRI and most complement system proteins were increased in serum of CKD patients compared to the healthy group. Among complement system proteins, the C8G subunit was significantly decreased three-fold in patients with CKD. However, only AAT and HSP90B2 were positively correlated with well-established markers and, therefore, could be proposed as potential biomarkers for CKD.\nRecent genome-wide association studies (GWAS) have identified 97 body-mass index (BMI) associated loci. We aimed to evaluate if dietary intake modifies BMI associations at these loci in the Singapore Chinese population. We utilized GWAS information from six data subsets from two adult Chinese population (N\u2009=\u20097817). Seventy-eight genotyped or imputed index BMI single nucleotide polymorphisms (SNPs) that passed quality control procedures were available in all datasets. Alternative Healthy Eating Index (AHEI)-2010 score and ten nutrient variables were evaluated. Linear regression analyses between z score transformed BMI (Z-BMI) and dietary factors were performed. Interaction analyses were performed by introducing the interaction term (diet x SNP) in the same regression model. Analysis was carried out in each cohort individually and subsequently meta-analyzed using the inverse-variance weighted method. Analyses were also evaluated with a weighted gene-risk score (wGRS) contructed by BMI index SNPs from recent large-scale GWAS studies. Nominal associations between Z-BMI and AHEI-2010 and some dietary factors were identified (P\u2009=\u20090.047-0.010). The BMI wGRS was robustly associated with Z-BMI (P\u2009=\u20091.55\u2009×\u200910) but not with any dietary variables. Dietary variables did not significantly interact with the wGRS to modify BMI associations. When interaction analyses were repeated using individual SNPs, a significant association between cholesterol intake and rs4740619 (CCDC171) was identified (β\u2009=\u20090.077, adjP\u2009=\u20090.043). The CCDC171 gene locus may interact with cholesterol intake to increase BMI in the Singaporean Chinese population, however most known obesity risk loci were not associated with dietary intake and did not interact with diet to modify BMI levels.
# """

            # do inference
            gene = 'CCDC171'
            response = get_inference(gene=gene, text_input=str_abstracts, log=True)
            print("got response: \n\n{}".format(response))
            # print("using GPT prompt: \n{}".format(str_prompt))

            # insert results and links
            # dcc_gpt_lib.insert_gpt_results(conn=conn, id_search=id_search, num_level=num_level, list_abstracts=list_abstracts, 
            #                                 gpt_abstract=str_chat, id_run=id_run, name_run=name_run, log=True)
            # time.sleep(30)
            break
            time.sleep(3)




