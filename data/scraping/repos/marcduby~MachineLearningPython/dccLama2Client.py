

# imports
from langchain import HuggingFacePipeline
from transformers import AutoTokenizer
import transformers
import torch
from langchain import PromptTemplate,  LLMChain
import os 
import pymysql as mdb
from time import gmtime, strftime
import time



# constants
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
PROMPT_LLM = """
Write a concise summary of the following text delimited by triple backquotes.
Return your response in sentences which covers the key points of the text.
```{input_text}```
BULLET POINT SUMMARY:
"""

PROMPT_COMMAND = "Summarize the information related to {} from the following text delimited by triple backquotes.\n"
PROMPT_LLM_GENE = """
```{input_text}```
SUMMARY:
"""


# DB constants
DB_PASSWD = os.environ.get('DB_PASSWD')
NUM_ABSTRACT_LIMIT = 5
SCHEMA_GPT = "gene_gpt"
DB_PAPER_TABLE = "pgpt_paper"
DB_PAPER_ABSTRACT = "pgpt_paper_abtract"

SQL_SELECT_ABSTRACT_BY_TITLE = "select id from {}.pgpt_paper_abstract where title = %s".format(SCHEMA_GPT)
SQL_SELECT_ABSTRACT_LIST_LEVEL_0 = """
select abst.id, abst.abstract 
from {}.pgpt_paper_abstract abst, {}.pgpt_search_paper seapaper 
where abst.document_level = 0 and seapaper.paper_id = abst.pubmed_id and seapaper.search_id = %s limit %s
""".format(SCHEMA_GPT, SCHEMA_GPT, SCHEMA_GPT)
# and abst.id not in (select child_id from {}.pgpt_gpt_paper where search_id = %s) limit %s

SQL_SELECT_ABSTRACT_LIST_LEVEL_HIGHER = """
select distinct abst.id, abst.abstract, abst.document_level
from {}.pgpt_paper_abstract abst, {}.pgpt_gpt_paper gpt
where abst.document_level = %s and gpt.parent_id = abst.id and gpt.search_id = %s
and abst.id not in (select child_id from {}.pgpt_gpt_paper where search_id = %s) limit %s
""".format(SCHEMA_GPT, SCHEMA_GPT, SCHEMA_GPT)

# SQL_INSERT_PAPER = "insert into {}.pgpt_paper (pubmed_id) values(%s)".format(SCHEMA_GPT)
SQL_INSERT_ABSTRACT = "insert into {}.pgpt_paper_abstract (abstract, title, journal_name, document_level) values(%s, %s, %s, %s)".format(SCHEMA_GPT)
SQL_INSERT_GPT_LINK = "insert into {}.pgpt_gpt_paper (search_id, parent_id, child_id, document_level) values(%s, %s, %s, %s)".format(SCHEMA_GPT)
SQL_UPDATE_ABSTRACT_FOR_TOP_LEVEL = "update {}.pgpt_paper_abstract set search_top_level_of = %s where id = %s".format(SCHEMA_GPT)

SQL_SELECT_SEARCHES = "select id, terms, gene from {}.pgpt_search where ready='Y' limit %s".format(SCHEMA_GPT)
SQL_UPDATE_SEARCH_AFTER_SUMMARY = "update {}.pgpt_search set ready = 'N', date_last_summary = sysdate() where id = %s ".format(SCHEMA_GPT)

# methods
def get_model_tokenizer(name, log=False):
    '''
    returns the associated model and tokenizer
    '''
    # initialize
    tokenizer = AutoTokenizer.from_pretrained(name)
    pipeline = transformers.pipeline(
        "text-generation", #task
        model=name,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        max_length=3000,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id
    )

    model = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0})

    # return
    return model, tokenizer


def summarize(model, gene, command, prompt, text_to_summarize, log=False):
    '''
    summarixes the test input using the model and prompt provided
    '''
    # initialize
    command_gene = command.format(gene)
    prompt_and_command = command_gene + prompt

    if log:
        print("prompt: \n{}".format(prompt_and_command))
    prompt_final = PromptTemplate(template=prompt_and_command, input_variables=["input_text"])
    llm_chain = LLMChain(prompt=prompt_final, llm=model)
    summary = None

    # log
    if log:
        print("using prompt: \n{}".format(prompt_and_command))

    # summarize
    summary = llm_chain.run(text_to_summarize)

    # return
    return summary

def get_list_abstracts(conn, id_search, num_level=0, num_abstracts=NUM_ABSTRACT_LIMIT, log=False):
    '''
    get a list of abstract map objects
    '''
    # initialize
    list_abstracts = []
    cursor = conn.cursor()

    # pick the sql based on level
    if log:
        print("searching for abstracts got input search: {}, doc_level: {}, limit: {}".format(id_search, num_level, num_abstracts))
    if num_level == 0:
        # cursor.execute(SQL_SELECT_ABSTRACT_LIST_LEVEL_0, (id_search, id_search, num_abstracts))
        cursor.execute(SQL_SELECT_ABSTRACT_LIST_LEVEL_0, (id_search, num_abstracts))
    else:
        cursor.execute(SQL_SELECT_ABSTRACT_LIST_LEVEL_HIGHER, (num_level, id_search, id_search, num_abstracts))

    # query 
    db_result = cursor.fetchall()
    for row in db_result:
        paper_id = row[0]
        abstract = row[1]
        list_abstracts.append({"id": paper_id, 'abstract': abstract})

    # return
    return list_abstracts

def get_connection():
    ''' 
    get the db connection 
    '''
    conn = mdb.connect(host='localhost', user='root', password=DB_PASSWD, charset='utf8', db=SCHEMA_GPT)

    # return
    return conn


# main
if __name__ == "__main__":
    # # initialize
    # num_level = 0
    # id_search = 2

    # # get the db connection
    # conn = get_connection()

    # # get the abstracts
    # list_abstracts = get_list_abstracts(conn=conn, id_search=id_search, num_level=num_level, num_abstracts=5, log=True)

    # # get the llm summary
    # str_input = ""
    # if len(list_abstracts) > 1:
    #     # top level is not this level if more than 2 abstracts found at this level
    #     found_top_level = False
    #     for item in list_abstracts:
    #         abstract = item.get('abstract')
    #         print("using abstract: \n{}".format(abstract))
    #         str_input = str_input + " " + abstract

    #     # log
    #     print("using {} for gpt query for level: {} and search: {}".format(len(list_abstracts), num_level, id_search))

    # print("using text: \n{}".format(str_input))

    # # get the model
    # llm_model, tokenizer = get_model_tokenizer(MODEL_NAME)
    # print("got model: {}".format(llm_model))

    # # get the summary
    # summary = summarize(model=llm_model, gene='UBE2NL', command=PROMPT_COMMAND, prompt=PROMPT_LLM_GENE, text_to_summarize=str_input, log=True)
    # print("got summary: \n{}".format(summary))


    model = "meta-llama/Llama-2-7b-chat-hf"

    tokenizer = AutoTokenizer.from_pretrained(model)

    # max length for pipeline indicates max input token 
    pipeline = transformers.pipeline(
        "text-generation", #task
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        max_length=2500,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id
    )

    llm = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0})

    template = """
                Write a concise summary of the following text delimited by triple backquotes.
                Return your response in bullet points which covers the key points of the text.
                ```{text}```
                BULLET POINT SUMMARY:
            """

#     # test for ube2nl gene summary test 
#     template2 = """
#                 Write a summary about UBE2NL from the following text delimited by triple backquotes.
#                 ```{text}```
#                 BULLET POINT SUMMARY:
#             """

#     prompt2 = PromptTemplate(template=template2, input_variables=["text"])

#     llm_chain = LLMChain(prompt=prompt2, llm=llm)

#     text2 = """ 
# Genetic association studies for gastroschisis have highlighted several candidate variants. However, genetic basis in gastroschisis from noninvestigated heritable factors could provide new insights into the human biology for this birth defect. We aim to identify novel gastroschisis susceptibility variants by employing whole exome sequencing (WES) in a Mexican family with recurrence of gastroschisis. We employed WES in two affected half-sisters with gastroschisis, mother, and father of the proband. Additionally, functional bioinformatics analysis was based on SVS-PhoRank and Ensembl-Variant Effect Predictor. The latter assessed the potentially deleterious effects (high, moderate, low, or modifier impact) from exome variants based on SIFT, PolyPhen, dbNSFP, Condel, LoFtool, MaxEntScan, and BLOSUM62 algorithms. The analysis was based on the Human Genome annotation, GRCh37/hg19. Candidate genes were prioritized and manually curated based on significant phenotypic relevance (SVS-PhoRank) and functional properties (Ensembl-Variant Effect Predictor). Functional enrichment analysis was performed using ToppGene Suite, including a manual curation of significant Gene Ontology (GO) biological processes from functional similarity analysis of candidate genes. No single gene-disrupting variant was identified. Instead, 428 heterozygous variations were identified for which SPATA17, PDE4DIP, CFAP65, ALPP, ZNF717, OR4C3, MAP2K3, TLR8, and UBE2NL were predicted as high impact in both cases, mother, and father of the proband. PLOD1, COL6A3, FGFRL1, HHIP, SGCD, RAPGEF1, PKD1, ZFHX3, BCAS3, EVPL, CEACAM5, and KLK14 were segregated among both cases and mother. Multiple interacting background modifiers may regulate gastroschisis susceptibility. These candidate genes highlight a role for development of blood vessel, circulatory system, muscle structure, epithelium, and epidermis, regulation of cell junction assembly, biological/cell adhesion, detection/response to endogenous stimulus, regulation of cytokine biosynthetic process, response to growth factor, postreplication repair/protein K63-linked ubiquitination, protein-containing complex assembly, and regulation of transcription DNA-templated. Considering the likely gene-disrupting prediction results and similar biological pattern of mechanisms, we propose a joint "multifactorial model" in gastroschisis pathogenesis. Cancer is characterized by abnormal growth of cells. Targeting ubiquitin proteins in the discovery of new anticancer therapeutics is an attractive strategy. The present study uses the structure-based drug discovery methods to identify new lead structures, which are selective to the putative ubiquitin-conjugating enzyme E2N-like (UBE2NL). The 3D structure of the UBE2NL was evaluated using homology modeling techniques. The model was validated using standard in silico methods. The hydrophobic pocket of UBE2NL that aids in binding with its natural receptor ubiquitin-conjugating enzyme E2 variant (UBE2V) was identified through protein-protein docking study. The binding site region of the UBE2NL was identified using active site prediction tools. The binding site of UBE2NL which is responsible for cancer cell progression is considered for docking study. Virtual screening study with the small molecular structural database was carried out against the active site of UBE2NL. The ligand molecules that have shown affinity towards UBE2NL were considered for ADME prediction studies. The ligand molecules that obey the Lipinski's rule of five and Jorgensen's rule of three pharmacokinetic properties like human oral absorption etc. are prioritized. The resultant ligand molecules can be considered for the development of potent UBE2NL enzyme inhibitors for cancer therapy. Migraine without aura (MWO) is the most common among migraine group, and is mainly associated with genetic, physical and chemical factors, and hormonal changes. We aimed to identify novel non-synonymous mutations predisposing to the susceptibility to MWO in a Chinese sample using exome sequencing. Four patients with MWO from a family and four non-migraine subjects unrelated with these patients were genotyped using whole-exome sequencing. Bioinformatics analysis was used to screen possible susceptibility gene mutations, which were then verified by PCR. In four patients with MWO, six novel rare non-synonymous mutations were observed, including EDA2R (G170A), UBE2NL (T266G), GBP2 (A907G), EMR1 (C264G), CLCNKB (A1225G), and ARHGAP28 (C413G). It is worth stressing that GBP2 (A907G) was absent in any control subject. Multiple genes predispose to the susceptibility to MWO. ARHGAP28-, EMR1-, and GBP2-encoded proteins may affect angiokinesis, which supports vasogenic theory for the etiological hypothesis of this disease. CLCNKB-encoded protein may affect cell membrane potential, which is consistent with the cortical spreading depression theory. UBE2NL-encoded protein may regulate cellular responses to 5-hydroxytryptamine, which is in accordance with trigeminovascular reflex theory. EDA2R and UBE2NL are located on the X chromosome, which supports that this disease may have gender differences in genetic predisposition. Replication in larger sample size would significantly strengthen these findings. Sporadic Alzheimer disease (SAD) is the most prevalent neurodegenerative disorder. With the development of new generation DNA sequencing technologies, additional genetic risk factors have been described. Here we used various methods to process DNA sequencing data in order to gain further insight into this important disease. We have sequenced the exomes of brain samples from SAD patients and non-demented controls. Using either method, we found a higher number of single nucleotide variants (SNVs), from SAD patients, in genes present at the X chromosome. Using the most stringent method, we validated these variants by Sanger sequencing. Two of these gene variants, were found in loci related to the ubiquitin pathway (UBE2NL and ATXN3L), previously do not described as genetic risk factors for SAD.
#     """
#     result2 = llm_chain.run(text2)
#     print(" ".join(result2.split()))
#     print(result2)

    # test for virgin gene set summary test 
    templateSet = """
                describe in 500 words or less with references the patterns in the gene set {text}.
                SUMMARY:
            """

    promptSet = PromptTemplate(template=templateSet, input_variables=["text"])

    llm_chain = LLMChain(prompt=promptSet, llm=llm)

    textSet = "AGPAT2, BSCL2, CAV1, LMNA, PLIN1"

    resultSet = llm_chain.run(textSet)
    print(" ".join(resultSet.split()))

    print("\n\n###################################################\n\n")

    textSet = "CEL, HNF1B, HNF4, KLF11, NEUROD1"

    resultSet = llm_chain.run(textSet)
    print(" ".join(resultSet.split()))

    print("\n\n###################################################\n\n")


    # test for gene set summary test 
    templateSet = """
                are there any patterns in the gene set AGPAT2, BSCL2, CAV1, LMNA, PLIN1 based on the following text delimited by triple backquotes.
                ```{text}```
                SUMMARY:
            """

    promptSet = PromptTemplate(template=templateSet, input_variables=["text"])

    llm_chain = LLMChain(prompt=promptSet, llm=llm)

    textSet = """ 
AGPAT2 is a gene critical for triacylglycerol biosynthesis and is associated with various forms of lipodystrophy that result in severe metabolic complications such as hypertriglyceridemia, diabetes, and hepatic steatosis. Loss-of-function mutations in AGPAT2 lead to a reduction in adipose tissue, resulting in a lipoatrophic phenotype. AGPAT2 also plays a significant role in lipid droplet synthesis and affects various tissues, resulting in steatosis and dysfunction. The severity of metabolic complications depends on the extent of fat loss, and patients with lipodystrophies require psychological support, low-fat diets, increased physical activity, cosmetic surgery, and metreleptin replacement therapy to alleviate metabolic complications. Leptin replacement therapy has been found to be effective in treating lipodystrophic patients with hypoleptinemia. Moreover, AGPAT2 is not implicated in other rare autosomal recessive disorders, such as mandibuloacral dysplasia (MAD), characterized by mandibular and clavicular hypoplasia, joint contractures, and mottled cutaneous pigmentation. AGPAT2 mutations have been found predominantly in African ancestry, while mutations in the Seipin gene have been found in patients from families originating from Europe and the Middle East. AGPAT2 mutations are a rare cause of lipodystrophies and help understand a common phenotype seen in complex disorders such as type 2 diabetes mellitus (T2DM) and obesity.
BSCL2 is a gene that plays a crucial role in lipid metabolism and the formation of lipid droplets. Mutations in the gene lead to various disorders, including Berardinelli-Seip congenital lipodystrophy type 2 (BSCL2), peripheral neuropathies like Charcot-Marie-Tooth disease (CMT) and hereditary motor neuropathy (HMN), and hereditary spastic paraplegia (HSP). The BSCL2 mutations also lead to insulin resistance, dyslipidemia, and fatty liver. Adipose tissue transplantation and leptin administration are effective treatments for metabolic disorders resulting from Seipin deficiency. Studies have demonstrated that expressing seipin in adipose tissue alone can rescue dyslipidemia, lipodystrophy, insulin resistance, and hepatic steatosis in CGL mice. Seipin's function remains unclear but is implicated in lipid metabolism and lipid droplet assembly and maintenance.
Furthermore, the article discusses the prevalence of SARS-CoV-2 infection in patients with congenital generalized lipodystrophy (CGL) and the effectiveness of leptin replacement therapy in treating metabolic complications in affected patients. The article provides valuable information on various genetic disorders and their underlying genetic mutations and phenotypic presentations. Understanding the role of seipin in disease development and progression can help identify novel therapeutic targets for CGL and other associated disorders. Overall, understanding seipin's function can offer vital insights into various diseases, including motor neuron diseases, adipose tissue disorders, and lipid metabolism disorders. Further studies are needed to understand the pathophysiology underlying the disease and its related disorders fully.
Caveolin-1 (CAV1) is a transmembrane protein that is involved in a range of physiological processes and diseases, making it a potential diagnostic biomarker and therapeutic target for multiple conditions. It has been associated with numerous medical conditions including Alzheimer's disease, cancer, cardiovascular disease, intervertebral disc degeneration, and obesity. Additionally, CAV1 is linked to calcium homeostasis and affects cell polarity and cancer metastasis. CAV1 may also function as a potential therapeutic target for various diseases including cancer, cardiovascular disease, autoimmune diseases, and neurological disorders. There is also potential for CAV1 to be used in cancer treatment as it controls the epithelial tension necessary for eliminating oncogene-transfected cells by apical extrusion and could aid in the accumulation of nanoparticles in tumors. Studies have shown that mutations in CaV1.1 cause hypokalemic periodic paralysis, and Cx43 might participate in atrial fibrillation pathogenesis. Various drugs have been found to modulate CAV1, including simvastatin which was found to alter estrogen signaling and Pravastatin which improves endothelial barrier disruption by modulating Cav-1/eNOS pathway. Integrated bioinformatics analysis identified CAV1 as an independent prognostic factor for clear cell renal cell carcinoma. Further research could explore the potential of CAV1-mediated transcellular routes that aid in efficient cancer treatment and the development of new treatments focused on targeting CAV1 for specific medical conditions.
LMNA is a gene that encodes for lamin A/C proteins, which are essential for maintaining nuclear stiffness and cell morphology. Mutations in this gene cause laminopathies that affect various parts of the body, including the heart, skeletal muscles, and skin/bone disorders, to name a few. Hutchinson-Gilford progeria syndrome (HGPS) is one of the most notable conditions caused by LMNA mutations, where patients experience severe aging symptoms. 
Research has identified several potential therapeutic targets for these disorders, such as tyrosine kinase inhibitors or BET bromodomain inhibitors for dilated cardiomyopathy caused by LMNA mutations, inhibiting the p38α signaling pathway for cardiomyopathy, and autophagy-inducing drugs for laminopathies. However, phenotype variations caused by LMNA mutations pose a challenge in diagnosis and treatment. 
Additionally, LMNA mutations are associated with various malignancies and neurodegenerative disorders. Research on the role of this gene in regulating cell fate and identifying biomarkers for disease status has been productive in better understanding disease mechanisms. Furthermore, studies show that culturing LMNA mutated cells on different substrates can predict patient-specific phenotypic development, enabling better patient management. 
Gene therapy using nucleotide vectors is a promising approach to correcting LMNA mutations, and RB1 is believed to play a significant role in regulating cellular phenotype in laminopathy-related cells. Overall, LMNA research has helped in understanding laminopathies and the development of effective therapies for these disorders.
Perilipin-1 (PLIN1) is a protein that plays a vital role in regulating lipid metabolism and lipolysis. It coats the lipid droplets in adipocytes and influences fat storage regulation. PLIN1 is involved in various pathological conditions, including obesity, metabolic disorders, and heart failure. Studies have identified genetic variations in the perilipin gene that may impact postprandial lipoprotein metabolism and atherogenic risk. Additionally, PLIN1 has been linked to bone loss, fatty liver disease, breast cancer, and age-related hearing loss. PLIN1 expression is regulated by PPAR and PI3K-Akt pathways, and its overexpression reportedly protects against atheroma progression. PLIN1 is also involved in the fragmentation and dispersion of cytoplasmic lipid droplets in response to β-adrenergic activation of adenylate cyclase. Studies have identified PLIN1 mutations that cause familial partial lipodystrophy, severe insulin resistance, diabetes, dyslipidemia, and fatty liver. PLIN1 has been targeted for potential therapeutic interventions for age-related hearing loss and adipogenesis-related conditions. Altering DCAD shifts adipose tissue metabolism from lipogenesis to lipolysis, while exposure to carbamazepine causes lipid metabolism disorder and mitochondrial damage. A combination of dietary bio-actives, grape seed proanthocyanidin extract, and retinoic acid has also been studied for their potential therapeutic effects on directing adipogenic differentiation in human cells. Overall, research indicates the importance of PLIN1 in lipid metabolism and the potential for therapeutic interventions in various diseases and conditions.
    """


    resultSet = llm_chain.run(textSet)
    print(" ".join(resultSet.split()))
    # print(resultSet)

    print("\n\n###################################################\n\n")
    # test for gene set summary test 
    templateSet = """
                summarize any patterns in the gene set CEL, HNF1B, HNF4, KLF11, NEUROD1 based on the following text delimited by triple backquotes.
                ```{text}```
                SUMMARY:
            """

    promptSet = PromptTemplate(template=templateSet, input_variables=["text"])

    llm_chain = LLMChain(prompt=promptSet, llm=llm)

    textSet = """ 
The article presents a range of medical studies covering various topics such as cancer treatment, infectious diseases, mental health, and environmental health risk assessment. One study focuses on Congenital Ectopia Lentis (CEL), which found that patients with CEL had higher levels of corneal horizontal coma and lower levels of corneal vertical coma primary spherical aberrations compared to healthy controls. Other studies explored the efficacy and safety of CAR T-cell therapies and their potential use in treating relapsed or refractory large B-cell lymphoma and multiple myeloma. The studies also provide valuable insights into the impact of various environmental and lifestyle factors on health outcomes and the potential benefits of targeting a systolic blood pressure of less than 120 mm Hg in patients at increased cardiovascular risk. Additionally, the studies examine the prevalence and clinical outcomes of the SARS-CoV-2 variant and investigate the impact of dietary glycation compounds on salivary concentrations of Maillard reaction products. The studies were conducted using various methods such as surveys, GWAS, and cohort studies. Overall, the studies suggest several promising treatment options and screening tools for different medical conditions and provide valuable insights into the impact of environmental and lifestyle factors on health outcomes.
Hepatocyte nuclear factor 1b (HNF1B) is a gene that regulates development in various organs, including the pancreas, liver, kidney, lung, and genitourinary tract. Mutations in HNF1B are associated with multiple health conditions, including maturity-onset diabetes of the young, polycystic kidney disease, exocrine pancreatic cancer, and MRKH syndrome. HNF1B haploinsufficiency is linked to diabetes and renal disease, and loss of HNF1B function leads to MRKH syndrome. The gene has also been associated with other health conditions, such as intraductal tubulopapillary neoplasms, hyperinsulinemic hypoglycemia, and congenital anomalies of the kidney and urinary tract.
HNF1B variants impact responses to insulin-sensitizing interventions, and whole-exome sequencing can help diagnose unexplained congenital anomalies in fetuses. The gene's role in regenerative medicine has also been investigated, inducing re-epithelialization of renal tubular epithelial cells and improving downstream applications and differentiations in cells committed towards the endoderm lineage. HNF1B mutations may also account for some cases of new-onset diabetes after transplantation in pediatric patients who undergo kidney transplantation.
The article highlights the importance of genetic testing in identifying related conditions, diagnosing subtypes of MODY, and individualizing treatment for patients. Moreover, HNF1B plays a complex and diverse role in regulating gene expressions in various organs and its association with different health conditions. The review also emphasizes the need for further research to understand the genetic architecture of diseases and develop effective prevention and treatment strategies.
Hepatocyte nuclear factor 4 (HNF4) was first identified as a DNA binding activity in rat liver nuclear extracts. Protein purification had then led to the cDNA cloning of rat HNF4, which was found to be an orphan member of the nuclear receptor superfamily. Binding sites for this factor were identified in many tissue-specifically expressed genes, and the protein was found to be essential for early embryonic development in the mouse. We have now isolated cDNAs encoding the human homolog of the rat and mouse HNF4 splice variant HNF4 alpha 2, as well as a previously unknown splice variant of this protein, which we called HNF alpha 4. More importantly, we also cloned a novel HNF4 subtype (HNF4 gamma) derived from a different gene and showed that the genes encoding HNF 4 alpha and HNF4 gamma are located on human chromosomes 20 and 8, respectively. Northern (RNA) blot analysis revealed that HNF4 GAMMA is expressed in the kidney, pancreas, small intestine, testis, and colon but not in the liver, while HNF4 alpha RNA was found in all of these tissues. By cotransfection experiments in C2 and HeLa cells, we showed that HNF4 gamma is significantly less active than HNF4 alpha 2 and that the novel HNF4 alpha splice variant HNF4 alpha 4 has no detectable transactivation potential. Therefore, the differential expression of distinct HNF4 proteins may play a key role in the differential transcriptional regulation of HNF4-dependent genes.
KLF11 is a transcription factor that is associated with various medical conditions, including endometriosis, lung adenocarcinoma, pancreatic beta cell function, maturity-onset diabetes of the young (MODY), chronic kidney disease (CKD), and malignant pleural mesothelioma (MPM). In pancreatic beta-cells, KLF11 activates and inhibits human insulin promoter activity and regulates insulin sensitivity and lipid metabolism. The downregulation of KLF11 can lead to fibrosis in ectopic endometrium lesions and inhibition of cell proliferation in lung adenocarcinoma. Pathogenic variants of KLF11 are the leading cause of MODY in the Trakya region of Turkey. KLF11 also inhibits endothelial activation, protects against stroke and venous thrombosis, and can regulate Twist1 expression, which increases cell invasion and migration in gastric cancer. 
KLF11 is a tumour suppressor that inhibits cellular growth and is critical in integrating progesterone receptor signaling and proliferation in uterine leiomyoma cells. KLF11 gene variants have been found to have a negative impact on insulin sensitivity and may play a role in type 2 diabetes. Additionally, KLF11 plays a role in regulating INS transcription by binding to the promoter via specific GC-rich sites and recruiting Sin3-histone deacetylase chromatin remodeling complexes. The article also discusses other members of the SP/XKLF transcription factor family, including KLF13, KLF14, and FKLF, which act as transcriptional regulators and have implications in various diseases and cellular processes. KLF11 and other members of the SP/XKLF transcription factor family play crucial roles in regulating gene expression and have significant implications for disease diagnosis and treatment.
NEUROD1 is a transcription factor that is essential for neuronal and endocrine differentiation. It plays a critical role in various cellular processes, including gene expression regulation, pancreatic beta cell function, differentiation of neuro-retinal cells, and potential therapeutic targets in certain diseases such as small cell lung carcinoma. Dysregulation of NEUROD1 is involved in various diseases, including SCLC, Parkinson's disease, and retinitis pigmentosa. Targeting NEUROD1 and its regulatory mechanisms may provide a therapeutic target for neuroblastoma and neuroinflammation-associated disorders.
NEUROD1 is crucial for neuronal differentiation, maturation, and survival in pancreatic endocrine cells, neuroblastoma, and during astrocyte-to-neuron reprogramming. Dysregulation of NEUROD1 is involved in various diseases such as SCLC, CRC, and severe neonatal diabetes. NEUROD1 is a potential therapeutic target for certain diseases and may play a crucial role in promoting neuronal differentiation and maturation. Mutations in NEUROD1 have been linked to the development of maturity onset diabetes of the young (MODY) and type 1 and type 2 diabetes. NEUROD1 regulates gene expression networks in several beta-cell function-related diseases and is a good neuroendocrine marker in gastric adenocarcinomas. Overall, further studies may help fully understand the potential of NEUROD1 as a therapeutic target and contribute to the development of novel treatments for related diseases.
    """


    resultSet = llm_chain.run(textSet)
    print(" ".join(resultSet.split()))
    # print(resultSet)


