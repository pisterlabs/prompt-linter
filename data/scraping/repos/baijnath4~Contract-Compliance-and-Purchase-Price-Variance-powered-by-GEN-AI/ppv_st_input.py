from pdfminer.high_level import extract_text
# import streamlit as st
import openai
import regex as re
import pandas as pd
import numpy as np
import openai


def questions(pdf1,i):
    question = i
    prompt = f"{question}\n{pdf1}\n\\n:"
    model = "text-davinci-003"
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=500,
        temperature=0,
        n = 1,
        stop=None
    )
    diffs = response.choices[0].text
    return diffs.strip()
# data_df = pd.DataFrame(columns = ['Contract_ID','Suppliers','Line Item Description', 'Contracted Payterms','Contracted unit price','Unit of Measure'])
# data_df = pd.read_excel("C:/Users/01934L744/Box/Baijnath Data/Project 2023/Nidhi/contract/upload_ppv/DummyData.xlsx")

def get_insight(uploaded_file):
    # uploaded_file = st.file_uploader("Upload Contract", "pdf")
    if uploaded_file is not None:
        openai.api_key = "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"
        element = extract_text(uploaded_file)
        prompt = ['What is the agreement name,provide only name','What is the supplier name','When is the start date and end date of the agreement',
        'What is the contract id mentioned in the agreement',
            'what are the payment terms agreed with supplier',
            'What are the prices of roles provided by supplier',
            'What is the unit of measure of the services']
        new_prompts = ['What is the agreement name,provide only name','What is the supplier name','When is the start date and end date of the agreement provide only dates','what is the contract id',
                'What are the number of days payment terms agreed',
                'What are the role wise price per unit not the total value provide in table','What is the unit of measure only']
        answers = ''
        exact_answers = ''
        for p in prompt:
            answers+= (questions(element,p))
        filtered_contract = str.join(" ", answers.splitlines())
        for c in new_prompts:
            exact_answers = exact_answers + 'Q)'+ c + '\n' + 'A)'+ questions(filtered_contract,c) + '\n\n' 
    return exact_answers
   


# uploaded_po = st.file_uploader("Upload PO", ["csv","xlsx"])
# def po_data(uploaded_po, data_df):
#     if uploaded_po is not None:
#         podf = pd.read_excel(uploaded_po,sheet_name='Sheet1')
#         final_po = pd.merge(podf,data_df,how='inner',on=['Contract_ID','Line Item Description'])
#         final_po = final_po[['POID','Line Item Description','SUPPLIER','PAYMENTTERMS_PO','POLINENUMBER','POQUANTITY','UNITOFMEASURE','UNITPRICEINORIGINALCURRENCY','POSPENDUSD','Contract_ID','Contracted Payterms','Contracted unit price']]
#         final_po['Contracted Payterms'] = final_po['Contracted Payterms'].astype(int)
#         final_po['Contracted unit price'] = final_po['Contracted unit price'].astype(int)
#         final_po['PAYMENTTERMS_PO'] = final_po['PAYMENTTERMS_PO'].astype(int)
#         final_po['UNITPRICEINORIGINALCURRENCY'] = final_po['UNITPRICEINORIGINALCURRENCY'].astype(int)
#         final_po['Purchase Price variance(PO vs Contract)'] = ((final_po['UNITPRICEINORIGINALCURRENCY'] - final_po['Contracted unit price'])/final_po['Contracted unit price'])*100
#         final_po['Purchase Price variance(PO vs Contract)'] = final_po['Purchase Price variance(PO vs Contract)'].astype(int)
#         final_po['Payterm Difference (PO vs Contract)'] = (final_po['Contracted Payterms'] - final_po['PAYMENTTERMS_PO'])
#     return final_po
    
# uploaded_invoice = st.file_uploader("Upload Open Invoice Data", ["csv","xlsx"])
uploaded_invoice = "C:/Users/01934L744/Box/Baijnath Data/Project 2023/Nidhi/contract/ppv_Script/Invoice data.xlsx"
def invoice_data(uploaded_invoice):
    if uploaded_invoice is not None:
        fdf = pd.read_excel(uploaded_invoice,sheet_name='Sheet1')
        final_df = pd.merge(fdf,data_df,how='inner',on=['Contract_ID','Line Item Description'])
        final_df = final_df[['INVOICE','PO','Contract_ID','Line Item Description','PAYMENTTERMS_PO','Invoice Payterms','Invoice unit price','Contracted Payterms','Contracted unit price','Invoice Gross Value','UNITPRICEINORIGINALCURRENCY']]
        final_df['Contracted Payterms'] = final_df['Contracted Payterms'].astype(int)
        final_df['PAYMENTTERMS_PO'] = final_df['PAYMENTTERMS_PO'].astype(int)
        final_df['Invoice Payterms'] = final_df['Invoice Payterms'].astype(int)
        final_df['Invoice Gross Value'] = final_df['Invoice Gross Value'].astype(int)
        final_df['Invoice unit price'] = final_df['Invoice unit price'].astype(int)
        final_df['Contracted unit price'] = final_df['Contracted unit price'].astype(int)
        final_df['UNITPRICEINORIGINALCURRENCY'] = final_df['UNITPRICEINORIGINALCURRENCY'].astype(int)
        
        # button_first = st.button("View Details",key="12")
        button_first=True
        if button_first:
            
        
            final_df['Purchase Price variance(Invoice vs Contract)'] = ((final_df['Invoice unit price'] - final_df['Contracted unit price'])/final_df['Contracted unit price'])*100
            final_df['Purchase Price variance(Invoice vs Contract)'] = final_df['Purchase Price variance(Invoice vs Contract)'].astype(int)
            final_df['Working Capital(Invoice vs Contract)'] = ((final_df['Contracted Payterms'] - final_df['Invoice Payterms'])*final_df['Invoice Gross Value'])/365

            final_df = final_df.rename(columns={'INVOICE': 'Document Number', 'PO': 'PO_ID','UNITPRICEINORIGINALCURRENCY':'PO Unit Price','Product Description':'Line item description'})
            final_df = final_df[['Document Number','Line Item Description','Invoice Payterms','Invoice unit price','PO_ID','PAYMENTTERMS_PO','PO Unit Price','Contract_ID','Contracted Payterms','Contracted unit price','Invoice Gross Value','Purchase Price variance(Invoice vs Contract)','Working Capital(Invoice vs Contract)']]        
            # st.dataframe(final_df.style.applymap(color_survived, subset=['Purchase Price variance(Invoice vs Contract)','Working Capital(Invoice vs Contract)']))
        # button_s = st.button("3-way-Match-Details",key="007")
        # if button_s:
            final_df = final_df.rename(columns={'INVOICE': 'Document Number', 'PO': 'PO_ID','UNITPRICEINORIGINALCURRENCY':'PO Unit Price','Product Description':'Line item description','PAYMENTTERMS_PO':'PO Payterms'})
            final_s = final_df[['Line Item Description','Contracted unit price','PO Unit Price','Invoice unit price','Contracted Payterms','PO Payterms','Invoice Payterms']]
            openai.api_key = "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"
            prompt_s = ['provide the differences in unit price across contract, PO and Invoice from the table in words and also provide summary of contract,po and invoice payterms']
            summary = ''
            summary+= (questions(final_s,prompt_s))
            # st.dataframe(final_s)
            # st.text_area('SUMMARY',summary,height=150)
    return summary


if __name__ == "__main__":
    # file="C:/Users/01934L744/Box/Baijnath Data/Project 2023/Nidhi/contract/upload_ppv/Contract1.pdf"
    # result = get_insight(file)
    # print("result:-------", result)

    file2="C:/Users/01934L744/Box/Baijnath Data/Project 2023/Nidhi/contract/upload_ppv/PO.xlsx"
    data_df = pd.read_excel("C:/Users/01934L744/Box/Baijnath Data/Project 2023/Nidhi/contract/upload_ppv/DummyData.xlsx")
    # result2 = po_data(file2, data_df)
    # print("result2:--", result2)