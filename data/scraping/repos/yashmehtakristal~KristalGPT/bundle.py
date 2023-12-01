# All imports

import streamlit as st
import openai
import os

# Importing functions
from core.loading import read_documents_from_directory, iterate_files_from_directory, save_uploaded_file, read_documents_from_uploaded_files, get_tables_from_uploaded_file, iterate_files_from_uploaded_files, iterate_excel_files_from_directory, iterate_uploaded_excel_files, print_file_details, show_dataframes, iterate_uploaded_excel_file
from core.pickle import save_to_pickle, load_from_pickle
from core.indexing import query_engine_function, query_engine_function_advanced, build_vector_index
from core.LLM_preprocessing import conditions_excel, extract_fund_variable, prompts_to_substitute_variable, storing_input_prompt_in_list
from core.querying import recursive_retriever_old, recursive_retriever
from core.LLM_prompting import individual_prompt, prompt_loop, prompt_loop_advanced, individual_prompt_advanced
from core.PostLLM_prompting import create_output_result_column, create_output_context_column, intermediate_output_to_excel
from core.parsing import create_schema_from_excel, parse_value
from core.Postparsing import create_filtered_excel_file, final_result_orignal_excel_file, reordering_columns
from core.Last_fixing_fields import find_result_fund_name, find_result_fund_house, find_result_fund_class, find_result_currency, find_result_acc_or_inc, create_new_kristal_alias, update_kristal_alias, update_sponsored_by, update_required_broker, update_transactional_fund, update_disclaimer, update_risk_disclaimer, find_nav_value, update_nav_value 
from core.output import output_to_excel, download_data_as_csv, download_data_as_excel_link, download_data_as_csv_link
from core.chroma import create_or_get_chroma_db, download_embedding_old, print_files_in_particular_directory, print_files_in_directory, download_embedding_zip, write_zip_files_to_directory, check_zipfile_directory, get_chroma_db, create_chroma_db


def no_embeddings_process_documents_individual(uploaded_files, chroma_file_path, prompt):
     
    with st.spinner("Reading uploaded PDF and Excel files"):
        
        docs = read_documents_from_uploaded_files(uploaded_files)
        # st.write("This is docs", docs)

        table_dfs = iterate_files_from_uploaded_files(uploaded_files)

        save_uploaded_file(uploaded_files)

        # print_file_details(uploaded_files)

        # orignal_excel_file, info_excel_file = iterate_uploaded_excel_file(uploaded_xlsx_files)

        # list_of_dataframes = [orignal_excel_file, info_excel_file]
        # show_dataframes(list_of_dataframes)

        directory_pickles = save_to_pickle(directory_pickles = "Pickle/table_dfs.pkl", table_dfs = table_dfs)

    st.success("Successfully read pdf file", icon="✅")


    with st.spinner("Conducting Indexing, Querying and Prompting"):

        # vector_store, storage_context = create_or_get_chroma_db(chroma_file_path)
        vector_store, storage_context = create_chroma_db(chroma_file_path)

        # Functions performing indexing
        llm, service_context, df_query_engines = query_engine_function(table_dfs = table_dfs)
        vector_index, vector_retriever, df_id_query_engine_mapping, nodes_to_retrieve = build_vector_index(service_context = service_context, df_query_engines = df_query_engines, docs = docs, nodes_to_retrieve = 3, storage_context = storage_context, vector_store = vector_store, is_chroma_loading = False)
        # vector_index, vector_retriever, df_id_query_engine_mapping, nodes_to_retrieve = build_vector_index(service_context = service_context, df_query_engines = df_query_engines, docs = docs, nodes_to_retrieve = 3)

        recursive_retriever, response_synthesizer, query_engine = recursive_retriever_old(vector_retriever = vector_retriever, df_id_query_engine_mapping = df_id_query_engine_mapping, service_context = service_context)

        # Calling individual_prompt function
        output_response, output_context = individual_prompt(query_engine = query_engine, prompt = prompt)

    st.success("Successfully finished Indexing, Querying and Prompting", icon="✅")

    st.markdown("#### Answer")
    st.markdown(f"{output_response}")

    download_embedding_zip(chroma_file_path, zip_filename = "embeddings")


def embeddings_process_documents_individual_advanced(uploaded_files, prompt, nodes_to_retrieve, model, temperature, request_timeout, max_retries, return_all_chunks, uploaded_zip_file):

    with st.spinner("Extract zip files"):
        master_folder, chroma_file_path, chroma_file_name = check_zipfile_directory()
        write_zip_files_to_directory(uploaded_zip_file, chroma_file_path)

    st.success("Successfully extracted zip files", icon="✅")
        
    with st.spinner("Reading uploaded PDF and Excel files"):
        
        docs = read_documents_from_uploaded_files(uploaded_files)
        # st.write("This is docs", docs)

        table_dfs = iterate_files_from_uploaded_files(uploaded_files)

        save_uploaded_file(uploaded_files)

        # print_file_details(uploaded_files)

        # orignal_excel_file, info_excel_file = iterate_uploaded_excel_file(uploaded_xlsx_files)

        # list_of_dataframes = [orignal_excel_file, info_excel_file]
        # show_dataframes(list_of_dataframes)

        directory_pickles = save_to_pickle(directory_pickles = "Pickle/table_dfs.pkl", table_dfs = table_dfs)

    st.success("Successfully read pdf file and excel file", icon="✅")


    with st.spinner("Conducting Indexing, Querying and Prompting"):

        # vector_store, storage_context = create_or_get_chroma_db(chroma_file_path)
        vector_store, storage_context = get_chroma_db(chroma_file_path)

        # Functions performing indexing
        llm, service_context, df_query_engines = query_engine_function_advanced(table_dfs = table_dfs, model = model, temperature = temperature, request_timeout = request_timeout, max_retries = max_retries)
        vector_index, vector_retriever, df_id_query_engine_mapping, nodes_to_retrieve = build_vector_index(service_context = service_context, df_query_engines = df_query_engines, docs = docs, nodes_to_retrieve = nodes_to_retrieve, storage_context = storage_context, vector_store = vector_store, is_chroma_loading = False)
        # vector_index, vector_retriever, df_id_query_engine_mapping, nodes_to_retrieve = build_vector_index(service_context = service_context, df_query_engines = df_query_engines, docs = docs, nodes_to_retrieve = 3)

        recursive_retriever, response_synthesizer, query_engine = recursive_retriever_old(vector_retriever = vector_retriever, df_id_query_engine_mapping = df_id_query_engine_mapping, service_context = service_context)

        # Calling individual_prompt function
        output_response, output_context, context_with_max_score_list, file_path_metadata_list, source_metadata_list = individual_prompt_advanced(query_engine = query_engine, prompt = prompt, nodes_to_retrieve = nodes_to_retrieve, return_all_chunks = return_all_chunks)

        return output_response, prompt, context_with_max_score_list, file_path_metadata_list, source_metadata_list, table_dfs, docs

        # st.markdown("#### Answer")
        # st.markdown(f"{output_response}")

def no_embeddings_process_documents_individual_advanced(uploaded_files, prompt, chroma_file_path, nodes_to_retrieve, model, temperature, request_timeout, max_retries, return_all_chunks):
        
    with st.spinner("Reading uploaded PDF and Excel files"):

        docs = read_documents_from_uploaded_files(uploaded_files)
        # st.write("This is docs", docs)

        table_dfs = iterate_files_from_uploaded_files(uploaded_files)

        save_uploaded_file(uploaded_files)

        # print_file_details(uploaded_files)

        # orignal_excel_file, info_excel_file = iterate_uploaded_excel_file(uploaded_xlsx_files)

        # list_of_dataframes = [orignal_excel_file, info_excel_file]
        # show_dataframes(list_of_dataframes)

        directory_pickles = save_to_pickle(directory_pickles = "Pickle/table_dfs.pkl", table_dfs = table_dfs)

    st.success("Successfully read pdf file and excel file", icon="✅")


    with st.spinner("Conducting Indexing, Querying and Prompting"):

        # vector_store, storage_context = create_or_get_chroma_db(chroma_file_path)
        vector_store, storage_context = create_chroma_db(chroma_file_path)


        # Functions performing indexing
        llm, service_context, df_query_engines = query_engine_function_advanced(table_dfs = table_dfs, model = model, temperature = temperature, request_timeout = request_timeout, max_retries = max_retries)
        vector_index, vector_retriever, df_id_query_engine_mapping, nodes_to_retrieve = build_vector_index(service_context = service_context, df_query_engines = df_query_engines, docs = docs, nodes_to_retrieve = nodes_to_retrieve, storage_context = storage_context, vector_store = vector_store, is_chroma_loading = False)
        # vector_index, vector_retriever, df_id_query_engine_mapping, nodes_to_retrieve = build_vector_index(service_context = service_context, df_query_engines = df_query_engines, docs = docs, nodes_to_retrieve = 3)

        recursive_retriever, response_synthesizer, query_engine = recursive_retriever_old(vector_retriever = vector_retriever, df_id_query_engine_mapping = df_id_query_engine_mapping, service_context = service_context)

        # Calling individual_prompt function
        output_response, output_context, context_with_max_score_list, file_path_metadata_list, source_metadata_list = individual_prompt_advanced(query_engine = query_engine, prompt = prompt, nodes_to_retrieve = nodes_to_retrieve, return_all_chunks = return_all_chunks)

    st.success("Successfully finished Indexing, Querying and Prompting", icon="✅")

    return output_response, prompt, context_with_max_score_list, file_path_metadata_list, source_metadata_list, table_dfs, docs
        # st.markdown("#### Answer")
        # st.markdown(f"{output_response}")


def no_embeddings_process_documents_loop_advanced(uploaded_files, uploaded_xlsx_files, chroma_file_path, nodes_to_retrieve, model, temperature, request_timeout, max_retries, sleep, return_all_chunks, fund_variable):

        
        with st.spinner("Reading uploaded PDF and Excel files"):

            docs = read_documents_from_uploaded_files(uploaded_files)
            # st.write("This is docs", docs)

            table_dfs = iterate_files_from_uploaded_files(uploaded_files)

            save_uploaded_file(uploaded_files)

            # print_file_details(uploaded_files)

            orignal_excel_file, info_excel_file = iterate_uploaded_excel_file(uploaded_xlsx_files)

            # list_of_dataframes = [orignal_excel_file, info_excel_file]
            # show_dataframes(list_of_dataframes)

            directory_pickles = save_to_pickle(directory_pickles = "Pickle/table_dfs.pkl", table_dfs = table_dfs)

        st.success("Successfully read pdf file and excel file", icon="✅")


        with st.spinner("Saving Embeddings"):

            # vector_store, storage_context = create_or_get_chroma_db(chroma_file_path)
            vector_store, storage_context = create_chroma_db(chroma_file_path)

        st.success("Successfully saved embeddings", icon="✅")


        with st.spinner("Conducting Indexing & LLM-preprocessing"):

            # Functions performing indexing
            llm, service_context, df_query_engines = query_engine_function_advanced(table_dfs = table_dfs, model = model, temperature = temperature, request_timeout = request_timeout, max_retries = max_retries)
            vector_index, vector_retriever, df_id_query_engine_mapping, nodes_to_retrieve = build_vector_index(service_context = service_context, df_query_engines = df_query_engines, docs = docs, nodes_to_retrieve = nodes_to_retrieve, storage_context = storage_context, vector_store = vector_store, is_chroma_loading = False)
            # vector_index, vector_retriever, df_id_query_engine_mapping, nodes_to_retrieve = build_vector_index(service_context = service_context, df_query_engines = df_query_engines, docs = docs, nodes_to_retrieve = 3)

            # Functions performing LLM-preprocessing
            LLM_inputs, Discretionary_inputs = conditions_excel(orignal_excel_file)
            # fund_variable = extract_fund_variable(info_excel_file = info_excel_file)
            orignal_excel_file, llm_full_index = prompts_to_substitute_variable(orignal_excel_file = orignal_excel_file, fund_variable = fund_variable, LLM_inputs = LLM_inputs)
            orignal_excel_file, llm_prompts_to_use, llm_prompts_index = storing_input_prompt_in_list(orignal_excel_file = orignal_excel_file, llm_full_index = llm_full_index)

            # Diagnostic purposes

            # st.write("Checking fund variable")
            # st.write(fund_variable)

            # st.write("Checking list - llm_prompts_to_use")
            # st.write(llm_prompts_to_use)

            # st.write("Checking list - llm_prompts_index")
            # st.write(llm_prompts_index)

            # Showing dataframes for diagnostic purposes
            # list_of_dataframes = [orignal_excel_file, info_excel_file]
            # show_dataframes(list_of_dataframes)

        st.success("Successfully finished indexing & LLM-preprocessing", icon="✅")

        with st.spinner("Conducting Querying"):

            recursive_retriever, response_synthesizer, query_engine = recursive_retriever_old(vector_retriever = vector_retriever, df_id_query_engine_mapping = df_id_query_engine_mapping, service_context = service_context)

            # Diagnostic purposes

            # st.write("Checking recursive_retriever")
            # st.write(type(recursive_retriever))
            # st.write(recursive_retriever)

            # st.write("Checking response_synthesizer")
            # st.write(type(response_synthesizer))
            # st.write(response_synthesizer)

            # st.write("Checking query engine")
            # st.write(type(query_engine))
            # st.write(query_engine)  
            
        st.success("Successfully finished Querying", icon="✅")

        with st.spinner("Conducting Prompting"):
            
            output_response, output_context, context_with_max_score_list, file_path_metadata_list, source_metadata_list = prompt_loop_advanced(query_engine = query_engine, llm_prompts_to_use = llm_prompts_to_use, nodes_to_retrieve = nodes_to_retrieve, sleep = sleep, return_all_chunks = return_all_chunks)

            # Showing list for diagnostic purposes
            # st.write("Final output")
            # st.write(output_response)
            # st.write(output_context)

        st.success("Successfully finished Prompting", icon="✅")


        with st.spinner("Conducting Post-LLM Prompting"):

            orignal_excel_file = create_output_result_column(orignal_excel_file = orignal_excel_file, llm_prompts_index = llm_prompts_index, output_response = output_response)
            orignal_excel_file = create_output_context_column(orignal_excel_file, llm_prompts_index, nodes_to_retrieve = nodes_to_retrieve, output_context = output_context)
            intermediate_output_to_excel(orignal_excel_file = orignal_excel_file, excel_directory = "Results", output_excel_filename = "results_output", file_extension = "xlsx")

        st.success("Successfully finished Post-LLM Prompting", icon="✅")


        with st.spinner("Parsing"):
            schema = create_schema_from_excel(orignal_excel_file, llm_prompts_index)
            orignal_excel_file = parse_value(output_response = output_response, llm_prompts_index = llm_prompts_index, orignal_excel_file = orignal_excel_file, schema = schema, llm = llm)

        st.success("Successfully finished Parsing", icon="✅")


        with st.spinner("Post-parsing"):
            filtered_excel_file = create_filtered_excel_file(orignal_excel_file = orignal_excel_file, llm_prompts_index = llm_prompts_index)
            orignal_excel_file = final_result_orignal_excel_file(filtered_excel_file = filtered_excel_file, orignal_excel_file = orignal_excel_file, llm_prompts_index = llm_prompts_index)
            orignal_excel_file = reordering_columns(orignal_excel_file)

        st.success("Successfully finished Post-Parsing", icon="✅")


        with st.spinner("Fixing LLM-post processing fields"):
            results_fund_name_value = find_result_fund_name(orignal_excel_file)
            result_fund_house_value = find_result_fund_house(orignal_excel_file)
            result_fund_class_value = find_result_fund_class(orignal_excel_file)
            result_currency_value = find_result_currency(orignal_excel_file)
            result_acc_or_inc_value = find_result_acc_or_inc(orignal_excel_file)
            kristal_alias = create_new_kristal_alias(results_fund_name_value, result_fund_house_value, result_fund_class_value, result_currency_value, result_acc_or_inc_value)
            orignal_excel_file = update_kristal_alias(orignal_excel_file = orignal_excel_file, kristal_alias = kristal_alias)
            orignal_excel_file = update_sponsored_by(orignal_excel_file = orignal_excel_file, sponsored_by = "backend-staging+hedgefunds@kristal.ai")
            orignal_excel_file = update_required_broker(orignal_excel_file = orignal_excel_file, required_broker = "Kristal Pooled")
            orignal_excel_file = update_transactional_fund(orignal_excel_file = orignal_excel_file, transactional_fund = "Yes")
            orignal_excel_file = update_disclaimer(
                orignal_excel_file = orignal_excel_file,
                disclaimer = '''
                The recommendations contained herein are for the exclusive use of investor and prohibits any form of disclosure or reproduction. The content cannot be relied upon by any other person for any other purpose. The recommendations are preliminary information to the investors, are subject to risks and may change based on investment objectives, financials, liabilities or the risk profile of an investor. Any recommendations including financial advice provided by Kristal.AI or its affiliates shall be subject to contractual understanding, necessary documentation, applicable laws, approvals and regulations. The recommendations contained herein may not be eligible for sale/purchase in some jurisdictions, in specific, are not intended for residents of the USA or within the USA.Though the recommendations are based on information obtained from reliable sources and are provided in good faith, they may be valid only on the date and time the recommendations are provided and shall be subject to change without notice. Kristal.AI
                '''
                )
            orignal_excel_file = update_risk_disclaimer(
                orignal_excel_file = orignal_excel_file,
                risk_disclaimer = '''
                The recommendations contained herein are for the exclusive use of investor and prohibits any form of disclosure or reproduction. The content cannot be relied upon by any other person for any other purpose. The recommendations are preliminary information to the investors, are subject to risks and may change based on investment objectives, financials, liabilities or the risk profile of an investor. Any recommendations including financial advice provided by Kristal.AI or its affiliates shall be subject to contractual understanding, necessary documentation, applicable laws, approvals and regulations. The recommendations contained herein may not be eligible for sale/purchase in some jurisdictions, in specific, are not intended for residents of the USA or within the USA.Though the recommendations are based on information obtained from reliable sources and are provided in good faith, they may be valid only on the date and time the recommendations are provided and shall be subject to change without notice. Kristal.AI
                '''
                )
            result_nav_value = find_nav_value(orignal_excel_file)
            orignal_excel_file = update_nav_value(orignal_excel_file = orignal_excel_file, result_nav_value = result_nav_value)
            output_to_excel(orignal_excel_file = orignal_excel_file, excel_directory = "Results", output_excel_filename = "results_output", file_extension = "xlsx")

        st.success("Successfully Fixed LLM-post processing fields", icon="✅")

        st.markdown("### Results")

        return output_response, llm_prompts_to_use, context_with_max_score_list, file_path_metadata_list, source_metadata_list, orignal_excel_file, table_dfs, docs 

                
        # @st.cache_data
        # def slider_state():
        #     return {"value": None}

        # prompt_result_selector = st.number_input(
        #         label="Select result of prompt to display", min_value = 1, max_value = len(output_response), step = 1
        #     )
        
        # # is_chosen = slider_state()  # gets our cached dictionary

        # # if prompt_result_selector:
        # #     # any changes need to be performed in place
        # #     prompt_result_selector.update({"value": prompt_result_selector})
        
        # if prompt_result_selector or st.session_state.load_prompt_result_selector_state:

        #     st.session_state.load_prompt_result_selector_state = True
        
        #     st.markdown(f"Displaying results for Prompt #{prompt_result_selector}: {llm_prompts_to_use[prompt_result_selector - 1]}")

        #     answer_col, sources_col = st.columns(2)

        #     # Displaying in answers columns
        #     with answer_col:
        #         st.markdown("#### Answer")
        #         st.markdown(output_response[prompt_result_selector - 1])


        #     # Displaying in sources columns
        #     with sources_col:

        #         # User selected option to display all chunks from vector search
        #         if return_all_chunks is True:

        #             # These are lists of corresponding question (as source was list of list)
        #             context_to_display = context_with_max_score_list[prompt_result_selector - 1]
        #             file_path_to_display = file_path_metadata_list[prompt_result_selector - 1]
        #             source_metadata_to_display = source_metadata_list[prompt_result_selector - 1]

        #             for i in range(nodes_to_retrieve):
        #                 st.markdown(context_to_display[i])
        #                 st.markdown(f"Document: {file_path_to_display[i]}")
        #                 st.markdown(f"Page Source: {source_metadata_to_display[i]}")
        #                 st.markdown("---")
                    
        #         # User selected option to display only 1 chunk
        #         if return_all_chunks is False:
                    
        #             # Display particular lists
        #             st.markdown(context_with_max_score_list[prompt_result_selector - 1])
        #             st.markdown(f"Document: {file_path_to_display[prompt_result_selector - 1]}")
        #             st.markdown(f"Page Source: {source_metadata_to_display[prompt_result_selector - 1]}")
                    

        #     st.markdown("### Bulk Prompt Results")

        #     # Display dataframe containing final results
        #     st.dataframe(data = orignal_excel_file, use_container_width = True, column_order = None)

        #     # Display button to download results to excel file
        #     download_data_as_excel(orignal_excel_file = orignal_excel_file)

        #     # Display button to download results to csv file
        #     download_data_as_csv(orignal_excel_file = orignal_excel_file)



def no_embeddings_process_documents_loop(uploaded_files, uploaded_xlsx_files, chroma_file_path, fund_variable):
        
        with st.spinner("Reading uploaded PDF and Excel files"):

            docs = read_documents_from_uploaded_files(uploaded_files)
            # st.write("This is docs", docs)

            table_dfs = iterate_files_from_uploaded_files(uploaded_files)

            save_uploaded_file(uploaded_files)

            # print_file_details(uploaded_files)

            orignal_excel_file, info_excel_file = iterate_uploaded_excel_file(uploaded_xlsx_files)

            # list_of_dataframes = [orignal_excel_file, info_excel_file]
            # show_dataframes(list_of_dataframes)

            directory_pickles = save_to_pickle(directory_pickles = "Pickle/table_dfs.pkl", table_dfs = table_dfs)

        st.success("Successfully read pdf file and excel file", icon="✅")


        with st.spinner("Saving Embeddings"):
            # vector_store, storage_context = create_or_get_chroma_db(chroma_file_path)
            vector_store, storage_context = create_chroma_db(chroma_file_path)

        st.success("Successfully saved embeddings", icon="✅")


        with st.spinner("Conducting Indexing & LLM-preprocessing"):

            # Functions performing indexing
            llm, service_context, df_query_engines = query_engine_function(table_dfs = table_dfs)
            vector_index, vector_retriever, df_id_query_engine_mapping, nodes_to_retrieve = build_vector_index(service_context = service_context, df_query_engines = df_query_engines, docs = docs, nodes_to_retrieve = 3, storage_context = storage_context, vector_store = vector_store, is_chroma_loading = False)
            # vector_index, vector_retriever, df_id_query_engine_mapping, nodes_to_retrieve = build_vector_index(service_context = service_context, df_query_engines = df_query_engines, docs = docs, nodes_to_retrieve = 3)

            # Functions performing LLM-preprocessing
            LLM_inputs, Discretionary_inputs = conditions_excel(orignal_excel_file)
            # fund_variable = extract_fund_variable(info_excel_file = info_excel_file)
            orignal_excel_file, llm_full_index = prompts_to_substitute_variable(orignal_excel_file = orignal_excel_file, fund_variable = fund_variable, LLM_inputs = LLM_inputs)
            orignal_excel_file, llm_prompts_to_use, llm_prompts_index = storing_input_prompt_in_list(orignal_excel_file = orignal_excel_file, llm_full_index = llm_full_index)

            # Diagnostic purposes

            # st.write("Checking fund variable")
            # st.write(fund_variable)

            # st.write("Checking list - llm_prompts_to_use")
            # st.write(llm_prompts_to_use)

            # st.write("Checking list - llm_prompts_index")
            # st.write(llm_prompts_index)

            # Showing dataframes for diagnostic purposes
            # list_of_dataframes = [orignal_excel_file, info_excel_file]
            # show_dataframes(list_of_dataframes)

        st.success("Successfully finished indexing & LLM-preprocessing", icon="✅")

        with st.spinner("Conducting Querying"):
            recursive_retriever, response_synthesizer, query_engine = recursive_retriever_old(vector_retriever = vector_retriever, df_id_query_engine_mapping = df_id_query_engine_mapping, service_context = service_context)

            # Diagnostic purposes

            # st.write("Checking recursive_retriever")
            # st.write(type(recursive_retriever))
            # st.write(recursive_retriever)

            # st.write("Checking response_synthesizer")
            # st.write(type(response_synthesizer))
            # st.write(response_synthesizer)

            # st.write("Checking query engine")
            # st.write(type(query_engine))
            # st.write(query_engine)  
            
        st.success("Successfully finished Querying", icon="✅")

        with st.spinner("Conducting Prompting"):

            output_response, output_context = prompt_loop(query_engine = query_engine, llm_prompts_to_use = llm_prompts_to_use)

            # Showing list for diagnostic purposes
            # st.write("Final output")
            # st.write(output_response)
            # st.write(output_context)

        st.success("Successfully finished Prompting", icon="✅")


        with st.spinner("Conducting Post-LLM Prompting"):

            orignal_excel_file = create_output_result_column(orignal_excel_file = orignal_excel_file, llm_prompts_index = llm_prompts_index, output_response = output_response)
            orignal_excel_file = create_output_context_column(orignal_excel_file, llm_prompts_index, nodes_to_retrieve = nodes_to_retrieve, output_context = output_context)
            intermediate_output_to_excel(orignal_excel_file = orignal_excel_file, excel_directory = "Results", output_excel_filename = "results_output", file_extension = "xlsx")

        st.success("Successfully finished Post-LLM Prompting", icon="✅")


        with st.spinner("Parsing"):
            schema = create_schema_from_excel(orignal_excel_file, llm_prompts_index)
            orignal_excel_file = parse_value(output_response = output_response, llm_prompts_index = llm_prompts_index, orignal_excel_file = orignal_excel_file, schema = schema, llm = llm)

        st.success("Successfully finished Parsing", icon="✅")


        with st.spinner("Post-parsing"):
            filtered_excel_file = create_filtered_excel_file(orignal_excel_file = orignal_excel_file, llm_prompts_index = llm_prompts_index)
            orignal_excel_file = final_result_orignal_excel_file(filtered_excel_file = filtered_excel_file, orignal_excel_file = orignal_excel_file, llm_prompts_index = llm_prompts_index)
            orignal_excel_file = reordering_columns(orignal_excel_file)

        st.success("Successfully finished Post-Parsing", icon="✅")


        with st.spinner("Fixing LLM-post processing fields"):
            results_fund_name_value = find_result_fund_name(orignal_excel_file)
            result_fund_house_value = find_result_fund_house(orignal_excel_file)
            result_fund_class_value = find_result_fund_class(orignal_excel_file)
            result_currency_value = find_result_currency(orignal_excel_file)
            result_acc_or_inc_value = find_result_acc_or_inc(orignal_excel_file)
            kristal_alias = create_new_kristal_alias(results_fund_name_value, result_fund_house_value, result_fund_class_value, result_currency_value, result_acc_or_inc_value)
            orignal_excel_file = update_kristal_alias(orignal_excel_file = orignal_excel_file, kristal_alias = kristal_alias)
            orignal_excel_file = update_sponsored_by(orignal_excel_file = orignal_excel_file, sponsored_by = "backend-staging+hedgefunds@kristal.ai")
            orignal_excel_file = update_required_broker(orignal_excel_file = orignal_excel_file, required_broker = "Kristal Pooled")
            orignal_excel_file = update_transactional_fund(orignal_excel_file = orignal_excel_file, transactional_fund = "Yes")
            orignal_excel_file = update_disclaimer(
                orignal_excel_file = orignal_excel_file,
                disclaimer = '''
                The recommendations contained herein are for the exclusive use of investor and prohibits any form of disclosure or reproduction. The content cannot be relied upon by any other person for any other purpose. The recommendations are preliminary information to the investors, are subject to risks and may change based on investment objectives, financials, liabilities or the risk profile of an investor. Any recommendations including financial advice provided by Kristal.AI or its affiliates shall be subject to contractual understanding, necessary documentation, applicable laws, approvals and regulations. The recommendations contained herein may not be eligible for sale/purchase in some jurisdictions, in specific, are not intended for residents of the USA or within the USA.Though the recommendations are based on information obtained from reliable sources and are provided in good faith, they may be valid only on the date and time the recommendations are provided and shall be subject to change without notice. Kristal.AI
                '''
                )
            orignal_excel_file = update_risk_disclaimer(
                orignal_excel_file = orignal_excel_file,
                risk_disclaimer = '''
                The recommendations contained herein are for the exclusive use of investor and prohibits any form of disclosure or reproduction. The content cannot be relied upon by any other person for any other purpose. The recommendations are preliminary information to the investors, are subject to risks and may change based on investment objectives, financials, liabilities or the risk profile of an investor. Any recommendations including financial advice provided by Kristal.AI or its affiliates shall be subject to contractual understanding, necessary documentation, applicable laws, approvals and regulations. The recommendations contained herein may not be eligible for sale/purchase in some jurisdictions, in specific, are not intended for residents of the USA or within the USA.Though the recommendations are based on information obtained from reliable sources and are provided in good faith, they may be valid only on the date and time the recommendations are provided and shall be subject to change without notice. Kristal.AI
                '''
                )
            result_nav_value = find_nav_value(orignal_excel_file)
            orignal_excel_file = update_nav_value(orignal_excel_file = orignal_excel_file, result_nav_value = result_nav_value)
            output_to_excel(orignal_excel_file = orignal_excel_file, excel_directory = "Results", output_excel_filename = "results_output", file_extension = "xlsx")

        st.success("Successfully Fixed LLM-post processing fields", icon="✅")

        # Display dataframe containing final results
        st.dataframe(data = orignal_excel_file, use_container_width = True, column_order = None)

        # Display button to download results to excel file
        download_data_as_excel_link(orignal_excel_file = orignal_excel_file)

        # Display button to download results to csv file
        download_data_as_csv_link(orignal_excel_file = orignal_excel_file)

        # print_files_in_particular_directory(chroma_file_path)

        # print_files_in_directory(chroma_file_path)

        # Display button to download embeddings from a given file path
        download_embedding_zip(chroma_file_path, zip_filename = "embeddings")
        #download_embedding_old(chroma_file_path)


def embeddings_process_documents_individual(uploaded_files, prompt, uploaded_zip_file):

    with st.spinner("Extract zip files"):
        master_folder, chroma_file_path, chroma_file_name = check_zipfile_directory()
        write_zip_files_to_directory(uploaded_zip_file, chroma_file_path)

    st.success("Successfully extracted zip files", icon="✅")
        
    with st.spinner("Reading uploaded PDF and Excel files"):
        
        docs = read_documents_from_uploaded_files(uploaded_files)
        # st.write("This is docs", docs)

        table_dfs = iterate_files_from_uploaded_files(uploaded_files)

        save_uploaded_file(uploaded_files)

        # print_file_details(uploaded_files)

        # orignal_excel_file, info_excel_file = iterate_uploaded_excel_file(uploaded_xlsx_files)

        # list_of_dataframes = [orignal_excel_file, info_excel_file]
        # show_dataframes(list_of_dataframes)

        directory_pickles = save_to_pickle(directory_pickles = "Pickle/table_dfs.pkl", table_dfs = table_dfs)

    st.success("Successfully read pdf file and excel file", icon="✅")        

    with st.spinner("Conducting Indexing, Querying and Prompting"):

        # vector_store, storage_context = create_or_get_chroma_db(chroma_file_path)
        vector_store, storage_context = get_chroma_db(chroma_file_path)
        llm, service_context, df_query_engines = query_engine_function(table_dfs = table_dfs)
        vector_index, vector_retriever, df_id_query_engine_mapping, nodes_to_retrieve = build_vector_index(service_context = service_context, df_query_engines = df_query_engines, docs = docs, nodes_to_retrieve = 3, storage_context = storage_context, vector_store = vector_store, is_chroma_loading = False)
        recursive_retriever, response_synthesizer, query_engine = recursive_retriever_old(vector_retriever = vector_retriever, df_id_query_engine_mapping = df_id_query_engine_mapping, service_context = service_context)
        output_response, output_context = individual_prompt(query_engine = query_engine, prompt = prompt)

    st.success("Successfully finished Indexing, Querying and Prompting", icon="✅")        

    st.markdown("#### Answer")
    st.markdown(f"{output_response}")


def embeddings_process_documents_loop(uploaded_files, uploaded_xlsx_files, fund_variable, uploaded_zip_file):
                
        with st.spinner("Extract zip files"):
            master_folder, chroma_file_path, chroma_file_name = check_zipfile_directory()
            write_zip_files_to_directory(uploaded_zip_file, chroma_file_path)

        st.success("Successfully extracted zip files", icon="✅")

        with st.spinner("Reading uploaded PDF and Excel files"):

            docs = read_documents_from_uploaded_files(uploaded_files)
            # st.write("This is docs", docs)

            table_dfs = iterate_files_from_uploaded_files(uploaded_files)

            save_uploaded_file(uploaded_files)

            # print_file_details(uploaded_files)

            orignal_excel_file, info_excel_file = iterate_uploaded_excel_file(uploaded_xlsx_files)

            # list_of_dataframes = [orignal_excel_file, info_excel_file]
            # show_dataframes(list_of_dataframes)

            directory_pickles = save_to_pickle(directory_pickles = "Pickle/table_dfs.pkl", table_dfs = table_dfs)

        st.success("Successfully read pdf file and excel file", icon="✅")        


        with st.spinner("Loading Embeddings"):
            # vector_store, storage_context = create_or_get_chroma_db(chroma_file_path)
            vector_store, storage_context = get_chroma_db(chroma_file_path)

        st.success("Successfully loaded embeddings", icon="✅")


        with st.spinner("Conducting Indexing & LLM-preprocessing"):

            # Functions performing indexing
            llm, service_context, df_query_engines = query_engine_function(table_dfs = table_dfs)
            vector_index, vector_retriever, df_id_query_engine_mapping, nodes_to_retrieve = build_vector_index(service_context = service_context, df_query_engines = df_query_engines, docs = docs, nodes_to_retrieve = 3, storage_context = storage_context, vector_store = vector_store, is_chroma_loading = False)
            # vector_index, vector_retriever, df_id_query_engine_mapping, nodes_to_retrieve = build_vector_index(service_context = service_context, df_query_engines = df_query_engines, docs = docs, nodes_to_retrieve = 3)

            # Functions performing LLM-preprocessing
            LLM_inputs, Discretionary_inputs = conditions_excel(orignal_excel_file)
            # fund_variable = extract_fund_variable(info_excel_file = info_excel_file)
            orignal_excel_file, llm_full_index = prompts_to_substitute_variable(orignal_excel_file = orignal_excel_file, fund_variable = fund_variable, LLM_inputs = LLM_inputs)
            orignal_excel_file, llm_prompts_to_use, llm_prompts_index = storing_input_prompt_in_list(orignal_excel_file = orignal_excel_file, llm_full_index = llm_full_index)

            # Diagnostic purposes

            # st.write("Checking fund variable")
            # st.write(fund_variable)

            # st.write("Checking list - llm_prompts_to_use")
            # st.write(llm_prompts_to_use)

            # st.write("Checking list - llm_prompts_index")
            # st.write(llm_prompts_index)

            # Showing dataframes for diagnostic purposes
            # list_of_dataframes = [orignal_excel_file, info_excel_file]
            # show_dataframes(list_of_dataframes)

        st.success("Successfully finished indexing & LLM-preprocessing", icon="✅")

        with st.spinner("Conducting Querying"):
            recursive_retriever, response_synthesizer, query_engine = recursive_retriever_old(vector_retriever = vector_retriever, df_id_query_engine_mapping = df_id_query_engine_mapping, service_context = service_context)

            # Diagnostic purposes

            # st.write("Checking recursive_retriever")
            # st.write(type(recursive_retriever))
            # st.write(recursive_retriever)

            # st.write("Checking response_synthesizer")
            # st.write(type(response_synthesizer))
            # st.write(response_synthesizer)

            # st.write("Checking query engine")
            # st.write(type(query_engine))
            # st.write(query_engine)  
            
        st.success("Successfully finished Querying", icon="✅")

        with st.spinner("Conducting Prompting"):

            output_response, output_context = prompt_loop(query_engine = query_engine, llm_prompts_to_use = llm_prompts_to_use)

            # Showing list for diagnostic purposes
            # st.write("Final output")
            # st.write(output_response)
            # st.write(output_context)

        st.success("Successfully finished Prompting", icon="✅")


        with st.spinner("Conducting Post-LLM Prompting"):

            orignal_excel_file = create_output_result_column(orignal_excel_file = orignal_excel_file, llm_prompts_index = llm_prompts_index, output_response = output_response)
            orignal_excel_file = create_output_context_column(orignal_excel_file, llm_prompts_index, nodes_to_retrieve = nodes_to_retrieve, output_context = output_context)
            intermediate_output_to_excel(orignal_excel_file = orignal_excel_file, excel_directory = "Results", output_excel_filename = "results_output", file_extension = "xlsx")

        st.success("Successfully finished Post-LLM Prompting", icon="✅")


        with st.spinner("Parsing"):
            schema = create_schema_from_excel(orignal_excel_file, llm_prompts_index)
            orignal_excel_file = parse_value(output_response = output_response, llm_prompts_index = llm_prompts_index, orignal_excel_file = orignal_excel_file, schema = schema, llm = llm)

        st.success("Successfully finished Parsing", icon="✅")


        with st.spinner("Post-parsing"):
            filtered_excel_file = create_filtered_excel_file(orignal_excel_file = orignal_excel_file, llm_prompts_index = llm_prompts_index)
            orignal_excel_file = final_result_orignal_excel_file(filtered_excel_file = filtered_excel_file, orignal_excel_file = orignal_excel_file, llm_prompts_index = llm_prompts_index)
            orignal_excel_file = reordering_columns(orignal_excel_file)

        st.success("Successfully finished Post-Parsing", icon="✅")


        with st.spinner("Fixing LLM-post processing fields"):
            results_fund_name_value = find_result_fund_name(orignal_excel_file)
            result_fund_house_value = find_result_fund_house(orignal_excel_file)
            result_fund_class_value = find_result_fund_class(orignal_excel_file)
            result_currency_value = find_result_currency(orignal_excel_file)
            result_acc_or_inc_value = find_result_acc_or_inc(orignal_excel_file)
            kristal_alias = create_new_kristal_alias(results_fund_name_value, result_fund_house_value, result_fund_class_value, result_currency_value, result_acc_or_inc_value)
            orignal_excel_file = update_kristal_alias(orignal_excel_file = orignal_excel_file, kristal_alias = kristal_alias)
            orignal_excel_file = update_sponsored_by(orignal_excel_file = orignal_excel_file, sponsored_by = "backend-staging+hedgefunds@kristal.ai")
            orignal_excel_file = update_required_broker(orignal_excel_file = orignal_excel_file, required_broker = "Kristal Pooled")
            orignal_excel_file = update_transactional_fund(orignal_excel_file = orignal_excel_file, transactional_fund = "Yes")
            orignal_excel_file = update_disclaimer(
                orignal_excel_file = orignal_excel_file,
                disclaimer = '''
                The recommendations contained herein are for the exclusive use of investor and prohibits any form of disclosure or reproduction. The content cannot be relied upon by any other person for any other purpose. The recommendations are preliminary information to the investors, are subject to risks and may change based on investment objectives, financials, liabilities or the risk profile of an investor. Any recommendations including financial advice provided by Kristal.AI or its affiliates shall be subject to contractual understanding, necessary documentation, applicable laws, approvals and regulations. The recommendations contained herein may not be eligible for sale/purchase in some jurisdictions, in specific, are not intended for residents of the USA or within the USA.Though the recommendations are based on information obtained from reliable sources and are provided in good faith, they may be valid only on the date and time the recommendations are provided and shall be subject to change without notice. Kristal.AI
                '''
                )
            orignal_excel_file = update_risk_disclaimer(
                orignal_excel_file = orignal_excel_file,
                risk_disclaimer = '''
                The recommendations contained herein are for the exclusive use of investor and prohibits any form of disclosure or reproduction. The content cannot be relied upon by any other person for any other purpose. The recommendations are preliminary information to the investors, are subject to risks and may change based on investment objectives, financials, liabilities or the risk profile of an investor. Any recommendations including financial advice provided by Kristal.AI or its affiliates shall be subject to contractual understanding, necessary documentation, applicable laws, approvals and regulations. The recommendations contained herein may not be eligible for sale/purchase in some jurisdictions, in specific, are not intended for residents of the USA or within the USA.Though the recommendations are based on information obtained from reliable sources and are provided in good faith, they may be valid only on the date and time the recommendations are provided and shall be subject to change without notice. Kristal.AI
                '''
                )
            result_nav_value = find_nav_value(orignal_excel_file)
            orignal_excel_file = update_nav_value(orignal_excel_file = orignal_excel_file, result_nav_value = result_nav_value)
            output_to_excel(orignal_excel_file = orignal_excel_file, excel_directory = "Results", output_excel_filename = "results_output", file_extension = "xlsx")

        st.success("Successfully Fixed LLM-post processing fields", icon="✅")

        # Display dataframe containing final results
        st.dataframe(data = orignal_excel_file, use_container_width = True, column_order = None)

        # Display button to download results to excel file
        download_data_as_excel_link(orignal_excel_file = orignal_excel_file)

        # Display button to download results to csv file
        download_data_as_csv_link(orignal_excel_file = orignal_excel_file)

        # download_embedding_zip(directory, zip_filename)



def embeddings_process_documents_loop_advanced(uploaded_files, uploaded_xlsx_files, nodes_to_retrieve, model, temperature, request_timeout, max_retries, sleep, return_all_chunks, fund_variable, uploaded_zip_file):

        with st.spinner("Extract zip files"):
            master_folder, chroma_file_path, chroma_file_name = check_zipfile_directory()
            write_zip_files_to_directory(uploaded_zip_file, chroma_file_path)

        with st.spinner("Reading uploaded PDF and Excel files"):

            docs = read_documents_from_uploaded_files(uploaded_files)
            # st.write("This is docs", docs)

            table_dfs = iterate_files_from_uploaded_files(uploaded_files)

            save_uploaded_file(uploaded_files)

            # print_file_details(uploaded_files)

            orignal_excel_file, info_excel_file = iterate_uploaded_excel_file(uploaded_xlsx_files)

            # list_of_dataframes = [orignal_excel_file, info_excel_file]
            # show_dataframes(list_of_dataframes)

            directory_pickles = save_to_pickle(directory_pickles = "Pickle/table_dfs.pkl", table_dfs = table_dfs)

        st.success("Successfully read pdf file and excel file", icon="✅")        


        with st.spinner("Loading Embeddings"):
            vector_store, storage_context = create_or_get_chroma_db(chroma_file_path)

        st.success("Successfully loaded embeddings", icon="✅")


        with st.spinner("Conducting Indexing & LLM-preprocessing"):

            # Functions performing indexing
            llm, service_context, df_query_engines = query_engine_function_advanced(table_dfs = table_dfs, model = model, temperature = temperature, request_timeout = request_timeout, max_retries = max_retries)
            vector_index, vector_retriever, df_id_query_engine_mapping, nodes_to_retrieve = build_vector_index(service_context = service_context, df_query_engines = df_query_engines, docs = docs, nodes_to_retrieve = nodes_to_retrieve, storage_context = storage_context, vector_store = vector_store, is_chroma_loading = False)
            # vector_index, vector_retriever, df_id_query_engine_mapping, nodes_to_retrieve = build_vector_index(service_context = service_context, df_query_engines = df_query_engines, docs = docs, nodes_to_retrieve = 3)

            # Functions performing LLM-preprocessing
            LLM_inputs, Discretionary_inputs = conditions_excel(orignal_excel_file)
            #fund_variable = extract_fund_variable(info_excel_file = info_excel_file)
            orignal_excel_file, llm_full_index = prompts_to_substitute_variable(orignal_excel_file = orignal_excel_file, fund_variable = fund_variable, LLM_inputs = LLM_inputs)
            orignal_excel_file, llm_prompts_to_use, llm_prompts_index = storing_input_prompt_in_list(orignal_excel_file = orignal_excel_file, llm_full_index = llm_full_index)

            # Diagnostic purposes

            # st.write("Checking fund variable")
            # st.write(fund_variable)

            # st.write("Checking list - llm_prompts_to_use")
            # st.write(llm_prompts_to_use)

            # st.write("Checking list - llm_prompts_index")
            # st.write(llm_prompts_index)

            # Showing dataframes for diagnostic purposes
            # list_of_dataframes = [orignal_excel_file, info_excel_file]
            # show_dataframes(list_of_dataframes)

        st.success("Successfully finished indexing & LLM-preprocessing", icon="✅")

        with st.spinner("Conducting Querying"):
            recursive_retriever, response_synthesizer, query_engine = recursive_retriever_old(vector_retriever = vector_retriever, df_id_query_engine_mapping = df_id_query_engine_mapping, service_context = service_context)

            # Diagnostic purposes

            # st.write("Checking recursive_retriever")
            # st.write(type(recursive_retriever))
            # st.write(recursive_retriever)

            # st.write("Checking response_synthesizer")
            # st.write(type(response_synthesizer))
            # st.write(response_synthesizer)

            # st.write("Checking query engine")
            # st.write(type(query_engine))
            # st.write(query_engine)  
            
        st.success("Successfully finished Querying", icon="✅")

        with st.spinner("Conducting Prompting"):
             
            output_response, output_context, context_with_max_score_list, file_path_metadata_list, source_metadata_list = prompt_loop_advanced(query_engine = query_engine, llm_prompts_to_use = llm_prompts_to_use, nodes_to_retrieve = nodes_to_retrieve, sleep = sleep, return_all_chunks = return_all_chunks)

            # Showing list for diagnostic purposes
            # st.write("Final output")
            # st.write(output_response)
            # st.write(output_context)

        st.success("Successfully finished Prompting", icon="✅")


        with st.spinner("Conducting Post-LLM Prompting"):

            orignal_excel_file = create_output_result_column(orignal_excel_file = orignal_excel_file, llm_prompts_index = llm_prompts_index, output_response = output_response)
            orignal_excel_file = create_output_context_column(orignal_excel_file, llm_prompts_index, nodes_to_retrieve = nodes_to_retrieve, output_context = output_context)
            intermediate_output_to_excel(orignal_excel_file = orignal_excel_file, excel_directory = "Results", output_excel_filename = "results_output", file_extension = "xlsx")

        st.success("Successfully finished Post-LLM Prompting", icon="✅")


        with st.spinner("Parsing"):
            schema = create_schema_from_excel(orignal_excel_file, llm_prompts_index)
            orignal_excel_file = parse_value(output_response = output_response, llm_prompts_index = llm_prompts_index, orignal_excel_file = orignal_excel_file, schema = schema, llm = llm)

        st.success("Successfully finished Parsing", icon="✅")


        with st.spinner("Post-parsing"):
            filtered_excel_file = create_filtered_excel_file(orignal_excel_file = orignal_excel_file, llm_prompts_index = llm_prompts_index)
            orignal_excel_file = final_result_orignal_excel_file(filtered_excel_file = filtered_excel_file, orignal_excel_file = orignal_excel_file, llm_prompts_index = llm_prompts_index)
            orignal_excel_file = reordering_columns(orignal_excel_file)

        st.success("Successfully finished Post-Parsing", icon="✅")


        with st.spinner("Fixing LLM-post processing fields"):
            results_fund_name_value = find_result_fund_name(orignal_excel_file)
            result_fund_house_value = find_result_fund_house(orignal_excel_file)
            result_fund_class_value = find_result_fund_class(orignal_excel_file)
            result_currency_value = find_result_currency(orignal_excel_file)
            result_acc_or_inc_value = find_result_acc_or_inc(orignal_excel_file)
            kristal_alias = create_new_kristal_alias(results_fund_name_value, result_fund_house_value, result_fund_class_value, result_currency_value, result_acc_or_inc_value)
            orignal_excel_file = update_kristal_alias(orignal_excel_file = orignal_excel_file, kristal_alias = kristal_alias)
            orignal_excel_file = update_sponsored_by(orignal_excel_file = orignal_excel_file, sponsored_by = "backend-staging+hedgefunds@kristal.ai")
            orignal_excel_file = update_required_broker(orignal_excel_file = orignal_excel_file, required_broker = "Kristal Pooled")
            orignal_excel_file = update_transactional_fund(orignal_excel_file = orignal_excel_file, transactional_fund = "Yes")
            orignal_excel_file = update_disclaimer(
                orignal_excel_file = orignal_excel_file,
                disclaimer = '''
                The recommendations contained herein are for the exclusive use of investor and prohibits any form of disclosure or reproduction. The content cannot be relied upon by any other person for any other purpose. The recommendations are preliminary information to the investors, are subject to risks and may change based on investment objectives, financials, liabilities or the risk profile of an investor. Any recommendations including financial advice provided by Kristal.AI or its affiliates shall be subject to contractual understanding, necessary documentation, applicable laws, approvals and regulations. The recommendations contained herein may not be eligible for sale/purchase in some jurisdictions, in specific, are not intended for residents of the USA or within the USA.Though the recommendations are based on information obtained from reliable sources and are provided in good faith, they may be valid only on the date and time the recommendations are provided and shall be subject to change without notice. Kristal.AI
                '''
                )
            orignal_excel_file = update_risk_disclaimer(
                orignal_excel_file = orignal_excel_file,
                risk_disclaimer = '''
                The recommendations contained herein are for the exclusive use of investor and prohibits any form of disclosure or reproduction. The content cannot be relied upon by any other person for any other purpose. The recommendations are preliminary information to the investors, are subject to risks and may change based on investment objectives, financials, liabilities or the risk profile of an investor. Any recommendations including financial advice provided by Kristal.AI or its affiliates shall be subject to contractual understanding, necessary documentation, applicable laws, approvals and regulations. The recommendations contained herein may not be eligible for sale/purchase in some jurisdictions, in specific, are not intended for residents of the USA or within the USA.Though the recommendations are based on information obtained from reliable sources and are provided in good faith, they may be valid only on the date and time the recommendations are provided and shall be subject to change without notice. Kristal.AI
                '''
                )
            result_nav_value = find_nav_value(orignal_excel_file)
            orignal_excel_file = update_nav_value(orignal_excel_file = orignal_excel_file, result_nav_value = result_nav_value)
            output_to_excel(orignal_excel_file = orignal_excel_file, excel_directory = "Results", output_excel_filename = "results_output", file_extension = "xlsx")

        st.success("Successfully Fixed LLM-post processing fields", icon="✅")

        #st.markdown("### Collective Prompt Results")
        st.markdown("### Results")

        return output_response, llm_prompts_to_use, context_with_max_score_list, file_path_metadata_list, source_metadata_list, orignal_excel_file, table_dfs, docs 


        # # Display dataframe containing final results
        # st.dataframe(data = orignal_excel_file, use_container_width = True, column_order = None)

        # # Display button to download results to excel file
        # download_data_as_excel(orignal_excel_file = orignal_excel_file)

        # # Display button to download results to csv file
        # download_data_as_csv(orignal_excel_file = orignal_excel_file)

