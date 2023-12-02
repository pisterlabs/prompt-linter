from langchain.document_loaders import DataFrameLoader
from langchain.vectorstores import Weaviate
import os, time

WEAVIATE_URL = os.getenv('WEAVIATE_URL') or 'http://localhost:8080'


data_array_counter = 0
current_duration_seconds = 0
current_records_processed = 0

def calculate_average_time_per_record(current_duration_seconds, current_records_processed):
    if current_duration_seconds == 0:
        return 0

    return round(current_duration_seconds / current_records_processed, 4)


def calculate_time_remaining(total_rows, current_duration_seconds, current_records_processed):
    if current_duration_seconds == 0:
        return 0

    remaining_rows = total_rows - current_records_processed
    remaining_duration_seconds = remaining_rows * calculate_average_time_per_record(current_duration_seconds, current_records_processed)
    
    if remaining_duration_seconds > 60:
        return f'{round(remaining_duration_seconds / 60, 2)}mins'
    
    return f'{remaining_duration_seconds}sec'


def process_group(group, embeddings_llm, start, end, row_count):
    loader = DataFrameLoader(group, page_content_column="body")
    data = loader.load()
    rows_processed = len(data)
    print(f'{row_count}-Start: {start}:{end} inserting {rows_processed} records')
    start_time = time.time()

    MAX_ROWS = 1000
    # check if data rows is more than 5000, if so split into chunks
    if rows_processed > MAX_ROWS:
        data_chunks = [data[x:x+MAX_ROWS] for x in range(0, len(data), MAX_ROWS)]
        print(f'{row_count}-Split: {rows_processed} records into chunks of {len(data_chunks)}')
        for i,chunk in enumerate(data_chunks):
            print(f'{row_count}-Split-Chunk: {i} of {len(data_chunks)}')
            db_empty_index = Weaviate.from_documents(
                chunk, 
                embeddings_llm, 
                weaviate_url=WEAVIATE_URL,
                index_name='test_index',
                by_text=False
            )
    else:
        db_empty_index = Weaviate.from_documents(
            data, 
            embeddings_llm, 
            weaviate_url=WEAVIATE_URL,
            index_name='test_index',
            by_text=False
        )
    
    end_time = time.time()
    duration = round(end_time - start_time, 1)
    index = row_count
    print(f'{index}-Row: inserted records: {rows_processed} in {duration}sec avg: {calculate_average_time_per_record(duration, rows_processed)} -  date: {start}-{end}:')
    return duration, rows_processed, index

    # return f'{len(data)}, {duration} for {start}:{end}'