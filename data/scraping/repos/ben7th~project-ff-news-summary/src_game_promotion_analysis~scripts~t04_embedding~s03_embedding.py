import os
import sys
SRC_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(SRC_PATH)

from setup.setup import connect_to_db
from logger.setup_logger import get_loguru_logger
logger = get_loguru_logger('check_topic', log_file_path='../logs/check_topic.log',
                           console_level='INFO')

from models.web_page_item import WebPageEmbeddingItem
from utils.embedding import openai_embeddings

def main():
    records = WebPageEmbeddingItem.objects.all()
    total = len(records)

    vectors_count = 0
    for index, record in enumerate(records, start=1):
        if record.block_embeddings:
            vectors_count += len(record.block_embeddings)
            continue

        print(f'{index}/{total} embedding: {record.url}')
        normalized_text_blocks = record.normalized_text_blocks
        embeddings = openai_embeddings(normalized_text_blocks)
        record.block_embeddings = embeddings
        record.save()

        vectors_count += len(normalized_text_blocks)
        # break
    
    print(f'共 {vectors_count} vectors')

if __name__ == '__main__':
    connect_to_db()
    main()