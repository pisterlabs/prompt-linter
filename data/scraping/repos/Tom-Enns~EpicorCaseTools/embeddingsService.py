# embeddingsService.py
import nltk
nltk.download('punkt')
from services.openAIService import OpenAIService
from services.pineconeService import PineconeService
from services.loggingService import LoggingService

logger = LoggingService.get_logger(__name__)

class EmbeddingsGeneratorService:
    def __init__(self):
        self.openai_service = OpenAIService()
        self.pinecone_service = PineconeService()

    def generate_and_store_embeddings(self, items, model="text-embedding-ada-002"):
        logger.info(f"Generating and storing embeddings for {len(items)} items...")
        try:
            for item in items:
                text, metadata = item
                chunks = self.split_text_into_chunks(text)

                for chunk_number, chunk in enumerate(chunks, start=1):
                    chunk_embedding = self.openai_service.generate_embeddings([chunk], model=model)

                    # Construct a unique ID using metadata and chunk number
                    case_number = metadata.get('CaseNum', 'Unknown')
                    component_type = metadata.get('ComponentType', 'Unknown')  # changed from 'Component' to 'ComponentType'
                    embedding_id = f"Case-{case_number}-{component_type}-{chunk_number}"

                    # Prepare the metadata to include with the embedding
                    metadata_with_chunk = {**metadata, "ChunkNumber": chunk_number}

                    # Prepare the data to upsert
                    data_to_upsert = [(embedding_id, chunk_embedding[0], metadata_with_chunk)]

                    self.pinecone_service.upsert_embeddings(data_to_upsert)

                logger.info(f"Successfully generated and stored embeddings for item: {case_number}")
        except Exception as e:
            logger.error(f"Error in generate_and_store_embeddings: {e}")
            raise



    def generate_and_find_similar_embeddings(self, text, top_k=5):
        logger.info(f"Generating and finding similar embeddings for text: {text}")
        try:
            query_embedding = self.openai_service.generate_embeddings([text])[0]
            search_results = self.pinecone_service.query_similar_embeddings(query_embedding, top_k)
            logger.info(f"Successfully found similar embeddings for the provided text.")
            return search_results
        except Exception as e:
            logger.error(f"Failed to generate and find similar embeddings: {str(e)}")
            raise

    def split_text_into_sentences(self, text):
        logger.info(f"Splitting text into sentences...")
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        sentences = tokenizer.tokenize(text)
        return sentences

    def split_text_into_chunks(self, text, max_tokens=8000):
        logger.info(f"Splitting text into chunks...")
        sentences = self.split_text_into_sentences(text)
        chunks = []
        current_chunk = []

        for sentence in sentences:
            # Check if adding the next sentence would exceed the max token limit
            test_chunk = current_chunk + [sentence]
            if self.openai_service.calculate_token_count(' '.join(test_chunk)) > max_tokens:
                # If adding the sentence exceeds the limit, finalize the current chunk
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
            else:
                # If not, add the sentence to the current chunk
                current_chunk.append(sentence)

        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks
