import jsonlines
import numpy as np
from typing import List, Dict, Tuple, DefaultDict, Any
from collections import defaultdict
import time
import random
import pickle
import os
import concurrent.futures
from pathlib import Path
from tqdm.auto import tqdm
from dateutil.parser import parse, ParserError
import openai

try:
    import config
    openai.api_key = config.OPENAI_API_KEY
except ImportError:
    openai.api_key = os.environ.get('OPENAI_API_KEY')


from .settings import PATH_TO_RAW_DATA, PATH_TO_DATASET_PKL, PATH_TO_DATASET_DICT_PKL, EMBEDDING_MODEL, LEN_EMBEDDINGS

from .text_splitter import TokenSplitter, split_into_sentences



error_count_dict = {
    "Entry has no source.": 0,
    "Entry has no title.": 0,
    "Entry has no text.": 0,
    "Entry has no URL.": 0,
    "Entry has wrong citation level.": 0
}


class MissingDataException(Exception):
    pass


class Dataset:
    def __init__(self,
            jsonl_data_path: str = PATH_TO_RAW_DATA,  # Path to the dataset .jsonl file.
            custom_sources: List[str] = None,  # List of sources to include, like "alignment forum", "lesswrong", "arxiv",etc.
            rate_limit_per_minute: int = 3_500,  # Rate limit for the OpenAI API.
            min_tokens_per_block: int = 300, # Minimum number of tokens per block.
            max_tokens_per_block: int = 400, # Maximum number of tokens per block.
            fraction_of_articles_to_use: float = 1.0,  # Fraction of articles to use. If 1.0, use all articles.
        ):
        self.jsonl_data_path = jsonl_data_path
        self.custom_sources = custom_sources
        self.rate_limit_per_minute = rate_limit_per_minute
        self.delay_in_seconds = 60.0 / self.rate_limit_per_minute
        self.fraction_of_articles_to_use = fraction_of_articles_to_use
        
        self.min_tokens_per_block = min_tokens_per_block  # for the text splitter
        self.max_tokens_per_block = max_tokens_per_block  # for the text splitter
        
        self.metadata: List[Tuple[str]] = []  # List of tuples, each containing the title, author, date, URL, and tags of an article.
        self.embedding_strings: List[str] = []  # List of strings, each being a few paragraphs from a single article (not exceeding max_tokens_per_block tokens).
        self.embeddings_metadata_index: List[int] = [] # List of integers, each being the index of the article from which the embedding string was taken.

        self.articles_count: DefaultDict[str, int] = defaultdict(int)  # Number of articles per source. E.g.: {'source1': 10, 'source2': 20, 'total': 30}

        if self.custom_sources is not None:
            for source in self.custom_sources:
                self.articles_count[source] = 0
        self.total_articles_count = 0
        
        self.total_char_count = 0
        self.total_word_count = 0
        self.total_sentence_count = 0
        self.total_block_count = 0
        
        self.sources_so_far: List[str] = []
        self.info_types: Dict[str, List[str]] = {}
    
    def extract_info_from_article(self, article: Dict[str, Any]) -> Tuple[str]:
        """
        This function extracts the title, author, date, URL, tags, and text from an article.
        
        Args:
            article (Dict[str, Any]): a dictionary containing the article's text and metadata.

        Returns:
            Tuple[str]: a tuple containing the title, author, date, URL, tags, and text of the article.
        """
        title: str = ""
        author: str = ""
        date_published: str = None
        url: str = None
        tags: str = None
        text: str = None
        
        # Get title
        if 'title' in article and 'book_title' in article and article['title']: title = article['title']
        elif 'book_title' in article and 'title' not in article and article['book_title']: 
            title = article['book_title']
        elif 'title' in article and article['title']: 
            title = article['title']
        title = title.strip('\n').replace('\n', ' ')[:100]

        # Get author
        if 'author' in article and 'authors' in article and article['author']: author = article['author']
        elif 'authors' in article and article['authors']: author = article['authors']
        elif 'author' in article and article['author']: author = article['author']
        if type(author) == str: author = get_authors_list(author)
        if type(author) == list: author = ', '.join(author)
        author = author.strip('\n').replace('\n', ' ')[:100]

        # Get date published
        if 'date_published' in article and article['date_published'] and len(article['date_published']) >= 10: date_published = article['date_published'][:10]
        elif 'published' in article and article['published'] and len(article['published']) >= 16: date_published = article['published'][:16]
        else: date_published = None
        if date_published is not None:
            date_published = standardize_date(date_published)
            
        # Get URL
        if 'link' in article and article['link']: url = article['link']
        elif 'url' in article and article['url']: url = article['url']
        elif 'doi' in article and article['doi']: url = article['doi']
        else: url = None
            
        # Get tags
        if 'tags' in article and article['tags']:
            if type(article['tags']) == list: tags = ', '.join([val['term'] for val in article['tags']])
            elif type(article['tags']) == str: tags = article['tags']
            else: tags = None
        
        # Get text
        if 'text' in article and article['text']: text = article['text']
        else:
            raise MissingDataException(f"Entry has no text.")

        return (title, author, date_published, url, tags, text)
           
    def get_alignment_texts(self):
        text_splitter = TokenSplitter(self.min_tokens_per_block, self.max_tokens_per_block)
        with jsonlines.open(self.jsonl_data_path, "r") as reader:
            for entry in tqdm(reader):
                try:
                    if 'source' not in entry: 
                        if 'url' in entry and entry['url'] == "https://www.cold-takes.com/": 
                            entry["source"] = "Cold Takes"
                        elif 'question' in entry and 'answer' in entry: 
                            entry["source"] = "printouts"
                            continue # for now, skip printouts
                        elif 'article_url' in entry and entry['article_url'] == "https://www.gwern.net":
                            entry["source"] = "gwern.net"
                        elif 'url' in entry and entry['url'] == "https://generative.ink/posts/":
                            entry["source"] = "generative.ink"
                        elif 'url' in entry and entry['url'][:24] == "https://greaterwrong.com":
                            entry["source"] = "greaterwrong.com"
                        else:
                            raise MissingDataException("Entry has no source.")
                    
                    # if we specified custom sources, only include articles from those sources
                    if (self.custom_sources is not None) and (entry['source'] not in self.custom_sources):
                        continue

                    
                    if entry["source"] == 'alignment forum':
                        if int(entry['score'].replace('−', '-')) < 70: continue
                    elif entry["source"] == 'lesswrong':
                        if int(entry['score'].replace('−', '-')) < 150: continue
                    elif entry["source"] == 'arxiv':
                        if 'citation_level' != '0': continue

                    # Dict describing the proportion of each source we want:
                    # E.g.: {'arxiv': 0.5, 'youtube': 0.5, 'lesswrong': 1.0}
                    desired_source_proportions = {
                        "https://aipulse.org": 1,
                        "ebook": 0,
                        "https://qualiacomputing.com": 0.02,
                        "alignment forum": .7,
                        "lesswrong": .5,
                        "manual": 1,
                        "arxiv": 0.1,
                        "https://deepmindsafetyresearch.medium.com/": 1,
                        "waitbutwhy.com": 1,
                        "GitHub": 1,
                        "https://aiimpacts.org": 0.2,
                        "arbital.com": 0.2,
                        "carado.moe": 0.3,
                        "nonarxiv_papers": 0.1,
                        "https://vkrakovna.wordpress.com": .5,
                        "https://jsteinhardt.wordpress.com": .5,
                        "audio-transcripts": 0.2,
                        "https://intelligence.org": .1,
                        "youtube": 0.07,
                        "reports": 0.4,
                        "https://aisafety.camp": 1,
                        "curriculum": 1,
                        "https://www.yudkowsky.net": 0.2,
                        "distill": 1,
                        "Cold Takes": 0.5,
                        "printouts": 1,
                        "gwern.net": 1,
                        "generative.ink": 1,
                        "greaterwrong.com": 0.2
                    }
                    
                    random_number = random.random()
                    if random_number > desired_source_proportions[entry['source']]:
                        continue
                    
                    # if we specified a fraction of articles to use, only use that fraction from the remaining articles
                    random_number = random.random()
                    if random_number > self.fraction_of_articles_to_use:
                        continue
                    
                    # Get title, author, date, URL, tags, and text
                    title, author, date_published, url, tags, text = self.extract_info_from_article(entry)
                    
                    # If there's less than 2 of 'title', 'author' and 'url', ignore this text
                    if (((title or '').strip() == '') + ((author or '').strip() == '') + ((url or '').strip() == '')) > 1:
                        print(f'{entry["source"]}')
                        continue
                    
                    #if the text is too short, ignore this text
                    if len(text) < 500:
                        continue

                    #we're keeping the text so we inc the aticle count
                    self.articles_count[entry['source']] += 1
                    self.total_articles_count += 1
                    
                    # Get signature
                    signature = ""
                    if title: signature += f"Title: {title}, "
                    else: signature += f"Title: None, "
                    if author: signature += f"Author: {author}"
                    else: signature += f"Author: None"
                    # if date_published: signature += f"Date published: {date_published}, "
                    # if url: signature += f"URL: {url}, "
                    # if tags: signature += f"Tags: {tags}, "  # Temporary decision to not include tags in the signature
                    # if signature: signature = signature[:-2]
                    signature = signature.replace("\n", " ")
                    
                    # Add info to metadata and embedding strings
                    self.metadata.append((title, author, date_published, url, tags))
                    blocks = text_splitter.split(text, signature)
                    self.embedding_strings.extend(blocks)
                    self.embeddings_metadata_index.extend([self.total_articles_count-1] * len(blocks))
                    
                    # Update counts
                    self.total_char_count += len(text)
                    self.total_word_count += len(text.split())
                    self.total_sentence_count += len(split_into_sentences(text))
                    self.total_block_count += len(blocks)
                    
                except MissingDataException as e:
                    if str(e) not in error_count_dict:
                        error_count_dict[str(e)] = 0
                    error_count_dict[str(e)] += 1

    def get_embeddings(self):
        def get_embeddings_at_index(texts: str, batch_idx: int, batch_size: int = 200): # int, np.ndarray
            embeddings = np.zeros((batch_size, 1536))
            openai_output = openai.Embedding.create(
                model=EMBEDDING_MODEL, 
                input=texts
            )['data']
            for i, embedding in enumerate(openai_output):
                embeddings[i] = embedding['embedding']
            return batch_idx, embeddings

        batch_size = 500
        rate_limit = 3500 / 60  # Maximum embeddings per second

        start = time.time()
        self.embeddings = np.zeros((len(self.embedding_strings), LEN_EMBEDDINGS))

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(
                get_embeddings_at_index, 
                self.embedding_strings[batch_idx:batch_idx+batch_size], 
                batch_idx,
                len(self.embedding_strings[batch_idx:batch_idx+batch_size])
            ) for batch_idx in range(0, len(self.embedding_strings), batch_size)]
            num_completed = 0
            for future in concurrent.futures.as_completed(futures):
                batch_idx, embeddings = future.result()
                num_completed += embeddings.shape[0]
                self.embeddings[batch_idx:batch_idx+embeddings.shape[0]] = embeddings

                elapsed_time = time.time() - start
                expected_time = num_completed / rate_limit
                sleep_time = max(expected_time - elapsed_time, 0)
                time.sleep(sleep_time)

                print(f"Completed {num_completed}/{len(self.embedding_strings)} embeddings in {elapsed_time:.2f} seconds.")

    def save_embeddings(self, path: str):
        np.save(path, self.embeddings)
        
    def load_embeddings(self, path: str):
        self.embeddings = np.load(path)
        
    def save_class(self, path: str = PATH_TO_DATASET_PKL):
        # Save the class to a pickle file
        print(f"Saving class to {path}...")
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    def save_data(self, path: str = PATH_TO_DATASET_DICT_PKL):
        # Save the data to a pickle file
        print(f"Saving data to {path}...")
        data = {
            "metadata": self.metadata,
            "embedding_strings": self.embedding_strings,
            "embeddings_metadata_index": self.embeddings_metadata_index,
            "embeddings": self.embeddings.astype(np.float32),
            "articles_count": self.articles_count,
            "total_articles_count": self.total_articles_count,
            "total_char_count": self.total_char_count,
            "total_word_count": self.total_word_count,
            "total_sentence_count": self.total_sentence_count,
            "total_block_count": self.total_block_count
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)


def get_authors_list(authors_string: str) -> List[str]:
    """
    Given a string of authors, return a list of the authors, even if the string contains a single author.
    """
    authors_string = authors_string.replace(" and ", ",")
    authors_string = authors_string.replace('\n', ' ')
    authors = []
    if authors_string is None:
        return []
    if "," in authors_string:
        authors = [author.strip() for author in authors_string.split(",")]
    else:
        authors = [authors_string.strip()]
    return authors

def standardize_date(date_string, default_date='n/a'):
    try:
        dt = parse(date_string)
        return dt.strftime('%Y-%m-%d')
    except (ParserError, ValueError):
        return default_date



"""
if __name__ == "__main__":
    # List of possible sources:
    all_sources = ["https://aipulse.org", "ebook", "https://qualiacomputing.com", "alignment forum", "lesswrong", "manual", "arxiv", "https://deepmindsafetyresearch.medium.com", "waitbutwhy.com", "GitHub", "https://aiimpacts.org", "arbital.com", "carado.moe", "nonarxiv_papers", "https://vkrakovna.wordpress.com", "https://jsteinhardt.wordpress.com", "audio-transcripts", "https://intelligence.org", "youtube", "reports", "https://aisafety.camp", "curriculum", "https://www.yudkowsky.net", "distill", "Cold Takes", "printouts", "gwern.net", "generative.ink", "greaterwrong.com"] # These sources do not have a source field in the .jsonl file

    # List of sources we are using for the test run:
    custom_sources = [
        # "https://aipulse.org", 
        # "ebook", 
        # "https://qualiacomputing.com", 
        # "alignment forum", 
        # "lesswrong", 
        "manual", 
        # "arxiv", 
        # "https://deepmindsafetyresearch.medium.com", 
        "waitbutwhy.com", 
        # "GitHub", 
        # "https://aiimpacts.org", 
        # "arbital.com", 
        # "carado.moe", 
        # "nonarxiv_papers", 
        # "https://vkrakovna.wordpress.com", 
        "https://jsteinhardt.wordpress.com", 
        # "audio-transcripts", 
        # "https://intelligence.org", 
        # "youtube", 
        # "reports", 
        "https://aisafety.camp", 
        "curriculum", 
        "https://www.yudkowsky.net", 
        # "distill",
        # "Cold Takes",
        # "printouts",
        # "gwern.net",
        # "generative.ink",
        # "greaterwrong.com"
    ]
    
    dataset = Dataset(
        jsonl_data_path=PATH_TO_RAW_DATA.resolve(), 
        custom_sources=custom_sources, 
        rate_limit_per_minute=3500, 
        min_tokens_per_block=200, max_tokens_per_block=300, 
        # fraction_of_articles_to_use=1/2000
    )
    dataset.get_alignment_texts()
    dataset.get_embeddings()
    # dataset.save_embeddings("data/embeddings.npy")
    
    dataset.save_class(PATH_TO_DATASET.resolve())
    # # dataset = pickle.load(open("dataset.pkl", "rb"))
    """
    