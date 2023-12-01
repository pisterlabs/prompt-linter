import os
import time
from utils import get_logger
from utils.db_helper import get_engine
from utils.open_ai import analize_text
import openai
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import text as texutal



logger = get_logger('OPENAI')

sql_server_engine = get_engine()

Session = sessionmaker(bind=sql_server_engine)



def get_unknown_text_df(table:str=None):
    if table == 'posts':
        sql = """SELECT top 100 * FROM twitter_posts where sentiment='unknown'"""

    elif table == 'comments':
        sql = """SELECT top 100 * FROM twitter_comments where sentiment='unknown'"""

    df = pd.read_sql(sql, sql_server_engine)
    return df


def update_sentiment(table:str=None, session:Session = None,
                     id:int = None, sentiment:str = None, tone:str = None):
    update_query = f"""
        UPDATE {table}
        SET sentiment = '{sentiment}', tone = '{tone}'
        WHERE id = '{id}'
    """

    # Execute the SQL query with parameters
    session.execute(texutal(update_query))

    # Step 5: Commit the changes to the database
    session.commit()

if __name__ == "__main__":
    logger.info(f'Starting sentiment job at:{datetime.now()}')
    try:
        posts = get_unknown_text_df('posts')
        count = 0
        if posts.shape[0]>0:
            session = Session()
            table = 'twitter_posts'
            logger.info('Starting sentiment analysis for Twitter Posts')
            for id, text in zip(posts['id'], posts['text']):
                print(id, text)
                result = analize_text(text)
                print(result)
                sentiment, tone = result.split('--|--')
                update_sentiment(table=table, session=session, id=id, sentiment=sentiment, tone=tone)
                count+=1
                time.sleep(1)
            logger.info(f'Analized {count} Twitter Posts!')
            session.close()

        comments = get_unknown_text_df('comments')
        count=0
        if comments.shape[0]>0:
            session = Session()
            table = 'twitter_comments'
            logger.info('Starting sentiment analysis for Twitter Comments')
            for id, text in zip(comments['id'],comments['text']):
                print(id, text)
                result = analize_text(text)
                print(result)
                sentiment, tone = result.split('--|--')
                update_sentiment(table=table, session=session, id=id, sentiment=sentiment, tone=tone)
                count+=1
                time.sleep(1)
            logger.info(f'Analized {count} Twitter Comments!')
            session.close()
        logger.info(f'Completed sentiment job at:{datetime.now()}')
    except Exception as e:
        print(e)
        logger.error(str(e))
