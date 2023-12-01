"""
This module provides functionality to load data into a
sqlite database stored in memory
"""

from sqlalchemy import create_engine
import pandas as pd
from langchain import SQLDatabase

def load_data(journey_pricing, flights):
    """
    Loads flight and journey pricing data into an in-memory SQLite database.

    This function creates an in-memory SQLite database, loads flight and journey pricing data
    into this database, and returns an SQLDatabase object that serves as the interface to this database.

    Parameters:
    journey_pricing (pd.DataFrame): A DataFrame containing journey pricing data.
    flights (pd.DataFrame): A DataFrame containing flight data.

    Returns:
    db (SQLDatabase): An SQLDatabase object that serves as the interface to the SQLite database.
    """
    
    engine = create_engine('sqlite:///:memory:')

    # Write the data to the SQLite database
    flights.to_sql('flights', engine, if_exists='replace', index=False)
    journey_pricing.to_sql('journey_pricing', engine, if_exists='replace', index=False)
    # Check if the data was loaded correctly
    df_loaded = pd.read_sql('SELECT * FROM flights', engine)
    db = SQLDatabase(engine)
    return db