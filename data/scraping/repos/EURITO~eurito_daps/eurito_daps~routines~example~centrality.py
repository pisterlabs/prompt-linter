'''
Centrality Pipeline
==============
Takes network from Neo4j database, calculates network centrality measures and updates each node in the database with new centrality attributes

'''

from nesta.core.luigihacks.mysqldb import MySqlTarget
from nesta.core.luigihacks.misctools import get_config

from eurito_daps.packages.utils import openaire_utils, centrality_utils
from eurito_daps.packages.utils import globals
from eurito_daps.core.orms.openaire_orm import Base, SoftwareRecord
from eurito_daps.packages.cordis.cordis_neo4j import _extract_name, orm_to_neo4j

from nesta.core.orms.orm_utils import get_mysql_engine
from nesta.core.orms.orm_utils import graph_session

from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_enginels

from py2neo.data import Node, Relationship

import igraph as ig

import luigi
import datetime
import os
import logging
import requests
import time

class RootTask(luigi.WrapperTask):
    '''The root task, which collects the supplied parameters and calls the SimpleTask.

    Args:
        date (datetime): Date used to label the outputs
        output_type (str): type of record to be extracted from OpenAIRE API. Accepts "software", "datasets", "publications", "ECProjects"
        production (bool): test mode or production mode
    '''
    date = luigi.DateParameter(default=datetime.datetime.today())
    output_type = luigi.Parameter()
    production = luigi.BoolParameter(default=False)

    def requires(self):
        '''Call the task to run before this in the pipeline.'''

        logging.getLogger().setLevel(logging.INFO)
        return CalcCentralityTask(date=self.date,
                          output_type=self.output_type,
                          test=not self.production)

class CalcCentralityTask(luigi.Task):
    '''Takes network from Neo4j database, calculates network centrality measures and updates each node in the database with new centrality attributes

    Args:
        date (datetime): Date used to label the outputs
        output_type (str): type of record to be extracted from OpenAIRE API. Accepts "software", "datasets", "publications", "ECProjects"
        test (bool): run a shorter version of the task if in test mode
    '''

    date = luigi.DateParameter(default=datetime.datetime.today())
    output_type = luigi.Parameter()
    test = luigi.BoolParameter()

    def output(self):
        '''Points to the output database engine where the task is marked as done.
        The luigi_table_updates table exists in test and production databases.
        '''
        db_config = get_config(os.environ["MYSQLDB"], 'mysqldb')
        db_config["database"] = 'dev' if self.test else 'production'
        db_config["table"] = "Example <dummy>"  # Note, not a real table
        update_id = "OpenAireToNeo4jTask_{}".format(self.date)
        return MySqlTarget(update_id=update_id, **db_config)

    def run(self):

        # Get connection settings
        conf = get_config('neo4j.config', 'neo4j')
        gkwargs = dict(host=conf['host'], secure=True,
                       auth=(conf['user'], conf['password']))

        igr = ig.Graph()
        with graph_session(**gkwargs) as tx:
            graph = tx.graph
            logging.info('getting relationships list')
            all_rels = list(graph.relationships.match().limit(30000) ) #finds all relationships in a graph

            #create tuple list (edgelist)
            logging.info("found %d relationships" % len(all_rels))
            tuplelist = list()
            for index,rel in enumerate(all_rels):
                if index % 1000 == 0:
                    print (index)
                #what is a better way of changing the main graph
                start_index, igr = centrality_utils.get_index(rel.start_node, graph, igr)
                target_index, igr = centrality_utils.get_index(rel.end_node, graph, igr)
                rel_tuple = (start_index, target_index)
                tuplelist.append(rel_tuple)

            igr.add_edges(tuplelist)

            density = igr.density(loops=False)

            logging.info("density:", density)

            betw = igr.betweenness(vertices=None, directed=False, cutoff=3, weights=None, nobigint=True)

            logging.info("betweenness:", betw)

            centrality_utils.add_betw_property(graph, igr, betw)


        logging.debug('Writing to DB complete')

        # mark as done
        logging.info("Task complete")
        self.output().touch()
