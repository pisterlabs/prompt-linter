'''
Centrality Pipeline
===================

Takes network from Neo4j database, calculates network centrality measures and updates each node in the database with new centrality attributes
'''

from nesta.core.luigihacks.mysqldb import MySqlTarget
from nesta.core.luigihacks.misctools import get_config

from eurito_daps.packages.utils import centrality_utils
from nesta.core.orms.orm_utils import graph_session

import igraph as ig
from py2neo import Graph as pgraph
import luigi
import datetime
import os
import logging


class RootTask(luigi.WrapperTask):
    '''The root task, which collects the supplied parameters and calls the main task.

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

        conf = get_config('neo4j.config', 'neo4j')
        gkwargs = dict(host=conf['host'], secure=True,
                       auth=(conf['user'], conf['password']))

        igr = ig.Graph()
        with graph_session(**gkwargs) as tx:
            graph = tx.graph
            all_rels = list(graph.relationships.match() ) #finds all relationships in a graph
            tuplelist = list()
            
            for index,rel in enumerate(all_rels):
                rel_tuple = (rel.start_node.identity, rel.end_node.identity)
                tuplelist.append(rel_tuple)

            newgraph = igr.TupleList(tuplelist)

            betw_list = newgraph.betweenness(vertices=None, directed=False, cutoff=3, weights=None, nobigint=True)

            centrality_utils.add_betw_property(graph, newgraph, betw_list)


        logging.debug('Writing to DB complete')

        # mark as done
        logging.info("Task complete")
        self.output().touch()
