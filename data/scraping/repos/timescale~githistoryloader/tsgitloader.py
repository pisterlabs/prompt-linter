import psycopg2
from dotenv import find_dotenv, load_dotenv
import os
import shutil
import subprocess
from git import Repo
import threading
import pandas as pd
from toolchainutils import LangChain, LlamaIndex
from multiprocessing import Lock, Process
import multiprocessing

TSV_TM_CATALOG_TABLE_NAME = "time_machine_catalog"
SCRATCH_REPO_DIR = "temprepo"
DEFAULT_TOOL_CHAIN = "langchain"

def github_url_to_table_name(github_url, toolchain):
    repository_path = github_url.replace("https://github.com/", "")
    table_name = toolchain+"_"+repository_path.replace("/", "_")
    return table_name

def record_catalog_info(repo_url, branch, toolchain):
    with psycopg2.connect(dsn=os.environ["TIMESCALE_SERVICE_URL"]) as connection:
        # Create a cursor within the context manager
        with connection.cursor() as cursor:
            # Define the Git catalog table creation SQL command
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {TSV_TM_CATALOG_TABLE_NAME} (
                repo_url TEXT,
                table_name TEXT,
                tool_chain TEXT,
                PRIMARY KEY(repo_url, tool_chain)
            );
            """
            cursor.execute(create_table_sql)

            delete_sql = f"DELETE FROM {TSV_TM_CATALOG_TABLE_NAME} WHERE repo_url = %s AND tool_chain = %s" 
            cursor.execute(delete_sql, (repo_url, toolchain))

            insert_data_sql = """
            INSERT INTO time_machine_catalog (repo_url, table_name, tool_chain)
            VALUES (%s, %s, %s);
            """
            
            table_name = github_url_to_table_name(repo_url, toolchain)
            cursor.execute(insert_data_sql, (repo_url, table_name, toolchain))
            connection.commit()
    connection.close()
    return table_name

def git_clone_url(repo_url, branch, tmprepo_dir):
    # Check if the clone directory exists, and if so, remove it
    if os.path.exists(tmprepo_dir):
        shutil.rmtree(tmprepo_dir)
    os.makedirs(tmprepo_dir)
    try:
        # Clone the Git repository with the specified branch
        res = subprocess.run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                "--single-branch",
                "--branch=" + branch,
                repo_url + ".git",
                tmprepo_dir,
            ],
            capture_output=True,
            text=True,
            cwd=".",  # Set the working directory here
        )

        if res.returncode != 0:
            raise ValueError(f"Git failed: {res.returncode}")
        return tmprepo_dir
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")

def tool_chain_factory(repo_dir, table_name, tool_chain):
    tool = None
    if (tool_chain == "langchain"):
        tool = LangChain(repo_dir, table_name)
    if (tool_chain == "llamaindex"):
        tool = LlamaIndex(repo_dir, table_name)
    return tool

def call_tool_chain_utils(lock, repo_dir, table_name, toolchain, params):
    with lock:
        toolchain_obj = tool_chain_factory(repo_dir, table_name, toolchain)
    toolchain_obj.process(params[0], params[1])
    with lock:
        toolchain_obj.save()

def setup_tables(table_name, toolchain):
    toolchain_obj = tool_chain_factory("", table_name, toolchain)
    toolchain_obj.create_tables()

def insert_rows_for_tool_chain(repo_dir, table_name, toolchain):
    repo = Repo(repo_dir)
    commit_count = len(list(repo.iter_commits()))
    print(f"Commit count: {commit_count}")
    process_count = multiprocessing.cpu_count()
    commits_per_process = commit_count//(process_count)
    remainder = commit_count % process_count
    process_commit_counts = [commits_per_process] * (process_count - 1) + [commits_per_process + remainder]
    print(process_commit_counts)
    skip = 0
    lock = Lock()
    processes = []
    for commit_count in process_commit_counts:
        data = [commit_count, skip]
        skip += commit_count
        p = Process(target=call_tool_chain_utils, args=(lock, repo_dir, table_name, toolchain, data))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()

def multi_load(repo_url, branch="master", tool_chain="langchain,llamaindex"):
    repo_dir = git_clone_url(repo_url, branch, SCRATCH_REPO_DIR)
    #repo_dir = SCRATCH_REPO_DIR
    tool_chain_list = tool_chain.split(",")
    for toolchain in tool_chain_list:
        table_name = record_catalog_info(repo_url, branch, toolchain)
        setup_tables(table_name, toolchain)
        insert_rows_for_tool_chain(repo_dir, table_name, toolchain)

def read_catalog_info(toolchain=DEFAULT_TOOL_CHAIN)->any:
    with psycopg2.connect(dsn=os.environ["TIMESCALE_SERVICE_URL"]) as connection:
        # Create a cursor within the context manager
        with connection.cursor() as cursor:
            try:
                select_data_sql = f"SELECT repo_url, table_name FROM time_machine_catalog WHERE tool_chain = \'{toolchain}\'"
                print(select_data_sql)
                cursor.execute(select_data_sql)
            except psycopg2.errors.UndefinedTable as e:
                return {}
            catalog_entries = cursor.fetchall()
            catalog_dict = {}
            for entry in catalog_entries:
                repo_url, table_name = entry
                catalog_dict[repo_url] = table_name
    connection.close()
    return catalog_dict

def load_git_history(repo_url:str, branch:str, toolchain:str):
    multi_load(repo_url, branch, toolchain)