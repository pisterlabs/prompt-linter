import openai
import faiss
import dotenv
import psycopg2
import os
from os import listdir
import validators
import re
import subprocess
import random
import shutil
import numpy as np
import time

from .utils.simplecst import SimpleCST
from .utils.simpletreesitter import SimpleTreeSitter

dotenv.load_dotenv()

openai.organization = os.environ["OPENAI_ORG_ID"]
openai.api_key = os.environ["OPENAI_API_KEY"]

# Threshold for maximum folder size (in bytes). This is set to 100MB
MAX_SIZE = 100000000

## NOTE:
# 1.) Library for AST:
#   - LibCST: Python
#   - TreeSitter: Javascript
# 2.) Indexing for libCST starts from 1, and for TreeSitter starts from 0 (accounted for by add 1 in append item)
# 3.) Write functional code, do not make any state variable that needs to be updated in the flow

## TODO:
# 1.) Add a re-index button, to re-index an already indexed library. Preferably store the SHA of the latest commit
# 2.) Add a button to delete a repo's embedding table (Only for the self-hosted version)
# 3.) Add a check that the number of tokens are less than the max allowed by OpenAI, if so then take those many tokens


class FlaskAPIHandler(object):
    def __init__(self, configs):
        self.configs = configs

        # Database
        self.conn = psycopg2.connect(
            database=configs["postgres"]["db_name"],
            user=os.environ["POSTGRES_USER"],
            password=os.environ["POSTGRES_PASSWORD"],
            host=configs["postgres"]["host"],
            port=configs["postgres"]["port"],
        )
        self._create_mapping_table()

        # FAISS
        self._create_faiss_index()

    def __del__(self):
        self.conn.close()

    def _create_faiss_index(self):
        cpu_index = faiss.IndexFlatIP(self.configs["model"]["dimension"])

        if faiss.get_num_gpus() == 0:
            print("No GPU detected. Using CPU.")
            self.index = cpu_index
        else:
            print("GPU detected. Using GPU.")
            self.res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(
                self.res, 0, cpu_index
            )  # Currently only supports one GPU

    def _create_embedding_table(self, table_name):
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings_{} (
                id SERIAL PRIMARY KEY,
                embedding FLOAT[{}],
                function_name TEXT,
                class_name TEXT,
                filepath TEXT,
                line_number INT
            );
            """.format(
                table_name, self.configs["model"]["dimension"]
            )
        )
        self.conn.commit()

    def _create_mapping_table(self):
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS mapping (
                id SERIAL PRIMARY KEY,
                url TEXT,
                is_public BOOLEAN
            )
            """
        )
        self.conn.commit()

    def _update_embedding_table(
        self, embedding, function_name, class_name, filepath, line_number, table_name
    ):
        cur = self.conn.cursor()
        try:
            cur.execute(
                """
                INSERT INTO embeddings_{} (embedding, function_name, class_name, filepath, line_number)
                VALUES (%s, %s, %s, %s, %s);
                """.format(
                    table_name
                ),
                (embedding, function_name, class_name, filepath, line_number),
            )
            self.conn.commit()

        except Exception as e:
            print("Error in FlaskAPIHandler._update_embedding_table: ", e)

    def _update_mapping_table(self, url, is_public):
        cur = self.conn.cursor()
        try:
            cur.execute(
                """
            INSERT INTO mapping (url, is_public) VALUES (%s, %s);""",
                (str(url), is_public),
            )
        except Exception as e:
            print("Error in FlaskAPIHandler._update_mapping_table: ", e)

        self.conn.commit()

        cur.execute("""SELECT id FROM mapping WHERE url = '{}'""".format(url))
        id = cur.fetchall()[0][0]

        return id

    def _get_embedding_table(self, table_name):
        cur = self.conn.cursor()
        cur.execute("""SELECT * FROM embeddings_{};""".format(table_name))
        return cur.fetchall()

    # This function currently only supports single query.
    def _get_nearest_neighbors(self, query_embedding, table_name):
        embedding_table = self._get_embedding_table(table_name=table_name)

        D, I = self.index.search(
            np.expand_dims(query_embedding, axis=0).astype("float32"),
            self.configs["model"]["num_nearest_neighbours"],
        )

        nearest_neighbors = []
        for i in I[0]:
            nearest_neighbors.append(embedding_table[i])

        result_dict = {}

        for i, result in enumerate(nearest_neighbors):
            result_dict["{}".format(i)] = {
                "function_name": result[2],
                "class_name": result[3],
                "filepath": result[4],
                "line_number": result[5],
            }

        return result_dict

    # TODO: Add support for other languages. Currently only supports Python, Javascript.
    def _generate_encoding_string(
        self, language, function_def, function_name, class_name
    ):
        if language == "python":
            encoding_string = "# Language: {}, Class name: {}, Function name: {} \n# Function definition: \n{}".format(
                language, class_name, function_name, function_def
            )
        elif language == "javascript":
            encoding_string = "// Language: {}, Class name: {}, Function name: {} \n# Function definition: \n{}".format(
                language, class_name, function_name, function_def
            )

        return encoding_string

    def _parse_file(self, filepath):
        # Using LibCST
        if filepath.endswith(".py"):
            simple_cst = SimpleCST()
            simple_cst.set_module(filepath=filepath)

            _ = simple_cst.wrapper.visit(simple_cst)

            return simple_cst.func_list

        # Using TreeSitter
        elif filepath.endswith(".js"):
            sts = SimpleTreeSitter()
            sts.set_language("javascript")
            sts.set_module(path=filepath)
            sts.traverse_tree(sts.module.root_node)

            return sts.func_list

    def _generate_index(self, table_name):
        embedding_table = self._get_embedding_table(table_name=table_name)

        if embedding_table == []:
            return

        embedding_np = np.vstack(
            np.asarray(embedding_table, dtype=object)[:, 1]
        ).astype("float32")
        print("embedding_np: ", embedding_np.dtype, " ", embedding_np.shape)
        self.index.add(embedding_np)
        print("Index size: ", self.index.ntotal)

    def _reset_database(self, table_name):
        try:
            cur = self.conn.cursor()
            cur.execute("DELETE FROM embeddings_{};".format(table_name))
            self.conn.commit()

        except Exception as e:
            print("Error in FlaskAPIHandler._reset_database: ", e)

    def _check_if_valid_url(self, url):
        validation = validators.url(url, public=True)

        if validation == True:
            pattern = "^https:\/\/github\.com\/[A-Za-z0-9-]+\/[A-Za-z0-9-_.]+$"
            match = re.match(pattern, url)

            if match:
                return True
            else:
                return False
        else:
            return False

    def _check_if_public(self, url):
        if self._check_if_valid_url(url=url):
            return "Yes"
        elif os.path.isdir(os.path.join("/mnt", url[1:])):
            return "No"
        else:
            return "Error"

    def _check_if_indexed(self, url):
        cur = self.conn.cursor()
        cur.execute("""SELECT id FROM mapping WHERE url = '{}'""".format(url))
        id = cur.fetchall()

        if len(id) == 0:
            return False, -1
        elif len(id) == 1:
            return True, id[0][0]
        else:
            print(
                "Error in FlaskAPIHandler._check_if_indexed, multiple entries in the DB for this Github URL"
            )
            raise Exception("There are multiple entries in the DB for this Github URL")

    def _get_folder_size(self, folder):
        total_size = 0
        for path, dirs, files in os.walk(folder):
            for f in files:
                fp = os.path.join(path, f)
                if os.path.exists(fp):
                    total_size += os.path.getsize(fp)
        return total_size

    # TODO: Put try except in git clone
    def _clone_repo(self, url):
        project_folder_name = random.randint(10000, 99999)
        project_path = os.path.join("/app/gitrepos", str(project_folder_name))

        while os.path.isdir(project_path):
            project_folder_name = random.randint(10000, 99999)
            project_path = os.path.join("/app/gitrepos", project_folder_name)

        os.makedirs(project_path)
        git_process = subprocess.Popen(["git", "clone", url, project_path])

        # Periodically check folder size
        while git_process.poll() is None:
            folder_size = self._get_folder_size(project_path)
            if folder_size > MAX_SIZE:
                git_process.terminate()
                print(
                    "Error: Git clone process terminated, folder size exceeded threshold."
                )
                shutil.rmtree(project_path)
                return False, ""

        return True, project_path

    def _get_embedding_from_input(self, input):
        response = openai.Embedding.create(
            input=input,
            model=self.configs["model"]["code_embedding"],
        )

        embedding = np.array(response["data"][0]["embedding"])

        return embedding

    def _encode_from_path(self, project_path, table_name, is_public, request):
        for root, dirs, files in os.walk(project_path):
            for file in files:
                if file.endswith(".py") or file.endswith(".js"):
                    file_path = os.path.join(root, file)
                    func_list = self._parse_file(file_path)

                    if file.endswith(".py"):
                        language = "python"
                    elif file.endswith(".js"):
                        language = "javascript"

                    for func in func_list:
                        function_name = func["func_name"]
                        class_name = func["class_name"]
                        line_number = func["line_number"]
                        function_def = func["source"]

                        encode_string = self._generate_encoding_string(
                            language=language,
                            function_def=function_def,
                            function_name=function_name,
                            class_name=class_name,
                        )

                        # This is added to retry after we fall into a RateLimit exception
                        while True:
                            try:
                                embedding = self._get_embedding_from_input(
                                    input=encode_string
                                )

                                embedding = embedding.tolist()

                                print("embedding: ", len(embedding))

                                self._update_embedding_table(
                                    embedding=embedding,
                                    function_name=function_name,
                                    class_name=class_name,
                                    filepath=file_path[
                                        20 if is_public else 4 :
                                    ],  # This is to remove unnecessary path from the filepath
                                    line_number=line_number,
                                    table_name=table_name,
                                )

                            except Exception as e:
                                # Adding a sleep to avoid rate limit.
                                print("Error in FlaskAPIHandler.handle_encode: ", e)

                                if type(e) == openai.error.RateLimitError:
                                    time.sleep(60)
                                    continue
                                else:
                                    self.handle_delete(self, request)
                                    return

                            break

    # Return Flag:
    #   0: Already encoded and stored in DB
    #   1: Encoding completed
    #   2: Incorrect is_public flag value entered
    def handle_encode(self, request):
        url = request.form["url"]  # This can be public URL or local file path
        is_public = self._check_if_public(
            url=url
        )  # This can be "Yes" or "No" or "Error" depending on whether public URL or not

        if is_public == "Yes":
            if self._check_if_valid_url(url=url):
                flag, table_name = self._check_if_indexed(url)

                if flag:
                    # TODO: Return with some proper flag which says that already indexed project
                    print("Already encoded project")
                    return 0
                else:
                    clone_flag, project_path = self._clone_repo(url=url)

                    if not clone_flag:
                        return 3

                    table_name = self._update_mapping_table(url=url, is_public=True)
                    self._create_embedding_table(table_name=table_name)
                    self._encode_from_path(
                        project_path=project_path,
                        table_name=table_name,
                        is_public=True,
                        request=request,
                    )
                    self._generate_index(table_name=table_name)
                    shutil.rmtree(project_path)

        elif is_public == "No":
            flag, table_name = self._check_if_indexed(url=url)
            if flag:
                # TODO: Return with some proper flag which says that already indexed project
                return 0
            else:
                project_path = os.path.join("/mnt", url[1:])

                table_name = self._update_mapping_table(url=url, is_public=False)
                self._create_embedding_table(table_name=table_name)
                self._encode_from_path(
                    project_path=project_path,
                    table_name=table_name,
                    is_public=False,
                    request=request,
                )
                self._generate_index(table_name=table_name)

        else:
            print("""Error: Wrong value in FlaskAPIHandler.handle_encode""")
            return 2

        return 1

    def handle_search(self, request):
        url = request.form["url"]  # This can be public URL or local file path
        query = request.form["query"]
        print("query: ", query)
        self._create_faiss_index()

        is_public = self._check_if_public(
            url=url
        )  # This can be "Yes" or "No" or "Error" depending on whether public URL or not

        if is_public == "Error":
            return {}

        if is_public == "Yes":
            if not self._check_if_valid_url(url=url):
                return {}

        encode_flag = self.handle_encode(request=request)

        if encode_flag == 0 or encode_flag == 1:
            flag, table_name = self._check_if_indexed(url)
            self._generate_index(table_name=table_name)

            query_embedding = self._get_embedding_from_input(input=query)
            result_dict = self._get_nearest_neighbors(
                query_embedding=query_embedding, table_name=table_name
            )

            return result_dict

        else:
            return {}

    def handle_delete(self, request):
        url = request.form["url"]
        print("URL: ", url)

        is_public = self._check_if_public(
            url=url
        )  # This can be "Yes" or "No" or "Error" depending on whether public URL or not

        if is_public == "Error":
            return 1

        if is_public == "Yes":
            if not self._check_if_valid_url(url=url):
                return 1

        flag, table_name = self._check_if_indexed(url=url)

        if flag:
            cur = self.conn.cursor()

            try:
                cur.execute("""DELETE FROM mapping WHERE id='{}'""".format(table_name))
                cur.execute("""DROP TABLE embeddings_{}""".format(table_name))
                self.conn.commit()

                return 0
            except Exception as e:
                print("Error in FlaskAPIHandler.handle_delete: ", e)
                return 1

    def handle_root(self):
        try:
            cur = self.conn.cursor()
            cur.execute("""SELECT * FROM mapping""")
            project_list = cur.fetchall()
        except Exception as e:
            print("Error in FlaskAPIHandler.handle_root: ", e)
            return False, {}

        project_dict = {}

        for project in project_list:
            project_dict["{}".format(project[0])] = project[1]

        return True, project_dict


if __name__ == "__main__":
    flask_api_handler = FlaskAPIHandler()
