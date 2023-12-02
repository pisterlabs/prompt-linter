import os
import time
import IPython

from . import firebase
from . import installer
from . import tools
from . import colors
from . import contexts


def init_config(config_id, collection, config):
    if config_id not in config:
        config[config_id] = {}

    if not collection.document(config_id).get().to_dict():
        firebase.save_config(config_id, config)
    config[config_id].update(collection.document(config_id).get().to_dict())

    return config


def init_prems(config):
    from collections import OrderedDict

    path = OrderedDict()

    def get_path(join=", "):
        info = "%s" % str(join.join(path.keys()))
        vals = [v if v != "" else "''" for v in path.values()]
        info += " = %s" % str(join.join(vals))
        return info

    path["db"] = (
        config["database"].split("@")[0] + "@" if "@" in config["database"] else config["database"].split("/")[-1]
    )
    path["subject"] = config["subject"]
    path["virtual_room"] = config["virtual_room"]
    path["nb_id"] = config["notebook_id"]
    path["user"] = (email := config["email"])

    if email is None:
        path["user"] = f"None ❌\x1b[41m\x1b[37m, email not configured\x1b[0m"
        return get_path()

    is_known_student = (
        ("virtual_room" in config and email in config["global"][config["virtual_room"]])
        or email in config["global"]["admins"]
        or email == "solution"
    )
    language = config["global"].get("language")
    if config["global"]["admins"] == "":
        is_known_student = True

    config["eparams"] = False

    if not is_known_student:
        if 0 and config["global"]["restricted"]:
            raise Exception.IndexError(
                f"❌\x1b[41m\x1b[37mL'email '{email}' n'est pas configuré dans la base de données. Contacter le professeur svp\x1b[0m"
                if language == "fr"
                else f"❌\x1b[41m\x1b[37mEmail '{email}' is not configured in the database. Please contact the teacher\x1b[0m"
            )
        path["user"] += (
            "❌ (\x1b[41m\x1b[37memail inconnu: contacter le professeur svp\x1b[0m), "
            if language == "fr"
            else "❌ (\x1b[41m\x1b[37munknown email: please contact the teacher\x1b[0m), "
        )

    else:
        path["user"] += "🎓" if email in config["global"]["admins"] or tools.is_admin(config) else "✅"

    return get_path(join="/")


def commercial(language="en", **kwargs):
    if language == "fr":
        text1 = 'Cette page utilise (<i>Package de support de cours interactif</i>). Si vous aimez cette approche🚀🏆🎯, <br/>vous pouvez supporter le projet sur <a href="https://github.com/gtherin/bulkhours">github</a> en donnant quelques etoiles.'
    else:
        text1 = 'This page is using (<i>A Support course package</i>). If you like this approach🚀🏆🎯, <br/>please support the project on <a href="https://github.com/gtherin/bulkhours">github</a> with some stars.'
    IPython.display.display(
        IPython.display.HTML(
            f"""
<table style="opacity:0.8;">
    <tr>
      <td style="background-color: white;"><a href="https://github.com/gtherin/bulkhours"><img style="background-color: white;" src="https://github.com/gtherin/bulkhours/blob/ce2ce67a250b396b7341becf7deb09da961f2698/data/BulkHours.png?raw=true" width="100"></a></td>
      <td style="background-color: white; text-align:left; color:black">{text1}</td>
      <td style="background-color: white;"><a href="https://github.com/gtherin/bulkhours">
      <img style="background-color: white;" src="https://github.com/gtherin/bulkhours/blob/main/data/github.png?raw=true" width="30"></a></td>
      <td style="background-color: white;"><a href="https://github.com/gtherin/bulkhours"><img style="background-color: white;margin-top: 0.5em;" src="https://github.com/gtherin/bulkhours/blob/main/data/like.gif?raw=true" width="150"></a></td>

</tr></table>
"""
        )
    )


def init_env(packages=None, plt_style="default", **kwargs):
    """Initialize the environment of the user

        Parameters:
        :param email: email of the student/teacher
        :param subject: subject of the course. Recommended to be unique for one entire subject
        :param notebook_id: id of the current notenook. Recommended to be unique per notebook
        :param database: database config can be:
     - A simple string pointing to a file. With this solution, users should deal with separating informations between students and teachers
    Examples:
    database ="data/cache/course2.json"
    database ="/content/mydatabase.json"
     - A string with an identifier token. This token gives you access to: a real-time database (with id check), api to openai, huggingface, etc.
     As this solution might generate reasonable costs, you should contact contact@bulkhours.eu to get a token.
    database = "bkloud@SUBJECT/teacher::eNq-XXXXXXXXXXXHXXXXX-L29tB"
    database = "bkloud@SUBJECT/CLASSROOM_student::eNq-XXXXXXXXXXXHXXXXX-9tB"
     - A dict of config information. Only firestore config are functional for the moment. Examples:
    database = {"type": "service_account", ..., "universe_domain": "googleapis.com"}

        :param openai_token: to activate openai functionalities [chatgpt, dall-e]
        :param huggingface_token: to activate huggingface functionalities [chatgpt, dall-e]
        :param packages: packages to be installed from pip or apt-get


    Examples:
        bulkhours.init_env(
            email="the.lordoflight@got.grrm",  # Email of the teacher
            subject="bulkhours_sessions",      # Recommended to be unique for one entire subject
            notebook_id="course_edition_nb",   # Recommended to be unique per notebook (id of the current notenook)
            database="data/cache/course2.json" # Database file
                          )
    """

    mode = "production" if "mode" not in kwargs else kwargs["mode"]

    quiet_mode = mode == "passive"

    cfg = firebase.init_database(kwargs)
    
    info = init_prems(cfg)
    start_time = time.time()

    if "BLK_PACKAGES_STATUS" not in os.environ:
        installer.install_dependencies(packages, start_time, tools.is_admin(cfg=cfg))
        os.environ["BLK_PACKAGES_STATUS"] = "INITIALIZED"
        os.environ["BLK_ROOT_DIR"] = os.environ["PWD"]

    colors.set_plt_style(plt_style)
    version = open(tools.abspath("bulkhours/__version__.py")).readlines()[0].split('"')[1]

    einfo = f", ⚠️\x1b[31m\x1b[41m\x1b[37m in admin/teacher🎓 mode\x1b[0m⚠️" if tools.is_admin(cfg=cfg) else ""
    if not quiet_mode:
        print(f"Import BULK Helper cOURSe (\x1b[0m\x1b[36mversion='{version}'\x1b[0m🚀{einfo}):", end="")
    if "bkloud" not in cfg["database"]:
        if not quiet_mode:
            print(
                f"⚠️\x1b[31mDatabase is local (security_level={cfg['security_level']}). Export your config file if you need persistency.\x1b[0m⚠️",
                end="",
            )
    if ipp := IPython.get_ipython():
        ipp.run_cell(
            """import IPython
import ipywidgets
        """
        )

        if tools.get_platform() == "colab":
            ipp.run_cell("%cd /content")

    if not quiet_mode:
        print("\n- session-info: " + info)

    if tools.get_value("openai_token") is not None or tools.get_value("huggingface_token") is not None:
        external_services = "- extra-services:\033[92m"
    if tools.get_value("openai_token") is not None:
        external_services += " openai 🤖, "
        if ipp := IPython.get_ipython() and False:  # Deactivate warning for the moment
            ipp.run_cell(
                """ try:
    import openai

    openai.api_key = "%s"
except ModuleNotFoundError:
    print("LOG import of openai failed 💥")
"""
                % tools.get_value("openai_token")
            )

    if tools.get_value("huggingface_token") is not None:
        import ipywidgets
        from .. import ml

        os.environ["BLK_HUGGINGFACE_TOKEN"] = tools.get_value("huggingface_token")
        external_services += "huggingface 🤗"

        with ipywidgets.Output():
            ml.PPOHugs()

    if not quiet_mode:
        if tools.get_value("openai_token") is not None or tools.get_value("huggingface_token") is not None:
            print(external_services + "\033[00m")
    contexts.generate_empty_context("student")
    contexts.generate_empty_context("teacher")
    os.environ["BLK_GLOBAL_STATUS"] = f"INITIALIZED"
