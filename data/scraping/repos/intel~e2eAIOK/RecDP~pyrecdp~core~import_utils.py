import os
from typing import Optional
import pip
import importlib
import pathlib
import pkg_resources

def list_requirements(requirements_path):
    with pathlib.Path(requirements_path).open() as requirements_txt:
        install_requires = [
            str(requirement)
            for requirement
            in pkg_resources.parse_requirements(requirements_txt)
        ]
    return install_requires

def fix_package_name(package):
    a = package.split('>')[0]
    a = a.split('<')[0]
    b = a.split('=')[0]
    
    package_name_map = {
        'scikit-learn' : 'sklearn',
        'pyyaml' : 'yaml'
    }
    
    if b in package_name_map:
        b = package_name_map[b]
    #print(b)
    return b
    
def import_with_auto_install(package):
    try:
        return importlib.import_module(fix_package_name(package))
    except ImportError:
        pip.main(['install', package])

def import_faiss(no_avx2: Optional[bool] = None, install_if_miss: bool = True):
    """
    Import faiss if available, otherwise raise error.
    If FAISS_NO_AVX2 environment variable is set, it will be considered
    to load FAISS with no AVX2 optimization.

    Args:
        no_avx2: Load FAISS strictly with no AVX2 optimization
            so that the vectorstore is portable and compatible with other devices.
    """
    if no_avx2 is None and "FAISS_NO_AVX2" in os.environ:
        no_avx2 = bool(os.getenv("FAISS_NO_AVX2"))

    try:
        if no_avx2:
            from faiss import swigfaiss as faiss
        else:
            import faiss
    except ImportError:
        if install_if_miss:
            os.system("pip install -q faiss-gpu")
            os.system("pip install -q faiss-cpu")
        else:
            raise ImportError(f"faiss package not found, "
                              f"please install it with 'pip install faiss-gpu && pip install faiss-cpu'")


def import_langchain(install_if_missing: bool = True):
    """
    Import langchain if available, otherwise raise error.
    """
    try:
        from langchain.document_loaders.base import BaseLoader
    except ImportError:
        if install_if_missing:
            os.system("pip install -q langchain")
        else:
            raise ImportError(f"langchain package not found, please install it with 'pip install langchain")


def import_sentence_transformers(install_if_missing: bool = True):
    """
    Import sentence_transformers if available, otherwise raise error.
    """
    try:
        import sentence_transformers

    except ImportError as exc:
        if install_if_missing:
            os.system("pip install -q sentence_transformers")
        else:
            raise ImportError(
                "Could not import sentence_transformers python package. "
                "Please install it with `pip install sentence-transformers`."
            ) from exc


def import_unstructured(install_if_missing: bool = True):
    try:
        import unstructured  # noqa:F401
        from unstructured.partition.auto import partition
    except ImportError:
        if install_if_missing:
            os.system("pip install -q  unstructured")
            os.system("pip install -q  unstructured[ppt, pptx, xlsx]")
        else:
            raise ValueError(
                "unstructured package not found, please install it with "
                "`pip install unstructured`"
            )


def import_openai(install_if_missing: bool = True):
    try:
        import openai
    except ImportError:
        if install_if_missing:
            os.system("pip install -q openai")
        else:
            raise ValueError(
                "openai package not found, please install it with "
                "`pip install openai`"
            )

def import_pysbd(install_if_missing: bool = True):
    try:
        import openai
    except ImportError:
        if install_if_missing:
            os.system("pip install -q pysbd")
        else:
            raise ValueError(
                "pysbd package not found, please install it with "
                "`pip install pysbd`"
            )

def import_markdownify():
    try:
        import markdownify
    except ImportError:
        pip.main(['install', 'markdownify'])

def import_beautiful_soup():
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        pip.main(['install', 'bs4'])
