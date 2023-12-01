"""lmss.enrich provides methods for enriching data in the knowledge graph, like:
* proposing pref labels or alt labels for entities
* proposing common translations for labels in different languages
* proposing definitions for entities
* proposing new relations between entities
* proposing new entities
"""

# imports
import argparse
import json
import os
import re
from pathlib import Path

# packages
import openai
from rdflib import URIRef, Literal
from rdflib.namespace import RDF, RDFS, OWL, SKOS

# project
import lmss.owl
from lmss.graph import LMSSGraph

try:
    openai.api_key = os.getenv("OPENAI_API_KEY", None)
    if openai.api_key is None:
        with open(Path.home() / ".openai" / "api_key") as api_key_file:
            openai.api_key = api_key_file.read().strip()
except Exception:
    openai.api_key = None


# regular expression to extract parenthetical text
PARENTHETICAL_REGEX = re.compile(r"\((.*?)\)", re.UNICODE)

SKIP_LABELS = {
    "Language",
    "Industry",
    "Service",
    "PACER NoS",
    "DEPRECATED",
}


def get_definition_prompt(
    lmss_graph: LMSSGraph,
    concept: dict,
) -> tuple[str, str]:
    """Get the system and user prompts for a concept definition.

    Args:
        lmss_graph (LMSSGraph): The LMSSGraph object.
        concept (dict): The concept dictionary.

    Returns:
        tuple[str, str]: The system and user prompts.
    """
    # get the labels for the parent and children
    parent_labels = [
        lmss_graph.concepts[parent]["label"]
        for parent in concept["parents"]
        if lmss_graph.concepts[parent]["label"] is not None
    ]
    child_labels = [
        lmss_graph.concepts[child]["label"]
        for child in concept["children"]
        if lmss_graph.concepts[child]["label"] is not None
    ]

    # get the prompt to send to the LLM
    system_prompt = """You are a legal knowledge management professional who works with ontologies.
    Please perform the following tasks:
    1. Review the top level OWL class in <TOP-LEVEL> provided by the user.
    2. Review the OWL Class information in <CONCEPT> provided by the user.
    3. Write a two-sentence definition in the style of the Black's Law Dictionary for CONCEPT in plain English.
    4. Only describe the CONCEPT specifically, not the top level class generally.
    5. Respond only with the definition.  Do not include any heading or other text.
    """

    top_concept_label = concept["top_concept"]
    try:
        top_concept_description = lmss_graph.concepts[
            lmss_graph.key_concepts[top_concept_label]
        ]["definitions"][0]
    except (KeyError, IndexError):
        top_concept_description = ""

    user_prompt = f"""<TOP-LEVEL>
Label: {top_concept_label}
Description: {top_concept_description}  
</TOP-LEVEL>

<CONCEPT>\n"""

    for key, value in concept.items():
        if key in ["iri", "parents", "children"]:
            continue
        if isinstance(value, list):
            if len(value) > 0:
                user_prompt += f"{key}: {', '.join(value)}\n"
        else:
            if value is not None and len(value.strip()) > 0:
                if key == "top_concept":
                    continue
                user_prompt += f"{key}: {value}\n"

    if len(parent_labels) > 0:
        user_prompt += f"Parents: {', '.join(parent_labels)}\n"
    if len(child_labels) > 0:
        user_prompt += f"Children: {', '.join(child_labels)}\n"

    user_prompt += "</CONCEPT>\n"

    return system_prompt, user_prompt


def get_translation_prompt(
        term: str,
        target_langs: list[str],
        source_lang: str = "en",
) -> tuple[str, str]:
    """Get the system and user prompts for a concept definition.

    Args:
        term (str): The term to translate.
        target_langs (list[str]): The target languages.
        source_lang (str): The source language.

    Returns:
        tuple[str, str]: The system and user prompts.
    """
    # get the prompt to send to the LLM
    system_prompt = """You are a legal knowledge management professional who works with ontologies.
    Please perform the following tasks:
    1. Translate the term <TERM> from <SOURCE_LANG> to each <TARGET_LANG> listed in ISO 639-1 format.
    2. Respond only with the list of translation in JSON.
    3. Do not include any heading or other text.
    """

    user_prompt = f"""<TERM>{term}</TERM>\n"""
    user_prompt += f"<SOURCE_LANG>{source_lang}</SOURCE_LANG>\n"
    user_prompt += "<TARGET_LANG>" + ", ".join(target_langs) + "</TARGET_LANG>\n"
    user_prompt += "JSON:"

    return system_prompt, user_prompt


def enrich_definitions(lmss_graph: LMSSGraph, progress: bool = True) -> LMSSGraph:
    """
    Enrich the definitions of concepts in the graph.

    Args:
        lmss_graph: The LMSSGraph object.
        progress: Whether to show progress.

    Returns:
        Graph: The enriched graph.
    """

    # get concepts and prog bar
    concepts = lmss_graph.concepts.items()
    if progress:
        try:
            import tqdm  # pylint: disable=C0415

            concepts = tqdm.tqdm(concepts, desc="Enriching definitions")
        except ImportError:
            progress = False

    # use rdflib subjects to get the list of classes
    for iri, concept in concepts:
        # update prog bar if progress set
        if progress:
            concepts.set_description(f"Enriching definitions: {concept['label']}")  # type: ignore

        # get the root concept beneath owl:Thing
        if len(concept["parents"]) == 0:
            continue

        if (
            "definitions" not in concept
            or concept["definitions"] is None
            or len(concept["definitions"]) == 0
        ):
            try:
                system_prompt, user_prompt = get_definition_prompt(g, concept)

                # get the definition with ChatCompletion API
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-0301",
                    temperature=0.0,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )

                # get the definition
                if len(response["choices"]) > 0:
                    definition = response["choices"][0]["message"]["content"].strip()
                    lmss_graph.add((URIRef(iri), SKOS.definition, Literal(definition)))
            except Exception as error:  # pylint: disable=W0718
                print(f"Unable to enrich definition for {concept['label']}: {error}")
                continue

    # return the graph
    return lmss_graph


# pylint: disable=R1702
def correct_labels(lmss_graph: LMSSGraph) -> LMSSGraph:
    """Correct rdfs:label and prefLabel values in the graph by
    removing any parenthetical text and moving it to altLabel.

    Args:
        lmss_graph: The LMSSGraph object.

    Returns:
        Graph: The corrected graph.
    """

    # get concepts and prog bar
    concepts = lmss_graph.concepts.items()
    try:
        import tqdm  # pylint: disable=C0415

        concepts = tqdm.tqdm(concepts, desc="Correcting labels")
    except ImportError:
        pass

    # use rdflib subjects to get the list of classes
    for iri, concept in concepts:
        # update prog bar if progress set
        concepts.set_description(f"Correcting labels: {concept['label']}")  # type: ignore

        # get the label
        label = concept["label"]
        if label is None:
            continue

        # check the rdfs:label
        if "(" in label:
            # get all the alt labels
            alt_labels = PARENTHETICAL_REGEX.findall(label)
            if len(alt_labels) > 0:
                # remove parens and drop double spaces
                label = PARENTHETICAL_REGEX.sub("", label).strip()
                label = " ".join(label.split())

                # add all the alt labels
                for alt_label in alt_labels:
                    if (
                        alt_label not in concept["alt_labels"]
                        and alt_label not in SKIP_LABELS
                    ):
                        lmss_graph.add((URIRef(iri), SKOS.altLabel, Literal(alt_label)))
                        concept["alt_labels"].append(alt_label)

                # remove and update the rdfs:label
                lmss_graph.remove((URIRef(iri), RDFS.label, Literal(concept["label"])))
                lmss_graph.add((URIRef(iri), RDFS.label, Literal(label)))

        # do the same thing with the pref labels
        for pref_label in concept["pref_labels"]:
            if "(" in pref_label:
                # get all the alt labels
                alt_labels = PARENTHETICAL_REGEX.findall(pref_label)
                if len(alt_labels) > 0:
                    # remove parens and drop double spaces
                    pref_label = PARENTHETICAL_REGEX.sub("", pref_label).strip()
                    pref_label = " ".join(pref_label.split())

                    # add all the alt labels
                    for alt_label in alt_labels:
                        if (
                            alt_label not in concept["alt_labels"]
                            and alt_label not in SKIP_LABELS
                        ):
                            lmss_graph.add(
                                (URIRef(iri), SKOS.altLabel, Literal(alt_label))
                            )
                            concept["alt_labels"].append(alt_label)

                    # remove and update the pref label
                    lmss_graph.remove(
                        (URIRef(iri), SKOS.prefLabel, Literal(concept["label"]))
                    )
                    lmss_graph.add((URIRef(iri), SKOS.prefLabel, Literal(pref_label)))

    # return the graph
    return lmss_graph


def translate_concepts(graph: LMSSGraph, concept_set: set[str], target_langs: list[str], progress: bool = True) -> list[dict]:
    """Translate the labels and definitions from en(-US) to one or more
     ISO 639-1 language codes, such as es-ES, de-DE, or en-UK.

    Args:
        graph: The LMSSGraph object.
        concept_set: The set of concepts to translate.
        target_langs: The target language(s) to translate to.
        progress: Whether to show a progress bar.

    Returns:
        Graph: The graph enriched with translations for labels and definitions.
    """
    result_data: list[dict] = []

    # iterate through concepts
    subjects = [s for s in
                g.subjects(RDF.type, OWL.Class)
                if str(s) in concept_set]

    if progress:
        try:
            import tqdm  # pylint: disable=C0415

            subjects = tqdm.tqdm(subjects, desc="Translating concepts")
        except ImportError:
            pass

    for concept in subjects:
        # check if the IRI is in the concept set
        try:
            iri = str(concept)
            if iri not in concept_set:
                continue

            # get rdfs:label and prefLabel
            label = g.value(concept, RDFS.label)
            record = {
                "iri": iri,
                "rdfs:label": str(label)
            }

            if label is not None:
                system_prompt, user_prompt = get_translation_prompt(label, target_langs)

                # get the definition with ChatCompletion API
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-0301",
                    temperature=0.0,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )

                # get the definition
                if len(response["choices"]) > 0:
                    try:
                        raw_response = response["choices"][0]["message"]["content"].strip()
                        json_response = json.loads(raw_response)
                        for lang_code in json_response:
                            if lang_code in target_langs:
                                record[lang_code] = json_response[lang_code]
                    except json.decoder.JSONDecodeError:
                        pass

                if len(record) > 2:
                    result_data.append(record)
        except Exception as e:
            print(iri, e)
            continue

    return result_data


if __name__ == "__main__":
    # setup argparser
    parser = argparse.ArgumentParser(description="Enrich LMSS OWL file.")

    # branch for graph
    parser.add_argument(
        "--branch",
        type=str,
        default=lmss.owl.DEFAULT_REPO_BRANCH,
        help="Branch to use for the LMSSGraph.",
    )

    # git repo for graph
    parser.add_argument(
        "--repo",
        type=str,
        default=lmss.owl.DEFAULT_REPO_ARTIFACT_URL,
        help="Git repo to use for the LMSSGraph.",
    )

    # local file for owl
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Local file to use for the LMSSGraph.",
    )

    # set options for labels and definitions
    parser.add_argument(
        "--labels",
        action="store_true",
        help="Correct labels in the OWL file.",
    )
    parser.add_argument(
        "--definitions",
        action="store_true",
        help="Enrich definitions in the OWL file.",
    )

    # set option for OpenAI key file
    parser.add_argument(
        "--openai-key-path",
        type=Path,
        default=".openai_key",
        help="Path to the OpenAI key file.",
    )

    # add output file
    parser.add_argument(
        "--output",
        type=Path,
        default="lmss.owl",
        help="Path to the output file.",
    )

    # parse args
    args = parser.parse_args()

    # get the graph based on file vs repo url/branch
    if args.file is not None:
        g = LMSSGraph(owl_path=args.file)
    else:
        g = LMSSGraph(owl_branch=args.branch, owl_repo_url=args.repo)

    # get the openai key
    if openai.api_key is None:
        if not args.openai_key_path.exists():
            raise RuntimeError("Unable to set OpenAI key from $OPENAI_API_KEY orfile.")

        with open(args.openai_key_path, "rt", encoding="utf-8") as f:
            openai.api_key = f.read().strip()

    # correct labels
    if args.labels:
        g = correct_labels(g)

    # enrich definitions
    if args.definitions:
        g = enrich_definitions(g)

    # serialize the graph
    g.serialize(args.output, format="xml")