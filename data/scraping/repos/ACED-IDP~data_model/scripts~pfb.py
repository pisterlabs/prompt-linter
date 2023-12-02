import json
import os
from pathlib import Path

import click
from fhirclient.models.bundle import Bundle


@click.command()
@click.option('--file_name_pattern',
              default='research_study*.json',
              show_default=True,
              help='File names to match.')
@click.option('--input_path',
              default='output/',
              show_default=True,
              help='Path to output data.')
def pfb(input_path, file_name_pattern):
    """Create PFB from coherent ResearchStudies."""

    # validate parameters
    assert os.path.isdir(input_path)
    input_path = Path(input_path)
    assert os.path.isdir(input_path)
    file_paths = list(input_path.glob(file_name_pattern))
    assert len(file_paths) >= 1, f"{str(input_path)}/{file_name_pattern} only returned {len(file_paths)} expected at least 1"

    # patients appear in multiple studies, don't duplicate
    already_added = []
    for file_path in file_paths:
        bundle = Bundle(json.load(open(file_path)))
        research_subjects = [bundle_entry.resource for bundle_entry in bundle.entry if bundle_entry.resource.resource_type == "ResearchSubject"]
        sources = sorted(research_subject.meta.source for research_subject in research_subjects if research_subject.meta.source not in already_added)
        sources = sources[:4]
        sources.append(file_path)  # add study bundle
        already_added.extend(sources)
        input_paths = [f"--input_path \"{source}\" \\\n" for source in sources]
        # input_paths.append([f"--input_path \"{source}\" \\\n" for source in sources][-1])
        pfb_file_name = f"{str(file_path).split('/')[-1].split('.')[0]}.pfb"
        cmd = f"pfb_fhir transform --simplify {' '.join(input_paths)} --pfb_path {pfb_file_name} "
        print(cmd)
        break


if __name__ == '__main__':
    pfb()

