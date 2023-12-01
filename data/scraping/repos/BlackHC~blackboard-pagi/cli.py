"""Console script for blackboard_pagi."""

#  Blackboard-PAGI - LLM Proto-AGI using the Blackboard Pattern
#  Copyright (c) 2023. Andreas Kirsch
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published
#  by the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import click
import langchain
from langchain.cache import SQLiteCache
from langchain.llms.openai import OpenAI

import blackboard_pagi.controller

langchain.llm_cache = SQLiteCache()


@click.command()
def main():
    """Main entrypoint."""
    click.echo("blackboard-pagi")
    click.echo("=" * len("blackboard-pagi"))
    click.echo("Proto-AGI using a Blackboard System (for the LLM Hackathon by Ben's Bites)")

    click.echo("What is your prompt?")
    prompt = click.prompt("Prompt", default="How many colleges are there in Oxford and Cambridge?")
    # default="What is the meaning of life?")

    kernel = blackboard_pagi.controller.Kernel(OpenAI())
    note = kernel(prompt)

    click.echo("Here is your note:")
    click.echo(note)


if __name__ == "__main__":
    main()  # pragma: no cover
