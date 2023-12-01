#    DAIA -  Digital Artificial Inteligence Agent
#    Copyright (C) 2023  Envedity
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.

from DAIA_GPT4V.run import run
from config import openai_api_key


def main(key: str = None) -> None:
    # Checking API Key
    if not key or len(key) <= 0 or key == "":
        input_api_key = ""

        while input_api_key == "":
            input_api_key = input("Enter Your API Key: ")

        # Saving API Key
        with open(".env", "a") as env_file:
            env_file.write(f"\nOPENAI_KEY={input_api_key}")
            env_file.close()

        key = input_api_key

    # User Interaction
    try:
        option = int(
            input(
                "\nOptions\n[1] DAIA_GPT-4-with-Vision\n[2] DAIA_Continues\n\nSelect Option: "
            )
        )

    except ValueError:
        print("Please enter a number.")
        exit()

    match option:
        case 1:
            return run(api_key=key)

        case 2:
            return print("Currently Unavaiable.")

        case _:
            return print("Invalid Option.")


if __name__ == "__main__":
    main(openai_api_key)
