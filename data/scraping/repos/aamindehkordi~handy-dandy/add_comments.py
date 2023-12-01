import os
import sys
import re
import openai

# Get the directory path from command line argument
dir_path = sys.argv[1]


# Define a function to add comments to a given code
def create_comments(code):
    openai.api_key = "YOUR_API_KEY"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[
            {
                "role": "system",
                "content": "You are a code commenter. Your task is to generate informative comments for the given Rust code. Create header comments for each function much like a javadoc. If the function does not appear complete, create a possible plan on how to implement with a todo comment.",
            },
            {
                "role": "user",
                "content": "Code: \n```rs\npub fn new(position: Position, color: Color, type_: PieceKind) -> Square {\n        Square {\n            position,\n            color,\n            type_,\n            piece: color as PieceAsByte + type_ as PieceAsByte,\n        }\n    }\n```",
            },
            {
                "role": "assistant",
                "content": "/// Creates a new square.\n///\n/// # Arguments\n/// * `position` - The position of the square.\n/// * `color` - The color of the piece.\n///\n/// # Returns\n/// A new square.\n///\n/// # Example\n/// ```rs\n///     let square = Square::new(4, Color::Black, PieceKind::King);\n/// ```",
            },
            {"role": "user", "content": f"Code: \n```rs\n{code}\n```"},
        ],
    )

    return response.choices[0].message["content"]


# Define regex pattern for Rust function
rust_function_pattern = re.compile(
    r"((?:pub\s)?fn\s[^\{]+\{(?:[^{}]*\{[^{}]*\}[^{}]*)*[^{}]*\})"
)

# Walk through the directory
for root, dirs, files in os.walk(dir_path):
    for file in files:
        if file.endswith(".rs"):
            file_path = os.path.join(root, file)
            with open(file_path, "r") as f:
                content = f.read()

                # Find all functions in the Rust code
                functions = rust_function_pattern.findall(content)

                for function in functions:
                    comments_for_function = create_comments(function)

                    # Add the comments to the function
                    commented_function = comments_for_function + "\n" + function + "\n"

                    # Replace the function with the commented function
                    content = content.replace(function, commented_function)

                    print(commented_function)

            # Write the updated content back to the file
            with open(file_path, "w") as f:
                f.write(content)
