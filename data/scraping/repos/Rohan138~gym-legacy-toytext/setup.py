from setuptools import setup

VERSION = "0.0.5"

long_description = ""
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="gym-legacy-toytext",
    version=VERSION,
    description="Legacy Toytext Environments from OpenAI Gym",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Rohan138/gym-legacy-toytext",
    author="Rohan Potdar",
    author_email="rohanpotdar138@gmail.com",
    license="",
    packages=["gym_toytext"],
    install_requires=[
        "gym>=0.19.0, <0.26.0",
    ],
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
