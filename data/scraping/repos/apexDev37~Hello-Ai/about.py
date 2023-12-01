"""
About script for Langchain demos.
"""

import langchain


def main() -> None:
  """ Script entry-point func."""
  package = langchain.__name__
  version = langchain.__version__
  
  print(f'Demo for {package}🦜 v{version}')


if __name__ == '__main__':
  main()
