import ast
from _ast import FunctionDef, ClassDef, AST
import logging
import random
import time
import astor

from openai.error import RateLimitError, APIConnectionError
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import DocGenDef
    from . import ASTAnalyzer


class MethodVisitor(ast.NodeTransformer):
    """
    A class that visits and transforms the AST nodes of a Python module.

    Attributes:
    -----------
    file_visitor : 'FileVisitor'
        An instance of the FileVisitor class.

    Methods:
    --------
    visit_FunctionDef(node: FunctionDef) -> FunctionDef:
        Visits and transforms the FunctionDef node of the AST.

    """

    def __init__(self, file_visitor: "FileVisitor"):
        """
        __init__(self, file_visitor: 'FileVisitor')

            Initializes an instance of the class with a given file visitor object.

            Parameters:
            -----------
            file_visitor: 'FileVisitor'
                An object of the FileVisitor class."""
        self.file_visitor = file_visitor

    def visit_FunctionDef(self, node: FunctionDef) -> FunctionDef:  # pylint: disable=invalid-name
        """
        visit_FunctionDef(self, node: FunctionDef) -> FunctionDef

            Visits a FunctionDef node and
             returns the result of the visit_def method of the file visitor object.

            Parameters:
            -----------
            node: FunctionDef
                A FunctionDef node to be visited.

            Returns:
            --------
            FunctionDef
                The result of the visit_def method of the file visitor object."""
        return self.file_visitor.visit_def(node, "Function")


class ClassVisitor(ast.NodeTransformer):
    """
    A class that visits and transforms the AST nodes of a Python module.

    Attributes:
    -----------
    file_visitor : 'FileVisitor'
        An instance of the FileVisitor class.

    Methods:
    --------
    visit_ClassDef(node: ClassDef) -> ClassDef:
        Visits and transforms the ClassDef node of the AST.

    """

    def __init__(self, file_visitor: "FileVisitor"):
        """
        __init__(self, file_visitor: 'FileVisitor')

            Initializes an instance of the class with a given file visitor object.

            Parameters:
            -----------
            file_visitor: 'FileVisitor'
                An object of the FileVisitor class."""
        self.file_visitor = file_visitor

    def visit_ClassDef(self, node: ClassDef) -> ClassDef:  # pylint: disable=invalid-name
        """
        visit_ClassDef(self, node: ClassDef) -> ClassDef

            Visits a ClassDef node
             and returns the result of the visit_def method of the file visitor object.

            Parameters:
            -----------
            node: ClassDef
                A ClassDef node to be visited.

            Returns:
            --------
            ClassDef
                The result of the visit_def method of the file visitor object."""
        return self.file_visitor.visit_def(node, "Class")


class FileVisitor:
    """
    A class that visits and transforms the AST nodes of a Python module.

    Attributes:
    -----------
    ast_analyzer : ASTAnalyzer
        An instance of the ASTAnalyzer class.
    class_visitor : ClassVisitor
        An instance of the ClassVisitor class.
    method_visitor : MethodVisitor
        An instance of the MethodVisitor class.

    Methods: -------- obtain_pydoc_wrapper(node: DocGenDef, source_code: str) -> str: Wraps the
    obtain_pydoc method of the ASTAnalyzer class and handles RateLimitError exceptions.

    visit_def(node: DocGenDef, str_type: str) -> DocGenDef:
        Visits and transforms the DocGenDef node of the AST.

    visit(tree: AST):
        Visits and transforms the AST nodes of a Python module.

    """

    def __init__(self, ast_analyzer: "ASTAnalyzer"):
        """
        __init__(self, ast_analyzer: ASTAnalyzer)

            Initializes an instance of the class with a given ASTAnalyzer object and creates
            instances of ClassVisitor and MethodVisitor classes.

            Parameters:
            -----------
            ast_analyzer: ASTAnalyzer
                An object of the ASTAnalyzer class."""
        self.ast_analyzer = ast_analyzer
        self.class_visitor = ClassVisitor(self)
        self.method_visitor = MethodVisitor(self)

    def obtain_pydoc_wrapper(self, node: "DocGenDef", source_code: str) -> str:
        """
        obtain_pydoc_wrapper(self, node: DocGenDef, source_code: str) -> str

            Wraps the obtain_pydoc method of the ASTAnalyzer class and handles RateLimitError
            exceptions by retrying the method after a random delay.

            Parameters:
            -----------
            node: DocGenDef
                A DocGenDef node.
            source_code: str
                A string containing the source code.

            Returns:
            --------
            str
                The PyDoc string obtained from the source code."""
        try:
            return self.ast_analyzer.obtain_pydoc(source_code)
        except RateLimitError:
            time.sleep(random.randint(5, 10))
            return self.obtain_pydoc_wrapper(node, source_code)
        except APIConnectionError as api_conn_error:
            logging.warning(
                "An api connection error has occurred: " + api_conn_error.error + "\nWaiting 30s to retry\n")
            time.sleep(30)
            return self.obtain_pydoc_wrapper(node, source_code)

    def visit_def(self, node: "DocGenDef", str_type: str) -> "DocGenDef":
        """
        visit_def(self, node: DocGenDef, str_type: str) -> DocGenDef

            Visits a DocGenDef node and adds a new docstring obtained from the source code to the
            node.

            Parameters:
            -----------
            node: DocGenDef
                A DocGenDef node to be visited.
            str_type: str
                A string representing the type of the node.

            Returns:
            --------
            DocGenDef
                The visited DocGenDef node with the new docstring added."""
        time.sleep(random.randint(2, 5))
        logging.info("%s name: %s", str_type, node.name)
        source_code = astor.to_source(node)
        response = self.obtain_pydoc_wrapper(node, source_code)
        logging.info(response)
        self.ast_analyzer.add_docstring_to_ast(node, new_docstring=response)
        return node

    def visit(self, tree: AST):
        """
        visit(self, tree: AST)

            Visits an AST object and visits its ClassDef and FunctionDef nodes using the
            ClassVisitor and MethodVisitor objects respectively. Removes any messages from the
            ASTAnalyzer object after each visit.

            Parameters:
            -----------
            tree: AST
                An AST object to be visited."""
        self.class_visitor.visit(tree)
        self.ast_analyzer.remove_messages()
        self.method_visitor.visit(tree)
        self.ast_analyzer.remove_messages()
