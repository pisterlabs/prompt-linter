import pytest
import os
import tempfile
import sys
from anthropicautodocstrings.main import (
    update_docstrings_in_directory,
    update_docstrings,
    _extract_exclude_list,
)
from unittest.mock import patch
with patch("anthropicautodocstrings.main.sys.exit"):
    import anthropicautodocstrings.main


@pytest.fixture(scope="function")
def setup_api_key():
    """
    setup_api_key()

        Provides a test API key as a fixture for tests.

        This fixture sets the ANTHROPIC_API_KEY environment variable to a fake key
        before each test runs. After the test completes it deletes the variable if
        it was the fake key. This ensures each test has a valid API key while keeping
        tests isolated from real keys.

        Raises:
            KeyError: If the real API key is not already set in the environment.

        Returns:
            str: The API key usable within the test.
    """
    try:
        os.environ["ANTHROPIC_API_KEY"]
    except KeyError:
        os.environ["ANTHROPIC_API_KEY"] = "FAKE_API_KEY"
    yield os.environ["ANTHROPIC_API_KEY"]
    if os.environ["ANTHROPIC_API_KEY"] == "FAKE_API_KEY":
        del os.environ["ANTHROPIC_API_KEY"]


def create_test_file_with_docstring(docstring: str):
    """
    create_test_file_with_docstring(docstring: str) -> tempfile.NamedTemporaryFile
        Creates a temporary test file with the given docstring.

        Parameters:
            docstring (str): The docstring to include in the test file

        Returns:
            tempfile.NamedTemporaryFile: A named temporary file containing a function with the provided docstring
    """
    file_contents = f"\ndef foo():\n{docstring}\npass\n"
    test_file = tempfile.NamedTemporaryFile(mode="w", delete=False)
    test_file.write(file_contents)
    test_file.close()
    return test_file


def create_test_file_with_constructor():
    """
    create_test_file_with_constructor()

        Creates a temporary test file with a simple __init__ method defined.

        Parameters:
            None

        Returns:
            tempfile.NamedTemporaryFile: A NamedTemporaryFile object representing the test file created.

        Raises:
            None

        This function creates a temporary test file on disk with a basic empty __init__ method defined. It returns the NamedTemporaryFile object representing the file, without deleting it immediately. This allows the created test file to be used for testing purposes without needing to manually write and clean up the file.
    """
    file_contents = """
    def __init__():
        pass
    """
    test_file = tempfile.NamedTemporaryFile(mode="w", delete=False)
    test_file.write(file_contents)
    test_file.close()
    return test_file


@pytest.mark.asyncio
async def test_update_docstrings_in_directory(mocker, setup_api_key: str) -> None:
    test_dir = tempfile.TemporaryDirectory()
    subdir_1 = os.path.join(test_dir.name, "subdir_1")
    os.makedirs(subdir_1)
    file_1 = os.path.join(test_dir.name, "file_1.py")
    file_2 = os.path.join(subdir_1, "file_2.py")
    open(file_1, "w").close()
    open(file_2, "w").close()
    mocker.patch.object(
        anthropicautodocstrings.main, "update_docstrings_in_file", return_value=None
    )
    await update_docstrings_in_directory(test_dir.name, True, False, [], [])
    anthropicautodocstrings.main.update_docstrings_in_file.assert_any_call(
        file_1, True, False
    )
    anthropicautodocstrings.main.update_docstrings_in_file.assert_any_call(
        file_2, True, False
    )
    test_dir.cleanup()


@pytest.mark.asyncio
async def test_update_docstrings_in_directory_with_exclude_files(mocker) -> None:
    test_dir = tempfile.TemporaryDirectory()
    file_1 = os.path.join(test_dir.name, "file_1.py")
    open(file_1, "w").close()
    mocker.patch.object(
        anthropicautodocstrings.main, "update_docstrings_in_file", return_value=None
    )
    await update_docstrings_in_directory(test_dir.name, True, False, [], ["file_1.py"])
    anthropicautodocstrings.main.update_docstrings_in_file.assert_not_called()
    test_dir.cleanup()


@pytest.mark.asyncio
async def test_update_docstrings_in_directory_with_exclude_dirs(mocker) -> None:
    test_dir = tempfile.TemporaryDirectory()
    subdir_1 = os.path.join(test_dir.name, "subdir_1")
    os.makedirs(subdir_1)
    file_2 = os.path.join(subdir_1, "file_2.py")
    open(file_2, "w").close()
    mocker.patch.object(
        anthropicautodocstrings.main, "update_docstrings_in_file", return_value=None
    )
    await update_docstrings_in_directory(test_dir.name, True, False, ["subdir_1"], [])
    anthropicautodocstrings.main.update_docstrings_in_file.assert_not_called()
    test_dir.cleanup()


@pytest.mark.asyncio
async def test_update_docstrings_input_is_valid_file(mocker) -> None:
    os.environ["ANTHROPIC_API_KEY"] = "test_key"
    open("test_file.py", "w").close()
    mocker.patch.object(
        anthropicautodocstrings.main, "update_docstrings_in_file", return_value=None
    )
    await update_docstrings(
        "test_file.py",
        replace_existing_docstrings=True,
        skip_constructor_docstrings=False,
        exclude_directories=[],
        exclude_files=["test_file.py"],
    )
    anthropicautodocstrings.main.update_docstrings_in_file.assert_not_called()
    await update_docstrings(
        "test_file.py",
        replace_existing_docstrings=True,
        skip_constructor_docstrings=False,
    )
    anthropicautodocstrings.main.update_docstrings_in_file.assert_called_once_with(
        "test_file.py", True, False
    )
    os.unlink("test_file.py")


@pytest.mark.asyncio
async def test_update_docstrings_input_is_valid_directory(mocker) -> None:
    os.environ["ANTHROPIC_API_KEY"] = "test_key"
    test_dir = tempfile.TemporaryDirectory()
    mocker.patch.object(
        anthropicautodocstrings.main,
        "update_docstrings_in_directory",
        return_value=None,
    )
    await update_docstrings(
        test_dir.name,
        replace_existing_docstrings=True,
        skip_constructor_docstrings=False,
        exclude_directories=[os.path.basename(test_dir.name)],
        exclude_files=[],
    )
    anthropicautodocstrings.main.update_docstrings_in_directory.assert_not_called()
    await update_docstrings(
        test_dir.name,
        replace_existing_docstrings=True,
        skip_constructor_docstrings=False,
    )
    anthropicautodocstrings.main.update_docstrings_in_directory.assert_called_once_with(
        test_dir.name, True, False, [], []
    )
    test_dir.cleanup()


def test_extract_exclude_list(mocker) -> None:
    """
    test_extract_exclude_list(mocker: Mocker) -> None:
       Extract a list of excluded file names from a comma separated string.

       Parameters:
          mocker: Mock object using for testing

       Returns:
          None

       Examples:
          assert _extract_exclude_list('') == []
          assert _extract_exclude_list('test.py') == ['test.py']
          assert _extract_extract_exclude_list('test1.py,test2.py,test3.py') == ['test1.py', 'test2.py', 'test3.py']
          assert _extract_exclude_list(' test1.py , test2.py , test3.py ') == ['test1.py', 'test2.py', 'test3.py']
    """
    assert _extract_exclude_list("") == []
    assert _extract_exclude_list("test.py") == ["test.py"]
    assert _extract_exclude_list("test1.py,test2.py,test3.py") == [
        "test1.py",
        "test2.py",
        "test3.py",
    ]
    assert _extract_exclude_list(" test1.py , test2.py , test3.py ") == [
        "test1.py",
        "test2.py",
        "test3.py",
    ]


@pytest.mark.asyncio
async def test_main(mocker):
    mocker.patch.object(
        anthropicautodocstrings.main, "update_docstrings", return_value=None
    )
    sys.argv = [
        "autodocstrings",
        "input_path",
        "--replace-existing-docstrings",
        "--skip-constructor-docstrings",
        "--exclude-directories",
        "dir1,dir2",
        "--exclude-files",
        "file1,file2",
    ]
    await anthropicautodocstrings.main.main()
    anthropicautodocstrings.main.update_docstrings.assert_called_once_with(
        "input_path", True, True, ["dir1", "dir2"], ["file1", "file2"]
    )
