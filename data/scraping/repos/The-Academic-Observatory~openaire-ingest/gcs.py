# Copyright 2023 Curtin University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Author: James Diprose, Aniek Roelofs, Alex Massen-Hane

import os
import logging
import pathlib
import multiprocessing
from typing import List, Tuple
from google.cloud import storage
from requests.exceptions import ChunkedEncodingError
from multiprocessing import BoundedSemaphore, cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed

from openaire.files import crc32c_base64_hash

# The chunk size to use when uploading / downloading a blob in multiple parts, must be a multiple of 256 KB.
DEFAULT_CHUNK_SIZE = 256 * 1024 * 4


def gcs_blob_name_from_path(relative_local_filepath: str) -> str:
    """Creates a blob name from a local file path.

    :param relative_local_filepath: The local filepath
    :return: The name of the blob on cloud storage
    """
    # Make sure that path is using forward slashes for Google Cloud Storage
    return pathlib.Path(relative_local_filepath).as_posix().strip("/")


def gcs_upload_file(
    *,
    bucket_name: str,
    blob_name: str,
    file_path: str,
    retries: int = 3,
    connection_sem: BoundedSemaphore = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    project_id: str = None,
    check_blob_hash: bool = True,
) -> Tuple[bool, bool]:
    """Upload a file to Google Cloud Storage.

    :param bucket_name: the name of the Google Cloud Storage bucket.
    :param blob_name: the name of the blob to save.
    :param file_path: the path of the file to upload.
    :param retries: the number of times to retry uploading a file if an error occurs.
    :param connection_sem: a BoundedSemaphore to limit the number of upload connections that can run at once.
    :param chunk_size: the chunk size to use when uploading a blob in multiple parts, must be a multiple of 256 KB.
    :param project_id: the project in which the bucket is located, defaults to inferred from the environment.
    :param check_blob_hash: check whether the blob exists and if the crc32c hashes match, in which case skip uploading.
    :return: whether the task was successful or not and whether the file was uploaded.
    """
    func_name = gcs_upload_file.__name__
    logging.info(f"{func_name}: bucket_name={bucket_name}, blob_name={blob_name}")

    # State
    upload = True
    success = False

    # Get blob
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)

    # Check if blob exists already and matches the file we are uploading
    if check_blob_hash and blob.exists():
        # Get blob hash
        blob.reload()
        expected_hash = blob.crc32c

        # Check file hash
        actual_hash = crc32c_base64_hash(file_path)

        # Compare hashes
        files_match = expected_hash == actual_hash
        logging.info(
            f"{func_name}: files_match={files_match}, expected_hash={expected_hash}, " f"actual_hash={actual_hash}"
        )
        if files_match:
            logging.info(
                f"{func_name}: skipping upload as files match. bucket_name={bucket_name}, blob_name={blob_name}, "
                f"file_path={file_path}"
            )
            upload = False
            success = True

    # Upload if file doesn't exist or exists and doesn't match
    if upload:
        # Get connection semaphore
        if connection_sem is not None:
            connection_sem.acquire()

        for i in range(0, retries):
            try:
                blob.chunk_size = chunk_size
                blob.upload_from_filename(file_path)
                success = True
                break
            except ChunkedEncodingError as e:
                logging.error(f"{func_name}: exception uploading file: try={i}, exception={e}")

        # Release connection semaphore
        if connection_sem is not None:
            connection_sem.release()

    return success, upload


def gcs_upload_files(
    *,
    bucket_name: str,
    file_paths: List[str],
    blob_names: List[str] = None,
    max_processes: int = cpu_count(),
    max_connections: int = cpu_count(),
    retries: int = 3,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> bool:
    """Upload a list of files to Google Cloud storage.

    :param bucket_name: the name of the Google Cloud storage bucket.
    :param file_paths: the paths of the files to upload as blobs.
    :param blob_names: the destination paths of blobs where the files will be uploaded. If not specified then these
    will be automatically generated based on the file_paths.
    :param max_processes: the maximum number of processes.
    :param max_connections: the maximum number of upload connections at once.
    :param retries: the number of times to retry uploading a file if an error occurs.
    :param chunk_size: the chunk size to use when uploading a blob in multiple parts, must be a multiple of 256 KB.
    :return: whether the files were uploaded successfully or not.
    """

    func_name = gcs_upload_files.__name__
    logging.info(f"{func_name}: uploading files")

    # Assert that files exist
    is_files = [os.path.isfile(file_path) for file_path in file_paths]
    if not all(is_files):
        not_found = []
        for file_path, is_file in zip(file_paths, is_files):
            if not is_file:
                not_found.append(file_path)
        raise Exception(f"{func_name}: the following files could not be found {not_found}")

    # Create blob names
    if blob_names is None:
        blob_names = [gcs_blob_name_from_path(file_path) for file_path in file_paths]

    # Assert that file_paths and blob_names have the same length
    assert len(file_paths) == len(blob_names), f"{func_name}: file_paths and blob_names have different lengths"

    # Upload each file in parallel
    manager = multiprocessing.Manager()
    connection_sem = manager.BoundedSemaphore(value=max_connections)
    with ProcessPoolExecutor(max_workers=max_processes) as executor:
        # Create tasks
        futures = []
        futures_msgs = {}
        for blob_name, file_path in zip(blob_names, file_paths):
            msg = f"{func_name}: bucket_name={bucket_name}, blob_name={blob_name}, file_path={str(file_path)}"
            logging.info(f"{func_name}: {msg}")
            future = executor.submit(
                gcs_upload_file,
                bucket_name=bucket_name,
                blob_name=blob_name,
                file_path=str(file_path),
                retries=retries,
                connection_sem=connection_sem,
                chunk_size=chunk_size,
            )
            futures.append(future)
            futures_msgs[future] = msg

        # Wait for completed tasks
        results = []
        for future in as_completed(futures):
            print(f"future type is {type(future)}")
            success, upload = future.result()
            results.append(success)
            msg = futures_msgs[future]
            if success:
                logging.info(f"{func_name}: success, {msg}")
            else:
                logging.info(f"{func_name}: failed, {msg}")

    return all(results)
