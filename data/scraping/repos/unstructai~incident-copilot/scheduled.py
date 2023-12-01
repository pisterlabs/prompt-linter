"""
.. Added by Kishore Jalleda
.. full list of modifications at https://github.com/unstructai
.. copyright: (c) 2023 Kishore Jalleda
.. author:: Kishore Jalleda <kjalleda@gmail.com>
"""
from dispatch.decorators import apply, timer, scheduled_project_task
from dispatch.plugin import service as plugin_service
from dispatch.project.models import Project
from dispatch.scheduler import scheduler
from dispatch.database.core import SessionLocal
from schedule import every
from dispatch.AI.fine_tune.models import FineTuningJob, Error, Hyperparameters, FileObject
from dispatch.AI.fine_tune import service as fine_tune_service
import os


import logging

log = logging.getLogger(__name__)
directory_path = "/root/training-data"


@scheduler.add(every(2).hours, name="fine-tune-model-job-create")
@apply(timer)
@scheduled_project_task
def fine_tune_model_job_create(
    db_session: SessionLocal,
    project: Project = None,
):
    """Create the fine tuning jobs from training files."""
    # Check /root/training-data directory for any new files to be fine tuned
    files = os.listdir(directory_path)
    if not files:
        log.debug("No files found to fine tune.")
        return

    # Create a new file object in the database if it doesn't already exist
    for filename in files:
        file_obj = fine_tune_service.get_file_object_by_internal_filename(
            db_session=db_session, internal_filename=filename
        )
        if not file_obj:
            # Create a new file object in the database. Fields will updated when the file is uploaded to OpenAI.
            file_obj_in = FileObject(
                id=f"{filename}-{project.organization.slug}-{project.slug}",
                bytes=os.path.getsize(f"{directory_path}/{filename}"),
                created_at=os.stat(f"{directory_path}/{filename}").st_ctime,
                filename=filename,
                internal_filename=filename,
                internal_status="pending",
                object="file",
                purpose="fine-tune",
                status="uploaded",
            )
            file_obj = fine_tune_service.create_file_object(
                db_session=db_session,
                file_object_in=file_obj_in,
            )

            log.debug(
                f"Created new file object in the database to be uploaded for finr tuning: {file_obj.filename}"
            )


@scheduler.add(every(2).hours, name="fine-tune-model-job-submit")
@apply(timer)
@scheduled_project_task
def fine_tune_model_job_submit(
    db_session: SessionLocal,
    project: Project = None,
):
    """Upload traning files to Fine tunes the model using the given training data."""
    plugin = plugin_service.get_preferred_active_instance(
        db_session=db_session, plugin_type="artificial-intelligence", project_id=project.id
    )

    if not plugin:
        log.warning("No active AI plugin found. Skipping fine tuning.")
        return

    # Check all the files that need to be fine tuned.
    files = fine_tune_service.get_all_file_objects(db_session=db_session)
    if not files:
        log.debug("No files found to fine tune.")
        return

    for file in files:
        # Upload any pending files to OpenAI for fine tuning
        if file.internal_status == "pending":
            external_file_obj = plugin.instance.upload(
                file=f"{directory_path}/{file.internal_filename}", purpose="fine-tune"
            )

            # Validate the return status of the file upload
            if not external_file_obj or external_file_obj.status not in [
                "uploaded",
                "processed",
                "ready",
            ]:
                log.warning(
                    f"Error while uploading file: {file.internal_filename} to OpenAI for fine tuning. Status: {external_file_obj.status}"
                )
                continue
            else:
                log.debug(
                    f"Uploaded file {file.internal_filename} to OpenAI for fine tuning: {external_file_obj.filename}, Status: {external_file_obj.status}"
                )

            # Update the internal file object with the external file object
            internal_file_obj = fine_tune_service.update_file_object(
                db_session=db_session,
                file_object=file,
                file_object_in=external_file_obj,
            )

            # update internal status of the file object to `uploaded`
            internal_file_obj.internal_status = "uploaded"

        # File uploaded. Create a job that will fine tune the model now.
        # "While the file is processing, you can still create a fine-tuning job but it will not start until the file processing has completed."
        elif file.internal_status == "uploaded":
            training_file = file.id
            fine_tuning_job_external = plugin.instance.create(
                training_file=training_file,
                model=plugin.instance.configuration.model,
            )

            log.debug(
                f"Created fine-tuning model job on OpenAI: {fine_tuning_job_external.id}, for training file: {training_file}. {fine_tuning_job_external}"
            )

            # Validate the return status of the fine tuning job
            if not fine_tuning_job_external or fine_tuning_job_external.status not in [
                "validating_files",
                "queued",
                "running",
                "succeeded",
            ]:
                # Create an error in the database
                log.warning(
                    f"Error while creating fine tuning job: {fine_tuning_job_external.id if fine_tuning_job_external else None}"
                )
                continue

            # Create a new fine tuning job in the database if it doesn't already exist
            fine_tuning_job_internal = fine_tune_service.get(
                db_session=db_session, fine_tuning_job_id=fine_tuning_job_external.id
            )
            if (
                not fine_tuning_job_internal
                or fine_tuning_job_internal.internal_status != "submitted"
            ):
                fine_tuning_job_internal = fine_tune_service.create(
                    db_session=db_session,
                    fine_tuning_job_in=fine_tuning_job_external,
                )

                # TODO. create hyperparameters and error objects in the database

            # Update the internal status of the fine tuning job, and associate the internal training file with the fine tuning job
            # External training file is the payload of the fine tuning job
            fine_tuning_job_internal.internal_status = "submitted"
            fine_tuning_job_internal.internal_training_file_id = file.id

    # Commit the changes to the database
    db_session.commit()


@scheduler.add(every(2).hours, name="fine-tune-model-job-retrieve")
@apply(timer)
@scheduled_project_task
def fine_tune_model_job_retrieve(
    db_session: SessionLocal,
    project: Project = None,
):
    """Retrieves the fine tuning jobs."""
    plugin = plugin_service.get_preferred_active_instance(
        db_session=db_session, plugin_type="artificial-intelligence", project_id=project.id
    )

    if not plugin:
        log.warning("No active AI plugin found. Skipping fine tuning.")
        return

    # Retrieve all the fine tuning jobs
    fine_tuning_jobs = plugin.instance.retrieve(limit=10)

    if not fine_tuning_jobs.data:
        log.debug("No fine tuning jobs found.")
        return
    else:
        log.debug(f"Found {len(fine_tuning_jobs.data)} fine tuning jobs from OpenAI")

    # Update the internal status of the fine tuning jobs
    for fine_tuning_job in fine_tuning_jobs:
        # Create a new fine tuning job in the database if it doesn't exist, else update it
        fine_tuning_job_internal = fine_tune_service.get(
            db_session=db_session, fine_tuning_job_id=fine_tuning_job.id
        )
        if not fine_tuning_job_internal:
            fine_tuning_job_internal = fine_tune_service.create(
                db_session=db_session,
                fine_tuning_job_in=fine_tuning_job,
            )
            log.debug(f"Created new fine tuning job in the database: {fine_tuning_job.id}")
        else:
            fine_tuning_job_internal = fine_tune_service.update(
                db_session=db_session,
                fine_tuning_job=fine_tuning_job_internal,
                fine_tuning_job_in=fine_tuning_job,
            )
            log.debug(f"Updated fine tuning job in the database: {fine_tuning_job.id}")

        # Update the internal status of the fine tuning job
        fine_tuning_job_internal.internal_status = "retrieved"

    # Commit the changes to the database
    db_session.commit()
