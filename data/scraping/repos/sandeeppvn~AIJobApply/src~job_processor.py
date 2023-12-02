import logging
import os
from typing import Optional

import pandas as pd
from dotenv import find_dotenv, load_dotenv
# from pandarallel import pandarallel
from tqdm import tqdm

from email_handler import EmailHandler
from google_api_handler import GoogleAPIClass
from openai_handler import Openai

# pandarallel.initialize()

tqdm.pandas()

logger = logging.getLogger(__name__)


class JobProcessor:
    def __init__(
        self,
        templates_path: str = "templates",
        gmail: Optional[str] = None,
        password: Optional[str] = None,
        credentials_file: Optional[str] = None,
        openapi_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize a JobProcessor instance with required handlers and credentials.
        """
        load_dotenv(find_dotenv())
        self.jobs_df = pd.DataFrame()
        self.templates_path = templates_path
        self.gmail = gmail or os.getenv("GMAIL_ADDRESS")
        self.password = password or os.getenv("GMAIL_PASSWORD")
        self.credentials_file = credentials_file or os.getenv("GOOGLE_API_CREDENTIALS_FILE")
        self.openapi_key = openapi_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_MODEL")
        


        self._validate_credentials()

        self.google_api_handler = GoogleAPIClass(self.credentials_file, use_service_account=False) # type: ignore
        self.openai = Openai(self.openapi_key, self.model) # type: ignore
        self.email_handler = EmailHandler(self.gmail, self.password) # type: ignore

        self.google_drive_folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
        if not self.google_drive_folder_id:
            raise ValueError("GOOGLE_DRIVE_FOLDER_ID is not set in environment variables.")

    def _validate_credentials(self):
        """
        Validate required credentials.
        """
        if not all([self.gmail, self.password, self.credentials_file, self.openapi_key, self.model, self.templates_path]):
            raise ValueError(
                "One or more of the required credentials is missing. Please check the .env file or the CLI arguments."
            )

    def _get_gsheet_name(self, gsheet_name: Optional[str]) -> str:
        """
        Get the Google Sheet name from the parameter or environment variables.
        """
        if not gsheet_name:
            load_dotenv(find_dotenv())
            gsheet_name = os.getenv("GOOGLE_SHEET_NAME")
            if not gsheet_name:
                raise ValueError(
                    "gsheet_name is not provided and GOOGLE_SHEET_NAME is not set in environment variables."
                )
        return gsheet_name

    def process_jobs(self, gsheet_name: Optional[str] = None):
        """
        Main function to process jobs based on their status.
        """
        try:
            logger.info("Processing jobs...")
            self.jobs_df = self.get_all_jobs(gsheet_name)
            logger.info(f"Found {len(self.jobs_df)} total jobs to process")

            logger.info("Processing new jobs...")
            self.process_new_jobs()

            logger.info("Processing Contact Ready jobs...")
            self.process_contact_ready_jobs()

            logger.info("Processing Message Approved jobs...")
            self.process_message_approved_jobs()

            logger.info("Updating Google Sheet...")
            self.update_dataframe_to_gsheet(gsheet_name)

            logger.info("Job processing complete.")
        except Exception as e:
            logger.error(f"Error while processing jobs: {str(e)}")
            raise

    def get_all_jobs(self, gsheet_name: Optional[str] = None) -> pd.DataFrame:
        """
        Get all jobs from the Google Sheet.
        """
        gsheet_name = self._get_gsheet_name(gsheet_name)
        gsheet = self.google_api_handler.get_gsheet(gsheet_name)
        return pd.DataFrame(gsheet.sheet1.get_all_records())

    def process_new_jobs(self):
        """
        Process jobs with "New Job" or "Contact Required" status.
        """
        for status in ['New Job', 'Contact Required']:
            self._process_jobs_with_status(status)

    def _process_jobs_with_status(self, status: str):
        """
        Process jobs with a given status.
        """
        # Create a copy of the jobs_df DataFrame
        jobs_df = self.jobs_df[self.jobs_df['Status'] == status].copy()
        if jobs_df.empty:
            return

        has_email = has_email = jobs_df['Email'].str.strip() != ""
        has_linedin = jobs_df['LinkedIn Contact'].str.strip() != ""
        jobs_df.loc[has_email | has_linedin, 'Status'] = 'Contact Ready'

        # has_contact_details = (jobs_df['Contact Details'].str.strip() != "") & ~has_email & ~has_linedin
        # found_emails = jobs_df.loc[has_contact_details, 'Contact Details'].progress_apply(self.openai.find_email)
        # jobs_df.loc[has_contact_details, 'Email'] = found_emails

        # jobs_df.loc[has_contact_details & (found_emails.str.strip() != ""), 'Status'] = 'Contact Ready'
        # jobs_df.loc[has_contact_details & (found_emails.str.strip() == ""), 'Status'] = 'Contact Required'
        # jobs_df.loc[~has_email & ~has_contact_details & ~has_linedin, 'Status'] = 'Contact Required'

        jobs_df.loc[~has_email & ~has_linedin, 'Status'] = 'Contact Required'

        self.jobs_df.update(jobs_df)

    def process_contact_ready_jobs(self):
        """
        Process jobs with "Message Ready" status.
        """

        jobs_df = self.jobs_df[self.jobs_df['Status'] == 'Contact Ready'].copy()
        if jobs_df.empty:
            return

        jobs_df.progress_apply(self.process_contact_ready_job_wrapper, axis=1) # type: ignore
        jobs_df['Status'] = 'Message Approval Required'
        self.jobs_df.update(jobs_df)

    def process_contact_ready_job_wrapper(self, row: pd.Series) -> pd.Series:
        """
        Wrapper function to process a single job with "Message Ready" status.
        """
        try:
            generated_json = self.openai.generate_custom_contents(row)
            row['Message Content'] = generated_json['Message Content']
            row['Message Subject'] = generated_json['Message Subject']
            row['Resume'] = generated_json['Resume']
            row['Cover Letter'] = generated_json['Cover Letter']
            row['Description'] = generated_json['Description']
            row['LinkedIn Note'] = generated_json['LinkedIn Note']

            row['Status'] = 'Message Approval Required'
        except Exception as e:
            logger.warning(f"Failed to generate custom contents for job at Company Name {row['Company Name']}. Error: {str(e)}")
            row['Status'] = 'Failed to generate custom contents'
        return row

    def process_message_approved_jobs(self):
        """
        Process jobs with "Message Approved" status.
        """
        
        # Send emails for jobs with "Message Approved" status
        jobs_df = self.jobs_df[self.jobs_df['Status'] == 'Message Approved'].copy()

        if jobs_df.empty:
            return
        
        logger.info("Creating drive files...")
        self.create_drive_files()
        
        # For all jobs with "Email" and "Message Approved" status, send an email
        email_jobs_df = jobs_df[jobs_df['Email'].str.strip() != ""].copy()
        if not email_jobs_df.empty:
            jobs_df.progress_apply(self.send_email, axis=1) # type: ignore
            jobs_df.loc[email_jobs_df.index, 'Status'] = 'Message Sent'
            jobs_df.loc[email_jobs_df.index, 'LinkedIn Status'] = 'Dont Send Message'

        
        # For all jobs with "LinkedIn Contact" and "Message Approved" status and not "Dont Send Message" LinkedIn Status, send a LinkedIn message
        linkedin_jobs_df = jobs_df[
            (jobs_df['LinkedIn Contact'].str.strip() != "") & 
            (jobs_df['LinkedIn Status'] != 'Dont Send Message') &
            (jobs_df['LinkedIn Status'] != 'Message Sent')
        ].copy()
        if not linkedin_jobs_df.empty:
            # linkedin_jobs_df.progress_apply(self.send_linkedin_message, axis=1)
            jobs_df.loc[linkedin_jobs_df.index, 'LinkedIn Status'] = 'Send Message'

        self.jobs_df.update(jobs_df)


    def send_email(self, row):
        """
        Send an email for a job.
        Parameters:
        - row (Series): Pandas Series containing the details of a job.
        """
        try:
            self.email_handler.send(
                content=row['Message Content'],
                recepient_email=row['Email'],
                subject=row['Message Subject'],
            )
            return 'Message Sent'
        except Exception as e:
            logger.warning(f"Failed to send email for job at Company Name {row['Company Name']}. Error: {str(e)}")
            return 'Failed to send message'


    def update_dataframe_to_gsheet(self, gsheet_name: Optional[str] = None):
        """
        Update the jobs_df DataFrame to the Google Sheet.
        """
        gsheet_name = self._get_gsheet_name(gsheet_name)
        if self.jobs_df.empty:
            return

        gsheet = self.google_api_handler.get_gsheet(gsheet_name)
        try:
            gsheet.sheet1.update([self.jobs_df.columns.values.tolist()] + self.jobs_df.values.tolist())
        except Exception as e:
            logger.error(f"Failed to update Google Sheet. Error: {str(e)}")

    def create_drive_files(self):
        """
        For each job, create a folder in Google Drive and upload the resume and cover letter and Message Content.
        """
        jobs_df = self.jobs_df[
            (self.jobs_df['Message Content'] != '') & 
            (self.jobs_df['Resume'] != '') &
            (self.jobs_df['Cover Letter'] != '') &
            (self.jobs_df['LinkedIn Note'] != '') &
            (self.jobs_df['Status'] == 'Message Approved')
        ].copy()

        if jobs_df.empty:
            return

        # For each job, create a folder in Google Drive and upload the resume and cover letter and Message Content
        jobs_df.progress_apply(self.create_drive_files_for_job, axis=1) # type: ignore

    def create_drive_files_for_job(self, row) -> pd.Series:
        """
        Create a folder in Google Drive and upload the resume and cover letter and Message Content for a job.
        Parameters:
        - row (Series): Pandas Series containing the details of a job.
        Returns:
        - Series: Pandas Series containing the details of the job.
        """
        folder_name = row['Company Name'] + " - " + row['Position']
        # Create folder on local machine
        os.mkdir('Jobs/' + folder_name)
        folder_id = self.google_api_handler.create_folder(folder_name, self.google_drive_folder_id) # type: ignore

        self.google_api_handler.upload_file(row['Resume'], "resume.docx", folder_id, folder_name)
        self.google_api_handler.upload_file(row['Cover Letter'],  "cover_letter.docx", folder_id, folder_name)
        self.google_api_handler.upload_file(row['Message Content'], "email.docx", folder_id, folder_name)

        return row