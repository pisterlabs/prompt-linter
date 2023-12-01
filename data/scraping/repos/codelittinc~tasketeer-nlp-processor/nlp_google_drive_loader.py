from langchain.document_loaders import GoogleDriveLoader
from google.oauth2.credentials import Credentials

class NlpGoogleDriveLoader(GoogleDriveLoader):
    """Google Drive Loader for NLP."""   
    
    # Google Drive Token
    google_token: str = None
    
    def __init__(self, google_token: str, *args, **kwargs):
        """Initialize Google Drive Loader for NLP."""
        super().__init__(*args, **kwargs)
        self.google_token = google_token
                  
    def _load_credentials(self):
        """Load credentials."""
        credentials = Credentials(self.google_token)
        return credentials