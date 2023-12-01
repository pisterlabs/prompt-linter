"""Definitions of different error classes and thier common names, messages, status code etc"""


class ChatErrorResponse(Exception):
    """Errors to be notified to user via bot's chat response"""

    def __init__(self, detail: str):
        super().__init__()
        self.name = "Chat Error Response"
        self.detail = detail
        self.status_code = 500


class UnprocessableException(Exception):
    """Format for Unprocessable error"""

    def __init__(self, detail: str):
        super().__init__()
        self.name = "Unprocessable Data"
        self.detail = detail
        self.status_code = 422


class PermissionException(Exception):
    """Format for permission error"""

    def __init__(self, detail: str):
        super().__init__()
        self.name = "Permission Denied"
        self.detail = detail
        self.status_code = 403


class AccessException(Exception):
    """Format for permission error"""

    def __init__(self, detail: str):
        super().__init__()
        self.name = "Access Denied"
        self.detail = detail
        self.status_code = 403


class OpenAIException(Exception):
    """Format for errors from OpenAI APIs"""

    def __init__(self, detail):
        super().__init__()
        self.name = "Error from OpenAI"
        self.detail = detail
        self.status_code = 502


class ChromaException(Exception):
    """Format for errors from ChromaDB's APIs"""

    def __init__(self, detail):
        super().__init__()
        self.name = "Error from ChromaDB"
        self.detail = detail
        self.status_code = 502


class PostgresException(Exception):
    """Format for errors from Postgres' APIs"""

    def __init__(self, detail):
        super().__init__()
        self.name = "Error from Postgres Database"
        self.detail = detail
        self.status_code = 502


class SupabaseException(Exception):
    """Format for errors from Supabase"""

    def __init__(self, detail):
        super().__init__()
        self.name = "Error from Supabase Authentication system"
        self.detail = detail
        self.status_code = 502


class GenericException(Exception):
    """Format for Database error"""

    def __init__(self, detail: str):
        super().__init__()
        self.name = "Error"
        self.detail = detail
        self.status_code = 500
