from src.OpenaiHandler import OpenAiHandler


class Digester:
    def __init__(self):
        self.handler = OpenAiHandler()

    def digest(self, commit_message: str, diff_message: str, digest_type: str):
        """
        Digests the commit and diff messages based on the specified digest type.

        Args:
            commit_message (str): The commit message.
            diff_message (str): The diff message.
            digest_type (str): The type of digest. Must be one of 'weekly', 'daily', or 'monthly'.

        Returns:
            str: The response from the handler.

        Raises:
            ValueError: If the digest type is not one of 'weekly', 'daily', or 'monthly'.
        """
        if digest_type not in ['weekly', 'daily', 'monthly']:
            raise ValueError('Digest type must be one of weekly, daily, or monthly.')
        self.handler.createAssistant('digest')
        content = f"Commit message: {commit_message}\nDiff message: {diff_message}\nDigest type: {digest_type}"
        response = self.handler.handleTask('digest', content)
        return response
        