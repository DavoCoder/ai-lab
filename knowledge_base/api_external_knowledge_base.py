import requests
from external_knowledge_base_interface import ExternalKnowledgeBase

class APIKnowledgeBase(ExternalKnowledgeBase):
    def __init__(self, api_url, api_key=None):
        """
        API-based Knowledge Base connector.

        Args:
            api_url (str): The endpoint URL of the external knowledge base API.
            api_key (str): Optional API key for authentication.
        """
        self.api_url = api_url
        self.api_key = api_key
        self.session = None

    def connect(self):
        """Establish a connection by creating a requests session."""
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})
        print(f"Connected to API: {self.api_url}")

    def fetch_documents(self):
        """
        Fetch all documents from the external knowledge base.

        Returns:
            List of document dictionaries.
        """
        if not self.session:
            raise ConnectionError("You must connect before fetching documents.")

        print("Fetching documents from the external knowledge base...")
        response = self.session.get(f"{self.api_url}/documents")
        response.raise_for_status()

        documents = response.json()
        print(f"Fetched {len(documents)} documents.")
        return documents

    def detect_updates(self):
        """
        Detect new or updated documents from the external API.

        Returns:
            List of updated document dictionaries.
        """
        if not self.session:
            raise ConnectionError("You must connect before detecting updates.")

        print("Detecting new or updated documents...")
        response = self.session.get(f"{self.api_url}/documents/updates")
        response.raise_for_status()

        updates = response.json()
        print(f"Detected {len(updates)} updated documents.")
        return updates
