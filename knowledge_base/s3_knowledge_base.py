import boto3
from knowledge_base.external_knowledge_base_interface import ExternalKnowledgeBase

class S3KnowledgeBase(ExternalKnowledgeBase):
    def __init__(self, bucket_name, aws_access_key, aws_secret_key):
        self.bucket_name = bucket_name
        self.s3_client = boto3.client(
            's3', aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key
        )

    def connect(self):
        print(f"Connected to S3 bucket: {self.bucket_name}")

    def fetch_documents(self):
        response = self.s3_client.list_objects_v2(Bucket=self.bucket_name)
        files = [obj['Key'] for obj in response.get('Contents', [])]
        print(f"Fetched {len(files)} files from S3.")
        return files

    def detect_updates(self):
        # Example: Compare with local metadata or detect modified files
        return []
