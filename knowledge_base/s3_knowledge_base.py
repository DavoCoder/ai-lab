# Copyright 2024-2025 DavoCoder
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# s3_knowledge_base.py
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
