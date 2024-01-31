# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pathlib
import yaml
from dataclasses import dataclass, asdict

from google.cloud import storage
from google.cloud.exceptions import NotFound

from utils.auth import OAUTH_CLIENT_ID, OAUTH_CLIENT_SECRET

LOCAL_CONFIG_FILE = (pathlib.Path(__file__).parents[2] / 'app_config.yaml').absolute()
GCS_CONFIG_FILE = 'neg_cleaner/app_config.yaml'

# GCS client
DEFAULT_BUCKET_NAME = os.getenv('DEFAULT_BUCKET_NAME')

# Default Google Vertex AI key
GOOGLE_VERTEXAI_API_KEY = os.getenv('GOOGLE_VERTEXAI_API_KEY', '')


def is_appengine():
    return os.getenv('RUNNING_ON_GAE') is not None


# NOTE: No need for a storage client in local env.
storage_client = storage.Client() if is_appengine() else None


@dataclass
class Config:
    # OAuth credentials
    client_id: str = OAUTH_CLIENT_ID
    client_secret: str = OAUTH_CLIENT_SECRET

    # Google Ads API
    developer_token: str = ''
    login_customer_id: str = ''
    use_proto_plus: bool = True

    # LLM platform API keys
    openai_api_key: str = ''
    google_api_key: str = GOOGLE_VERTEXAI_API_KEY

    gemini_enabled: str = False

    # Tool settings
    batch_size: int = 15

    @classmethod
    def from_disk(cls, config_path=None):
        if config_path is None:
            config_path = LOCAL_CONFIG_FILE
        if not os.path.exists(config_path):
            return cls()
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
        return cls(**config)

    def save_to_disk(self, config_path=None):
        if config_path is None:
            config_path = LOCAL_CONFIG_FILE
        config = asdict(self)
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        print(f"Configurations updated in {config_path}")

    @classmethod
    def from_gcs(cls):
        bucket = storage_client.bucket(DEFAULT_BUCKET_NAME)
        blob = bucket.blob(GCS_CONFIG_FILE)
        try:
            blob.download_to_file('/tmp/app_config.yaml')
        except NotFound:
            print(f'The file {GCS_CONFIG_FILE} does not exist in bucket {DEFAULT_BUCKET_NAME}.')
            return cls()
        return cls.from_disk(config_path='/tmp/app_config.yaml')

    def save_to_gcs(self):
        self.save_to_disk(config_path='/tmp/app_config.yaml')
        with open('/tmp/app_config.yaml', 'r') as f:
            content = f.read()
        bucket = storage_client.bucket(DEFAULT_BUCKET_NAME)
        blob = bucket.blob(GCS_CONFIG_FILE)
        blob.upload_from_string(content, content_type='application/x-yaml')
