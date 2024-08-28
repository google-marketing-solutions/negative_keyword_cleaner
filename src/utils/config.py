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
import logging
import os
import pathlib
from dataclasses import dataclass, asdict

import yaml
from google.cloud import exceptions
from google.cloud import storage

from utils.auth import _OAUTH_CLIENT_ID, _OAUTH_CLIENT_SECRET

LOCAL_CONFIG_FILE = (
    pathlib.Path(__file__).parents[2] / "app_config.yaml"
).absolute()
GCS_CONFIG_FILE = "neg_cleaner/app_config.yaml"
# GCS client
DEFAULT_BUCKET_NAME = os.getenv("DEFAULT_BUCKET_NAME")
# Default Google Vertex AI key
GOOGLE_VERTEXAI_API_KEY = os.getenv("GOOGLE_VERTEXAI_API_KEY", "")


def is_cloudrun():
  return os.getenv("K_SERVICE") is not None


# NOTE: No need for a storage client in local env.
storage_client = storage.Client() if is_cloudrun() else None


@dataclass
class Config:
  """Config class handling all keys, tokens and config for the application.
  """

  # OAuth credentials
  client_id: str = _OAUTH_CLIENT_ID
  client_secret: str = _OAUTH_CLIENT_SECRET
  # Google Ads API
  developer_token: str = os.getenv("GOOGLE_ADS_API_TOKEN", "")
  login_customer_id: str = os.getenv("MCC_ID", "")
  use_proto_plus: bool = True
  # LLM platform API keys
  openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
  google_api_key: str = GOOGLE_VERTEXAI_API_KEY
  # Tool settings
  batch_size: int = 15

  @classmethod
  def from_disk(cls, config_path=None):
    if config_path is None:
      config_path = LOCAL_CONFIG_FILE
    if not os.path.exists(config_path):
      return cls()
    with open(config_path, "r", encoding="utf-8") as f:
      config = yaml.load(f, Loader=yaml.SafeLoader)
    return cls(**config)

  def save_to_disk(self, config_path=None):
    if config_path is None:
      config_path = LOCAL_CONFIG_FILE
    config = asdict(self)
    with open(config_path, "w", encoding="utf-8") as f:
      yaml.dump(config, f)
    print(f"Configurations updated in {config_path}")

  @classmethod
  def from_gcs(cls):
    """Loads the config file from GCS and returns an instance of the class.

    Returns:
        An instance of the class, either loaded from the downloaded file
        or a default configuration if the file is not found.
    """
    bucket = storage_client.bucket(DEFAULT_BUCKET_NAME)
    blob = bucket.blob(GCS_CONFIG_FILE)
    try:
      with open("/tmp/app_config.yaml", "wb", encoding="utf-8") as file_obj:
        blob.download_to_file(file_obj)
    except exceptions.NotFound:
      logging.error(
          "The file %s does not exist in bucket %s.",
          GCS_CONFIG_FILE,
          DEFAULT_BUCKET_NAME,
      )
      return cls()
    return cls.from_disk(config_path="/tmp/app_config.yaml")

  def save_to_gcs(self):
    self.save_to_disk(config_path="/tmp/app_config.yaml")
    with open("/tmp/app_config.yaml", "r", encoding="utf-8") as f:
      content = f.read()
    bucket = storage_client.bucket(DEFAULT_BUCKET_NAME)
    blob = bucket.blob(GCS_CONFIG_FILE)
    blob.upload_from_string(content, content_type="application/x-yaml")
