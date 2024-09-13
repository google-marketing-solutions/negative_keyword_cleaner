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
import tempfile
from dataclasses import dataclass, asdict
from typing import Dict

import yaml
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

# MCC ID
MCC_ID = os.getenv("MCC_ID")

# Google Ads API token
GOOGLE_ADS_API_TOKEN = os.getenv("GOOGLE_ADS_API_TOKEN")

# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def is_cloudrun():
  return os.getenv("K_SERVICE") is not None


# NOTE: No need for a storage client in local env.
storage_client = storage.Client() if is_cloudrun() else None


@dataclass
class Config:
  # OAuth credentials
  client_id: str = _OAUTH_CLIENT_ID
  client_secret: str = _OAUTH_CLIENT_SECRET

  # Google Ads API
  developer_token: str = GOOGLE_ADS_API_TOKEN
  login_customer_id: str = MCC_ID
  use_proto_plus: bool = True

  # LLM platform API keys
  openai_api_key: str = OPENAI_API_KEY
  google_api_key: str = GOOGLE_VERTEXAI_API_KEY

  # Tool settings
  batch_size: int = 15

  @classmethod
  def from_disk(cls, config_path=None):
    if config_path is None:
      config_path = LOCAL_CONFIG_FILE
    if not os.path.exists(config_path):
      return cls()
    with open(config_path, "r") as f:
      config = yaml.load(f, Loader=yaml.SafeLoader)
    return cls(**config)

  def save_to_disk(self, config_path=None):
    if config_path is None:
      config_path = LOCAL_CONFIG_FILE
    config = asdict(self)
    with open(config_path, "w") as f:
      yaml.dump(config, f)
    print(f"Configurations updated in {config_path}")

  @classmethod
  def from_gcs(self):
    storage_client = storage.Client()
    bucket = storage_client.bucket(DEFAULT_BUCKET_NAME)
    blob = bucket.blob(GCS_CONFIG_FILE)

    logging.warning("Bucket :", DEFAULT_BUCKET_NAME)
    logging.warning("Blob :", GCS_CONFIG_FILE)
    with tempfile.NamedTemporaryFile(mode="w+b", delete=False) as tmp:
      logging.warning("TMP FILE :", tmp.name)
      blob.download_to_filename(tmp.name)
      with open(tmp.name, "rb") as f:
        logging.warning("FILE CONTENT :", f)
        config_data = yaml.safe_load(f)

    # os.unlink(tmp.name)  # Ensure temp file is deleted
    self.from_dict(config_data)

  def save_to_gcs(self):
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp:
      yaml.dump(asdict(self), tmp)

    try:
      storage_client = storage.Client()
      bucket = storage_client.bucket(DEFAULT_BUCKET_NAME)
      blob = bucket.blob(GCS_CONFIG_FILE)
      blob.upload_from_filename(tmp.name)
    finally:
      os.unlink(tmp.name)

  @classmethod
  def from_dict(cls, data: Dict):
    """Create a Config instance from a dictionary."""
    return cls(
        client_id=data.get("client_id"),
        client_secret=data.get("client_secret"),
        developer_token=data.get("developer_token"),
        login_customer_id=data.get("login_customer_id"),
        use_proto_plus=data.get("use_proto_plus"),
        openai_api_key=data.get("openai_api_key"),
        google_api_key=data.get("google_api_key"),
        batch_size=data.get("batch_size"),
    )
