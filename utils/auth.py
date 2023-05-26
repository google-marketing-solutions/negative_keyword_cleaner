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

from typing import Any, Dict
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request


def get_credentials(config: Dict[str, Any]):
    creds = None
    user_info = {
        "client_id": config['client_id'],
        "refresh_token": config['refresh_token'],
        "client_secret": config['client_secret']
    }
    creds = Credentials.from_authorized_user_info(user_info)

    # If credentials are expired, refresh.
    if creds.expired:
        try:
            creds.refresh(Request())
        except Exception as error:
            if 'invalid_scope' in error.args[0]:
                creds = None
                
    if not creds.valid:
        creds = None

    return creds
