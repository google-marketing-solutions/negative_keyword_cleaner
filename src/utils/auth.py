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
from typing import Any, Dict

try:
    # Loads local env variables.
    from dotenv import find_dotenv, load_dotenv
    env_file = find_dotenv()
    print("Found .env file:", env_file)
    load_dotenv(env_file)
except ImportError as err:
    print("ImportError:", err)
except Exception as err:
    print(f"Failed to import local env variables: {err}")

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
import streamlit as st
from st_oauth import st_oauth
from st_oauth.st_oauth import _STKEY as ST_OAUTH_TOKEN_KEY

# Google OAuth2.0 configuration
AUTHORIZE_URL = os.getenv('AUTHORIZE_URL', 'https://accounts.google.com/o/oauth2/auth')
TOKEN_URL = os.getenv('TOKEN_URL', 'https://oauth2.googleapis.com/token')
REFRESH_TOKEN_URL = os.getenv('REFRESH_TOKEN_URL', 'https://oauth2.googleapis.com/token')
REVOKE_TOKEN_URL = os.getenv('REVOKE_TOKEN_URL', 'https://accounts.google.com/o/oauth2/revoke')
JWKS_URI = os.getenv('JWKS_URI', 'https://www.googleapis.com/oauth2/v3/certs')
OAUTH_CLIENT_ID = os.getenv('OAUTH_CLIENT_ID')
OAUTH_CLIENT_SECRET = os.getenv('OAUTH_CLIENT_SECRET')
OAUTH_REDIRECT_URI = os.getenv('OAUTH_REDIRECT_URI')

print("OAUTH_REDIRECT_URI:", OAUTH_REDIRECT_URI)


def authenticate_user():
    oauth2_params = {
        'authorization_endpoint': AUTHORIZE_URL,
        'token_endpoint': TOKEN_URL,
        'redirect_uri': OAUTH_REDIRECT_URI,
        'jwks_uri': JWKS_URI,
        'client_id': OAUTH_CLIENT_ID,
        'client_secret': OAUTH_CLIENT_SECRET,
        'scope': 'email profile https://www.googleapis.com/auth/adwords',
        'audience': OAUTH_CLIENT_ID,
        'identity_field_in_token': 'sub',
    }
    st_oauth(oauth2_params)


def get_access_token():
    oauth_result = st.session_state[ST_OAUTH_TOKEN_KEY]
    return oauth_result['access_token']


def get_credentials(config: Dict[str, Any]):
    creds = None
    user_info = {
        'client_id': config['client_id'],
        'refresh_token': config['refresh_token'],
        'client_secret': config['client_secret']
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
