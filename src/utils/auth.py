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
GOOGLE_SIGN_IN_CODE = '''
<button class="gsi-material-button">
  <div class="gsi-material-button-state"></div>
  <div class="gsi-material-button-content-wrapper">
    <div class="gsi-material-button-icon">
      <svg version="1.1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48" xmlns:xlink="http://www.w3.org/1999/xlink" style="display: block;">
        <path fill="#EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z"></path>
        <path fill="#4285F4" d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z"></path>
        <path fill="#FBBC05" d="M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z"></path>
        <path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z"></path>
        <path fill="none" d="M0 0h48v48H0z"></path>
      </svg>
    </div>
    <span class="gsi-material-button-contents">Continue with Google</span>
    <span style="display: none;">Continue with Google</span>
  </div>
</button>
'''

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
    st_oauth(oauth2_params, label=GOOGLE_SIGN_IN_CODE)


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
