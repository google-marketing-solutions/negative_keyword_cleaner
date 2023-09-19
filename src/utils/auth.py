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

# Retrieves the app url and configures the OAuth callback url.
try:
    # NOTE: locally we have access to Cloudflare Quick Tunnel domain on the /metrics endpoint.
    # local_response = requests.get('http://localhost:8081/metrics')
    # local_response.raise_for_status()
    # st_base_url = re.compile(r'userHostname\="([^"]+)"').search()
    st_base_url = "http://localhost:8080"
except requests.ConnectionError as e:
    session = st.runtime.get_instance()._session_mgr.list_active_sessions()[0]
    st_base_url = urllib.parse.urlunparse(['https', session.client.request.host, '', '', '', ''])

# REDIRECT_URI = os.path.join(st_base_url, 'component/streamlit_oauth.authorize_button/index.html')
REDIRECT_URI = st_base_url
SCOPES = ' '.join([
    # Default scope
    'https://www.googleapis.com/auth/userinfo.profile',
    'https://www.googleapis.com/auth/userinfo.email',
    # AdWords scope
    'https://www.googleapis.com/auth/adwords',
])


def authenticate_user():
    oauth2_params = {
        'authorization_endpoint': AUTHORIZE_URL,
        'token_endpoint': TOKEN_URL,
        'redirect_uri': REDIRECT_URI,
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
