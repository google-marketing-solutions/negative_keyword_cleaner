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

import yaml
from dataclasses import dataclass, asdict

CONFIG_FILE = './app_config.yaml'


@dataclass
class Config:
    # OAuth credentials
    client_id: str = ''
    client_secret: str = ''
    refresh_token: str = ''

    # Google Ads API
    developer_token: str = ''
    login_customer_id: str = ''
    use_proto_plus: bool = True

    # LLM platform API keys
    openai_api_key: str = ''
    google_api_key: str = ''

    @classmethod
    def from_disk(cls):
        with open(CONFIG_FILE, 'r') as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
        return cls(**config)

    def save_to_disk(self):
        config = asdict(self)
        with open(CONFIG_FILE, 'w') as f:
            yaml.dump(config, f)
        print(f"Configurations updated in {CONFIG_FILE}")
