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
    client_id: str
    client_secret: str
    refresh_token: str
    developer_token: str
    login_customer_id: int
    ai_api_token: str
    use_proto_plus: bool

    def __init__(self):
        self.file_path = CONFIG_FILE
        config = self.load_config_from_file()
        if config is None:
            config = {}

        self.client_id = config.get('client_id', '')
        self.client_secret = config.get('client_secret', '')
        self.refresh_token = config.get('refresh_token', '')
        self.developer_token = config.get('developer_token', '')
        self.login_customer_id = config.get('login_customer_id', '')
        self.ai_api_token = config.get('ai_api_token', '')
        self.use_proto_plus = True


    def load_config_from_file(self):
        with open(self.file_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
        return config


    def save_to_file(self):
        config = asdict(self)
        with open(self.file_path, 'w') as f:
            yaml.dump(config, f)
        print(f"Configurations updated in {self.file_path}")
