# Copyright 2024 Google LLC
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

import langchain_google_genai as google_genai

GEMINI_MODEL = "gemini-3.1-flash-lite-preview"


# Select the LLM to use based on the settings set in the UI.
def select_llm(config):
    if not config.google_api_key:
        raise ValueError("Google API Key is required for Gemini.")
    print("Picked Gemini model for summarizing")
    os.environ["GOOGLE_API_KEY"] = config.google_api_key
    return google_genai.ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        max_output_tokens=8192,
        temperature=0.2,
        top_p=0.98,
        top_k=40,
    )
