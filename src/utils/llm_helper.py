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
import langchain_google_vertexai as google_vertexai
import langchain_openai as openai

_GEMINI_MODEL = "gemini-1.5-flash-latest"
_NON_GEMINI_MODEL = "text-bison"
_OPENAI_MODEL = "gpt-3.5-turbo"


# Select the LLM to use based on the settings set in the UI.
def select_llm(config):
  if config.google_api_key and config.gemini_enabled:
    print("Picked Gemini 1.5 Flash model for summarizing")
    os.environ["GOOGLE_API_KEY"] = config.google_api_key
    return google_genai.ChatGoogleGenerativeAI(
        model=_GEMINI_MODEL,
        max_output_tokens=8192,
        temperature=0.2,
        top_p=0.98,
        top_k=40,
    )
  elif config.google_api_key:
    print("Picked Google PaLM 2 model for summarizing")
    os.environ["GOOGLE_API_KEY"] = config.google_api_key
    return google_vertexai.VertexAI(
        model_name=_NON_GEMINI_MODEL,
        temperature=0.2,
        top_p=0.98,
        top_k=40,
        max_output_tokens=1024,
    )
  else:
    print("Picked OpenAI 3.5-turbo model for summarizing")
    return openai.OpenAI(
        model_name=_OPENAI_MODEL,
        temperature=0.2,
        max_tokens=1024,
        openai_api_key=config.openai_api_key,
    )
