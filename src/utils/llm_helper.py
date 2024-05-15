import os

import langchain_google_genai as google_genai
import langchain_google_vertexai as google_vertexai
from langchain_community import llms

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
        temperature=0.1,
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
    return llms.OpenAI(
        model_name=_OPENAI_MODEL,
        temperature=0.2,
        max_tokens=1024,
        openai_api_key=config.openai_api_key,
    )
