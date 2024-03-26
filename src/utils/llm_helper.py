import os

from langchain_community.llms import OpenAI
from langchain_google_vertexai import VertexAI


# Select the LLM to use based on the settings set in the UI.
def select_llm(config):
    if config.google_api_key and config.gemini_enabled:
        print("Picked Gemini Pro model for summarizing")
        return VertexAI(model="gemini-pro",
                        max_output_tokens=2048, )
    elif config.google_api_key:
        print("Picked Google PaLM 2 model for summarizing")
        os.environ["GOOGLE_API_KEY"] = config.google_api_key
        return VertexAI(
            model_name="text-bison",
            temperature=0.2,
            top_p=0.98,
            top_k=40,
            max_output_tokens=1024,
        )
    else:
        print("Picked OpenAI 3.5-turbo model for summarizing")
        return OpenAI(
            temperature=0.2,
            max_tokens=1024,
            openai_api_key=config.openai_api_key,
        )
