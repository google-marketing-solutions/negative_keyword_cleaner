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

import dataclasses
import enum
import logging
import random
import textwrap
from collections import OrderedDict
from typing import Optional, Any, Dict, List

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
import yaml
from bs4 import BeautifulSoup
from langchain.chains import summarize
from langchain.docstore.document import Document
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from requests.exceptions import HTTPError, SSLError

logger = logging.getLogger(__name__)

_HTML_EXCLUDE_TAGS = [
    "[document]",
    "noscript",
    "header",
    "html",
    "meta",
    "head",
    "input",
    "script",
    "style",
]

_SUMMARY_PROMPT_TEMPLATE = textwrap.dedent("""\
    Use the following company homepage and personal knowledge to write a concise bullet point summary.
    Explain what they sell as product, service or added value:

    {text}

    CONSCISE SUMMARY IN BULLET POINTS:
    """)


def get_random_state(force_new: bool = False) -> int:
  """
  Returns a random state integer. If force_new is True or no state exists, generates a new state.

  Parameters:
  force_new (bool): Force the generation of a new random state.

  Returns:
  int: The random state integer.
  """
  random_state = st.session_state.get("random_seed")
  if force_new or not random_state:
    random_state = random.randint(1000, 100000)
    st.session_state["random_seed"] = random_state
  return random_state


@st.cache_data(ttl=600, show_spinner=False)
def fetch_landing_page_text(url: str) -> List[Document]:
  """
  Fetches the visible text from a landing page URL and returns a list of Documents.

  Parameters:
  url (str): The URL of the landing page.

  Returns:
  List[Document]: A list of Documents containing the text from the landing page.
  """
  content = ""
  try:
    # Since we are making requests from a datacenter,
    # the following request sometimes get rejected.
    #
    # To avoid this, we are mimicking an actual browser
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
            " AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0"
            " Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
        "DNT": "1",  # Do Not Track Request Header
        "Upgrade-Insecure-Requests": "1",
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    text_content = response.text
    logger.info(text_content)
    soup = BeautifulSoup(text_content, features="html.parser")
    texts = soup.find_all(text=True)
    content = "\n".join(
        str(t).strip() for t in texts if t.parent.name not in _HTML_EXCLUDE_TAGS
    ).strip()
  except (HTTPError, SSLError) as err:
    # Fallback to Client-Side Fetch
    st.warning("Server-side fetch failed. Trying client-side fetch...")

    js_code = """
            <script>
            const url = window.Streamlit.getUrlParams()['url'];  

            fetch(url)
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`HTTP error: ${response.status}`);
                    }
                    return response.text();
                })
                .then(html => {
                    const parser = new DOMParser();
                    const doc = parser.parseFromString(html, 'text/html');
                    const textContent = Array.from(doc.body.querySelectorAll('*')) // Select all elements
                        .filter(el => !_HTML_EXCLUDE_TAGS.includes(el.tagName.toLowerCase())) 
                        .map(el => el.textContent.trim())
                        .join('\\n');

                    window.Streamlit.setComponentValue(textContent);
                })
                .catch(error => {
                    window.Streamlit.setComponentValue('Error: ' + error.message);
                });
            </script>
        """

    component_value = components.html(js_code, height=0, width=0)

    if component_value.text.startswith("Error:"):
      st.error(component_value)
      st.stop()

    content = component_value

  # Splits into multiple documents to fit into LLM context.
  text_splitter = CharacterTextSplitter()
  return text_splitter.create_documents([content])


def summarize_text(docs: list[Document], llm: LLM, verbose: bool = False) -> (
    str):
  """Summarizes the given text using LangChain's summary chain.

  Args:
    docs (Document): Documents to summarize.
    llm (LLM): Language model to use for summarization.
    verbose (bool): If True, enables verbose mode.

  Returns:
    str: The summarized text.
  """
  prompt = PromptTemplate(
      template=_SUMMARY_PROMPT_TEMPLATE, input_variables=["text"]
  )
  chain = summarize.load_summarize_chain(
      llm, chain_type="map_reduce", map_prompt=prompt, verbose=verbose
  )
  return chain.run(docs)


class ScoreDecision(enum.StrEnum):
  """Enumeration for scoring decisions.

  Attributes:
    UNKNOWN (str): When the LLM didn't know which decision to take.
    REMOVE (str): Decision to remove a keyword.
    KEEP (str): Decision to keep a keyword.
  """

  UNKNOWN = "UNKNOWN"
  REMOVE = "REMOVE"
  KEEP = "KEEP"


def asdict_enum_factory(data: Dict[str, Any]) -> Dict[str, Any]:
  """Factory function to create a custom serialization function for dataclasses.

  This function handles enum members by converting them to their respective
  values.

  Args:
    data (Dict[str, Any]): The data dictionary to serialize.

  Returns:
    Dict[str, Any]: A dictionary with enum members serialized to their values.
  """

  def serialize_enum_value(obj: Any) -> Any:
    """Serializes an enum member to its value if the object is an enum.

    It otherwise returns the object as is.

    Args:
     obj (Any): The object to serialize.

    Returns:
     Any: The serialized object.
    """
    if isinstance(obj, enum.Enum):
      return obj.value
    return obj

  return dict((k, serialize_enum_value(v)) for k, v in data)


@dataclasses.dataclass
class KeywordEvaluation:
  """LLM or Human evaluation."""

  keyword: str
  decision: ScoreDecision = ScoreDecision.UNKNOWN
  reason: str = ""

  def should_remove(self) -> bool:
    return self.decision == ScoreDecision.REMOVE

  @property
  def opposite_decision(self) -> ScoreDecision:
    match self.decision:
      case ScoreDecision.REMOVE:
        return ScoreDecision.KEEP
      case ScoreDecision.KEEP:
        return ScoreDecision.REMOVE
      case ScoreDecision.UNKNOWN:
        return ScoreDecision.UNKNOWN

  @classmethod
  def from_dict(cls, data: dict[str, str]):
    return cls(
        keyword=data.get("keyword", data.get("Keyword", "")),
        decision=ScoreDecision(
            data.get("decision", ScoreDecision.UNKNOWN.value)
        ),
        reason=data.get("reason", "Unspecified"),
    )

  def to_dict(self) -> dict:
    d = dataclasses.asdict(self, dict_factory=asdict_enum_factory)
    # Enforces order of keys to allow the LLM to produce tokens in the right order.
    ordered_keys = ["keyword", "reason", "decision"]
    return dict(sorted(d.items(), key=lambda pair: ordered_keys.index(pair[0])))


@dataclasses.dataclass
class EvaluationPair:
  llm_decision: ScoreDecision
  human_decision: ScoreDecision


def sample_batch(
    df: pd.DataFrame,
    batch_size: int,
    exclude_keywords: Optional[set[str]] = None,
    random_state: int = 0,
) -> pd.DataFrame:
  """
  Samples a batch of keywords from the provided DataFrame.

  Parameters:
  df (pd.DataFrame): The DataFrame to sample from.
  batch_size (int): The size of the batch to sample.
  exclude_keywords (Optional[set[str]]): Keywords to exclude from sampling.
  random_state (int): Random state for reproducibility.

  Returns:
  pd.DataFrame: A DataFrame containing the sampled batch.
  """
  df_filtered = df.query("keyword not in @exclude_keywords")
  return df_filtered.sample(
      min(len(df_filtered), batch_size), random_state=random_state
  )


def format_scoring_fragment(
    evaluations: OrderedDict[str, KeywordEvaluation],
) -> str:
  """
  Formats the given evaluations for the LLM.

  Parameters:
  evaluations (OrderedDict[str, KeywordEvaluation]): Evaluations to format.

  Returns:
  str: A string representation of the evaluations in YAML format.
  """
  data = [e.to_dict() for e in evaluations.values()]
  yaml_str = yaml.dump(
      data, sort_keys=False, allow_unicode=True, default_flow_style=False
  )

  # Modify the REASON to use a Representer that allows "'" and "\""
  modified_lines = []
  for line in yaml_str.split("\n"):
    if line.strip().startswith("reason:"):
      modified_lines.append(line.replace("reason:", "reason::", 1))
    else:
      modified_lines.append(line)
  return "\n".join(modified_lines)


def _clean_yaml_response(response: str) -> str:
  """Cleans the LLM response for YAML parsing."""
  return (
      response.replace("```yaml", "")
      .replace("```YAML", "")
      .replace("```", "")
      .replace("---", "")
      .strip()
  )


def parse_scoring_response(response: str) -> List[KeywordEvaluation]:
  """Parses the LLM response in YAML format into KeywordEvaluation objects.

  Args:
      response (str): The LLM response.

  Returns:
      List[KeywordEvaluation]: A list of parsed keyword evaluations.
  """

  response = _clean_yaml_response(response)

  if not response:
    return []

  try:
    data = yaml.safe_load(response)
  except yaml.parser.ParserError as e:
    logger.error(f"Error parsing YAML data: {e}")
    logger.error(f"Original Content: {data}")
    return []

  outputs = []
  for item_yaml in response.split("\n-"):
    item_yaml = item_yaml.strip()
    if not item_yaml:
      continue

    try:
      item_yaml = (
          "- " + item_yaml if not item_yaml.startswith("-") else item_yaml
      )
      item = yaml.safe_load(item_yaml)
      if isinstance(item[0], dict):
        outputs.append(KeywordEvaluation.from_dict(item[0]))
    except (yaml.parser.ParserError, TypeError, ValueError) as e:
      logger.error(
          f"Failed to parse keyword: {e}. Original Content: {item_yaml}"
      )

  return outputs
