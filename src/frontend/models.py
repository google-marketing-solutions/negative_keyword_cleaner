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


"""Module containing data models and helper functions for the frontend."""

import collections
import dataclasses
import enum
import logging
import random
import textwrap
from typing import Any, Dict, List, Optional

from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import CharacterTextSplitter
import pandas as pd
from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator
import requests
import streamlit as st
import yaml

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
  """Returns a random state integer.

  If force_new is True or no state exists, generates a new state.

  Args:
    force_new: Force the generation of a new random state.

  Returns:
    The random state integer.
  """
  random_state = st.session_state.get("random_seed")
  if force_new or not random_state:
    random_state = random.randint(1000, 100000)
    st.session_state["random_seed"] = random_state
  return random_state


@st.cache_data(ttl=600, show_spinner=False)
def fetch_landing_page_text(url: str) -> List[Document]:
  """Fetches the visible text from a landing page URL.

  Returns a list of Documents.

  Parameters:
  url (str): The URL of the landing page.

  Returns:
  List[Document]: A list of Documents containing the text from the
  landing page.
  """
  headers = {
      "User-Agent": (
          "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
          " AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0"
          " Safari/537.36"
      ),
      "Accept": (
          "text/html,application/xhtml+xml,application/xml;q=0.9,"
          "image/webp,image/apng,*/*;q=0.8,"
          "application/signed-exchange;v=b3;q=0.9"
      ),
      "Accept-Language": "en-US,en;q=0.9",
      "Connection": "keep-alive",
      "DNT": "1",
      "Upgrade-Insecure-Requests": "1",
  }
  response = requests.get(url, headers=headers, timeout=15)
  response.raise_for_status()

  soup = BeautifulSoup(response.text, features="html.parser")
  texts = soup.find_all(string=True)
  content = "\n".join(
      str(t).strip() for t in texts if t.parent.name not in _HTML_EXCLUDE_TAGS
  ).strip()

  # Splits into multiple documents to fit into LLM context.
  text_splitter = CharacterTextSplitter()
  return text_splitter.create_documents([content])


def summarize_text(docs: list[Document], llm: BaseLanguageModel) -> str:
  """Summarizes the given text using LangChain's summary chain.

  Args:
    docs (Document): Documents to summarize.
    llm (BaseLanguageModel): Language model to use for summarization.

  Returns:
    str: The summarized text.
  """
  prompt = PromptTemplate(
      template=_SUMMARY_PROMPT_TEMPLATE, input_variables=["text"]
  )
  text = "\n\n".join([doc.page_content for doc in docs])
  chain = prompt | llm | StrOutputParser()
  return chain.invoke({"text": text})


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
  """Factory function to create custom serialization for dataclasses.

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
  priority_score: int = 0

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
  def from_dict(cls, data: dict[str, Any]):
    return cls(
        keyword=data.get("keyword", data.get("Keyword", "")),
        decision=ScoreDecision(
            data.get("decision", ScoreDecision.UNKNOWN.value)
        ),
        reason=data.get("reason", "Unspecified"),
        priority_score=data.get("priority_score", 0),
    )

  def to_dict(self) -> dict[str, Any]:
    d = dataclasses.asdict(self, dict_factory=asdict_enum_factory)
    # Enforces order of keys to allow the LLM to produce tokens in the right
    # order.
    ordered_keys = ["keyword", "reason", "decision", "priority_score"]
    return dict(sorted(d.items(), key=lambda pair: ordered_keys.index(pair[0])))


@dataclasses.dataclass
class EvaluationPair:
  llm_decision: ScoreDecision
  human_decision: ScoreDecision


class KeywordEvaluationModel(BaseModel):
  """Pydantic model for a single LLM keyword evaluation."""

  keyword: str = Field(description="The negative keyword analyzed")
  reason: str = Field(description="Concise explanation of the decision")
  decision: ScoreDecision = Field(description="KEEP or REMOVE")
  priority_score: int = Field(
      default=0,
      description=(
          "Priority score (1-5, 5 being highest) for addressing this keyword"
      ),
  )

  @field_validator("decision", mode="before")
  @classmethod
  def _normalize_decision(cls, value: Any) -> ScoreDecision:
    if isinstance(value, ScoreDecision):
      return value
    if isinstance(value, str):
      upper_val = value.strip().upper()
      if upper_val in ScoreDecision.__members__:
        return ScoreDecision(upper_val)
    return ScoreDecision.UNKNOWN


class KeywordEvaluations(BaseModel):
  """Pydantic container for a list of LLM keyword evaluations."""

  evaluations: List[KeywordEvaluationModel]


def sample_batch(
    df: pd.DataFrame,
    batch_size: int,
    exclude_keywords: Optional[set[str]] = None,
    random_state: int = 0,
) -> pd.DataFrame:
  """Samples a batch of keywords from the provided DataFrame.

  Args:
    df: The DataFrame to sample from.
    batch_size: The size of the batch to sample.
    exclude_keywords: Keywords to exclude from sampling.
    random_state: Random state for reproducibility.

  Returns:
    A DataFrame containing the sampled batch.
  """
  if exclude_keywords:
    df_filtered = df[~df["keyword"].isin(exclude_keywords)]
  else:
    df_filtered = df

  return df_filtered.sample(
      min(len(df_filtered), batch_size), random_state=random_state
  )


def format_scoring_fragment(
    evaluations: collections.OrderedDict[str, KeywordEvaluation],
) -> str:
  """Formats the given evaluations for the LLM.

  Args:
    evaluations: Evaluations to format.

  Returns:
    A string representation of the evaluations in YAML format.
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


def prepare_human_dataframe(
    filtered_keywords: pd.DataFrame,
    evaluations: dict[str, KeywordEvaluation],
) -> pd.DataFrame:
  """Prepares a DataFrame of human evaluations.

  Args:
    filtered_keywords: DataFrame containing the filtered keywords.
    evaluations: Mapping of keyword string to human KeywordEvaluation.

  Returns:
    A pandas DataFrame containing the human evaluations, or an empty DataFrame
    if no matching filtered keywords are available.
  """
  if filtered_keywords is None or filtered_keywords.empty or not evaluations:
    return pd.DataFrame()

  first_by_kw = (
      filtered_keywords.drop_duplicates(subset=["keyword"])
      .set_index("keyword")[
          ["original_keyword", "campaign_name", "campaign_id", "adgroup_id"]
      ]
      .to_dict("index")
  )
  formatted_evals = [
      {
          "keyword": kw,
          "original_keyword": first_by_kw[kw]["original_keyword"],
          "human_decision": str(human_eval.decision),
          "human_reason": human_eval.reason,
          "campaign_name": first_by_kw[kw]["campaign_name"],
          "campaign_id": first_by_kw[kw]["campaign_id"],
          "adgroup_id": first_by_kw[kw]["adgroup_id"],
          "Source": "Human",
      }
      for kw, human_eval in evaluations.items()
      if kw in first_by_kw
  ]
  return pd.DataFrame(formatted_evals)


def _format_criterion_type(adgroup_name: str, match_type: str) -> str:
  """Formats the Google Ads Editor criterion type string."""
  prefix = "Negative " if adgroup_name else "Campaign Negative "
  match_type_map = {
      "BROAD": "Broad",
      "PHRASE": "Phrase",
  }
  return prefix + match_type_map.get(match_type, "Exact")


def prepare_dataframes(
    cached_scoring_kws_evals: List[KeywordEvaluation],
    filtered_keywords: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
  """Prepares DataFrames for keywords to remove and keywords to keep.

  Args:
    cached_scoring_kws_evals: List of student evaluations.
    filtered_keywords: DataFrame containing the filtered keywords.

  Returns:
    Tuple of (df_to_remove, df_to_keep).
  """
  if filtered_keywords is None or filtered_keywords.empty:
    return pd.DataFrame(), pd.DataFrame()

  first_by_kw = (
      filtered_keywords.drop_duplicates(subset=["keyword"])
      .set_index("keyword")[
          ["original_keyword", "campaign_id", "adgroup_id", "account_id"]
      ]
      .to_dict("index")
  )
  rows_by_kw: dict[str, list[Any]] = {}
  for df_entry in filtered_keywords.itertuples():
    if df_entry.campaign_name or df_entry.adgroup_id:
      rows_by_kw.setdefault(df_entry.keyword, []).append(df_entry)

  formatted_evals_to_remove = []
  formatted_evals_to_keep = []
  for student_eval in cached_scoring_kws_evals:
    for df_entry in rows_by_kw.get(student_eval.keyword, []):
      criterion_type = _format_criterion_type(
          df_entry.adgroup_name, df_entry.match_type
      )
      if student_eval.decision == ScoreDecision.REMOVE:
        formatted_evals_to_remove.append({
            "Account": df_entry.account_name,
            "Campaign": df_entry.campaign_name,
            "Ad Group": df_entry.adgroup_name,
            "Keyword": student_eval.keyword,
            "Criterion Type": criterion_type,
            "Status": "Removed",
            "Student Reason": student_eval.reason,
            "Source": "Student",
            "campaign_id": df_entry.campaign_id,
            "adgroup_id": df_entry.adgroup_id,
            "match_type": df_entry.match_type,
            "Priority Score": student_eval.priority_score,
        })
      elif student_eval.decision == ScoreDecision.KEEP:
        formatted_evals_to_keep.append({
            "Account": df_entry.account_name,
            "Campaign": df_entry.campaign_name,
            "Ad Group": df_entry.adgroup_name,
            "Keyword": student_eval.keyword,
            "Criterion Type": criterion_type,
            "Status": "Enabled",
            "Student Reason": student_eval.reason,
            "Source": "Student",
            "Priority Score": student_eval.priority_score,
            **first_by_kw[student_eval.keyword],
        })

  df_to_remove = pd.DataFrame(formatted_evals_to_remove)
  df_to_keep = pd.DataFrame(formatted_evals_to_keep)

  if not df_to_remove.empty:
    df_to_remove = df_to_remove.sort_values(
        by=["Priority Score"], ascending=False
    )
  if not df_to_keep.empty:
    df_to_keep = df_to_keep.sort_values(by=["Priority Score"], ascending=False)

  return df_to_remove, df_to_keep
