from collections import OrderedDict
import dataclasses
import enum
import logging
import random
import textwrap
from typing import Optional, Any, Dict, List

from bs4 import BeautifulSoup
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.llms.base import LLM
from langchain.text_splitter import CharacterTextSplitter

import pandas as pd
import requests
from requests.exceptions import HTTPError, SSLError
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
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'DNT': '1',  # Do Not Track Request Header
            'Upgrade-Insecure-Requests': '1',
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        text_content = response.text
        logger.info(text_content)
        soup = BeautifulSoup(text_content, features="html.parser")
        texts = soup.find_all(text=True)
        content = "\n".join(str(t).strip() for t in texts if
                            t.parent.name not in _HTML_EXCLUDE_TAGS).strip()
        metadata = {
            "title": soup.find("title").string,
            "description": soup.find("meta", property="description",
                                     content=True),
        }
    except HTTPError as http_err:
        # Handle HTTP errors (status code 4xx or 5xx)
        logger.error(f"HTTP error occurred: {http_err}")
        logger.error(f"Status code: {http_err.response.status_code}")
        st.error("HTTP error: This site in unreachable.")
        st.stop()
    except (Exception, SSLError) as err:
        # Handle other types of errors
        logger.error(f"An error occurred: {err}")
        st.error(
            "SSL Error: This site is not secure and its content could not be retrieved.")
        st.stop()

    # Splits into multiple documents to fit into LLM context.
    text_splitter = CharacterTextSplitter()
    return text_splitter.create_documents([content])


def summarize_text(_docs: Document, _llm: LLM, verbose: bool = False) -> str:
    """
    Summarizes the given text using LangChain's summary chain.

    Parameters:
    docs (Document): Documents to summarize.
    llm (LLM): Language model to use for summarization.
    verbose (bool): If True, enables verbose mode.

    Returns:
    str: The summarized text.
    """
    prompt = PromptTemplate(
        template=_SUMMARY_PROMPT_TEMPLATE,
        input_variables=["text"])
    chain = load_summarize_chain(
        _llm,
        chain_type="map_reduce",
        map_prompt=prompt,
        verbose=verbose)
    return chain.run(_docs)


class ScoreDecision(enum.StrEnum):
    """
    Enumeration for scoring decisions.

    Attributes:
    UNKNOWN (str): Represents the case where the LLM didn't know which decision to take.
    REMOVE (str): Represents a decision to remove a keyword.
    KEEP (str): Represents a decision to keep a keyword.
    """
    UNKNOWN = "UNKNOWN"
    REMOVE = "REMOVE"
    KEEP = "KEEP"


def asdict_enum_factory(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Factory function to create a custom serialization function for dataclasses.
    This function handles enum members by converting them to their respective values.

    Parameters:
    data (Dict[str, Any]): The data dictionary to serialize.

    Returns:
    Dict[str, Any]: A dictionary with enum members serialized to their values.
    """

    def serialize_enum_value(obj: Any) -> Any:
        """
        Serializes an enum member to its value if the object is an enum, otherwise returns the object as is.

        Parameters:
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
            keyword=data.get("keyword", data.get("Keyword")),
            decision=ScoreDecision(
                data.get("decision", ScoreDecision.UNKNOWN.value)),
            reason=data.get("reason", "Unspecified")
        )

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self, dict_factory=asdict_enum_factory)
        # Enforces order of keys to allow the LLM to produce tokens in the right order.
        ordered_keys = ["keyword", "reason", "decision"]
        return dict(
            sorted(d.items(), key=lambda pair: ordered_keys.index(pair[0])))


@dataclasses.dataclass
class EvaluationPair:
    llm_decision: ScoreDecision
    human_decision: ScoreDecision


def sample_batch(df: pd.DataFrame, batch_size: int,
                 exclude_keywords: Optional[set[str]] = None,
                 random_state: int = 0) -> pd.DataFrame:
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
    return df_filtered.sample(min(len(df_filtered), batch_size),
                              random_state=random_state)


def format_scoring_fragment(
        evaluations: OrderedDict[str, KeywordEvaluation]) -> str:
    """
    Formats the given evaluations for the LLM.

    Parameters:
    evaluations (OrderedDict[str, KeywordEvaluation]): Evaluations to format.

    Returns:
    str: A string representation of the evaluations in YAML format.
    """
    data = [e.to_dict() for e in evaluations.values()]
    return yaml.dump(data, sort_keys=False, allow_unicode=True)


def parse_scoring_response(response: str) -> List[KeywordEvaluation]:
    """
    Parses the LLM response expected in YAML format into a list of KeywordEvaluation objects.

    Parameters:
    response (str): The LLM response in YAML format.

    Returns:
    List[KeywordEvaluation]: A list of KeywordEvaluation objects parsed from the response.
    """

    # PaLM 2 cleaning
    response = (response
                .replace('```yaml', '')
                .replace('```YAML', '')
                .replace('```', '')
                .replace('---', '')
                .strip()
                )

    data = yaml.safe_load(response)
    outputs = []
    for d in data:
        try:
            outputs.append(KeywordEvaluation.from_dict(d))
        except (TypeError, ValueError) as inst:
            logger.error(f"Failed to parse keyword: {inst}")
    return outputs
