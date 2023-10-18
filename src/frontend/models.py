import base64
from collections import OrderedDict
import dataclasses
import enum
import hashlib
import logging
import random
import textwrap
from typing import Optional

from bs4 import BeautifulSoup
from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.llms.base import LLM
from langchain.text_splitter import CharacterTextSplitter
import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
from streamlit_lottie import st_lottie
import yaml

logger = logging.getLogger(__name__)

HTML_EXCLUDE_TAGS = [
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

SUMMARY_PROMPT_TEMPLATE = textwrap.dedent("""\
    Write a concise bullet point summary of the following company homepage and explain what do they sell as product or added value:

    {text}

    CONSCISE SUMMARY IN BULLET POINTS:
    """)


def get_random_state(force_new: bool = False):
    random_state = st.session_state.get("random_seed")
    if force_new or not random_state:
        random_state = random.randint(1000, 100000)
        st.session_state["random_seed"] = random_state
    return random_state


def render_custom_spinner_animation():
    animation_response = requests.get('https://assets1.lottiefiles.com/packages/lf20_vykpwt8b.json')
    animation_response.raise_for_status()
    return st_lottie(animation_response.json(), height=200, width=300)


@st.cache_data(ttl=600, show_spinner=False)
def fetch_landing_page_text(url: str) -> list[Document]:
    """Fetches the visible text from a landing page url."""
    html_content = requests.get(url).text
    soup = BeautifulSoup(html_content, features="html.parser")
    texts = soup.find_all(text=True)
    content = "\n".join(str(t).strip() for t in texts if t.parent.name not in HTML_EXCLUDE_TAGS).strip()
    metadata = {
        "title": soup.find("title").string,
        "description": soup.find("meta", property="description", content=True),
    }
    # Splits into multiple documents to fit into LLM context.
    text_splitter = CharacterTextSplitter()
    return text_splitter.create_documents([content])


def summarize_text(_docs: Document, _llm: LLM, verbose: bool = False) -> str:
    """Summarizes the given text using LangChain summary chain."""

    prompt = PromptTemplate(
        template=SUMMARY_PROMPT_TEMPLATE,
        input_variables=["text"])
    chain = load_summarize_chain(
        _llm,
        chain_type="map_reduce",
        map_prompt=prompt,
        verbose=verbose)
    return chain.run(_docs)


class ScoreDecision(enum.StrEnum):
    UNKNOWN = "unknown"
    REMOVE = "remove"
    KEEP = "keep"


class ScoreCategory(enum.StrEnum):
    UNKNOWN = "unknown"
    COMPETITOR = "competitor"
    BRAND_SAFETY = "brand safety"
    MISSPELLING = "misspelling"
    OTHER = "other"


def asdict_enum_factory(data):
    def serialize_enum_value(obj):
        if isinstance(obj, enum.Enum):
            return obj.value
        return obj
    return dict((k, serialize_enum_value(v)) for k, v in data)


@dataclasses.dataclass
class KeywordEvaluation:
    """LLM or Human evaluation."""
    keyword: str
    decision: ScoreDecision = ScoreDecision.UNKNOWN
    #category: ScoreCategory = ScoreCategory.UNKNOWN
    reason: str = ""

    # def __getstate__(self):
    #   return self.to_dict()

    # def __setstate__(self, state):
    #   for k,v in state.items():
    #     setattr(self, k, v)

    def should_remove(self) -> bool:
        return self.decision == ScoreDecision.REMOVE

    @property
    def opposite_decision(self) -> ScoreDecision:
        match self.decision:
            case ScoreDecision.REMOVE:
              return ScoreDecision.KEEP
            case ScoreDecision.KEEP:
              return ScoreDecision.REMOVE
            case _:
              return ScoreDecision.UNKNOWN

    @classmethod
    def from_dict(cls, data: dict[str, str]):
        return cls(
            keyword=data.get("keyword", data.get("Keyword")),
            decision=ScoreDecision(data.get("decision", ScoreDecision.UNKNOWN.value)),
            #category=ScoreCategory(data.get("category", ScoreCategory.UNKNOWN.value)),
            #category=ScoreCategory.UNKNOWN.value,
            reason=data.get("reason", "Unspecified")
        )

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self, dict_factory=asdict_enum_factory)
        # Enforces order of keys to allow the LLM to produce tokens in the right order.
        ordered_keys = ["keyword", "category", "reason", "decision"]
        return dict(sorted(d.items(), key=lambda pair: ordered_keys.index(pair[0])))

    @property
    def hash(self):
        md5bytes = hashlib.md5(self.keyword.encode("utf-8")).digest()
        return base64.urlsafe_b64encode(md5bytes).decode("ascii")


@dataclasses.dataclass
class EvaluationPair:
    llm_decision: ScoreDecision
    human_decision: ScoreDecision


def sample_batch(df: pd.DataFrame, batch_size: int, exclude_keywords: Optional[set[str]] = None, random_state: int = 0) -> list[str]:
    """Draw a new sample from our data source."""
    df_filtered = df.query("keyword not in @exclude_keywords")
    return df_filtered.sample(min(len(df_filtered), batch_size), random_state=random_state)


def format_scoring_fragment(evaluations: OrderedDict[str, KeywordEvaluation]) -> str:
    """Formats the given evaluations for the LLM."""
    data = [e.to_dict() for e in evaluations.values()]
    return yaml.dump(data, sort_keys=False, allow_unicode=True)


def parse_scoring_response(response: str) -> list[KeywordEvaluation]:
    """Parses the LLM response, expecting a YAML format."""

    # PALM 2 cleaning
    response = (response
        .replace('```yaml', '')
        .replace('```YAML', '')
        .replace('```', '')
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


def sparkline(data, figsize=(3, 0.5), **kwargs):
    """Sparkline is a plot without any axes."""
    ylim = kwargs.pop('ylim')

    fig, ax = plt.subplots(1, 1, figsize=figsize, **kwargs)
    ax.plot(data)
    ax.fill_between(range(len(data)), y1=data, y2=0, alpha=0.35)
    ax.set_ylim(*ylim)

    # Removes all the axes.
    for k,v in ax.spines.items():
      v.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Adds a red dot on the latest data point.
    plt.plot(len(data) - 1, data[len(data) - 1], 'r.')

    plt.tight_layout()
    return fig
