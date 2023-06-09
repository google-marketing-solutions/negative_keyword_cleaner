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

from collections import defaultdict, OrderedDict
import dataclasses
import json
import pickle
import random
import textwrap
import time
from typing import Callable

from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.llms.fake import FakeListLLM
from openai.error import OpenAIError
import pandas as pd
import streamlit as st
from streamlit_elements import dashboard, elements, lazy, mui
import yaml

from .components.sidebar import display_sidebar_component
from . import models

SAMPLE_BATCH_SIZE = 20

SCHEMA_EVALUATIONS = {
    "bad keyword": models.KeywordEvaluation(
        "bad keyword",
        decision=models.ScoreDecision.KEEP,
        category=models.ScoreCategory.BRAND_SAFETY,
        reason="Keep as a negative for brand safety reasons"),
    "good keyword": models.KeywordEvaluation(
        "good keyword",
        decision=models.ScoreDecision.REMOVE,
        category=models.ScoreCategory.OTHER,
        reason="It is safe to target this keyword"),
}

DEBUG_SUMMARY = True
DEBUG_SCORING = False


def get_neg_keywords(config):
  # TODO: extract the negative keywords directly from here once streamlit 1.23 is released
  # https://docs.streamlit.io/library/changelog
  # https://github.com/streamlit/streamlit/pull/6622

  # googleads_api_client = GoogleAdsApiClient(
  # config_dict=config.__dict__,
  # version=_GOOGLE_ADS_API_VERSION)
  # report_fetcher = AdsReportFetcher(
  #     googleads_api_client,
  #     [config.login_customer_id])
  # adgroup_neg_kws = report_fetcher.fetch(AdgroupNegativeKeywords())
  # campaign_neg_kws = report_fetcher.fetch(CampaignNegativeKeywords())

  # neg_keywords_report = adgroup_neg_kws + campaign_neg_kws
  # displayable_report = neg_keywords_report[['keyword',
  #                                         'match_type',
  #                                         'level',
  #                                         'adgroup_name',
  #                                         'campaign_name',
  #                                         'account_name']]
  pass


def display_page():
  st.header("AI Student — for Google Ads Negative Keywords")
  st.info("Hi, I am your AI student ready to learn from your client to clean their negative keywords. Let's dive in.", icon="🤖")

  display_sidebar_component()

  if DEBUG_SUMMARY:
    responses = [
        "HTML summary 1",
        "HTML summary 2",
        "HTML summary 3",
        "Renault sells a range of vehicles, including electric, full hybrid, and mild hybrid models, with prices ranging from €11,400 to €61,900. Customers can discover, configure, and compare two models in 3D, and select and discover models based on criteria such as electric, family, city, SUV, and 7+ seats. Renault also offers services such as long-term rental, contract Sérénité Renault, and delivery time. Additionally, they provide services such as MY Renault, service client, FAQ, access for deaf and hard of hearing, ordering attestations, configure, test drive, accessories, original museum & store, renew, mobilize share, garages and concessions.",
    ]
    llm = FakeListLLM(responses=responses)
  else:
    llm = OpenAI(
        temperature=0.2,
        max_tokens=1024,
        openai_api_key=st.session_state.config.google_api_key or st.session_state.config.open_api_key,
    )


  ##
  # Company Details

  with st.expander("1. Company Details", expanded=st.session_state.get('context_open', True)):
    company_homepage_url = st.text_input(
        "Company Homepage URL",
        placeholder="https://...",
        value="https://renault.fr" if DEBUG_SUMMARY else "",
    )
    if not company_homepage_url:
      st.info("Once I have their website url, I can directly read and understand "
              "who is this customer", icon="🤖")
      st.stop()
    else:
      #st.info("Thanks!")
      pass

    with st.spinner("I'm browsing their website..."):
      homepage_docs = models.fetch_landing_page_text(company_homepage_url)
      if st.session_state.get("homepage_fetched", False):
        if DEBUG_SUMMARY: time.sleep(2)  # Too fast, we can slow it a little bit
        st.session_state.homepage_fetched = True

    st.success("Browsing done, I've collected enough info", icon="🤖")

    with st.spinner("I'm now summarizing everything into an executive summary "
                    "(this will take a minute) ..."):
      #models.render_custom_spinner_animation()
      if not st.session_state.get("homepage_summary", None):
        homepage_summary = models.summarize_text(homepage_docs, llm, verbose=DEBUG_SUMMARY).strip()
        if DEBUG_SUMMARY: time.sleep(10)  # Too fast, we can slow it a little bit
        st.session_state.homepage_summary = homepage_summary
      else:
        homepage_summary = st.session_state.homepage_summary

    st.success("Summarizing done, feel free to correct anything that I've written "
               "here. I'm just a student.", icon="🤖")

    company_pitch = st.text_area(
        "Company Business Pitch",
        placeholder="Describe what the company is selling in a few words",
        value=homepage_summary,
        height=150
    )


  def handle_continue_with_context():
    st.session_state.context_ready = True
    st.session_state.context_open = True
    st.session_state.epoch_eval_pairs = []


  st.button("Continue with this context", on_click=handle_continue_with_context)
  if not st.session_state.get('context_ready'):
    st.stop()
  elif st.session_state.get('context_open'):
    st.session_state.context_open = False
    time.sleep(0.05)
    st.experimental_rerun()


  ##
  # Loads keywords

  with st.expander("2. Load negative keywords", expanded=st.session_state.get('load_keywords_open', True)):
    sheets_url = st.text_input(
        "Spreadsheet URL",
        placeholder="https://docs.google.com/...",
        value=st.secrets.get("private_gsheets_url", ""),
        help="This spreadsheet needs to contain the negative keywords",
    )

    # Read in data from the Google Sheet.
    # Uses st.cache_data to only rerun when the query changes or after 10 min.
    @st.cache_data(ttl=600)
    def load_data(sheets_url):
        csv_url = sheets_url.replace("/edit#gid=", "/export?format=csv&gid=")
        return pd.read_csv(csv_url, on_bad_lines='skip')

    if not sheets_url:
      st.stop()

    df = load_data(sheets_url)
    train_mask = (df['HumanRelevancy'].fillna(0) != 0)
    df_test = df[~train_mask]
    df_train = df[train_mask]
    df_train = df_train.assign(relevancy=df_train['HumanRelevancy']).drop(columns=['HumanRelevancy'])

    st.success(f"I've loaded {len(df_test)} negative keywords. Teach me now!", icon="🤖")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total negative keywords", len(df))
    col2.metric("Reviewed", len(df_train))
    col3.metric("Not reviewed", len(df_test))


  def reset_batch_props():
    st.session_state.batch_eval_pairs: list[models.EvaluationPair] = list()
    st.session_state.batch_scored_keywords = set()
    st.session_state.keyword_feedback_eval = None


  def handle_sample_batch():
    # Stores batch eval pairs.
    current_batch_eval_pairs = st.session_state.get("batch_eval_pairs", None)
    if current_batch_eval_pairs:
      epoch_eval_pairs = st.session_state.epoch_eval_pairs
      epoch_eval_pairs.append(current_batch_eval_pairs)

    # Resets variables.
    st.session_state.sample_new_batch = True
    st.session_state.load_keywords_open = True
    st.session_state.scored_keywords = None
    st.session_state.random_state = models.get_random_state(force_new=True)
    reset_batch_props()


  class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
      if dataclasses.is_dataclass(o):
        return dataclasses.asdict(o)
      return super().default(o)


  class EnhancedJSONDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
      super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, data):
      if 'keyword' in data:
        return models.KeywordEvaluation.from_dict(data)
      return data


  def save_evaluations():
    evaluations = st.session_state.evaluations
    with open(".streamlit/evaluations.json", "w") as fp:
      json.dump(evaluations, fp, cls=EnhancedJSONEncoder, indent=2)


  def load_evaluations():
    with open(".streamlit/evaluations.json", "r") as fp:
      evaluations = json.load(fp, cls=EnhancedJSONDecoder)
      print(f"Loaded #{len(evaluations)} evaluations")
      st.session_state.evaluations = evaluations


  if st.session_state.get('load_keywords_open'):
    st.session_state.load_keywords_open = False
    time.sleep(0.05)
    st.experimental_rerun()


  ##
  # Samples and Scores the sampled batch.

  if "evaluations" not in st.session_state:
    st.session_state.evaluations = OrderedDict()

  evaluations = st.session_state.evaluations
  random_state = st.session_state.get("random_state", models.get_random_state())
  df_keywords = models.sample_batch(
      df_test,
      batch_size=SAMPLE_BATCH_SIZE,
      exclude_keywords=set(evaluations.keys()),
      random_state=random_state)

  col1, col2 = st.columns(2)
  with col1:
    st.button("Save evaluations", on_click=save_evaluations)
  with col2:
    st.button("Load evaluations", on_click=load_evaluations)

  formatted_facts = models.format_scoring_fragment(st.session_state.evaluations or SCHEMA_EVALUATIONS)
  formatted_keywords = yaml.dump(df_keywords['keyword'].tolist(), allow_unicode=True)

  template = textwrap.dedent("""\
      You are an agent working for a Google Ads agency asked to score keyword against .

      {company_segment}

      Learn from these examples scored by an expert, formatted as a yaml output, especially learn from the reason column:

      {facts_segment}

      The category field can only take one of the following values: {category_allowed_values}.
      The decision field can only take one of the following values: {decision_allowed_values}.
      The reason field is a free form string that explains why it picked this category and decision.

      The decision to keep a keyword means that this keyword should be excluded from targeting.
      The decision to remove a keyword means that we want to target this keyword.

      Given this context and examples, score this new list of keywords with relevancy and add a detailed reason why you scored that way inspired by the reason used in our examples above:

      {keywords_segment}
      """)
  prompt = PromptTemplate(
      template=template,
      input_variables=["company_segment", "facts_segment", "keywords_segment", "category_allowed_values", "decision_allowed_values"],
  )

  if DEBUG_SCORING:
    scoring_llm = FakeListLLM(responses=[
        textwrap.dedent("""\
            - keyword: taille coffre renault arkana
              category: brand-safety
              reason: It is safe to target this keyword as it is related to a Renault product
              decision: remove
            - keyword: "location v\xE9hicule utilitaire"
              category: other
              reason: Can target this generic query
              decision: remove
            - keyword: acheter une renault captur hybride
              category: other
              reason: Safe to target hybrid models
              decision: remove
            - keyword: accessoires renault trafic
              category: other
              reason: We don't want to sell accessories
              decision: keep
            - keyword: essai citadine
              category: other
              reason: Safe to target generic model queries
              decision: remove
            """)
        ])
  else:
    scoring_llm = OpenAI(
        temperature=0.1,
        max_tokens=2048,
        openai_api_key=st.session_state.config.google_api_key or st.session_state.config.openai_api_key,
    )

  scored_keywords = st.session_state.get("scored_keywords", None)

  if not scored_keywords:
    with st.spinner("Scoring this batch of keywords..."):
      llm_chain = LLMChain(prompt=prompt, llm=scoring_llm, verbose=True)
      try:
        scored_keywords = llm_chain.run({
            "company_segment": company_pitch,
            "facts_segment": formatted_facts,
            "keywords_segment": formatted_keywords,
            "category_allowed_values": ", ".join(x.value for x in models.ScoreCategory),
            "decision_allowed_values": ", ".join(x.value for x in models.ScoreDecision),
        })
      # TODO(dulacp): catch the same exception for PALM 2
      except OpenAIError as inst:
        st.error(f"Failed to run OpenAI LLM due to error: {inst}")
        st.stop()
      else:
        st.session_state.scored_keywords = scored_keywords

  parsed_scored_keywords = models.parse_scoring_response(scored_keywords)
  if "batch_scored_keywords" not in st.session_state:
    st.session_state.batch_scored_keywords = set()
  if "batch_eval_pairs" not in st.session_state:
    st.session_state.batch_eval_pairs = list()
  scored_set = st.session_state.get("batch_scored_keywords", set())
  eval_pairs = st.session_state.get("batch_eval_pairs", list())

  st.header("Teach me")
  if not evaluations:
    st.info("Help me improve my knowledge by correcting my following guesses. I will get better over time.", icon="🤖")
  else:
    st.success(f"I've learned from {len(evaluations)} human evaluations. Keep on correcting me to improve my accuracy!", icon="🤖")

  # Computes each batch evaluation accuracy.
  epoch_eval_pairs = st.session_state.get("epoch_eval_pairs", [])
  epoch_accurracies = []
  for eval_pair in epoch_eval_pairs:
    accuracy = sum(p.llm_decision == p.human_decision for p in eval_pair) / len(eval_pair)
    epoch_accurracies.append(accuracy)
  print("epoch_accurracies:", epoch_accurracies)

  col1, col2, col3, col4 = st.columns(4)
  with col3:
    if epoch_accurracies:
      if len(epoch_accurracies) > 1:
        delta_accuracy = (epoch_accurracies[-1] - epoch_accurracies[-2]) / epoch_accurracies[-2]
        delta_accuracy = f"{delta_accuracy:.0%}"
      else:
        delta_accuracy = None
      st.metric(label="Accuracy (last batch)", value=f"{epoch_accurracies[-1]:.0%}", delta=delta_accuracy)
  with col1:
    st.button("Sample a new batch", on_click=handle_sample_batch)
    st.button("Reset", on_click=reset_batch_props)
  with col2:
    if epoch_accurracies:
      st.pyplot(models.sparkline(epoch_accurracies, ylim=[0, max(epoch_accurracies)*1.1]), transparent=True)
  with col4:
    batch_count = len(scored_set)
    batch_size = len(parsed_scored_keywords)
    st.progress(batch_count / batch_size, text=f"Batch completion {batch_count:d}/{batch_size:d}")

  # Splits keywords on the LLM decision, to simplify the review process
  keywords_to_remove = []
  keywords_to_keep = []
  keywords_unknown = []
  for item in parsed_scored_keywords:
    if item.keyword in scored_set:
      continue
    match item.decision:
      case models.ScoreDecision.KEEP:
        keywords_to_keep.append(item)
      case models.ScoreDecision.REMOVE:
        keywords_to_remove.append(item)
      case models.ScoreDecision.UNKNOWN:
        keywords_unknown.append(item)

  # Tracks keyword when a user disagrees.
  keyword_feedback_eval = st.session_state.get("keyword_feedback_eval", None)


  def save_human_eval(*, human_eval: models.KeywordEvaluation, llm_eval: models.KeywordEvaluation):
    evaluations[human_eval.keyword] = human_eval
    scored_set.add(human_eval.keyword)
    eval_pairs.append(models.EvaluationPair(llm_decision=llm_eval.decision, human_decision=human_eval.decision))


  def define_handler_scoring(llm_eval: models.KeywordEvaluation, human_agree_with_llm: bool) -> Callable:
    def _inner():
      if not human_agree_with_llm:
        st.session_state.keyword_feedback_eval = models.KeywordEvaluation(
            llm_eval.keyword,
            category=llm_eval.category,
            reason=llm_eval.reason,
            decision=llm_eval.opposite_decision)
        return

      human_eval = models.KeywordEvaluation(
          keyword=llm_eval.keyword,
          category=llm_eval.category,
          decision=llm_eval.decision,
          reason=llm_eval.reason)
      save_human_eval(human_eval=human_eval, llm_eval=llm_eval)
    return _inner


  def handler_cancel_human_eval():
    st.session_state.keyword_feedback_eval = None


  def handler_human_category(event, value):
    keyword_feedback_eval.category = value["props"]["value"]


  def handler_human_decision(event, value):
    keyword_feedback_eval.decision = value["props"]["value"]


  def handler_human_reason(event):
    keyword_feedback_eval.reason = event.target.value


  def define_handler_save_human_eval(llm_eval: models.KeywordEvaluation) -> Callable:
    def _inner():
      human_eval = models.KeywordEvaluation(
          keyword=llm_eval.keyword,
          category=keyword_feedback_eval.category,
          decision=keyword_feedback_eval.decision,
          reason=keyword_feedback_eval.reason)
      save_human_eval(human_eval=human_eval, llm_eval=llm_eval)
    return _inner


  def render_item_card(item: models.KeywordEvaluation):
    with mui.Card(key="first_item", sx={"display": "flex", "flexDirection": "column", "borderRadius": 3}, elevation=1):
      mui.CardHeader(
          title=item.keyword,
          sx={"background": "rgba(255, 255, 255, 0.1)"},
      )

      if keyword_feedback_eval is not None and keyword_feedback_eval.keyword == item.keyword:
        with mui.CardContent(sx={"flex": 1}):
          with mui.Table(), mui.TableBody():
            with mui.TableRow(sx={'&:last-child td, &:last-child th': { 'border': 0 } }):
              with mui.TableCell(component="th", scope="row", sx={'p': 0}):
                mui.Chip(label="Human Category")
              with mui.TableCell(), mui.Select(value=keyword_feedback_eval.category, onChange=handler_human_category):
                for cat in models.ScoreCategory:
                  mui.MenuItem(cat.name, value=cat.value)
            with mui.TableRow(sx={'&:last-child td, &:last-child th': { 'border': 0 } }):
              with mui.TableCell(component="th", scope="row", sx={'p': 0}):
                mui.Chip(label="Human Reason")
              with mui.TableCell():
                mui.TextField(
                    multiline=True,
                    placeholder="Explain your rating (e.g. 'Competitor product')",
                    defaultValue=keyword_feedback_eval.reason,
                    onChange=lazy(handler_human_reason))
            with mui.TableRow(sx={'&:last-child td, &:last-child th': { 'border': 0 } }):
              with mui.TableCell(component="th", scope="row", sx={'p': 0}):
                mui.Chip(label="Human Decision")
              with mui.TableCell(), mui.Select(value=keyword_feedback_eval.decision, onChange=handler_human_decision):
                for dec in models.ScoreDecision:
                  mui.MenuItem(dec.name, value=dec.value)
        with mui.CardActions(disableSpacing=True, sx={"margin-top": "auto"}):
          mui.Button("Save human feedback", color="success", onClick=define_handler_save_human_eval(item), sx={"margin-right": "auto"})
          mui.Button("Cancel", onClick=handler_cancel_human_eval, sx={"color": "#999999"})
      else:
        with mui.CardContent(sx={"flex": 1}):
          with mui.Table(), mui.TableBody():
            with mui.TableRow(sx={'&:last-child td, &:last-child th': { 'border': 0 } }):
              with mui.TableCell(component="th", scope="row", sx={'p': 0}):
                mui.Chip(label="AI Category")
              with mui.TableCell():
                mui.Typography(item.category.name)
            with mui.TableRow(sx={'&:last-child td, &:last-child th': { 'border': 0 } }):
              with mui.TableCell(component="th", scope="row", sx={'p': 0}):
                mui.Chip(label="AI Reason")
              with mui.TableCell():
                mui.Typography(item.reason or "Empty")
        with mui.CardActions(disableSpacing=True, sx={"margin-top": "auto"}):
          mui.Button("Disagree with Student", color="error", onClick=define_handler_scoring(item, human_agree_with_llm=False), sx={"margin-right": "auto"})
          mui.Button("Agree with Student", color="success", onClick=define_handler_scoring(item, human_agree_with_llm=True))


  # Display cards
  with elements("cards"):
    with mui.Grid(container=True):
      with mui.Grid(item=True, xs=True):
        mui.Typography(f"Candidates to remove ({len(keywords_to_remove)})", variant="h4", sx={"mb": 2})
        with mui.Stack(spacing=2, direction="column", useFlexGap=True):
          if not keywords_to_remove:
            mui.Typography("No more.")
          for item in keywords_to_remove:
            render_item_card(item)
      mui.Divider(orientation="vertical", flexItem=True, sx={"mx": 4})
      with mui.Grid(item=True, xs=True):
        mui.Typography(f"Candidates to keep ({len(keywords_to_keep)})", variant="h4", sx={"mb": 2})
        with mui.Stack(spacing=2, direction="column", useFlexGap=True):
          if not keywords_to_keep:
            mui.Typography("No more.")
          for item in keywords_to_keep:
            render_item_card(item)
