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
import logging
import os
import re
import pickle
import random
import textwrap
import time
from typing import Callable

from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import VertexAI, OpenAI
from langchain.llms.fake import FakeListLLM
from openai.error import OpenAIError
import pandas as pd
import streamlit as st
from streamlit_elements import dashboard, elements, lazy, mui
from st_oauth import st_oauth
import yaml

from .components.sidebar import display_sidebar_component
from . import models
from utils import auth
from utils.keyword_helper import KeywordHelper

logging.getLogger().setLevel(logging.DEBUG)
logger = logging.Logger(__name__)


SAMPLE_BATCH_SIZE = 15

SCHEMA_EVALUATIONS = {
    "bad keyword": models.KeywordEvaluation(
        "bad keyword",
        decision=models.ScoreDecision.KEEP,
        reason="Keep as a negative, not relevant"),
    "good keyword": models.KeywordEvaluation(
        "good keyword",
        decision=models.ScoreDecision.REMOVE,
        reason="It is safe to target this keyword"),
}

DEBUG_SUMMARY = False
DEBUG_SCORING = False
DEBUG_SCORING_LIMIT = 500  # No limit: -1

URL_REGEX = r"^((http|https)://)[-a-zA-Z0-9@:%._\\+~#?&//=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%._\\+~#?&//=]*)$"


def display_page():
  auth.authenticate_user()

  st.header("AI Student — for Google Ads Neg Keywords")
  st.info("Hi, I am your AI student ready to learn from your client to clean their negative keywords. Let's dive in.", icon="🎓")

  display_sidebar_component()

  if DEBUG_SUMMARY:
    responses = [
        "HTML summary 1",
        "HTML summary 2",
        "HTML summary 3",
        "HTML summary 4",
        #"Renault sells a range of vehicles, including electric, full hybrid, and mild hybrid models, with prices ranging from €11,400 to €61,900. Customers can discover, configure, and compare two models in 3D, and select and discover models based on criteria such as electric, family, city, SUV, and 7+ seats. Renault also offers services such as long-term rental, contract Sérénité Renault, and delivery time. Additionally, they provide services such as MY Renault, service client, FAQ, access for deaf and hard of hearing, ordering attestations, configure, test drive, accessories, original museum & store, renew, mobilize share, garages and concessions.",
        "Nike is a global sportswear and lifestyle brand that sells a variety of products including shoes, apparel, and accessories, as well as services such as Nike App, Nike Run Club, Nike Training Club, SNKRS, and Factory Store. They offer exclusive collections and collaborations with top athletes and influencers, as well as guides on their products and support on order status, shipping and delivery, returns, payment methods, and promo codes. They also provide information about their company, news, careers, investors, sustainability, and legal information."
    ]
    llm = FakeListLLM(responses=responses)
  elif st.session_state.config.google_api_key:
    print("Picked Google PALM model for summarizing")
    os.environ["GOOGLE_API_KEY"] = st.session_state.config.google_api_key
    llm = VertexAI(
        model_name="text-bison",
        temperature=0.2,
        top_p=0.98,
        top_k=40,
        max_output_tokens=1024,
    )
  else:
    print("Picked OpenAI 3.5-turbo model for summarizing")
    llm = OpenAI(
        temperature=0.2,
        max_tokens=1024,
        openai_api_key=st.session_state.config.openai_api_key,
    )


  ##
  # 1. Company Details
  #

  with st.expander("1. Company Details", expanded=st.session_state.get('context_open', True)):
    company_homepage_url = st.text_input(
        "Company Homepage URL",
        placeholder="https://...",
        value="https://nike.com" if DEBUG_SUMMARY else st.session_state.get("company_homepage_url", ""),
    )
    if not company_homepage_url:
      st.info("Once I have their website url, I can directly read and understand "
              "who is this customer", icon="🎓")
      st.stop()
    else:
      st.session_state.company_homepage_url = company_homepage_url

    if len(company_homepage_url) > 0: 
      if not re.match(URL_REGEX, company_homepage_url):
         st.error("Invalid URL") 

    with st.spinner("I'm browsing their website..."):
      homepage_docs = models.fetch_landing_page_text(company_homepage_url)
      if st.session_state.get("homepage_fetched", False):
        if DEBUG_SUMMARY: time.sleep(2)  # Too fast, we can slow it a little bit
        st.session_state.homepage_fetched = True

    st.success("Browsing done, I've collected enough info", icon="🎓")

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
               "here. I'm just a student.", icon="🎓")

    company_pitch = st.text_area(
        "Company Business Pitch",
        placeholder="Describe what the company is selling in a few words",
        value=homepage_summary,
        height=150
    )

    st.info("Happy to know more about what you don't want to target ads for", icon="🎓")

    exclude_pitch = st.text_area(
        "Exclude summary",
        placeholder="Describe what you don't want to target ads for",
        height=50
    )


  def handle_continue_with_context():
    st.session_state.context_ready = True
    st.session_state.context_open = True
    st.session_state.epoch_eval_pairs = []


  if not st.session_state.get('context_ready'):
    st.button("Continue with this context", on_click=handle_continue_with_context)
    st.stop()
  elif st.session_state.get('context_open'):
    st.session_state.context_open = False
    time.sleep(0.05)
    st.rerun()


  ##
  # 2. Loads keywords
  #

  @st.cache_resource(show_spinner=False)
  def load_keywords():
    kw_helper = KeywordHelper(st.session_state.config)
    if not kw_helper:
        st.error("An internal error occurred. Try again later")
        return

    with st.spinner(text='Loading negative keywords... This may take a few minutes'):
        negative_kws_report = kw_helper.get_neg_keywords()
        if not negative_kws_report:
            st.warning("No negative keywords found")
            st.stop()

    negative_kws = kw_helper.clean_and_dedup(negative_kws_report)
    df = pd.DataFrame(
        [(kw.get_clean_keyword_text(), kw.kw_text, kw.campaign_name, kw.campaign_id, kw.adgroup_id) 
        for keywords in negative_kws.values() 
        for kw in keywords],
        columns=['keyword', 'original_keyword', 'campaign_name', 'campaign_id', 'adgroup_id']
    )
    return df


  with st.expander("2. Load negative keywords", expanded=st.session_state.get('load_keywords_open', True)):
    df = load_keywords()
    # st.dataframe(df.sample(5))
    st.success(f"I've loaded {len(df)} negative keywords from all campaigns. Filter only the relevant campaigns!", icon="🎓")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total negative keywords", len(df))
    col2.metric("Total unique keywords", df.keyword.nunique())
    col3.metric("Total campaigns", df.campaign_id.nunique())


  def save_evaluations():
    evaluations = st.session_state.evaluations
    with open(".streamlit/evaluations.json", "w") as fp:
      json.dump(evaluations, fp, cls=EnhancedJSONEncoder, indent=2)


  def load_evaluations():
    with open(".streamlit/evaluations.json", "r") as fp:
      evaluations = json.load(fp, cls=EnhancedJSONDecoder)
      print(f"Loaded #{len(evaluations)} evaluations")
      st.session_state.evaluations = evaluations


  def reset_evaluations():
    st.session_state.evaluations = OrderedDict()


  def reset_batch_props():
    st.session_state.batch_scored_keywords = set()
    st.session_state.keyword_feedback_eval = None


  def handle_selected_campaigns():
    print("Reset evaluations")
    reset_evaluations()
    reset_batch_props()
    st.session_state.scored_keywords = None


  with st.expander("3. Filter on campaigns", expanded=st.session_state.get('filter_campaigns_open', True)):
    st.multiselect(
        "Selected Campaigns",
        df.groupby(['campaign_name'])['keyword'].count().reset_index(name='count').sort_values(["count"], ascending=False),
        [],
        on_change=handle_selected_campaigns,
        key='selected_campaigns',
    )

    df_filtered = df.copy()
    if st.session_state.get('selected_campaigns', None):
      options = st.session_state.selected_campaigns
      # print("Options:", options)
      df_filtered = df_filtered.query("campaign_name in @options")

    col1, col2, col3 = st.columns(3)
    col1.metric("Selected negative keywords", len(df_filtered))
    col2.metric("Selected Unique keywords", df_filtered.keyword.nunique())
    col3.metric("Selected campaigns", df_filtered.campaign_id.nunique())


  def handle_continue_with_filters():
    st.session_state.filters_ready = True


  if not st.session_state.get('filters_ready', False):
    st.button("Continue with these filters", on_click=handle_continue_with_filters)
    st.stop()


  def score_batch_evals():
    # Stores batch eval pairs.
    current_batch_eval_pairs = st.session_state.get("batch_eval_pairs", None)
    if current_batch_eval_pairs:
      epoch_eval_pairs = st.session_state.epoch_eval_pairs
      epoch_eval_pairs.append(current_batch_eval_pairs)
      st.session_state.batch_eval_pairs: list[models.EvaluationPair] = list()


  def handle_sample_batch():
    # Resets variables.
    st.session_state.sample_new_batch = True
    st.session_state.load_keywords_open = True
    st.session_state.scored_keywords = None
    st.session_state.random_state = models.get_random_state(force_new=True)
    reset_batch_props()
    score_batch_evals()


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


  if st.session_state.get('load_keywords_open'):
    st.session_state.load_keywords_open = False
    time.sleep(0.05)
    st.experimental_rerun()


  ##
  # 3. Samples and Scores the sampled batch.
  #

  if "evaluations" not in st.session_state:
    reset_evaluations()

  evaluations = st.session_state.evaluations

  if not st.session_state.get('stop_training'):
    random_state = st.session_state.get("random_state", models.get_random_state())
    df_keywords = models.sample_batch(
        df_filtered,
        batch_size=SAMPLE_BATCH_SIZE,
        exclude_keywords=set(evaluations.keys()),
        random_state=random_state)

    # col1, col2 = st.columns(2)
    # with col1:
    #   st.button("Save evaluations", on_click=save_evaluations)
    # with col2:
    #   st.button("Load evaluations", on_click=load_evaluations)

    formatted_facts = models.format_scoring_fragment(st.session_state.evaluations or SCHEMA_EVALUATIONS)
    formatted_keywords = yaml.dump(df_keywords['keyword'].tolist(), allow_unicode=True)

    template = textwrap.dedent("""\
        You are an agent working for a Google Ads agency asked to score keyword against .

        {company_segment}

        Learn from these examples scored by an expert, formatted as YAML output, especially learn from the reason column:

        {facts_segment}

        The category field can only take one of the following values: {category_allowed_values}.
        The decision field can only take one of the following values: {decision_allowed_values}.
        The reason field is a free form string that explains why it picked this category and decision.

        The decision to keep a keyword means that this keyword should be excluded from targeting.
        The decision to remove a keyword means that we want to target this keyword.

        Given this context and examples, score this new list of keywords with relevancy and add a detailed reason why you scored that way inspired by the reason used in our examples above formatted as valid YAML output:

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
                reason: It is safe to target this keyword as it is related to a Renault product
                decision: remove
              - keyword: "location v\xE9hicule utilitaire"
                reason: Can target this generic query
                decision: remove
              - keyword: acheter une renault captur hybride
                reason: Safe to target hybrid models
                decision: remove
              - keyword: accessoires renault trafic
                reason: We don't want to sell accessories
                decision: keep
              - keyword: essai citadine
                reason: Safe to target generic model queries
                decision: remove
              """)
          ])
    elif st.session_state.config.google_api_key:
      print("Picked Google PALM model for scoring")
      os.environ["GOOGLE_API_KEY"] = st.session_state.config.google_api_key
      scoring_llm = VertexAI(
          model_name="code-bison",
          temperature=0.1,
          top_p=0.98,
          top_k=40,
          max_output_tokens=2048,
      )
    else:
      print("Picked OpenAI 3.5-turbo model for scoring")
      scoring_llm = OpenAI(
          model_name="gpt-3.5-turbo",  # Defaults to "text-davinci-003" (not an instruction-tuned model).
          temperature=0.1,
          max_tokens=2048,
          openai_api_key=st.session_state.config.openai_api_key,
      )

    scored_keywords = st.session_state.get("scored_keywords", None)
    if not scored_keywords:
      session_state.load_keywords_open = False
      with st.spinner("Scoring this batch of keywords..."):
        llm_chain = LLMChain(prompt=prompt, llm=scoring_llm, verbose=True)
        try:
          scored_keywords = llm_chain.run({
              "company_segment": "\n\n".join(filter(None, [company_pitch, exclude_pitch])),
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

    logger.error(scored_keywords)
    parsed_scored_keywords = models.parse_scoring_response(scored_keywords)
    if "batch_scored_keywords" not in st.session_state:
      st.session_state.batch_scored_keywords = set()
    if "batch_eval_pairs" not in st.session_state:
      st.session_state.batch_eval_pairs = list()
    scored_set = st.session_state.get("batch_scored_keywords", set())
    eval_pairs = st.session_state.get("batch_eval_pairs", list())

    # st.header("Teach me")
    if not evaluations:
      st.info("Help me improve my knowledge by correcting my following guesses. I will get better over time.", icon="🎓")
    else:
      st.success(f"I've learned from {len(evaluations)} human evaluations. Keep on correcting me to improve my accuracy!", icon="🎓")

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
              #category=llm_eval.category,
              reason=llm_eval.reason,
              decision=llm_eval.opposite_decision)
          return

        human_eval = models.KeywordEvaluation(
            keyword=llm_eval.keyword,
            #category=llm_eval.category,
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
            #category=keyword_feedback_eval.category,
            decision=keyword_feedback_eval.decision,
            reason=keyword_feedback_eval.reason)
        save_human_eval(human_eval=human_eval, llm_eval=llm_eval)
      return _inner


    def render_item_card(item: models.KeywordEvaluation):
      kw_lines = df_keywords.loc[df_keywords.keyword == item.keyword]
      kw_campaigns = kw_lines.campaign_name.tolist()

      with mui.Card(key="first_item", sx={"display": "flex", "flexDirection": "column", "borderRadius": 3}, elevation=1):
        mui.CardHeader(
            title=item.keyword,
            titleTypographyProps={"variant": "h6"},
            sx={"background": "rgba(250, 250, 250, 0.1)"},
        )

        if keyword_feedback_eval is not None and keyword_feedback_eval.keyword == item.keyword:
          with mui.CardContent(sx={"flex": 1, "pt": 0, "pb": 0}):
            with mui.Table(), mui.TableBody():
              # with mui.TableRow(sx={'&:last-child td, &:last-child th': { 'border': 0 } }):
              #   with mui.TableCell(component="th", scope="row", sx={'p': 0}):
              #     mui.Chip(label="Human Category")
              #   with mui.TableCell(), mui.Select(value=keyword_feedback_eval.category, onChange=handler_human_category):
              #     for cat in models.ScoreCategory:
              #       mui.MenuItem(cat.name, value=cat.value)
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
          with mui.CardContent(sx={"flex": 1, "pt": 0, "pb": 0}):
            with mui.Table(), mui.TableBody():
              with mui.TableRow(sx={'&:last-child td, &:last-child th': { 'border': 0 } }):
                with mui.TableCell(component="th", scope="row", sx={'p': 0}):
                  mui.Chip(label=f"{len(kw_campaigns)} Campaign{'s' if len(kw_campaigns) > 1 else ''}")
                with mui.TableCell():
                  mui.Typography(",".join(kw_campaigns))
              with mui.TableRow(sx={'&:last-child td, &:last-child th': { 'border': 0 } }):
                with mui.TableCell(component="th", scope="row", sx={'p': 0}):
                  mui.Chip(label="AI Reason")
                with mui.TableCell():
                  mui.Typography(item.reason or "Empty", paragraph=True)
          with mui.CardActions(disableSpacing=True, sx={"margin-top": "auto"}):
            mui.Button("Disagree with Student", color="error", onClick=define_handler_scoring(item, human_agree_with_llm=False), sx={"margin-right": "auto"})
            mui.Button("Agree with Student", color="success", onClick=define_handler_scoring(item, human_agree_with_llm=True))


    # Display cards
    if keywords_to_remove or keywords_to_keep:
      with elements("cards"):
        with mui.Grid(container=True):
          with mui.Grid(item=True, xs=True):
            mui.Typography(f"Candidates to remove ({len(keywords_to_remove)})", variant="h5", sx={"mb": 2})
            with mui.Stack(spacing=2, direction="column", useFlexGap=True):
              if not keywords_to_remove:
                mui.Typography("No more.")
              for item in keywords_to_remove:
                render_item_card(item)
          mui.Divider(orientation="vertical", flexItem=True, sx={"mx": 4})
          with mui.Grid(item=True, xs=True):
            mui.Typography(f"Candidates to keep ({len(keywords_to_keep)})", variant="h5", sx={"mb": 2})
            with mui.Stack(spacing=2, direction="column", useFlexGap=True):
              if not keywords_to_keep:
                mui.Typography("No more.")
              for item in keywords_to_keep:
                render_item_card(item)
    else:
      # Scores the batch if needed.
      score_batch_evals()

      # Computes each batch evaluation accuracy.
      epoch_eval_pairs = st.session_state.get("epoch_eval_pairs", [])
      epoch_accurracies = []
      for eval_pair in epoch_eval_pairs:
        accuracy = sum(p.llm_decision == p.human_decision for p in eval_pair) / len(eval_pair)
        epoch_accurracies.append(accuracy)
      print("epoch_accurracies:", epoch_accurracies)

      col3, col4 = st.columns(2)
      with col3:
        if epoch_accurracies:
          if len(epoch_accurracies) > 1:
            delta_accuracy = (epoch_accurracies[-1] - epoch_accurracies[-2]) / epoch_accurracies[-2]
            delta_accuracy = f"{delta_accuracy:.0%}"
          else:
            delta_accuracy = None
          st.metric(label="Accuracy (last batch)", value=f"{epoch_accurracies[-1]:.0%}", delta=delta_accuracy)
      # with col1:
      #   st.button("Reset", on_click=reset_batch_props)
      # with col2:
      #   if epoch_accurracies:
      #     st.pyplot(models.sparkline(epoch_accurracies, ylim=[0, max(epoch_accurracies)*1.1]), transparent=True)
      with col4:
        batch_count = len(scored_set)
        batch_size = len(parsed_scored_keywords)
        st.progress(batch_count / batch_size, text=f"Batch completion {batch_count:d}/{batch_size:d}")

      with elements("placeholder"):
        mui.Typography("You can fetch a new batch to improve the accuracy of the Student", align="center", sx={"mt": 2})
      st.button("Sample a new batch", on_click=handle_sample_batch)


  ##
  # 4. Download scored keywords
  #

  st.header("Download negative keywords to remove")
  st.markdown("Identified by the AI Student **and confirmed by a human expert**!")

  stop_training = st.button(
      "Stop the training",
      key="stop_training"
  )
  if stop_training:
    keyword_scored = []
    # for kw, human_eval in evaluations.items():
    #   if human_eval.should_remove():
    #     keyword_scored.append(kw)
    # df_output = df_filtered.query("keyword in @keyword_scored")
    formatted_evals = [
        {
            "keyword": kw,
            "original_keyword": df_filtered.loc[df_filtered['keyword'] == kw, 'original_keyword'].values[0],
            "human_decision": human_eval.decision,
            "human_reason": human_eval.reason,
            "campaign_name": df_filtered.loc[df_filtered['keyword'] == kw, 'campaign_name'].values[0],
            "campaign_id": df_filtered.loc[df_filtered['keyword'] == kw, 'campaign_id'].values[0],
            "adgroup_id": df_filtered.loc[df_filtered['keyword'] == kw, 'adgroup_id'].values[0]
        }
        for kw, human_eval in evaluations.items()
        if kw in df_filtered['keyword'].values
    ]
    df_output = pd.DataFrame(formatted_evals)

    st.dataframe(
      df_output,
      height=200,
      column_config={
        "campaign_id": st.column_config.TextColumn("campaign_id"),
        "adgroup_id": st.column_config.TextColumn("adgroup_id")
    })

    st.download_button(
        "Download fine-tuning examples", df_output.to_csv(index=False), file_name="negative_keywords_used_to_train_student.csv")

  ##
  # 5. Run the student on the remaining keywords
  #

  count_remaining = df_filtered.keyword.nunique() - len(evaluations)
  st.header(f"Score the remaining {count_remaining:,} keywords")
  st.markdown("**Will NOT modify your account**, it's purely a scoring procedure.")

  score_remaining = st.button(
      "Score remaining keywords",
      key="score_remaining"
  )
  if score_remaining:
    scoring_progress_text = "Your fine-tuned AI-Student is now scoring the remaining keywords..."
    scoring_bar = st.progress(0, text=scoring_progress_text)
    scoring_seen_kws = set(evaluations.keys())
    scoring_kws_evals = list()

    if DEBUG_SCORING_LIMIT > 0:
      df_to_score = df_filtered.iloc[:DEBUG_SCORING_LIMIT]
    else:
      df_to_score = df_filtered

    while True:
      random_state = st.session_state.get("random_state", models.get_random_state())
      df_keywords = models.sample_batch(
          df_to_score,
          batch_size=50,
          exclude_keywords=scoring_seen_kws,
          random_state=random_state)
      if len(df_keywords) == 0:
        scoring_bar.progress(1.0, text="")
        st.markdown("Done ✅")
        break

      formatted_facts = models.format_scoring_fragment(st.session_state.evaluations or SCHEMA_EVALUATIONS)
      formatted_keywords = yaml.dump(df_keywords['keyword'].tolist(), allow_unicode=True)
      llm_chain = LLMChain(prompt=prompt, llm=scoring_llm, verbose=True)
      try:
        latest_scored_keywords = llm_chain.run({
            "company_segment": "\n\n".join(filter(None, [company_pitch, exclude_pitch])),
            "facts_segment": formatted_facts,
            "keywords_segment": formatted_keywords,
            "category_allowed_values": ", ".join(x.value for x in models.ScoreCategory),
            "decision_allowed_values": ", ".join(x.value for x in models.ScoreDecision),
        })
        logger.warning(latest_scored_keywords)
      # TODO(dulacp): catch the same exception for PALM 2
      except OpenAIError as inst:
        st.error(f"Failed to run OpenAI LLM due to error: {inst}")
        st.stop()

      # Parses the results.
      try:
        parsed_scored_keywords = models.parse_scoring_response(latest_scored_keywords)
      except yaml.scanner.ScannerError as inst:
        # Skips this failed batch.
        logger.error(f"Failed batch with error: {inst}")
        continue

      scoring_kws_evals.extend(parsed_scored_keywords)

      # Marks them as seen.
      scoring_seen_kws.update(df_keywords['keyword'].tolist())

      # Updates the progress bar
      curr = len(scoring_kws_evals)
      N = len(df_to_score)
      scoring_bar.progress(curr/N, text=scoring_progress_text + f" {curr}/{N}")

    # Keeps the results in cache
    if scoring_kws_evals:
      st.session_state.scoring_kws_evals = scoring_kws_evals

  # Displays the download button if a scoring has been run.
  if st.session_state.get("scoring_kws_evals", None):
    cached_scoring_kws_evals = st.session_state["scoring_kws_evals"]

    # Prepares the keywords to remove for download.
    formatted_evals_to_remove = [
        {
            "keyword": student_eval.keyword,
            "student_decision": student_eval.decision,
            "student_reason": student_eval.reason,
            "original_keyword": df_entry.original_keyword,
            "campaign_name": df_entry.campaign_name,
            "campaign_id": df_entry.campaign_id,
            "adgroup_id": df_entry.adgroup_id,
        }
        for student_eval in cached_scoring_kws_evals
        if student_eval.decision == models.ScoreDecision.REMOVE
        for df_entry in df_filtered.itertuples()
        if df_entry.keyword == student_eval.keyword
    ]
    formatted_evals_to_keep = [
        {"keyword": student_eval.keyword, "student_decision": student_eval.decision, "student_reason": student_eval.reason}
        for student_eval in cached_scoring_kws_evals if student_eval.decision == models.ScoreDecision.KEEP
    ]
    df_to_remove = pd.DataFrame(formatted_evals_to_remove)
    df_to_keep = pd.DataFrame(formatted_evals_to_keep)
    st.dataframe(df_to_remove, height=200)
    st.download_button(
        "Download keywords to remove found by Student",
        df_to_remove.to_csv(index=False), file_name="negative_keywords_to_remove.csv"
    )
    st.dataframe(df_to_keep, height=200)
    st.download_button(
        "Download keywords to keep with reason written by Student",
        df_to_keep.to_csv(index=False), file_name="negative_keywords_to_keep.csv"
    )
