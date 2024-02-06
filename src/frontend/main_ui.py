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

from collections import OrderedDict
import dataclasses
import json
import logging
import os
import re
import textwrap
import time
from typing import Callable

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_vertexai import VertexAI
from langchain_community.llms import OpenAI, BaseLLM
from langchain.docstore.document import Document
from openai.error import OpenAIError
import pandas as pd
import streamlit as st
from streamlit_elements import elements, lazy, mui
import yaml

from .components.sidebar import display_sidebar_component
from . import models
from utils import auth
from utils.keyword_helper import KeywordHelper, Customer

logging.getLogger().setLevel(logging.DEBUG)
logger = logging.Logger(__name__)

_SCHEMA_EVALUATIONS = {
    "bad keyword": models.KeywordEvaluation(
        "bad keyword",
        decision=models.ScoreDecision.KEEP,
        reason="Irrelevant because ..."),
    "good keyword": models.KeywordEvaluation(
        "good keyword",
        decision=models.ScoreDecision.REMOVE,
        reason="Relevant because ..."),
}

_DEBUG_SCORING_LIMIT = -1  # No limit: -1

_URL_REGEX = r"^((http|https)://)[-a-zA-Z0-9@:%._\\+~#?&//=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%._\\+~#?&//=]*)$"


def display_page() -> None:
    st.header("Welcome to Negative Keyword Cleaner")
    st.info(
        "I am your AI student, ready to learn from your account to identify "
        "negative keywords that block relevant ads.",
        icon="🧑‍🎓")

    auth.authenticate_user()

    display_sidebar_component()

    if st.session_state.config.google_api_key and st.session_state.config.gemini_enabled:
        print("Picked Gemini Pro model for summarizing")
        llm = VertexAI(
            model="gemini-pro"
        )
    elif st.session_state.config.google_api_key and not st.session_state.config.gemini_enabled:
        print("Picked Google PaLM 2 model for summarizing")
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

    with st.expander("1. Advertiser Information",
                     expanded=st.session_state.get('context_open', True)):
        company_homepage_url = st.text_input(
            "Company Homepage URL",
            placeholder="https://...",
            value=st.session_state.get("company_homepage_url", ""),
        )

        if not company_homepage_url:
            st.info(
                "Once I have their website URL, I can directly read and "
                "understand who this customer is.", icon="🧑‍🎓")
            st.stop()
        else:
            st.session_state.company_homepage_url = company_homepage_url

        if len(company_homepage_url) > 0:
            if not re.match(_URL_REGEX, company_homepage_url):
                st.error(
                    "The URL is not in a valid format. Please check it and "
                    "try again.")
                st.stop()

        with st.spinner("I'm browsing their website..."):
            homepage_docs = models.fetch_landing_page_text(
                company_homepage_url)
            if st.session_state.get("homepage_fetched", False):
                st.session_state.homepage_fetched = True

        st.success("Browsing done, I've collected enough info", icon="🧑‍🎓")

        with st.spinner(
                "I'm now consolidating everything into an executive summary "
                "(this will take a minute) ..."):
            if not st.session_state.get("homepage_summary", None):
                homepage_summary = models.summarize_text(homepage_docs, llm,
                                                         verbose=True).strip()
                st.session_state.homepage_summary = homepage_summary
            else:
                homepage_summary = st.session_state.homepage_summary

        st.success(
            "Summarizing done but feel free to correct anything that I've "
            "written. "
            , icon="🎓")

        company_pitch = st.text_area(
            "✅ [Positive prompt] Advertiser's executive summary",
            help="You can add campaign information below",
            placeholder="Describe what the company is selling in a few words",
            value=homepage_summary,
            height=150
        )

        st.info(
            "Happy to know more about what you don't want to target ads for",
            icon="🧑‍🎓")

        exclude_pitch = st.text_area(
            "❌ [Negative prompt] Exclude summary",
            placeholder="Describe what you don't want to target ads for",
            height=50
        )

    def _handle_continue_with_context():
        """
        Sets the session state for context continuation.
        """
        st.session_state.context_ready = True
        st.session_state.context_open = False
        st.session_state.epoch_eval_pairs = []

    if not st.session_state.get('context_ready'):
        st.button("Continue with this context",
                  on_click=_handle_continue_with_context)
        st.stop()
    elif st.session_state.get('context_open'):
        st.session_state.context_open = False
        st.rerun()

    ##
    # 2. Loads keywords
    #

    @st.cache_resource(show_spinner=False)
    def _load_keywords(selected_customers: list) -> pd.DataFrame:
        """
        Load negative keywords based on the selected customers.

        This function retrieves negative keywords using the KeywordHelper class,
        cleans and deduplicates them, and returns them in a DataFrame.

        Parameters:
        selected_customers (list): A list of selected customer identifiers.

        Returns:
        pd.DataFrame: A DataFrame containing the negative keywords along with additional information.
        """
        kw_helper = KeywordHelper(st.session_state.config)

        if not kw_helper:
            st.error(
                "An internal error occurred. Could not load KeywordHelper.")
            return

        with st.spinner(
                text='Loading negative keywords... This may take a few minutes'):
            clean_selected_customers = [s.replace("-", "") for s in
                                        selected_customers]
            negative_kws_report = kw_helper.get_neg_keywords(
                clean_selected_customers)
            if not negative_kws_report:
                st.warning("No negative keywords found")
                st.stop()

        negative_kws = kw_helper.clean_and_dedup(negative_kws_report)
        negative_keyword_list = pd.DataFrame(
            [(kw.get_clean_keyword_text(), kw.kw_text, kw.match_type,
              kw.campaign_name, kw.campaign_id, kw.adgroup_id, kw.adgroup_name,
              kw.account_id, kw.account_name)
             for keywords in negative_kws.values()
             for kw in keywords],
            columns=['keyword', 'original_keyword', 'match_type',
                     'campaign_name', 'campaign_id', 'adgroup_id',
                     'adgroup_name', 'account_id', 'account_name']
        )
        return negative_keyword_list

    @st.cache_resource(show_spinner=False)
    def _load_customers() -> pd.DataFrame:
        """
        Load customers using the KeywordHelper.

        This function retrieves a list of customers from KeywordHelper and
        returns them in a DataFrame.

        Returns:
        pd.DataFrame: A DataFrame containing customer IDs and names.
        """
        kw_helper = KeywordHelper(st.session_state.config)
        if not kw_helper:
            st.error(
                "An internal error occurred. Could not load KeywordHelper.")
            return

        with st.spinner(text="Loading customers under MCC, please wait..."):
            customers_report = kw_helper.get_customers()
            if not customers_report:
                st.warning("No Customers found under MCC.")
                st.stop()

            all_customers = []
            for customer in customers_report:
                customer_obj = Customer(customer.customer_client_id,
                                        customer.customer_client_descriptive_name)
                all_customers.append(customer_obj)

            customer_data = pd.DataFrame(
                [(cust.customer_id, cust.customer_name)
                 for cust in all_customers],
                columns=['customer_id', 'customer_name'])

        return customer_data

    def _reset_batch_props() -> None:
        """
        Reset the properties related to batch scoring in the session state.
        """
        st.session_state.batch_scored_keywords = set()
        st.session_state.keyword_feedback_eval = None

    def _handle_selected_customers() -> None:
        """
        Handle the event when customers are selected by resetting batch properties
        and clearing scored keywords.
        """
        _reset_batch_props()
        st.session_state.scored_keywords = None

    def _handle_selected_campaigns() -> None:
        """
        Handle the event when campaigns are selected by resetting batch properties
        and clearing scored keywords.
        """
        _reset_batch_props()
        st.session_state.scored_keywords = None

    def _handle_continue_with_customers() -> None:
        """
        Handle the continuation with the selected customers by setting the customers
        as ready and closing the customer loader.
        """
        st.session_state.customers_ready = True
        st.session_state.load_customers_open = False

    def _format_customer_id(cid: int) -> str:
        """
        Format the customer ID into a more readable string format.

        Parameters:
        cid (int): The customer ID to be formatted.

        Returns:
        str: The formatted customer ID.
        """
        str_cid = str(cid)
        return f"{str_cid[:3]}-{str_cid[3:6]}-{str_cid[6:]}"

    with st.expander("2. Load Customers",
                     expanded=st.session_state.get('load_customers_open',
                                                   True)):
        df = _load_customers()
        selected_customers = st.multiselect(
            "Selected Customers",
            df['customer_id'].apply(_format_customer_id) + " | " + df[
                'customer_name'].astype(str),
            [],
            on_change=_handle_selected_customers,
            key='selected_customers',
        )

        if not st.session_state.get('customers_ready', False):
            st.button("Continue with these customers",
                      on_click=_handle_continue_with_customers)
            st.stop()

    with st.expander("3. Load negative keywords",
                     expanded=st.session_state.get('load_keywords_open',
                                                   True)):
        df = _load_keywords(selected_customers).query('keyword != ""')
        number_of_neg_kw: str = "{:.0f}".format(len(df))
        st.success(
            f"I've loaded {number_of_neg_kw} negative keywords from all campaigns. Filter only the relevant campaigns!",
            icon="🎓")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total negative keywords",
                    "{:.0f}".format(len(df)).replace(",", " "))
        col2.metric("Total unique keywords",
                    "{:.0f}".format(df.keyword.nunique()).replace(",", " "))
        col3.metric("Total campaigns",
                    "{:.0f}".format(df.campaign_id.nunique()).replace(",",
                                                                      " "))

    with st.expander("4. Filter on campaigns",
                     expanded=st.session_state.get('filter_campaigns_open',
                                                   True)):
        st.multiselect(
            "Selected Campaigns",
            df.groupby(['campaign_name'])['keyword'].count().reset_index(
                name='count').sort_values(["count"], ascending=True),
            [],
            on_change=_handle_selected_campaigns,
            key='selected_campaigns',
        )

        df_filtered = df.copy()
        if st.session_state.get('selected_campaigns', None):
            options = st.session_state.selected_campaigns
            df_filtered = df_filtered.query("campaign_name in @options")

        col1, col2, col3 = st.columns(3)
        col1.metric("Selected negative keywords",
                    "{:.0f}".format(len(df_filtered)).replace(",", " "))
        col2.metric("Selected Unique keywords",
                    "{:.0f}".format(df_filtered.keyword.nunique()).replace(
                        ",", " "))
        col3.metric("Selected campaigns", "{:.0f}".format(
            df_filtered.campaign_id.nunique()).replace(",", " "))

    def _handle_continue_with_filters() -> None:
        """
        Set the session state to indicate that filters are ready.
        """
        st.session_state.filters_ready = True

    if not st.session_state.get('filters_ready', False):
        st.button("Continue with these filters",
                  on_click=_handle_continue_with_filters)
        st.stop()

    def _score_batch_evals() -> None:
        """
        Score the batch evaluations. Adds the current batch evaluation pairs to the
        epoch evaluation pairs and resets the batch evaluation pairs.
        """
        # Stores batch eval pairs.
        current_batch_eval_pairs = st.session_state.get("batch_eval_pairs",
                                                        None)
        if current_batch_eval_pairs:
            epoch_eval_pairs = st.session_state.epoch_eval_pairs
            epoch_eval_pairs.append(current_batch_eval_pairs)
            st.session_state.batch_eval_pairs: list[
                models.EvaluationPair] = list()

    def _handle_sample_batch() -> None:
        """
        Handle the sampling of a new batch by resetting relevant variables and
        scoring batch evaluations.
        """
        # Resets variables.
        st.session_state.sample_new_batch = True
        st.session_state.load_keywords_open = True
        st.session_state.scored_keywords = None
        st.session_state.random_state = models.get_random_state(force_new=True)
        _reset_batch_props()
        _score_batch_evals()

    class EnhancedJSONEncoder(json.JSONEncoder):
        """
        A JSON encoder class that handles the serialization of dataclass objects.
        """

        def default(self, o):
            if dataclasses.is_dataclass(o):
                return dataclasses.asdict(o)
            return super().default(o)

    class EnhancedJSONDecoder(json.JSONDecoder):
        """
        A JSON decoder class that includes custom decoding logic for specific types
        of objects.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(object_hook=self._object_hook, *args, **kwargs)

        def _object_hook(self, data):
            if 'keyword' in data:
                return models.KeywordEvaluation.from_dict(data)
            return data

    if st.session_state.get('load_keywords_open'):
        st.session_state.load_keywords_open = False
        time.sleep(0.05)
        st.rerun()

    ##
    # 3. Samples and Scores the sampled batch.
    #
    def _reset_evaluations() -> None:
        """
        Reset the evaluations in the session state by initializing it to an empty OrderedDict.
        This is used to clear any existing evaluations stored in the session state.
        """
        st.session_state.evaluations = OrderedDict()

    if "evaluations" not in st.session_state:
        _reset_evaluations()

    evaluations = st.session_state.evaluations

    if not st.session_state.get('stop_training'):
        random_state = st.session_state.get("random_state",
                                            models.get_random_state())
        df_keywords = models.sample_batch(
            df_filtered,
            batch_size=st.session_state.batch_size,
            exclude_keywords=set(evaluations.keys()),
            random_state=random_state)

        formatted_facts = models.format_scoring_fragment(
            st.session_state.evaluations or _SCHEMA_EVALUATIONS)
        formatted_keywords = yaml.dump(df_keywords['keyword'].tolist(),
                                       allow_unicode=True)

        template = textwrap.dedent("""\
        You are an agent working for an agency that optimizes Google Ads campaigns. 
        Your job is to evaluate if keywords set as "negative targeting" from an Advertiser's Google Ads account are relevant or not.
        To do that, you first need to take into consideration the following advertiser's business context:

        {company_segment}

        Then you need to learn from the following examples scored by an expert, formatted as YAML, especially learn from the reason column:

        {facts_segment}

        The "decision" field can only take one of the following values: {decision_allowed_values}.
        The "reason" field is a free form string where you can explain in details why you decided to KEEP or REMOVE a keyword.

        The "decision" to KEEP a keyword means that this keyword should be excluded from targeting.
        The "decision" to REMOVE a keyword means that we want to target this keyword.

        Given the above context and examples, score all keywords from the list below for each and every element of this list.
        Format the output in the same way as the above example and as valid YAML according to YAML specifications.
        Here is the list of keywords:
        
        {keywords_segment}
        """)
        prompt = PromptTemplate(
            template=template,
            input_variables=["company_segment", "facts_segment",
                             "keywords_segment", "decision_allowed_values"],
        )

        if st.session_state.config.google_api_key and st.session_state.config.gemini_enabled:
            print("Picked Gemini Pro model for scoring")
            os.environ[
                "GOOGLE_API_KEY"] = st.session_state.config.google_api_key
            scoring_llm = VertexAI(
                model="gemini-pro"
            )
        elif st.session_state and not st.session_state.config.gemini_enabled:
            print("Picked Google PaLM model for scoring")
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
                model_name="gpt-3.5-turbo",
                temperature=0.1,
                max_tokens=2048,
                openai_api_key=st.session_state.config.openai_api_key,
            )

        scored_keywords = st.session_state.get("scored_keywords", None)
        if not scored_keywords:
            with st.spinner("Scoring this batch of keywords..."):
                llm_chain = LLMChain(prompt=prompt, llm=scoring_llm,
                                     verbose=True)
                print(llm_chain, company_pitch, exclude_pitch, formatted_facts,
                      formatted_keywords)
                try:
                    scored_keywords = llm_chain.run({
                        "company_segment": "\n\n".join(
                            filter(None, [company_pitch, exclude_pitch])),
                        "facts_segment": formatted_facts,
                        "keywords_segment": formatted_keywords,
                        "decision_allowed_values": ", ".join(
                            x.value for x in models.ScoreDecision),
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
            st.info(
                "Help me improve my knowledge by correcting my following "
                "guesses. I will get better over time.",
                icon="🎓")
        else:
            st.success(
                f"I've learned from {len(evaluations)} human evaluations. "
                f"Keep on correcting me to improve my accuracy!",
                icon="🎓")

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
        keyword_feedback_eval = st.session_state.get("keyword_feedback_eval",
                                                     None)

        def _save_human_eval(*, human_eval: models.KeywordEvaluation,
                             llm_eval: models.KeywordEvaluation) -> None:
            """
            Save the human evaluation alongside the corresponding LLM evaluation.
            Updates the evaluations dictionary and the scored set with the human evaluation,
            and appends the evaluation pair to the eval_pairs list.

            Parameters:
            human_eval (models.KeywordEvaluation): The human evaluation.
            llm_eval (models.KeywordEvaluation): The LLM (Language Model) evaluation.
            """
            evaluations[human_eval.keyword] = human_eval
            scored_set.add(human_eval.keyword)
            eval_pairs.append(
                models.EvaluationPair(llm_decision=llm_eval.decision,
                                      human_decision=human_eval.decision))

        def _define_handler_scoring(llm_eval: models.KeywordEvaluation,
                                    human_agree_with_llm: bool) -> Callable:
            """
            Define a handler for scoring, which returns an inner function to handle the logic
            of saving human evaluations based on agreement with LLM evaluations.

            Parameters:
            llm_eval (models.KeywordEvaluation): The LLM (Language Model) evaluation.
            human_agree_with_llm (bool): Indicates if the human agrees with the LLM evaluation.

            Returns:
            Callable: A function that handles the scoring logic.
            """

            def _inner():
                if not human_agree_with_llm:
                    st.session_state.keyword_feedback_eval = models.KeywordEvaluation(
                        llm_eval.keyword,
                        reason=llm_eval.reason,
                        decision=llm_eval.opposite_decision)
                    return

                human_eval = models.KeywordEvaluation(
                    keyword=llm_eval.keyword,
                    decision=llm_eval.decision,
                    reason=llm_eval.reason)
                _save_human_eval(human_eval=human_eval, llm_eval=llm_eval)

            return _inner

        def _handler_cancel_human_eval() -> None:
            """
            Handler for canceling a human evaluation.
            """
            st.session_state.keyword_feedback_eval = None

        def _handler_human_decision(event, value) -> None:
            """
            Handler for setting the human decision in the evaluation.

            Parameters:
            event: The event triggered by the UI component.
            value: The value from the UI component.
            """
            keyword_feedback_eval.decision = value["props"]["value"]

        def _handler_human_reason(event) -> None:
            """
            Handler for setting the human reason in the evaluation.

            Parameters:
            event: The event triggered by the UI component.
            """
            keyword_feedback_eval.reason = event.target.value

        def _define_handler_save_human_eval(
                llm_eval: models.KeywordEvaluation) -> Callable:
            """
            Define a handler for saving human evaluations, which returns an inner function to handle
            the logic of saving human evaluations.

            Parameters:
            llm_eval (models.KeywordEvaluation): The LLM (Language Model) evaluation.

            Returns:
            Callable: A function that handles the saving of human evaluations.
            """

            def _inner():
                human_eval = models.KeywordEvaluation(
                    keyword=llm_eval.keyword,
                    # category=keyword_feedback_eval.category,
                    decision=keyword_feedback_eval.decision,
                    reason=keyword_feedback_eval.reason)
                _save_human_eval(human_eval=human_eval, llm_eval=llm_eval)

            return _inner

        def _render_item_card(item: models.KeywordEvaluation) -> None:
            """
            Render a card UI component for a keyword evaluation item.

            Parameters:
            item (models.KeywordEvaluation): The keyword evaluation to be rendered in the card.
            """
            kw_lines = df_keywords.loc[df_keywords.keyword == item.keyword]
            kw_campaigns = kw_lines.campaign_name.tolist()

            with mui.Card(key="first_item",
                          sx={"display": "flex", "flexDirection": "column",
                              "borderRadius": 3}, elevation=1):
                mui.CardHeader(title=item.keyword,
                               titleTypographyProps={"variant": "h6"},
                               sx={"background": "rgba(250, 250, 250, 0.1)"})

                if keyword_feedback_eval and keyword_feedback_eval.keyword == item.keyword:
                    with mui.CardContent(sx={"flex": 1, "pt": 0, "pb": 0}):
                        with mui.Table(), mui.TableBody():
                            for label, value, onChange in [
                                ("Human Reason", keyword_feedback_eval.reason,
                                 lazy(_handler_human_reason)),
                                ("Human Decision",
                                 keyword_feedback_eval.decision,
                                 _handler_human_decision),
                            ]:
                                with mui.TableRow(sx={
                                    '&:last-child td, &:last-child th': {
                                        'border': 0}}):
                                    with mui.TableCell(component="th",
                                                       scope="row",
                                                       sx={'p': 0}):
                                        mui.Chip(label=label)
                                    with mui.TableCell():
                                        if onChange:
                                            mui.TextField(multiline=True,
                                                          placeholder=f"Explain your rating (e.g. '{label.lower()}')",
                                                          defaultValue=value,
                                                          onChange=onChange)
                                        else:
                                            mui.DisplayText(value)
                        with mui.CardActions(disableSpacing=True,
                                             sx={"margin-top": "auto"}):
                            mui.Button("Save human feedback", color="success",
                                       onClick=_define_handler_save_human_eval(
                                           item), sx={"margin-right": "auto"})
                            mui.Button("Cancel",
                                       onClick=_handler_cancel_human_eval,
                                       sx={"color": "#999999"})


                else:
                    with mui.CardContent(sx={"flex": 1, "pt": 0, "pb": 0}):
                        with mui.Table(), mui.TableBody():
                            campaigns_label = f"{len(kw_campaigns)} Campaign{'s' if len(kw_campaigns) > 1 else ''}"
                            mui.TableRow(sx={
                                '&:last-child td, &:last-child th': {
                                    'border': 0}})(
                                mui.TableCell(component="th", scope="row",
                                              sx={'p': 0})(
                                    mui.Chip(label=campaigns_label)),
                                mui.TableCell()(
                                    mui.Typography(",".join(kw_campaigns),
                                                   noWrap=True))
                            )

                            mui.TableRow(sx={
                                '&:last-child td, &:last-child th': {
                                    'border': 0}})(
                                mui.TableCell(component="th", scope="row",
                                              sx={'p': 0})(
                                    mui.Chip(label="AI Reason")),
                                mui.TableCell()(
                                    mui.Typography(item.reason or "Empty",
                                                   paragraph=True))
                            )

                        with mui.CardActions(disableSpacing=True,
                                             sx={"margin-top": "auto"}):
                            disagree_button = mui.Button(
                                "Disagree with Student", color="error",
                                onClick=_define_handler_scoring(item,
                                                                human_agree_with_llm=False),
                                sx={"margin-right": "auto"})
                            agree_button = mui.Button("Agree with Student",
                                                      color="success",
                                                      onClick=_define_handler_scoring(
                                                          item,
                                                          human_agree_with_llm=True))

        # Display cards
        if keywords_to_remove or keywords_to_keep:
            with elements("cards"):
                with mui.Grid(container=True):
                    with mui.Grid(item=True, xs=True):
                        mui.Typography(
                            f"I think you should target ({len(keywords_to_remove)}):",
                            variant="h5", sx={"mb": 2})
                        with mui.Stack(spacing=2, direction="column",
                                       useFlexGap=True):
                            if not keywords_to_remove:
                                mui.Typography("No more.")
                            for item in keywords_to_remove:
                                _render_item_card(item)
                    mui.Divider(orientation="vertical", flexItem=True,
                                sx={"mx": 4})
                    with mui.Grid(item=True, xs=True):
                        mui.Typography(
                            f"I think you shouldn't target ({len(keywords_to_keep)}):",
                            variant="h5", sx={"mb": 2})
                        with mui.Stack(spacing=2, direction="column",
                                       useFlexGap=True):
                            if not keywords_to_keep:
                                mui.Typography("No more.")
                            for item in keywords_to_keep:
                                _render_item_card(item)
        else:
            # Scores the batch if needed.
            _score_batch_evals()

            # Computes each batch evaluation accuracy.
            epoch_eval_pairs = st.session_state.get("epoch_eval_pairs", [])
            epoch_accurracies = []
            for eval_pair in epoch_eval_pairs:
                accuracy = sum(p.llm_decision == p.human_decision for p in
                               eval_pair) / len(eval_pair)
                epoch_accurracies.append(accuracy)
            print("epoch_accurracies:", epoch_accurracies)

            col3, col4 = st.columns(2)
            with col3:
                if epoch_accurracies:
                    if len(epoch_accurracies) > 1:
                        delta_accuracy = (epoch_accurracies[-1] -
                                          epoch_accurracies[-2]) / \
                                         epoch_accurracies[-2]
                        delta_accuracy = f"{delta_accuracy:.0%}"
                    else:
                        delta_accuracy = None
                    st.metric(label="Accuracy (last batch)",
                              value=f"{epoch_accurracies[-1]:.0%}",
                              delta=delta_accuracy)
            with col4:
                batch_count = len(scored_set)
                batch_size = len(parsed_scored_keywords)
                st.progress(batch_count / batch_size,
                            text=f"Batch completion {batch_count:d}/{batch_size:d}")

            with elements("placeholder"):
                mui.Typography(
                    "You can fetch a new batch to improve the accuracy of the Student",
                    align="center", sx={"mt": 2})
            st.button("Sample a new batch", on_click=_handle_sample_batch)

    ##
    # 4. Run the student on the remaining keywords
    #

    count_remaining = df_filtered.keyword.nunique() - len(evaluations)
    st.header(f"Score the remaining {count_remaining:,} keywords")
    st.markdown(
        "**Will NOT modify your account**, it's purely a scoring procedure.")

    score_remaining = st.button(
        "Score remaining keywords",
        key="score_remaining"
    )
    if score_remaining:
        scoring_progress_text = "Your fine-tuned AI-Student is now scoring " \
                                "the remaining keywords... "
        scoring_bar = st.progress(0, text=scoring_progress_text)
        scoring_seen_kws = set(evaluations.keys())
        scoring_kws_evals = list()

        if _DEBUG_SCORING_LIMIT > 0:
            df_to_score = df_filtered.iloc[:_DEBUG_SCORING_LIMIT]
        else:
            df_to_score = df_filtered

        while True:
            random_state = st.session_state.get("random_state",
                                                models.get_random_state())
            df_keywords = models.sample_batch(
                df_to_score,
                batch_size=50,
                exclude_keywords=scoring_seen_kws,
                random_state=random_state)
            if len(df_keywords) == 0:
                scoring_bar.progress(1.0, text="")
                st.markdown("Done ✅")
                break

            formatted_facts = models.format_scoring_fragment(
                st.session_state.evaluations or _SCHEMA_EVALUATIONS)
            formatted_keywords = yaml.dump(df_keywords['keyword'].tolist(),
                                           allow_unicode=True)
            llm_chain = LLMChain(prompt=prompt, llm=scoring_llm, verbose=True)
            try:
                latest_scored_keywords = llm_chain.run({
                    "company_segment": "\n\n".join(
                        filter(None, [company_pitch, exclude_pitch])),
                    "facts_segment": formatted_facts,
                    "keywords_segment": formatted_keywords,
                    "decision_allowed_values": ", ".join(
                        x.value for x in models.ScoreDecision),
                })
                logger.warning(latest_scored_keywords)
            # TODO(dulacp): catch the same exception for PALM 2
            except OpenAIError as inst:
                st.error(f"Failed to run OpenAI LLM due to error: {inst}")
                st.stop()

            # Parses the results.
            try:
                parsed_scored_keywords = models.parse_scoring_response(
                    latest_scored_keywords)
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
            scoring_bar.progress(curr / N,
                                 text=scoring_progress_text + f" {curr}/{N}")

        # Keeps the results in cache
        if scoring_kws_evals:
            st.session_state.scoring_kws_evals = scoring_kws_evals

    # Displays the download button if a scoring has been run.
    if st.session_state.get("scoring_kws_evals", None):
        cached_scoring_kws_evals = st.session_state["scoring_kws_evals"]
        # Prepares the keywords to remove for download.
        formatted_evals_to_remove = [
            {
                "Account": df_entry.account_name,
                "Campaign": df_entry.campaign_name,
                "Ad Group": df_entry.adgroup_name,
                "Keyword": student_eval.keyword,
                "Criterion Type": (
                                      "Negative " if df_entry.adgroup_name else "Campaign Negative ") +
                                  (
                                      "Broad" if df_entry.match_type == "BROAD" else
                                      "Phrase" if df_entry.match_type == "PHRASE" else
                                      "Exact"),
                "Status": "Removed",
                "Student Reason": student_eval.reason,
            }
            for student_eval in cached_scoring_kws_evals
            if student_eval.decision == models.ScoreDecision.REMOVE
            for df_entry in df_filtered.itertuples()
            if df_entry.keyword == student_eval.keyword and (
                       df_entry.campaign_name or df_entry.adgroup_id)
        ]

        def get_df_values(df, keyword, columns):
            # Fetches values for specified columns based on the keyword
            return {col: df.loc[df['keyword'] == keyword, col].values[0] for
                    col in columns}

        formatted_evals_to_keep = [
            {
                "Account": df_entry.account_name,
                "Campaign": df_entry.campaign_name,
                "Ad Group": df_entry.adgroup_name,
                "Keyword": student_eval.keyword,
                "Criterion Type": ("Broad" if df_entry.match_type == "BROAD"
                                   else "Phrase" if df_entry.match_type == "PHRASE"
                else "Exact"),
                "Status": "Enabled",
                "Student Reason": student_eval.reason,
                **get_df_values(df_filtered, student_eval.keyword,
                                ['original_keyword', 'campaign_id',
                                 'adgroup_id', 'account_id'])
            }
            for student_eval in cached_scoring_kws_evals
            if student_eval.decision == models.ScoreDecision.KEEP
            for df_entry in df_filtered.itertuples()
            if df_entry.keyword == student_eval.keyword and (
                       df_entry.campaign_name or df_entry.adgroup_id)
        ]

        df_to_remove = pd.DataFrame(formatted_evals_to_remove)
        df_to_keep = pd.DataFrame(formatted_evals_to_keep)
        st.dataframe(
            df_to_remove,
            height=200,
            column_config={
                "campaign_id": st.column_config.TextColumn("campaign_id"),
                "adgroup_id": st.column_config.TextColumn("adgroup_id"),
                "account_id": st.column_config.TextColumn("account_id"),
            })
        st.download_button(
            "Download keywords to remove found by Student",
            df_to_remove.to_csv(index=False),
            file_name="negative_keywords_to_remove.csv"
        )
        st.dataframe(
            df_to_keep,
            height=200,
            column_config={
                "campaign_id": st.column_config.TextColumn("campaign_id"),
                "adgroup_id": st.column_config.TextColumn("adgroup_id"),
                "account_id": st.column_config.TextColumn("account_id"),
            })
        st.download_button(
            "Download keywords to keep with reason written by Student",
            df_to_keep.to_csv(index=False),
            file_name="negative_keywords_to_keep.csv"
        )

    ##
    # 5. Download scored keywords
    #

    st.header("Download negative keywords to remove")
    st.markdown(
        "Identified by the AI Student **and confirmed by a human expert**!")

    stop_training = st.button(
        "Stop the training",
        key="stop_training"
    )
    if stop_training:
        keyword_scored = []
        formatted_evals = [
            {
                "keyword": kw,
                "original_keyword": df_filtered.loc[
                    df_filtered['keyword'] == kw, 'original_keyword'].values[
                    0],
                "human_decision": human_eval.decision,
                "human_reason": human_eval.reason,
                "campaign_name": df_filtered.loc[
                    df_filtered['keyword'] == kw, 'campaign_name'].values[0],
                "campaign_id": df_filtered.loc[
                    df_filtered['keyword'] == kw, 'campaign_id'].values[0],
                "adgroup_id": df_filtered.loc[
                    df_filtered['keyword'] == kw, 'adgroup_id'].values[0]
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
            "Download human scorings", df_output.to_csv(index=False),
            file_name="negative_keywords_used_to_train_student.csv")
