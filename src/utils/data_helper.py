import pandas as pd
import streamlit as st

from utils.keyword_helper import KeywordHelper, Customer


@st.cache_resource(show_spinner=False)
def load_keywords(selected_customers: list) -> pd.DataFrame:
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
    st.error("An internal error occurred. Could not load KeywordHelper.")
    return

  with st.spinner(text="Loading negative keywords... This may take a few minutes"):
    clean_selected_customers = [s.replace("-", "") for s in selected_customers]
    negative_kws_report = kw_helper.get_neg_keywords(clean_selected_customers)
    if not negative_kws_report:
      st.warning("No negative keywords found")
      st.stop()

  negative_kws = kw_helper.clean_and_dedup(negative_kws_report)
  negative_keyword_list = pd.DataFrame(
      [
          (
              kw.get_clean_keyword_text(),
              kw.kw_text,
              kw.match_type,
              kw.campaign_name,
              kw.campaign_id,
              kw.adgroup_id,
              kw.adgroup_name,
              kw.account_id,
              kw.account_name,
          )
          for keywords in negative_kws.values()
          for kw in keywords
      ],
      columns=[
          "keyword",
          "original_keyword",
          "match_type",
          "campaign_name",
          "campaign_id",
          "adgroup_id",
          "adgroup_name",
          "account_id",
          "account_name",
      ],
  )
  return negative_keyword_list


@st.cache_resource(show_spinner=False)
def load_customers() -> pd.DataFrame:
  """
  Load customers using the KeywordHelper.

  This function retrieves a list of customers from KeywordHelper and
  returns them in a DataFrame.

  Returns:
  pd.DataFrame: A DataFrame containing customer IDs and names.
  """
  kw_helper = KeywordHelper(st.session_state.config)
  if not kw_helper:
    st.error("An internal error occurred. Could not load KeywordHelper.")
    return

  with st.spinner(text="Loading customers under MCC, please wait..."):
    customers_report = kw_helper.get_customers()
    if not customers_report:
      st.warning("No Customers found under MCC.")
      st.stop()

    all_customers = []
    for customer in customers_report:
      customer_obj = Customer(
          customer.customer_client_id, customer.customer_client_descriptive_name
      )
      all_customers.append(customer_obj)

    customer_data = pd.DataFrame(
        [(cust.customer_id, cust.customer_name) for cust in all_customers],
        columns=["customer_id", "customer_name"],
    )

  return customer_data
