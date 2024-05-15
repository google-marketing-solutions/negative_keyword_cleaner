import textwrap

import streamlit as st

from .faq import display_faq_component


def display_sidebar_component():
  with st.sidebar:
    st.markdown(
        textwrap.dedent(
            """\
            # About
            Our AI Student aims at helping you maintain healthy negative keywords.

            Please share your feedback or any suggestion at negatives@google.com 💡

            _Made by Google Ads Professional Services_

            ---
            """
        )
    )

    display_faq_component()
