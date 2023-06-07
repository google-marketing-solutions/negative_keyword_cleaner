import textwrap

import streamlit as st


def display_faq_component():
    st.markdown(textwrap.dedent("""\
        # FAQ

        ## How does AI Student work?
        We help you build the most efficient prompt to solve your marketing
        maintenance task. Unlocking the full capabilities of LLMs.

        ## Is my data safe?
        Yes, your data is safe. AI Student calls LLMs with a private flag to
        ensure that none of the data is reused to train future versions.
        """))
