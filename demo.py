import streamlit as st
from yext import YextClient
from test_semantic_vertical_ranking import get_new_vertical_ranks

"""
# Vertical Ranking - Embedding Prototype
This app is an interactive demo of the new vertical ranking approach of
embedding the highlighted fields of the top result of each vertical.
"""

YEXT_API_KEY = "01db1d1e5ebbaa7ea2e6807ad2196ab3"
yext_client = YextClient(YEXT_API_KEY)

query = st.text_input("Query")

verticals = [
    "developer_documents",
    "discourse",
    "faqs",
    "guides",
    "help_articles",
    "hh_blog",
    "hh_event",
    "hh_module",
    "hh_track",
    "hh_unit",
    "promotion",
]

st.sidebar.write("## Vertical Boosts")
vertical_boost_map = {}
for vertical in verticals:
    vertical_boost_map[vertical] = st.sidebar.slider(
        label=vertical, value=0.0, min_value=-0.5, max_value=0.5, step=0.05
    )


if query:
    results = yext_client.search_answers_universal(query=query, experience_key="yext_hh_community")
    original_results = results.raw_response
    first_results = [module["results"][0] for module in original_results["response"]["modules"]]

    def get_boosts_vector(original_results):
        """Turns the dictionary of user-defined boosts into a list to be added."""
        ordered_vertical_keys = [
            module["verticalConfigId"] for module in original_results["response"]["modules"]
        ]
        boosts = [vertical_boost_map[key] for key in ordered_vertical_keys]
        return boosts

    boosts_vector = get_boosts_vector(original_results)

    new_ranks, similarities, _ = get_new_vertical_ranks(query, first_results, boosts_vector)

    def _reorder_modules(modules, new_rank):
        return [module for i, module in sorted(zip(new_rank, modules), key=lambda x: x[0])]

    original_modules = original_results["response"]["modules"]
    reordered_modules = _reorder_modules(original_modules, new_ranks)
    left_col, right_col = st.columns(2)

    with left_col:
        st.write("## Original Results")
    with right_col:
        st.write("## Reordered Results")

    for old_rank, new_rank in enumerate(new_ranks):
        original_module = original_results["response"]["modules"][old_rank]
        reordered_module = original_results["response"]["modules"][new_rank]
        delta = new_rank - old_rank

        if delta > 0:
            renderer = st.success
        elif delta < 0:
            renderer = st.error
        elif delta == 0:
            renderer = st.warning

        with left_col:
            st.info(
                f"""
            **Vertical Key:** {original_module['verticalConfigId']}\n
            **Top Result:** {original_module['results'][0]['data']['name']}\n
            **Original Rank:** {old_rank}\n
            """
            )
        with right_col:
            renderer(
                f"""
            **Vertical Key:** {reordered_module['verticalConfigId']}\n
            **Top Result:** {reordered_module['results'][0]['data']['name']}\n
            **Reordered Rank:** {old_rank} (Δ{delta})\n
            **Similarity:** {sorted(similarities, key=lambda x: -x)[old_rank]}
            """
            )
