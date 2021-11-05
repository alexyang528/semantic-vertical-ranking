import streamlit as st
from yext import YextClient
from test_semantic_vertical_ranking import get_new_vertical_ranks, get_liveapi_response

"""
# Vertical Ranking - Embedding Prototype
This app is an interactive demo of the new vertical ranking approach of
embedding the highlighted fields of the top result of each vertical.
"""

YEXT_API_KEY = st.text_input("API Key")
EXPERIENCE_KEY = st.text_input("Experience Key")
QUERY = st.text_input("Query")
VERTICALS = st.text_input("Verticals for Boosting (Comma Separated)")

VERTICALS = [v.strip() for v in VERTICALS.split(",") if v]

st.sidebar.write("## Vertical Boosts")
vertical_boost_map = {}
for vertical in VERTICALS:
    vertical_boost_map[vertical] = st.sidebar.slider(
        label=vertical, value=0.0, min_value=-1.0, max_value=1.0, step=0.05
    )


if YEXT_API_KEY and EXPERIENCE_KEY and QUERY:
    try:
        yext_client = YextClient(YEXT_API_KEY)
        response = get_liveapi_response(QUERY, yext_client, EXPERIENCE_KEY)
    except:
        raise ValueError("Invalid Experience Key or API Key.")
    first_results = [
        module["results"][0]
        for module in response["modules"]
        if module["source"] == "KNOWLEDGE_MANAGER"
    ]
    vertical_ids = [
        module["verticalConfigId"]
        for module in response["modules"]
        if module["source"] == "KNOWLEDGE_MANAGER"
    ]

    boost_vector = [
        vertical_boost_map[id_] if id_ in vertical_boost_map else 0 for id_ in vertical_ids
    ]

    new_ranks, _, max_fields, max_values, max_similarities, embeddings = get_new_vertical_ranks(
        QUERY, vertical_ids, first_results, boost_vector
    )

    left_col, right_col = st.columns(2)
    with left_col:
        st.write("## Original Results")
    with right_col:
        st.write("## Reordered Results")

    for old_rank, _ in enumerate(new_ranks):
        new_rank = new_ranks.index(old_rank)
        original_module = response["modules"][old_rank]
        reordered_module = response["modules"][new_rank]
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
            **Similarity:** {max_similarities[old_rank]}\n
            **Max Field:** {max_fields[old_rank]}\n
            **Max Field Value:** {max_values[old_rank]}
            """
            )
        with right_col:
            renderer(
                f"""
            **Vertical Key:** {reordered_module['verticalConfigId']}\n
            **Top Result:** {reordered_module['results'][0]['data']['name']}\n
            **Reordered Rank:** {old_rank} (Δ{delta})\n
            **Similarity:** {sorted(max_similarities, reverse=True)[old_rank]}\n
            **Max Field:** {max_fields[new_rank]}\n
            **Max Field Value:** {max_values[new_rank]}
            """
            )
    st.write("### Embeddings Calculated: {}".format(embeddings))
