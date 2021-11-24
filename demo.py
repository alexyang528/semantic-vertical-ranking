import streamlit as st
from yext import YextClient
from math import floor
from semantic_vertical_ranking import (
    get_liveapi_response,
    get_new_rank,
    svr,
    svr_dcg_result_name,
    svr_max_result_name,
)

"""
# Vertical Ranking 2.0 Prototype
This app is an interactive demo of a new vertical ranking approach, including vertical score using 
semantic similarity of top results, vertical boosts, and vertical intents.
"""

# BUSINESS_ID = st.text_input("Business ID")
YEXT_API_KEY = st.text_input("API Key")
EXPERIENCE_KEY = st.text_input("Experience Key")
QUERY = st.text_input("Query")
VERTICALS = st.text_input("Vertical Keys for Boosting and Intents (Comma Separated)")
VERTICALS = [v.strip() for v in VERTICALS.split(",") if v]
PRIORITY_FIELDS = ["name", "filter_value"]

st.sidebar.write("## Vertical Ranking Components")
has_entity_type_filter = st.sidebar.checkbox("Has Entity Type Filter")
has_near_me_filter = st.sidebar.checkbox("Has Near Me Filter")
has_location_filter = st.sidebar.checkbox("Has Location Filter")
score_method = st.sidebar.radio(label="Scoring Method", options=("SVR", "SVR Names", "SVR DCG"))

st.sidebar.write("## General")
bucket = st.sidebar.checkbox("Bucket Similarities")

st.sidebar.write("## Vertical Boosts")
vertical_boosts = {}
for vertical in VERTICALS:
    vertical_boosts[vertical] = st.sidebar.slider(
        label=vertical, value=0.0, min_value=-1.0, max_value=1.0, step=0.05
    )

st.sidebar.write("## Vertical Intents")
vertical_intents = {}
for vertical in VERTICALS:
    vertical_intents_str = st.sidebar.text_input("{} (Comma Separated)".format(vertical), key=1)
    vertical_intents[vertical] = [i.strip() for i in vertical_intents_str.split(",") if i]

st.sidebar.write("## Semantic Fields")
semantic_fields = {}
for vertical in VERTICALS:
    semantic_fields_str = st.sidebar.text_input("{} (Comma Separated)".format(vertical), key=2)
    semantic_fields[vertical] = [i.strip() for i in semantic_fields_str.split(",") if i]

if YEXT_API_KEY and EXPERIENCE_KEY and QUERY:
    try:
        yext_client = YextClient(YEXT_API_KEY)
        response = get_liveapi_response(QUERY, yext_client, EXPERIENCE_KEY)
    except:
        raise ValueError("Invalid Experience Key or API Key.")

    # Get general inputs
    modules = [module for module in response["modules"] if module["source"] == "KNOWLEDGE_MANAGER"]
    vertical_ids = [module["verticalConfigId"] for module in modules]
    query_filters = [module["appliedQueryFilters"] for module in modules]

    # Get filter criteria:
    filter_criteria = []
    if has_entity_type_filter:
        entity_type_filter = [any([filter.get("filter", {}).get("builtin.entityType") for filter in l if filter]) for l in query_filters]
        filter_criteria.append(entity_type_filter)
    if has_near_me_filter:
        near_me_filter = [any([filter.get("filter", {}).get("builtin.location", {}).get("$near") for filter in l if filter]) for l in query_filters]
        filter_criteria.append(near_me_filter)
    if has_location_filter:
        location_filter = [any([filter.get("filter", {}).get("builtin.location", {}).get("$eq") for filter in l if filter]) for l in query_filters]
        filter_criteria.append(location_filter)

    # Get Semantic Vertical Relevance (SVR)
    if score_method == "SVR":
        # Collect inputs
        first_results = [module["results"][0] for module in modules]
        filter_values = [[f_i["displayValue"] for f_i in f] for f in query_filters]

        # Get Semantic Vertical Relevance score and priority field tiebreaker
        scores, max_values, max_fields, embeds = svr(
            QUERY, vertical_ids, first_results, filter_values, vertical_intents, semantic_fields
        )
        is_priority_fields = [
            0 if set(v_max_fields).isdisjoint(PRIORITY_FIELDS) else 1 for v_max_fields in max_fields
        ]
        # Apply boosts
        scores = [
            score + vertical_boosts.get(vertical_id, 0)
            for score, vertical_id in zip(scores, vertical_ids)
        ]
        # Apply bucketing
        if bucket:
            scores = [floor(score * 10) / 10 for score in scores]

        # Get new rank and templates
        new_ranks = get_new_rank(*filter_criteria, scores, is_priority_fields)
        template = """
            **Vertical Key:** {}\n
            **Top Result:** {}\n
            **Original Rank:** {}\n
            **Similarity:** {}\n
            **Max Fields:** {}\n
            **Max Field Value:** {}
        """

    # Get Semantic Vertical Relevance (SVR) with just Result Names
    elif score_method == "SVR Names":
        # Collect inputs
        result_names = [
            [
                result.get("data", {}).get("name", None)
                for result in module["results"]
                if result.get("data", {}).get("name", None) is not None
            ]
            for module in modules
        ]

        # Limit to 10 results max per vertical
        result_names = [names[:10] for names in result_names]

        # Get scores
        scores, max_values, max_position, embeds = svr_max_result_name(QUERY, result_names)

        # Get new rank and templates
        new_ranks = get_new_rank(*filter_criteria, scores)
        template = """
            **Vertical Key:** {}\n
            **Original Rank:** {}\n
            **Similarity:** {}\n
            **Top Result Name:** {}\n
            **Top Result Position:** {}
        """

    # Get Semantic Vertical Relevance (SVR) with Result Names and Apply DCG
    elif score_method == "SVR DCG":
        # Collect inputs
        result_names = [
            [
                result.get("data", {}).get("name", None)
                for result in module["results"]
                if result.get("data", {}).get("name", None) is not None
            ]
            for module in modules
        ]

        # Limit to 10 results max per vertical
        result_names = [names[:10] for names in result_names]

        # Get scores
        scores, max_values, max_position, embeds = svr_dcg_result_name(QUERY, result_names)

        # Get new rank and templates
        new_ranks = get_new_rank(*filter_criteria, scores)
        template = """
            **Vertical Key:** {}\n
            **Original Rank:** {}\n
            **Similarity:** {}\n
            **Top Result Name:** {}\n
            **Top Result Position:** {}
        """

    # Remove line breaks from values for better presentation
    max_values = [i.replace("\n", " ") if i else None for i in max_values]

    left_col, right_col = st.columns(2)
    with left_col:
        st.write("## Original Results")
    with right_col:
        st.write("## Reordered Results")

    old_rank = 0
    while old_rank in new_ranks:
        # Get the index of the new item at old_rank
        new_rank = new_ranks.index(old_rank)

        original_module = modules[old_rank]
        reordered_module = modules[new_rank]
        delta = new_rank - old_rank

        if delta > 0:
            renderer = st.success
        elif delta < 0:
            renderer = st.error
        elif delta == 0:
            renderer = st.warning

        if score_method == "SVR":
            left_output = template.format(
                original_module["verticalConfigId"],
                original_module["results"][0]["data"]["name"],
                old_rank,
                scores[old_rank],
                max_fields[old_rank],
                max_values[old_rank],
            )
            right_output = template.format(
                reordered_module["verticalConfigId"],
                reordered_module["results"][0]["data"]["name"],
                f"{old_rank} (Δ {delta})",
                scores[new_rank],
                max_fields[new_rank],
                max_values[new_rank],
            )
        else:
            left_output = template.format(
                original_module["verticalConfigId"],
                old_rank,
                scores[old_rank],
                max_values[old_rank],
                max_position[old_rank],
            )
            right_output = template.format(
                reordered_module["verticalConfigId"],
                f"{old_rank} (Δ {delta})",
                scores[new_rank],
                max_values[new_rank],
                max_position[new_rank],
            )

        with left_col:
            st.info(left_output)
        with right_col:
            renderer(right_output)
        
        old_rank += 1
    st.write("### Embeddings Calculated: {}".format(embeds))
