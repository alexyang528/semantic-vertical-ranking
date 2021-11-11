import logging

LOGGER = logging.getLogger(__name__)

import re
import requests
from scipy.spatial.distance import cosine
from scipy.stats import rankdata


def get_liveapi_response(search_term, yext_client, experience_key):
    results = yext_client.search_answers_universal(query=search_term, experience_key=experience_key)
    response = results.raw_response["response"]
    return response


def _embed(strings: list) -> list:

    embeddings = requests.post(
        "http://cuda-serv01.dev.us2.yext.com:30034/embed", json={"queries": strings, "lang": "en"}
    ).json()["embeddings"]

    embeddings = [[float(i) for i in l] for l in embeddings]
    return embeddings


def _get_embeddings(query, values):

    # Save length of first result values lists
    lengths = [len(_flatten(l)) for l in values]
    values = _flatten(values)

    # Get embeddings of query and all first result values
    embeds = _embed([query] + values)

    # Get embedding of query
    query_embed = embeds[0]
    embeds = embeds[1:]

    # Get embeddings of first result values
    value_embeds = []
    for length in lengths:
        value_embeds.append(embeds[:length])
        embeds = embeds[length:]

    return query_embed, value_embeds


def _similarity(embed_a, embed_b):
    return 1 - cosine(embed_a, embed_b)


def _flatten(values):
    out = []
    for value in values:
        if isinstance(value, list):
            out.extend(_flatten(value))
        else:
            out.append(value)
    return out


def get_snippet(value, matched_subs, chars_before=50, chars_after=50, use_dense=True):
    if not matched_subs:
        return value[:chars_after]
    if len(value) < (chars_before + chars_after):
        return value

    best_substring = matched_subs[0]

    # Select snippet by density strategy
    if use_dense:
        most_subs = 1
        for i, matched_sub in enumerate(matched_subs):
            start_offset = matched_sub["offset"]
            remaining_subs = matched_subs[i:]
            inrange_subs = 0
            for sub in remaining_subs:
                if start_offset + sub["offset"] + sub["length"] > chars_after:
                    inrange_subs += 1
                else:
                    break
            if inrange_subs > most_subs:
                best_substring = matched_sub
                most_subs = inrange_subs

    offset = best_substring["offset"]
    display_start = max(offset - chars_before, 0)
    display_end = min(offset + chars_after, len(value) - 1)

    # Adjusting Start - Go forward until space
    if display_start != 0 and value[display_start - 1] != " ":
        while value[display_start] != " ":
            display_start += 1

    # Adjusting End - Go backward until space
    while value[display_end] != " ":
        display_end -= 1

    return value[display_start:display_end]


def _parse_value_recursively(field, value):

    # 1) Single dict of {matchedSubstrings / value} - if there is just one match
    if type(value) == dict and set(value.keys()) == set(["value", "matchedSubstrings"]):
        processed_value = get_snippet(value["value"], value["matchedSubstrings"])
        processed_field = field + " (snipped)" if value["value"] != processed_value else field
        return [processed_value], [processed_field]

    # 2) Dict of {field: {matchedSubstrings / value}} - if field is an object
    if type(value) == dict:
        values_and_fields = [_parse_value_recursively(f, v) for f, v in value.items()]
        processed_values = _flatten([i[0] for i in values_and_fields])
        processed_fields = _flatten([i[1] for i in values_and_fields])
        return processed_values, processed_fields

    # 3) List of {matchedSubstrings / value} or {field: {matchedSubstring / value}}
    if type(value) == list:
        values_and_fields = [_parse_value_recursively(field, v) for v in value]
        processed_values = _flatten([i[0] for i in values_and_fields])
        processed_fields = _flatten([i[1] for i in values_and_fields])
        return processed_values, processed_fields


def parse_highlighted_fields(vertical_id, first_result, vertical_intents):

    # Initialize with vertical intents
    matched_values = vertical_intents
    matched_fields = ["vertical_intent"] * len(matched_values)

    # Return just vertical intents if the result JSON is None
    if not first_result:
        LOGGER.error("Received an empty first result JSON for vertical ID: {}.".format(vertical_id))
        return matched_values, matched_fields

    # Try to append the name of the first result by default
    name_value = first_result.get("data", {}).get("name", None)
    if name_value:
        matched_values.append(name_value)
        matched_fields.append("name")

    # Begin JSON parsing highlighted fields
    highlights = first_result.get("highlightedFields", {})

    # Recursively get field, value pairs from highlightedFields
    highlighted_values, highlighted_fields = _parse_value_recursively(None, highlights)
    matched_values.extend(highlighted_values)
    matched_fields.extend(highlighted_fields)

    assert len(matched_values) == len(matched_fields)
    return matched_values, matched_fields


def get_new_vertical_ranks(
    query, vertical_ids, first_results, vertical_intents={}, vertical_boosts={}
):
    # Get highlighed field values of the top entity result for each vertical
    all_values_and_fields = [
        parse_highlighted_fields(vertical_id, result, vertical_intents.get(vertical_id, []))
        for vertical_id, result in zip(vertical_ids, first_results)
    ]
    all_values = [i[0] for i in all_values_and_fields]
    all_fields = [i[1] for i in all_values_and_fields]

    # Embed the query and the values of each first result
    query_embed, value_embeds = _get_embeddings(query, all_values)
    embeddings_calculated = len(_flatten(all_values)) + 1

    # Compute similarities between query and matched values
    similarities = [[_similarity(query_embed, v_i) for v_i in v] for v in value_embeds]

    # Boost if a boost vector is provided
    boost_vector = [vertical_boosts.get(id_, 0) for id_ in vertical_ids]
    similarities = [[sim + boost for sim in l] for l, boost in zip(similarities, boost_vector)]

    # Get the index of the max similarity, and the corresponding field and value
    max_similarities = [max(l, default=-1) for l in similarities]
    idx_max_similarities = [
        l.index(i) if i != -1 else -1 for l, i in zip(similarities, max_similarities)
    ]
    max_values = [l[i] if i != -1 else None for l, i in zip(all_values, idx_max_similarities)]
    max_fields = [l[i] if i != -1 else None for l, i in zip(all_fields, idx_max_similarities)]

    # Get the new vertical rankings by sorting on similarities
    new_rank = rankdata(max_similarities, method="ordinal")
    # Flip new rank, so it is highest to lowest similarity
    new_rank = [len(new_rank) - x for x in new_rank]

    assert len(first_results) == len(max_values) == len(max_fields) == len(new_rank)
    return (
        new_rank,
        _flatten(all_fields),
        max_fields,
        max_values,
        max_similarities,
        embeddings_calculated,
    )
