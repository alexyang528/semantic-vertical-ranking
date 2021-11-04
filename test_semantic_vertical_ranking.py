import logging
from rich.logging import RichHandler
from rich.progress import Progress

LOG_FORMAT = "%(message)s"
logging.basicConfig(level="INFO", format=LOG_FORMAT, datefmt="[%X] ", handlers=[RichHandler()])
LOGGER = logging.getLogger(__name__)

import argparse
import os
import json
import requests
import pandas as pd
from pyhive import presto
from scipy.spatial.distance import cosine

PRESTO_USER = os.getenv("PRESTO_USER")
PRESTO_HOST = os.getenv("PRESTO_HOST")


def _get_data_from_presto(query, PRESTO_USER=PRESTO_USER, PRESTO_HOST=PRESTO_HOST):
    presto_conn = presto.connect(
        host=PRESTO_HOST,
        username=PRESTO_USER,
        port=8889,
        catalog="hive",
        schema="answers2",
    )

    query_export = pd.read_sql(query, presto_conn)
    return query_export


def _get_liveapi_response(search_term, api_key, experience_key):
    # Get URL and headers of Live API request
    url = (
        "https://liveapi.yext.com/v2/accounts/me/answers/query?v=20190101&api_key={}&"
        "jsLibVersion=v1.7.0&sessionTrackingEnabled=true&input={}&experienceKey={}&"
        "version=PRODUCTION&locale=en&referrerPageUrl=&source=STANDARD".format(
            api_key, search_term, experience_key
        )
    )
    headers = {
        "user-agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko)"
            " Chrome/87.0.4280.88 Safari/537.36"
        )
    }

    # Make get request to Live API and track JSON response
    r = requests.get(url=url, headers=headers)
    output = json.loads(r.text)

    live_api_response = {
        "query": search_term,
        "answers_key": experience_key,
        "result": output["response"],
    }

    return live_api_response


def _embed_single_query(query):
    out = requests.post(
        "http://cuda-serv01.dev.us2.yext.com:30034/embed",
        json={"queries": [query], "lang": "en"},
    ).json()["embeddings"]

    return out[0]


def _compare_embedded_queries(str_a, str_b):
    embed_a = [float(i) for i in _embed_single_query(str_a)]
    embed_b = [float(i) for i in _embed_single_query(str_b)]
    return 1 - cosine(embed_a, embed_b)


def _flatten(values):
    out = []
    for value in values:
        if isinstance(value, list):
            out.extend(_flatten(value))
        else:
            out.append(value)
    return out


def get_snippet(value, matched_subs, chars_before=50, chars_after=200, use_dense=True):
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


def parse_highlighted_fields(result_json):
    if result_json is None:
        return [], [], [], []

    # Initialize output lists
    highlighted_fields = []
    matched_values = []

    # Begin JSON parsing highlighted fields
    highlighted_field = result_json["highlightedFields"]
    highlighted_fields.append(highlighted_field)

    # For all highlighted fields, pull out the matched substrings and values
    for k, v in highlighted_field.items():
        if v == []:
            continue

        ### For each highlighted field, there are three possible formats:
        # 1) List of {matchedSubstrings / value} pairs - if there are multiple matches
        if type(v) == list:
            try:
                values = [v_i["value"] for v_i in v]
                substrings = [v_i["matchedSubstrings"] for v_i in v]

                processed_values = [
                    get_snippet(value, sub) for value, sub in zip(values, substrings)
                ]
                matched_values.extend(processed_values)
            # 2) List of {field: matchedSubstring / value} dicts - if highlighted field is an object
            except:
                values = [v_i[k_i]["value"] for v_i in v for k_i in v_i]
                substrings = [v_i[k_i]["matchedSubstrings"] for v_i in v for k_i in v_i]

                processed_values = [
                    get_snippet(value, sub) for value, sub in zip(values, substrings)
                ]
                matched_values.extend(processed_values)
        # 3) Single dict of {matchedSubstrings / value} - if there is just one match
        else:
            processed_values = get_snippet(v["value"], v["matchedSubstrings"])
            matched_values.append(processed_values)

    # If there are no highlighted field values, then use the name data field instead
    if len(matched_values) == 0:
        matched_values = [result_json["data"]["name"]]

    return matched_values


def get_new_vertical_ranks(query, first_results):
    # Get highlighed field values of the top entity result for each vertical
    values = [parse_highlighted_fields(result) for result in first_results]

    # Compute similarities between query and matched values
    similarities = [
        max([_compare_embedded_queries(query, v_i) for v_i in _flatten(value)], default=None)
        for value in values
    ]

    # Get the new vertical rankings by sorting on similarities
    new_rank = [sorted(similarities, reverse=True).index(x) for x in similarities]

    assert len(first_results) == len(values) == len(similarities) == len(new_rank)
    return new_rank, len(_flatten(values))


def compile_comparison_json(df):
    def _reorder_modules(modules, new_rank):
        return [module for i, module in sorted(zip(new_rank, modules), key=lambda x: x[0])]

    def _replace_modules(response, reordered_modules):
        return {
            "query": response["query"],
            "answers_key": response["answers_key"],
            "result": {
                "businessId": response["result"]["businessId"],
                "failedVerticals": [],
                "modules": reordered_modules,
                "queryId": response["result"]["queryId"],
                "searchIntents": response["result"]["searchIntents"],
                "locationBias": response["result"]["locationBias"],
            },
        }

    # Get new order of modules
    df["modules"] = df["old_results"].apply(lambda response: response["result"]["modules"])
    df["reordered_modules"] = [
        _reorder_modules(module, new_rank)
        for module, new_rank in zip(df["modules"], df["new_vertical_rank"])
    ]

    # Compile new liveAPI response format with reordered modules
    df["new_results"] = [
        _replace_modules(old_result, new_modules)
        for old_result, new_modules in zip(df["old_results"], df["reordered_modules"])
    ]

    # Aggregate by answers key
    agg_df = df.groupby("answers_key").agg(
        {"query": list, "old_results": list, "new_results": list}
    )
    agg_df = agg_df.reset_index()

    comparison_json = {}
    for _, row in agg_df.iterrows():
        answersKey = row["answers_key"]
        queries = row["query"]
        oldResults = row["old_results"]
        newResults = row["new_results"]

        comparison_json[answersKey] = [
            {
                "query": query,
                "oldResult": {"query": query, "result": oldResult["result"]},
                "newResult": {"query": query, "result": newResult["result"]},
            }
            for query, oldResult, newResult in zip(queries, oldResults, newResults)
        ]

    with open("comparison.json", "w") as f:
        json.dump(comparison_json, f)
    return df


def main(args):

    # Initialize DataFrame with queries and answers key
    df = pd.DataFrame(args.search_terms, columns=["query"])
    df["answers_key"] = [args.experience_key] * len(df.index)

    # Get LiveAPI responses for all queries
    LOGGER.info("Getting LiveAPI responses...")
    with Progress() as progress:
        response_progress = progress.add_task("[green]Querying...", total=len(df.index))
        responses = []
        for query in df["query"]:
            response = _get_liveapi_response(query, args.api_key, args.experience_key)
            responses.append(response)
            progress.update(response_progress, advance=1)
        df["old_results"] = responses

    # Pull out the first entity result for each vertical module returned for each query
    df["vertical_ids"] = df["old_results"].apply(
        lambda response: [i["verticalConfigId"] for i in response["result"]["modules"]]
    )
    df["entity_results"] = df["old_results"].apply(
        lambda response: [i["results"] for i in response["result"]["modules"]]
    )
    df["first_results"] = df["entity_results"].apply(
        lambda results: [result[0] for result in results]
    )

    # Calculate similarities to highlighted fields of first results and rerank verticals
    LOGGER.info("Calculating new vertical ranks...")
    with Progress() as progress:
        rerank_progress = progress.add_task("[green]Calculating...", total=len(df.index))
        new_ranks = []
        total_embeddings = []
        for query, first_results in zip(df["query"], df["first_results"]):
            new_rank, embeddings = get_new_vertical_ranks(query, first_results)
            new_ranks.append(new_rank)
            total_embeddings.append(embeddings)
            progress.update(rerank_progress, advance=1)
        df["new_vertical_rank"] = new_ranks
        df["total_embeddings"] = total_embeddings
    LOGGER.info("Done.")

    # Generate comparison JSON for diff tool
    LOGGER.info("Compiling final comparison JSON...")
    compile_comparison_json(df)
    LOGGER.info("Done.")

    LOGGER.info(
        "{} embeddings calculated on average per query.".format(df["total_embeddings"].mean())
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Rerank verticals using semantic similarity to highlighted fields of top results."
        )
    )
    parser.add_argument(
        "-a", "--api_key", type=str, help="API key of the experience to test.", required=True
    )
    parser.add_argument(
        "-e",
        "--experience_key",
        type=str,
        help="Experience key of the experience to test.",
        required=True,
    )
    parser.add_argument(
        "-s",
        "--search_terms",
        nargs="+",
        help="The search terms for which to compute vertical rankings.",
        required=False,
    )
    args = parser.parse_args()

    if not args.search_terms:
        LOGGER.info("Getting top 50 search terms by search volume...")
        query = """
            select tokenizer_normalized_query, count(distinct query_id)
            from log_federator_requests
            where date(dd) > date_add('day', -30, now())
            and answers_key = '{}'
            group by 1
            order by 2 desc
            limit 50
        """.format(
            args.experience_key
        )
        df = _get_data_from_presto(query, PRESTO_USER, PRESTO_HOST)
        args.search_terms = df["tokenizer_normalized_query"].values.tolist()
        LOGGER.info("Done.")

    main(args)
