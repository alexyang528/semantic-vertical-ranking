import logging
from rich.logging import RichHandler
from rich.progress import Progress

LOG_FORMAT = "%(message)s"
logging.basicConfig(level="INFO", format=LOG_FORMAT, datefmt="[%X] ", handlers=[RichHandler()])
LOGGER = logging.getLogger(__name__)

import argparse
import json
import os
import pandas as pd
from pyhive import presto
from yext import YextClient
from semantic_vertical_ranking import get_new_vertical_ranks, get_liveapi_response

PRESTO_USER = os.getenv("PRESTO_USER")
PRESTO_HOST = os.getenv("PRESTO_HOST")


def _flatten(values):
    out = []
    for value in values:
        if isinstance(value, list):
            out.extend(_flatten(value))
        else:
            out.append(value)
    return out


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


def compile_comparison_json(df):
    def _reorder_modules(modules, new_rank):
        return [module for i, module in sorted(zip(new_rank, modules), key=lambda x: x[0])]

    def _replace_modules(response, reordered_modules):
        return {
            "businessId": response["businessId"],
            "failedVerticals": [],
            "modules": reordered_modules,
            "queryId": response["queryId"],
            "searchIntents": response["searchIntents"],
            "locationBias": response["locationBias"],
        }

    # Get new order of modules
    df["modules"] = df["old_results"].apply(lambda response: response["modules"])
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
                "oldResult": {"query": query, "result": oldResult},
                "newResult": {"query": query, "result": newResult},
            }
            for query, oldResult, newResult in zip(queries, oldResults, newResults)
        ]

    with open("comparison.json", "w") as f:
        json.dump(comparison_json, f)
    return df


def main(args):

    yext_client = YextClient(args.api_key)

    # Initialize DataFrame with queries and answers key
    df = pd.DataFrame(args.search_terms, columns=["query"])
    df["answers_key"] = [args.experience_key] * len(df.index)

    # Get LiveAPI responses for all queries
    LOGGER.info("Getting LiveAPI responses...")
    with Progress() as progress:
        response_progress = progress.add_task("[green]Querying...", total=len(df.index))
        responses = []
        for query in df["query"]:
            response = get_liveapi_response(query, yext_client, args.experience_key)
            responses.append(response)
            progress.update(response_progress, advance=1)
        df["old_results"] = responses

    # Pull out the first entity result for each vertical module returned for each query
    df["vertical_ids"] = df["old_results"].apply(
        lambda response: [i["verticalConfigId"] for i in response["modules"]]
    )

    df["first_results"] = df["old_results"].apply(
        lambda response: [i["results"][0] for i in response["modules"]]
    )

    # Calculate similarities to highlighted fields of first results and rerank verticals
    LOGGER.info("Calculating new vertical ranks...")
    with Progress() as progress:
        rerank_progress = progress.add_task("[green]Calculating...", total=len(df.index))
        new_ranks = []
        all_fields = []
        max_fields = []
        total_embeddings = []
        for q, v, f in zip(df["query"], df["vertical_ids"], df["first_results"]):
            new_rank, all_fs, max_fs, _, _, embeddings = get_new_vertical_ranks(q, v, f)
            new_ranks.append(new_rank)
            all_fields.append(all_fs)
            max_fields.append(max_fs)
            total_embeddings.append(embeddings)
            progress.update(rerank_progress, advance=1)
        df["new_vertical_rank"] = new_ranks
        df["all_fields"] = all_fields
        df["max_fields"] = max_fields
        df["total_embeddings"] = total_embeddings
    LOGGER.info("Done.")

    # Generate comparison JSON for diff tool
    LOGGER.info("Compiling final comparison JSON...")
    compile_comparison_json(df)
    LOGGER.info("Done.")

    LOGGER.info("{} embeddings on average per query.".format(df["total_embeddings"].mean()))

    # Get the max fields
    max_field_counts = dict()
    for field in _flatten(df["max_fields"].values.tolist()):
        max_field_counts[field] = max_field_counts.get(field, 0) + 1
    all_field_counts = dict()
    for field in _flatten(df["all_fields"].values.tolist()):
        all_field_counts[field] = all_field_counts.get(field, 0) + 1
    field_relevance = dict()
    for k in all_field_counts.keys():
        field_relevance[k] = round(max_field_counts.get(k, 0) / all_field_counts[k], 2)

    LOGGER.info("Field relevance (%% of instances where field was max similarity):")
    LOGGER.info(field_relevance)


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
        LOGGER.info("Getting top 100 search terms by search volume...")
        query = """
            select tokenizer_normalized_query, count(distinct query_id)
            from log_federator_requests
            where date(dd) > date_add('day', -1, now())
            and answers_key = '{}'
            group by 1
            order by 2 desc
            limit 100
        """.format(
            args.experience_key
        )
        df = _get_data_from_presto(query, PRESTO_USER, PRESTO_HOST)
        args.search_terms = df["tokenizer_normalized_query"].values.tolist()
        LOGGER.info("Done.")

    main(args)
