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


def compile_comparison_json(answers_key, queries, responses, new_ranks, output_file):
    def _reorder(modules, new_rank):
        return [module for i, module in sorted(zip(new_rank, modules), key=lambda x: x[0])]

    def _replace(response, reordered_modules):
        return {
            "businessId": response["businessId"],
            "failedVerticals": [],
            "modules": reordered_modules,
            "queryId": response["queryId"],
            "searchIntents": response["searchIntents"],
            "locationBias": response["locationBias"],
            "directAnswer": response.get("directAnswer", "")
        }


    # Get new order of modules
    modules = [response["modules"] for response in responses]
    reordered_modules = [_reorder(module, new_rank) for module, new_rank in zip(modules, new_ranks)]

    comparison_json = {}
    comparison_json[answers_key] = [
        {
            "query": query,
            "oldResult": {"query": query, "result": response},
            "newResult": {"query": query, "result": _replace(response, reordered_module)},
        }
        for query, response, reordered_module in zip(queries, responses, reordered_modules)
    ]

    with open(output_file, "w") as f:
        json.dump(comparison_json, f)


def get_top_search_terms_from_presto(answers_key, business_id, count=500):
    LOGGER.info(f"Getting top {count} search terms by search volume...")
    query = f"""
        select tokenizer_normalized_query, count(distinct query_id)
        from log_federator_requests
        where date(dd) > date_add('day', -7, now())
        and answers_key = '{answers_key}'
        and business_id = {business_id}
        group by 1
        order by 2 desc
        limit {count}
    """
    df = _get_data_from_presto(query, PRESTO_USER, PRESTO_HOST)
    search_terms = df["tokenizer_normalized_query"].values.tolist()
    LOGGER.info("Done. Received {} search terms from Presto.".format(len(search_terms)))
    return search_terms


def main(args):

    search_terms = get_top_search_terms_from_presto(
        args.experience_key, args.business_id, args.limit
    )

    yext_client = YextClient(args.api_key)
    vertical_boosts = {}
    vertical_intents = {}
    if args.boost_file:
        vertical_boosts = json.load(open(args.boost_file))
    if args.intents_file:
        vertical_intents = json.load(open(args.intents_file))

    responses = []
    new_ranks = []

    # Get LiveAPI responses for all queries
    LOGGER.info("Reranking...")
    with Progress() as progress:
        rerank_progress = progress.add_task("[green]Reranking...", total=len(search_terms))
        for query in search_terms:
            # Get Live API response for each search term
            response = get_liveapi_response(query, yext_client, args.experience_key)
            responses.append(response)

            # Pull out top results, vertical IDs, and filter values for each module returned
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
            query_filters = [
                module["appliedQueryFilters"]
                for module in response["modules"]
                if module["source"] == "KNOWLEDGE_MANAGER"
            ]
            filter_values = [[f_i["displayValue"] for f_i in f] for f in query_filters]

            # Get new rank of verticals for query
            new_rank = get_new_vertical_ranks(
                query,
                vertical_ids,
                first_results,
                filter_values,
                semantic_fields={},
                vertical_intents=vertical_intents,
                vertical_boosts=vertical_boosts,
            )[0]
            new_ranks.append(new_rank)

            progress.update(rerank_progress, advance=1)

    compile_comparison_json(args.experience_key, search_terms, responses, new_ranks, args.output)
    LOGGER.info("Done.")


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
        "-b",
        "--business_id",
        type=int,
        help="Business ID of the experience to test.",
        required=True,
    )
    parser.add_argument(
        "-l",
        "--limit",
        type=int,
        help="Number of search terms to include.",
        required=False,
        default=500,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="The name of the output JSON file.",
        default="comparison.json",
    )
    parser.add_argument(
        "--boost_file",
        type=str,
        help="The name of the JSON file to read boosts dict from.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--intents_file",
        type=str,
        help="The name of the JSON file to read intents dict from.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--semantic_file",
        type=str,
        help="The name of the JSON file to read semantic fields dict from.",
        required=False,
        default=None,
    )
    args = parser.parse_args()
    LOGGER.info(args)

    main(args)
