from typing import Iterable, Tuple
from time import time
import sys
from dataclasses import dataclass
import jax
import jax.experimental.sparse as sparse
import jax.numpy as jnp
import numpy as np
import polars as pl
import nltk
from nltk.stem import WordNetLemmatizer

from utils import perr

adj_tags = ["JJ", "JJR", "JJS"]
noun_tags = ["NN", "NNS", "NNP", "NNPS"]
skip_words = ["abstract", "%", "paper", "study", "use"]


def get_terms_from_docs(
    df: pl.DataFrame, doc_id: str, text_cols: Iterable[str]
) -> pl.DataFrame:
    def nltk_tags(col: str, alias: str | None = None):
        return (
            pl.col(col)
            .alias(alias if alias is not None else col.lower() + "_tokens_tags")
            .map_elements(
                nltk.word_tokenize,
                return_dtype=pl.List(pl.String),
                strategy="thread_local",
            )
            .map_batches(
                lambda series: pl.Series(nltk.pos_tag_sents(series)),
                return_dtype=pl.List(pl.List(pl.String)),
            )
            # .cast(pl.Struct({"yo": pl.String, "y": pl.Categorical}))
        )

    lemmatizer = WordNetLemmatizer()

    q = (
        df.lazy()
        .select(
            pl.col(doc_id).alias("doc_id"),
            pl.concat_list(text_cols).alias("text"),
        )
        .explode("text")
        .with_columns(
            pl.arange(0, pl.len()).alias("text_id"),
            nltk_tags("text", "tokens_tags"),
        )
        .explode("tokens_tags")
        .select(
            pl.exclude("tokens_tags"),
            pl.col("tokens_tags").list.get(0).alias("token"),
            pl.col("tokens_tags").list.get(1).alias("tag").cast(pl.Categorical),
        )
        .with_columns(
            pl.col("tag")
            .is_in(adj_tags + noun_tags)
            .alias("keep")  # keep only nouns and adjectives
        )
        .with_columns(group_id=pl.col("keep").rle_id())
        .filter(pl.col("keep") == True)
        # Lemmatize plural forms to singular form
        .with_columns(
            pl.struct("token", "tag")
            .map_elements(
                lambda t: (
                    lemmatizer.lemmatize(t["token"])
                    if t["tag"] in noun_tags
                    else t["token"]
                ),
                return_dtype=pl.String,
            )
            .alias("token"),
        )
        .group_by("doc_id", "text_id", "group_id")
        .agg("token", "tag")
        # Keep only word sequences that end with a noun
        .with_columns(
            pl.col("tag")
            .list.eval(pl.element().is_in(adj_tags))
            .list.reverse()
            .alias("adjs_reversed")
        )
        .with_columns(
            (
                pl.col("adjs_reversed").list.len()
                - 1
                - pl.col("adjs_reversed").list.arg_min()
            ).alias("last_noun_pos"),
        )
        .with_columns(pl.col("token").list.head(pl.col("last_noun_pos") + 1))
        .filter(
            ~pl.col("adjs_reversed").list.all()
        )  # remove groups that are only adjectives
        # Final data preparation
        .select(
            pl.col("doc_id").alias("doc_ids"),
            pl.col("token").list.join(" ").str.to_lowercase().alias("term"),
            pl.col("tag").alias("tags"),
        )
        .filter(
            ~pl.col("term").is_in(skip_words),
        )
        .group_by("term")
        .agg(
            pl.col("doc_ids"),
            pl.col("term").len().alias("count"),
            pl.col("tags").first(),
        )
        .filter(pl.col("count") > 5)  # keep only words that appear 5 times
        .with_columns(pl.col("doc_ids").list.unique().list.len().alias("paper_count"))
        .sort("paper_count", "count", descending=True)
        .select(pl.arange(0, pl.len(), dtype=pl.UInt32).alias("id"), pl.all())
    )

    return q.collect()


# The dimensions for the dot product (contracting dimensions)
# Since numpy.dot(A, B) sums over the last axis of A and the second-to-last of B,
# these are the dimensions to contract.
dimension_numbers = (([1], [0]), ([], []))


@jax.jit
def get_cooccurences(a: jax.Array) -> jax.Array:
    return a @ a.T


@jax.jit
def get_second_order_cooccurences(a: jax.Array) -> jax.Array:
    fo = a @ a.T
    return fo @ fo  # no need to transpose because symetric


@dataclass
class AssociationDfParams:
    df: pl.DataFrame
    id: str = "id"
    other_id: str = "other_id"
    count: str = "count"


def get_mat_from_association_df(params: AssociationDfParams):
    data = params.df[params.count].to_numpy()
    indices = np.concatenate(
        params.df.select(pl.concat_list(params.id, params.other_id).alias("indices"))[
            "indices"
        ].to_numpy()
    ).reshape(len(params.df), 2)
    n = params.df.select(pl.col(params.id).max())[0, 0] + 1
    p = params.df.select(pl.col(params.other_id).max())[0, 0] + 1
    kw_paper = sparse.BCOO(
        (jnp.array(data), jnp.array(indices)), shape=(n, p)
    ).todense()
    return kw_paper


def get_overall_counts_from_association_df(params: AssociationDfParams):
    id_ = params.id
    count = params.count

    df = (
        params.df.lazy().group_by(id_).agg(pl.col(count).sum()).select(id_, count)
    ).collect()

    n = df.select(pl.col(id_).max())[0, 0] + 1
    overall_counts = sparse.BCOO(
        (
            jnp.array(df[count].to_numpy()),
            jnp.array(df[id_].to_numpy()[:, jnp.newaxis]),
        ),
        shape=(n,),
    )
    return overall_counts.todense()


def get_distrib(a: jax.Array):
    # no need to compute the overall cooccurence matrix because all terms share the same distribution
    # overall_cooc = get_cooccurences(overall_counts)
    overall_distrib = a / a.sum(axis=1)[:, jnp.newaxis]
    return overall_distrib


@dataclass
class VOSViewerReturn:
    relevant_terms: pl.DataFrame
    cooccurences_mat: jax.Array


def get_relevant_terms(
    params: AssociationDfParams, relevance_threshold=0.1
) -> Tuple[pl.DataFrame, jax.Array]:
    assoc_mat = get_mat_from_association_df(params)
    cooc = get_cooccurences(assoc_mat)
    snd_cooc = get_cooccurences(cooc)
    snd_distrib = get_distrib(snd_cooc)

    overall_counts = get_overall_counts_from_association_df(params)
    overall_distrib = get_distrib(overall_counts[jnp.newaxis, :])

    dist = jax.scipy.special.kl_div(snd_distrib, overall_distrib)

    relevance = dist.sum(axis=1)
    relevant_kw = jnp.where(relevance >= relevance_threshold)[0]
    kw_relevance = relevance[relevant_kw]

    df_relevant_terms = pl.DataFrame(
        [
            pl.Series("id", np.array(relevant_kw), dtype=pl.UInt32),
            pl.Series("relevance", np.array(kw_relevance)),
        ]
    )
    return df_relevant_terms, cooc


def vos_viewer(
    df: pl.DataFrame, doc_id: str, text_cols: Iterable[str], relevance_threshold: float
) -> VOSViewerReturn:
    print("Identifying terms...", file=sys.stderr)
    t0 = time()
    df_terms = get_terms_from_docs(df=df, doc_id=doc_id, text_cols=text_cols)
    df_term_doc = (
        df_terms.lazy()
        .explode("doc_ids")
        .with_columns(pl.col("term").cast(pl.Categorical))
        .group_by("id", "term", "doc_ids")
        .agg(pl.col("term").len().alias("count"))
        .rename({"id": "term_id", "doc_ids": "doc_id"})
    ).collect()
    perr("Done in {:.2}s.".format(time() - t0))
    print("Computing relevant terms...", file=sys.stderr)
    t0 = time()
    df_relevant_terms, cooc = get_relevant_terms(
        AssociationDfParams(df=df_term_doc, id="term_id", other_id="doc_id"),
        relevance_threshold=relevance_threshold,
    )
    print("Done in {:.2}s.".format(time() - t0), file=sys.stderr)
    res_df = (
        df_relevant_terms.lazy()
        .join(df_terms.lazy(), on="id")
        .select("id", "term", "doc_ids", "count", "paper_count", "relevance")
    ).collect()

    return VOSViewerReturn(relevant_terms=res_df, cooccurences_mat=cooc)
