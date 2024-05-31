import polars as pl
from itertools import combinations
from graph_tool.all import Graph, graph_draw

bim_dataset_path = "bim_dataset.xlsx"

# publications = pl.read_excel(bim_dataset_path, sheet_name="Publications")
# keywords = pl.read_excel(bim_dataset_path, sheet_name="Keywords")


def get_publications() -> pl.DataFrame:
    publications = pl.read_excel(bim_dataset_path, sheet_name="Publications")
    return publications


def get_keywords() -> pl.DataFrame:
    keywords = pl.read_excel(bim_dataset_path, sheet_name="Keywords")
    return keywords


def get_papers() -> pl.DataFrame:
    papers = pl.read_excel(
        bim_dataset_path,
        sheet_name="Papers",
        engine="calamine",
        schema_overrides={"Year": pl.UInt16 },
    ).rename({"__UNNAMED__0": "Id"})
    nb_end_cols_to_drop = 12
    to_drop = papers.columns[-nb_end_cols_to_drop:]
    #print(f"These columns will be dropped : {", ".join(to_drop)}.")
    papers = papers.drop(to_drop)
    papers = papers.with_columns(
            pl.col("Keywords").str.split(";").list.eval(pl.element().str.strip_chars().str.to_lowercase())
            )
    return papers

def get_keywords_cooccurrences() -> pl.Series:
    c_kw = pl.col("Keywords")
    c_kw_pairs = pl.col("KW_pairs")

    q = (
    get_papers().lazy()
    .filter(~c_kw.list.eval(pl.element() == "_").list.all())  # remove papers without keywords
    .select(
        c_kw.alias(c_kw_pairs.meta.output_name()).map_elements(
            lambda a: list(map(tuple, combinations(a, 2))),
            return_dtype=pl.List(pl.List(pl.String))
        ).cast(pl.List(pl.Array(pl.String, 2))).list.explode(),
    )
    )
    return q.collect().get_column(c_kw_pairs.meta.output_name())

def get_keywords_graph() -> Graph:
    g = Graph(directed=False)
    g.add_edge_list(get_keywords_cooccurrences().to_list(), hashed=True)
    
    return g


if __name__ == "__main__":
    # print(get_papers().select("Keywords").head())
    print(get_keywords_graph())
