import polars as pl
import sys

def df_path(name: str) -> str:
    return "dataframes/" + name + ".parquet"

def scan_df(name: str) -> pl.LazyFrame:
    return pl.scan_parquet(df_path(name))

def write_df(df: pl.DataFrame, name: str) -> None:
    df.write_parquet(df_path(name))


def perr(*args):
    print(*args, file=sys.stderr)
