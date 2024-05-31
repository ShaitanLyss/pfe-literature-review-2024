import polars as pl

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
    papers = (
        pl
        .read_excel(bim_dataset_path, sheet_name="Papers", engine="calamine", schema_overrides={"Year": pl.UInt16})
        .rename({"__UNNAMED__0": "Id"})
    )
    nb_end_cols_to_drop = 12
    to_drop = papers.columns[-nb_end_cols_to_drop:]
    print(f"These columns will be dropped : {", ".join(to_drop)}.")
    papers = papers.drop(to_drop)
    return papers



if __name__ == '__main__':
    print(get_keywords().describe())
