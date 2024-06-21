import pickle

import click
import polars as pl


@click.command()
@click.option("--input-list", required=True)
@click.option("--output", required=True)
def main(input_list, output):
    file_list = pl.read_csv(input_list, separator="\t", has_header=False)["column_1"]
    pl.concat(
        pl.read_csv(str(t), separator="\t", infer_schema_length=None) for t in file_list
    ).with_columns(
        pl.min_horizontal(["ipsa_l", "ipsa_r"]).alias("ipsa_min")
    ).write_parquet(
        output
    )


if __name__ == "__main__":
    main()
