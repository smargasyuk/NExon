import click
import polars as pl

REQUIRED_COLUMNS = [
    "seqname",
    "source",
    "feature",
    "start",
    "end",
    "score",
    "strand",
    "frame",
    "attribute",
]


# primitive parser, ignores repeating tags
# replace this with gffutils.parser._split_keyvals ?
def dictify(row):
    return dict(elem.split(" ") for elem in row[:-1].replace('"', "").split("; "))


def parse_gtf2(fname):
    df0 = pl.read_csv(
        fname,
        separator="\t",
        comment_char="#",
        has_header=False,
        new_columns=REQUIRED_COLUMNS,
        use_pyarrow=True,
    )

    ds = [dictify(x) for x in df0["attribute"]]

    df0_attrs = pl.DataFrame(ds)
    df1 = pl.concat([df0, df0_attrs], how="horizontal")
    return df1


@click.command()
@click.option("--input", required=True)
@click.option("--output", required=True)
def main(input, output):
    df_gencode_full = parse_gtf2(input)
    df_gencode_full.write_parquet(output)


if __name__ == "__main__":
    main()
