import click
import polars as pl


def get_stats(df):
    return (
        f"Total events: {df.shape[0]};\n Novel events: {df.filter(~pl.col('is_annotated')).shape[0]};\n"
        f"Unique events: {df['exon_id'].n_unique()};\n"
        f"Novel unique events {df.filter(~pl.col('is_annotated'))['exon_id'].n_unique()}\n\n"
    )


@click.command()
@click.option("--input", required=True)
@click.option("--output", required=True)
def main(input, output):
    df1 = pl.read_parquet(input, use_pyarrow=True)

    print("Initial statistics:")
    print(get_stats(df1))

    df1 = df1.filter(pl.col("is_annotated_right"))
    print("After removal of non-annotated pairs:")
    print(get_stats(df1))

    columns_groupby = [
        "exon_id",
        "seqname",
        "start",
        "end",
        "strand",
        "event_type",
        "sample_name",
        "is_annotated",
    ]
    columns_aggregate = [
        "coord_prev",
        "coord_next",
        "junction_id_l",
        "junction_id_r",
        "is_annotated_right",
        "cov",
        "ipsa_min",
        "exon_id_right",
        "cov_right",
        "start_right",
        "end_right",
    ]

    df2 = (
        df1.sort(by="ipsa_min")
        .groupby(columns_groupby)
        .agg([pl.col(e).last() for e in columns_aggregate])
        .with_columns(
            pl.when(pl.col("event_type") != "AR")
            .then(pl.col("start"))
            .otherwise(pl.col("end_right"))
            .cast(pl.Int64)
            .alias("novel_start"),
            pl.when(pl.col("event_type") != "AL")
            .then(pl.col("end"))
            .otherwise(pl.col("start_right"))
            .cast(pl.Int64)
            .alias("novel_end"),
        )
        .with_columns(
            (pl.col("novel_end") - pl.col("novel_start")).alias("novel_length")
        )
    )

    print("After aggregating right elements:")
    print(get_stats(df2))

    df2.write_parquet(output)


if __name__ == "__main__":
    main()
