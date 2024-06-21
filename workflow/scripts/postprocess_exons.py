import click
import polars as pl


def get_unique_event_stats(df):
    return (
        df.select(["event_type", "exon_id"])
        .unique()
        .group_by("event_type")
        .agg(pl.col("exon_id").count())
    )


GTF_REQUIRED_COLUMNS = [
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


@click.command()
@click.option("--input-pq", required=True)
@click.option("--input-ann", required=True)
@click.option("--input-meta", required=True)
@click.option("--quantile-expr", default=0.15)
@click.option("--quantile-ce-cons", default=0.15)
@click.option("--quantile-as-cons", default=0.15)
@click.option("--output-table", required=True)
def main(
    input_pq,
    input_ann,
    input_meta,
    quantile_expr,
    quantile_as_cons,
    quantile_ce_cons,
    output_table,
):
    df5 = pl.read_parquet(input_pq, use_pyarrow=True)

    print(f"Number of unique events in input DF:")
    print(get_unique_event_stats(df5))

    gencode_exons = (
        pl.read_parquet(input_ann, use_pyarrow=True)
        .filter(pl.col("feature") == "exon")
        .select(["seqname", "start", "end", "strand", "score"])
        .unique()
    )


    # left-join with GENCODE DF on left and right sites separately, use score as indicator
    df5 = df5.join(
        gencode_exons.select(["seqname", "start", "score"]).unique(),
        on=["seqname", "start"],
        how="left",
    ).join(
        gencode_exons.select(["seqname", "end", "score"]).unique(),
        on=["seqname", "end"],
        how="left",
    )
    df5 = df5.with_columns(
        pl.col("score").is_null().alias("is_left_novel"),
        pl.col("score_right").is_null().alias("is_right_novel"),
    ).drop(["score", "score_right"])

    df5 = df5.filter(~pl.col("is_annotated"))
    print(f"Number of novel events:")
    print(get_unique_event_stats(df5))


    df5 = df5.filter(
        ~(
            (pl.col("event_type") == "CE")
            & ~pl.col("is_annotated")
            & (pl.col("ann_frac") > 0)
        )
    )
    print(f"Number of novel events after removal of partially annotated CE:")
    print(get_unique_event_stats(df5))

    df5 = df5.filter(pl.col("is_left_novel") | pl.col("is_right_novel"))
    print(
        f"Number of novel events after removal of exons with both sites annotated separately:"
    )
    print(get_unique_event_stats(df5))

    df7_ce = df5.filter(
        (pl.col("event_type") == "CE")
        & (
            (pl.col("ipsa_min_ann_cdf") > quantile_expr)
            & (pl.col("cov_ann_cdf") > quantile_expr)
            & (pl.col("cons_avg_ann_cdf") > quantile_ce_cons)
        )
    )
    df7_as = df5.filter(
        (pl.col("event_type") != "CE")
        & (
            (pl.col("ipsa_min_ann_cdf") > quantile_expr)
            & (pl.col("cov_ann_cdf") > quantile_expr)
            & (pl.col("cons_avg_ann_cdf") > quantile_as_cons)
        )
    )
    df7 = pl.concat([df7_ce, df7_as])
    print(
        f"Number of novel events after cutting by annotated eCDF thresholds \
          (expression: {quantile_expr}, CE conservation: {quantile_ce_cons}, AS conservation: {quantile_as_cons})"
    )
    print(get_unique_event_stats(df7))

    # add metadata column
    meta_df = pl.read_csv(input_meta)
    df7 = df7.join(meta_df.select(['name', 'meta']), left_on="sample_name", right_on='name', how="left")

    df7.write_csv(output_table, separator="\t")


if __name__ == "__main__":
    main()
