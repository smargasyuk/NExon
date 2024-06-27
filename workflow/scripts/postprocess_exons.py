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
@click.option("--input-ann-is", required=True)
@click.option("--input-meta", required=True)
@click.option("--output-table", required=True)
def main(
    input_pq,
    input_ann_is,
    input_meta,
    output_table,
):
    df5 = pl.read_parquet(input_pq, use_pyarrow=True)

    df_ann_is = pl.read_csv(input_ann_is, separator='\t', has_header=False, new_columns=['exon_id', 'ann_frac'])
    df5 = df5.join(df_ann_is, on='exon_id')

    print(f"Number of unique events in input DF:")
    print(get_unique_event_stats(df5))

    df5 = df5.filter(~pl.col("is_annotated"))
    print(f"Number of novel events:")
    print(get_unique_event_stats(df5))


    df5 = df5.filter(
        ~(
            (pl.col("event_type") == "CE")
            & (pl.col("ann_frac") > 0)
        )
    )
    print(f"Number of novel events after removal of partially annotated CE:")
    print(get_unique_event_stats(df5))

    # add metadata column
    meta_df = pl.read_csv(input_meta)
    df7 = df5.join(meta_df.select(['name', 'meta']), left_on="sample_name", right_on='name', how="left")

    df7.write_csv(output_table, separator="\t")


if __name__ == "__main__":
    main()
