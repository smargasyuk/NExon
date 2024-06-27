import click
import polars as pl


def get_stats(df):
    return (
        f"Total events: {df.shape[0]};\n Novel events: {df.filter(~pl.col('is_annotated')).shape[0]};\n"
        f"Unique events: {df['exon_id'].n_unique()};\n"
        f"Novel unique events {df.filter(~pl.col('is_annotated'))['exon_id'].n_unique()}\n\n"
    )


def read_gencode_introns(dfa):
    dft = dfa.filter((pl.col('feature') == 'transcript') & (pl.col('transcript_type') == "protein_coding"))['transcript_id']
    dfa2 = dfa\
        .filter(
            (pl.col('feature') == 'exon') & 
            pl.col('transcript_id').is_in(dft))\
        .sort(by=['transcript_id', 'start'])\
        .with_columns(
            pl.col("start").shift(-1).over("transcript_id").alias("coord_next"),
            pl.col("end").shift(1).over("transcript_id").alias("coord_prev"),
            pl.col('exon_number').cast(pl.Int16)
        )\
        .with_columns(
            (pl.col('seqname') + "_" + pl.col('end').cast(str) + "_" + pl.col('coord_next').cast(str) + "_" + pl.col('strand') + "_").alias('intron_r')
        )

    return dfa2['intron_r']



@click.command()
@click.option("--input", required=True)
@click.option("--annotation-pq", required=True)
@click.option("--output", required=True)
def main(input, annotation_pq, output):
    df1 = pl.read_parquet(input, use_pyarrow=True)\
        .with_columns(
        (pl.col('seqname') + "_" + pl.col('coord_prev').cast(str) + "_" + 
         pl.col('coord_next').cast(str) + "_" + pl.col('strand') + "_").alias('junction_id_o')
    )
    dfa = pl.read_parquet(annotation_pq)
    dfi = read_gencode_introns(dfa)

    print("Initial statistics:")
    print(get_stats(df1))

    df1 = df1.filter(
        pl.col("is_annotated_right") & 
        (pl.col('junction_id_o').is_in(dfi) | pl.col('event_type').is_in(['AR', 'AL']))
    )
    print("After removal of non-annotated pairs and non-coding transcripts:")
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
        "junction_id_o",
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
