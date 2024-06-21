import gzip

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


IPSA_COLUMNS = [
    "junction_id",
    "total_count",
    "staggered_count",
    "entropy",
    "annotation_status",
    "splice_site",
]


def dictify(row):
    return dict(elem.split(" ") for elem in row[:-1].replace('"', "").split("; "))


def parse_gtf(fname):
    df0 = pl.read_csv(
        fname,
        separator="\t",
        comment_char="#",
        has_header=False,
        new_columns=REQUIRED_COLUMNS,
    )
    ds = [dictify(x) for x in df0["attribute"]]
    df0_attrs = pl.from_records(ds, infer_schema_length=None)
    df1 = pl.concat([df0, df0_attrs], how="horizontal")
    return df1


def parse_ipsa(ipsa_fname):
    dfj1 = pl.read_csv(
        ipsa_fname, separator="\t", has_header=False, new_columns=IPSA_COLUMNS
    )
    return dfj1.filter(
        (pl.col("splice_site") == "GTAG") & (pl.col("annotation_status") > 0)
    ).select(["junction_id", "total_count"])


def get_exons_from_gtf(df1):
    # get only exons; for each exon find left and right junctions from transcripts; remove terminal
    df2 = df1.filter(pl.col("feature") == "exon").drop(["FPKM", "TPM"])
    df2 = df2.with_columns(
        pl.col("end").shift(1).over("transcript_id").alias("coord_prev"),
        pl.col("start").shift(-1).over("transcript_id").alias("coord_next"),
    )  # .filter(pl.any_horizontal(pl.col(['coord_prev', 'coord_next']).is_null()).not_())
    return df2


def aggregate_exons_by_transcripts(df2, anno_exons):
    # merge with annotation df and add is_annotated flag; aggregate cov from different transcripts; add junction ids
    df3 = df2.join(
        anno_exons.select(["seqname", "start", "end", "score"]).unique(),
        on=["seqname", "start", "end"],
        how="left",
    )
    df3 = df3.with_columns(~pl.col("score_right").is_null()).rename(
        {"score_right": "is_annotated"}
    )
    df3agg = df3.group_by(["seqname", "start", "end", "strand"]).agg(pl.sum("cov"))
    df3 = (
        df3.select(
            [
                "seqname",
                "start",
                "end",
                "strand",
                "gene_id",
                "ref_gene_id",
                "is_annotated",
                "coord_prev",
                "coord_next",
            ]
        )
        .unique()
        .join(df3agg, on=["seqname", "start", "end", "strand"])
    )
    df3 = df3.with_columns(
        (
            pl.col("seqname")
            + "_"
            + pl.col("coord_prev").cast(str)
            + "_"
            + pl.col("start").cast(str)
            + "_"
            + pl.col("strand")
        ).alias("junction_id_l"),
        (
            pl.col("seqname")
            + "_"
            + pl.col("end").cast(str)
            + "_"
            + pl.col("coord_next").cast(str)
            + "_"
            + pl.col("strand")
        ).alias("junction_id_r"),
        (
            pl.col("seqname")
            + "_"
            + pl.col("start").cast(str)
            + "_"
            + pl.col("end").cast(str)
            + "_"
            + pl.col("strand")
        ).alias("exon_id"),
    )
    return df3


def merge_w_ipsa(df3, dfj2):
    return (
        df3.join(dfj2, left_on="junction_id_l", right_on="junction_id")
        .join(dfj2, left_on="junction_id_r", right_on="junction_id")
        .rename({"total_count": "ipsa_l", "total_count_right": "ipsa_r"})
    )


def find_events(df6, df6_introns):
    df7_ce = df6.join(
        df6_introns,
        left_on=["seqname", "coord_prev", "coord_next"],
        right_on=["seqname", "end", "coord_next"],
    ).with_columns(
        pl.lit("CE").alias("event_type")
    )  # .with_columns(pl.lit(True).alias('is_annotated_right'))
    df7_al = (
        df6.join(
            df6.select(
                [
                    "seqname",
                    "coord_prev",
                    "start",
                    "end",
                    "cov",
                    "is_annotated",
                    "exon_id",
                ]
            ).unique(),
            on=["seqname", "coord_prev", "end"],
        )
        .filter(pl.col("start") < pl.col("start_right"))
        .with_columns(pl.lit("AL").alias("event_type"))
    )
    df7_ar = (
        df6.join(
            df6.select(
                [
                    "seqname",
                    "coord_next",
                    "start",
                    "end",
                    "cov",
                    "is_annotated",
                    "exon_id",
                ]
            ).unique(),
            on=["seqname", "coord_next", "start"],
        )
        .filter(pl.col("end") > pl.col("end_right"))
        .with_columns(pl.lit("AR").alias("event_type"))
    )

    return pl.concat([df7_ce, df7_al, df7_ar], how="diagonal")


def parse_gencode_table(gencode_parquet):
    df_anno_full = pl.read_parquet(gencode_parquet)
    df_anno_full2 = df_anno_full.filter(pl.col("feature") == "exon")
    df_anno_full2 = df_anno_full2.sort(by=["seqname", "start"]).with_columns(
        pl.col("end").shift(1).over("transcript_id").alias("coord_prev"),
        pl.col("start").shift(-1).over("transcript_id").alias("coord_next"),
    )
    return (
        df_anno_full2.filter(pl.col("feature") == "exon"),
        df_anno_full2.select(
            ["seqname", "end", "coord_next", "strand", "score"]
        ).unique(),
    )


@click.command()
@click.option("--stringtie-gtf", required=True)
@click.option("--ipsa-junctions", required=True)
@click.option("--annotation-gtf", required=True)
@click.option("--output", required=True)
@click.option("--sample-name", required=True)
def main(stringtie_gtf, ipsa_junctions, annotation_gtf, output, sample_name):
    anno_exons, anno_introns = parse_gencode_table(annotation_gtf)

    df1 = parse_gtf(stringtie_gtf)
    df1 = df1.with_columns(
        pl.col(["cov", "FPKM", "TPM"]).cast(pl.Float32),
        pl.col("exon_number").cast(pl.Int32),
    )
    df2 = get_exons_from_gtf(df1)
    print(df2.head())
    df3 = aggregate_exons_by_transcripts(df2, anno_exons)
    dfj2 = parse_ipsa(ipsa_junctions)

    introns_df = (
        df3.select(["seqname", "end", "coord_next", "strand"])
        .unique()
        .join(anno_introns, on=["seqname", "end", "coord_next", "strand"], how="left")
        .with_columns(~pl.col("score").is_null())
        .rename({"score": "is_annotated"})
    )

    df6 = merge_w_ipsa(df3, dfj2)
    df7 = find_events(df6, introns_df).with_columns(
        pl.lit(sample_name).alias("sample_name")
    )

    with gzip.open(output, "wb") as f:
        df7.write_csv(f, separator="\t")


if __name__ == "__main__":
    main()
