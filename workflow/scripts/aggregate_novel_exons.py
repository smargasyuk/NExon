import click
import polars as pl

LINK_TEMPLATE = (
    "https://www.genome-euro.ucsc.edu/cgi-bin/hgTracks?db={g}&position={c}%3A{s}%2D{e}"
)


@click.command()
@click.option("--input", required=True)
@click.option("--output-bed", required=True)
@click.option("--output-tsv", required=True)
@click.option("--genome-UCSC", default="hg38")
@click.option("--track-name", default="NExon")
def main(input, output_bed, output_tsv, genome_ucsc, track_name):
    df7 = pl.read_csv(input, separator="\t")

    df1_info = (
        df7.sort(by="ipsa_min")
        .group_by(["seqname", "start", "end", "exon_id", "strand", "event_type"])
        .agg(
            pl.col("junction_id_l").last(),
            pl.col("junction_id_r").last(),
            pl.col("coord_prev").last(),
            pl.col("coord_next").last(),
            pl.col("novel_length").last(),
            pl.col("cov").mean(),
            pl.col("cons_avg").mean(),
            pl.col("ipsa_min").mean(),
            pl.col("meta")
            .filter(~pl.col("meta").is_null())
            .unique()
            .map_elements(lambda l: ",".join(sorted(l))),
            pl.col("sample_name").n_unique(),
        )
    )

    df11 = df1_info.with_columns(
        df1_info.map_rows(
            lambda r: LINK_TEMPLATE.format(
                g=genome_ucsc, c=r[0], s=r[8] - 10, e=r[9] + 10
            )
        )
        .get_columns()[-1]
        .alias("GB_link")
    )
    df11 = df11.rename(
        {
            "cov": "expr_exon",
            "ipsa_min": "expr_junction",
            "cons_avg": "phastcons",
            "sample_name": "sample_count",
        }
    )
    df11 = df11.select(
        [
            "seqname",
            "start",
            "end",
            "exon_id",
            "strand",
            "event_type",
            "junction_id_l",
            "junction_id_r",
            "coord_prev",
            "coord_next",
            "novel_length",
            "expr_exon",
            "expr_junction",
            "phastcons",
            "meta",
            "sample_count",
            "GB_link",
        ]
    )
    df11 = df11.with_columns(
        [pl.col(c).round(2) for c in ["expr_exon", "expr_junction", "phastcons"]]
    )
    df11.write_csv(output_tsv, separator="\t")

    df9_bed = df11.select(
        pl.col("seqname"),
        pl.col("coord_prev") - 5,
        pl.col("coord_next") + 4,
        pl.col("exon_id"),
        pl.lit(0).alias("score"),
        pl.col("strand"),
        (pl.col("coord_prev") - 5).alias("thickStart"),
        (pl.col("coord_next") + 4).alias("thickEnd"),
        pl.lit("0,0,0").alias("itemRgb"),
        pl.lit(3).alias("blockCount"),
        ("5," + (pl.col("end") - pl.col("start") + 1).cast(str) + ",5").alias(
            "blockSizes"
        ),
        (
            "0,"
            + (pl.col("start") - pl.col("coord_prev") + 4).cast(str)
            + ","
            + (pl.col("coord_next") - pl.col("coord_prev") + 4).cast(str)
        ).alias("blockStarts"),
    )

    with open(output_bed, "w") as f:
        f.write(f'track name={track_name} description="Novel exon predictions"\n')
    with open(output_bed, "a") as f:
        df9_bed.write_csv(f, separator="\t", has_header=False)


if __name__ == "__main__":
    main()
