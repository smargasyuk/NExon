import click
import polars as pl


@click.command()
@click.option("--input", required=True)
@click.option("--output", required=True)
@click.option("--radius", default=1)
def main(input, output, radius):
    df2 = pl.scan_parquet(input)
    dfnr1 = (
        df2.filter(pl.col("event_type") == "CE")
        .select("seqname", "start", "end", "exon_id", "event_type")
        .unique()
    )
    dfnr2 = (
        df2.filter(pl.col("event_type") == "AL")
        .select("seqname", "start", "start_right", "exon_id", "event_type")
        .unique()
        .with_columns(pl.col("start_right").cast(pl.Int64))
    )
    dfnr2 = dfnr2.rename({"start_right": "end"})
    dfnr3 = (
        df2.filter(pl.col("event_type") == "AR")
        .select("seqname", "end_right", "end", "exon_id", "event_type")
        .unique()
        .with_columns(pl.col("end_right").cast(pl.Int64))
    )
    dfnr3 = dfnr3.rename({"end_right": "start"})

    dfnr = pl.concat([dfnr1, dfnr2, dfnr3])
    dfnr = dfnr.with_columns(pl.col("start") - radius - 1, pl.col("end") + radius)

    dfnr.sink_csv(output, has_header=False, separator="\t")


if __name__ == "__main__":
    main()
