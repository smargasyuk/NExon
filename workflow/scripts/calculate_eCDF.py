import click
import numpy as np
import polars as pl
from scipy.stats import ecdf
from tqdm import tqdm


def get_eventtype_stats(df):
    return df["event_type"].value_counts()


def get_unique_event_stats(df):
    return (
        df.select(["event_type", "exon_id"])
        .unique()
        .group_by("event_type")
        .agg(pl.col("exon_id").count())
    )


def xor(b1, b2):
    return b1 != b2


def map_event_type(r):
    if r[5] == "CE":
        return "CE"
    if xor((r[4] == "+"), (r[5] == "AR")):
        return "3'AS"
    else:
        return "5'AS"


@click.command()
@click.option("--input-pq", required=True)
@click.option("--input-bed", required=True)
@click.option("--max-as-length", default=150)
@click.option("--output", required=True)
def main(input_pq, input_bed, max_as_length, output):
    input_bed_columns = [
        "seqname",
        "start",
        "end",
        "exon_id",
        "event_type",
        "ann_bp",
        "ann_frac",
        "cons_wmean",
        "cons_cov",
        "pc_gene_ind",
    ]
    dfc1 = pl.read_csv(
        input_bed,
        separator="\t",
        use_pyarrow=True,
        has_header=False,
        new_columns=input_bed_columns,
    )
    print(f"Number of unique events in BED:")
    print(get_eventtype_stats(dfc1))

    dfc1 = dfc1.with_columns(
        (pl.col("cons_wmean") * pl.col("cons_cov")).alias("cons_avg")
    ).filter(~pl.col("cons_avg").is_nan())
    print(f"Number of unique events in BED after removing non-conserved:")
    print(get_eventtype_stats(dfc1))

    dfc1 = dfc1.filter(pl.col("pc_gene_ind") == 1).drop("pc_gene_ind")
    print(f"Number of unique events in BED after removing non-protein-coding:")
    print(get_eventtype_stats(dfc1))

    dfc1 = dfc1.filter(
        ~(
            (pl.col("event_type").is_in(["AL", "AR"]))
            & ((pl.col("end") - pl.col("start") > max_as_length))
        )
    )
    print(
        f"Number of unique events in BED after removing AS longer than {max_as_length}:"
    )
    print(get_eventtype_stats(dfc1))

    df2 = pl.read_parquet(input_pq, use_pyarrow=True)
    print(f"Number of unique events in full dataset:")
    print(get_unique_event_stats(df2))

    df2 = df2.join(
        dfc1.select("exon_id", "event_type", "cons_avg", "ann_frac"),
        on=["exon_id", "event_type"],
    )
    print(f"Number of unique events in full dataset after merge with BED data:")
    print(get_unique_event_stats(df2))

    df2 = df2.filter(~pl.col("cov").is_nan())
    print(f"Number of unique events in BED after removing elements with cov = NaN:")
    print(get_unique_event_stats(df2))

    # convert event_type from AL, AR to A3, A5
    df2 = df2.with_columns(
        df2.map_rows(map_event_type).get_columns()[-1].alias("event_type")
    )
    print(f"Number of unique events after conversion to 5'|3' AS notation:")
    print(get_unique_event_stats(df2))

    dfs = []
    for n, df in tqdm(df2.group_by(["sample_name", "event_type"], maintain_order=True)):
        dfm1_known = df.filter(pl.col("is_annotated"))
        ecdfs = {k: ecdf(dfm1_known[k].view()) for k in ["cons_avg", "ipsa_min", "cov"]}

        res = df.with_columns(
            [
                pl.Series(np.round(v.cdf.evaluate(df[k].view()), decimals=4)).alias(
                    f"{k}_ann_cdf"
                )
                for k, v in ecdfs.items()
            ]
        )
        # res = res.with_columns(df2[f'{k}_ann_cdf'] = np.round(v.cdf.evaluate(df2[k].view()), decimals=4))
        res = res.with_columns(
            pl.min_horizontal([f"{k}_ann_cdf" for k in ecdfs.keys()]).alias(
                "ann_cdf_min"
            )
        )
        dfs.append(res)

    pl.concat(dfs).write_parquet(output, use_pyarrow=True)


if __name__ == "__main__":
    main()
