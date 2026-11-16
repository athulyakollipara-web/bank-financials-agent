"""Bronze -> Silver -> Gold pipeline in PySpark.

Runs locally (``local[*]``) and unchanged as an AWS Glue 4/5 job: the only thing
that differs is the input/output paths (local folder vs. ``s3://``).

* Bronze: raw CSVs landed as strings + ingestion metadata. Nothing is dropped.
* Silver: typed, cleaned, de-duplicated (latest amended filing wins), snake_case.
* Gold:   ``dim_bank`` and ``fact_bank_quarter`` - the only tables the agent may query.
"""
from __future__ import annotations

from pathlib import Path

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

NUMERIC_COLUMNS = {
    "ASSET": "total_assets",
    "DEP": "total_deposits",
    "LNLSNET": "total_loans",
    "NETINC": "net_income",
    "EQ": "equity_capital",
    "NET_INT_INC": "net_interest_income",
    "NONINT_INC": "noninterest_income",
    "NONINT_EXP": "noninterest_expense",
    "EARNING_ASSETS": "earning_assets",
    "TIER1_CAP": "tier1_capital",
    "RWA": "risk_weighted_assets",
    "NONCURRENT_LOANS": "noncurrent_loans",
}
GOLD_TABLES = ("dim_bank", "fact_bank_quarter")


def get_spark(app_name: str = "bank-agent") -> SparkSession:
    return (
        SparkSession.builder.master("local[2]")
        .appName(app_name)
        .config("spark.ui.enabled", "false")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )


def bronze(spark: SparkSession, raw_dir: Path, warehouse: Path) -> dict[str, int]:
    institutions = (
        spark.read.option("header", True).option("inferSchema", False).csv(str(raw_dir / "institutions.csv"))
        .withColumn("_ingested_at", F.current_timestamp())
        .withColumn("_source_file", F.lit("institutions.csv"))
    )
    # Batch files are sorted by name so amendments (batch2) get a higher load sequence.
    files = sorted(str(p) for p in raw_dir.glob("financials_*.csv"))
    frames = []
    for seq, path in enumerate(files):
        frames.append(
            spark.read.option("header", True).option("inferSchema", False).csv(path)
            .withColumn("_load_seq", F.lit(seq))
            .withColumn("_source_file", F.lit(Path(path).name))
        )
    financials = frames[0]
    for frame in frames[1:]:
        financials = financials.unionByName(frame)
    financials = financials.withColumn("_ingested_at", F.current_timestamp())

    institutions.write.mode("overwrite").parquet(str(warehouse / "bronze" / "institutions"))
    financials.write.mode("overwrite").parquet(str(warehouse / "bronze" / "financials"))
    return {"bronze_institutions": institutions.count(), "bronze_financials": financials.count()}


def _to_number(col: str) -> F.Column:
    cleaned = F.regexp_replace(F.trim(F.col(col)), ",", "")
    return F.when(cleaned == "", None).otherwise(cleaned.cast("double"))


def silver(spark: SparkSession, warehouse: Path) -> dict[str, int]:
    fin = spark.read.parquet(str(warehouse / "bronze" / "financials"))
    typed = fin.select(
        F.col("CERT").cast("int").alias("cert"),
        F.to_date(F.col("REPDTE"), "yyyyMMdd").alias("report_date"),
        *[_to_number(src).alias(dst) for src, dst in NUMERIC_COLUMNS.items()],
        F.col("_load_seq"),
        F.col("_source_file"),
    ).where(F.col("cert").isNotNull() & F.col("report_date").isNotNull() & F.col("total_assets").isNotNull())

    latest = Window.partitionBy("cert", "report_date").orderBy(F.col("_load_seq").desc())
    deduped = (
        typed.withColumn("_rn", F.row_number().over(latest))
        .where(F.col("_rn") == 1)
        .drop("_rn")
        .withColumn("quarter", F.concat(F.year("report_date").cast("string"), F.lit("Q"), F.quarter("report_date").cast("string")))
    )

    inst = spark.read.parquet(str(warehouse / "bronze" / "institutions")).select(
        F.col("CERT").cast("int").alias("cert"),
        F.trim("NAME").alias("bank_name"),
        F.trim("CITY").alias("city"),
        F.upper(F.trim("STALP")).alias("state"),
        F.trim("BKCLASS").alias("charter_class"),
    ).dropDuplicates(["cert"])

    deduped.write.mode("overwrite").parquet(str(warehouse / "silver" / "financials"))
    inst.write.mode("overwrite").parquet(str(warehouse / "silver" / "institutions"))
    return {"silver_financials": deduped.count(), "silver_institutions": inst.count()}


def gold(spark: SparkSession, warehouse: Path) -> dict[str, int]:
    fin = spark.read.parquet(str(warehouse / "silver" / "financials"))
    inst = spark.read.parquet(str(warehouse / "silver" / "institutions"))

    latest_assets = (
        fin.withColumn("_rn", F.row_number().over(Window.partitionBy("cert").orderBy(F.col("report_date").desc())))
        .where("_rn = 1")
        .select("cert", "total_assets")
    )
    dim_bank = inst.join(latest_assets, "cert", "left").withColumn(
        "asset_size_bucket",
        F.when(F.col("total_assets") < 1_000_000, "under_1b")
        .when(F.col("total_assets") < 10_000_000, "1b_to_10b")
        .otherwise("over_10b"),
    ).drop("total_assets")

    fact = fin.select(
        "cert", "report_date", "quarter", *NUMERIC_COLUMNS.values()
    )

    dim_bank.write.mode("overwrite").parquet(str(warehouse / "gold" / "dim_bank"))
    fact.write.mode("overwrite").parquet(str(warehouse / "gold" / "fact_bank_quarter"))
    return {"dim_bank": dim_bank.count(), "fact_bank_quarter": fact.count()}


def register_gold_views(spark: SparkSession, warehouse: Path) -> None:
    for table in GOLD_TABLES:
        spark.read.parquet(str(warehouse / "gold" / table)).createOrReplaceTempView(table)


def run_all(spark: SparkSession, raw_dir: Path, warehouse: Path) -> dict[str, int]:
    stats: dict[str, int] = {}
    stats.update(bronze(spark, raw_dir, warehouse))
    stats.update(silver(spark, warehouse))
    stats.update(gold(spark, warehouse))
    return stats


def read_gold(spark: SparkSession, warehouse: Path, table: str) -> DataFrame:
    return spark.read.parquet(str(warehouse / "gold" / table))
