from bank_agent.config import get_settings
from bank_agent.pipeline.medallion import get_spark, read_gold
from bank_agent.pipeline.quality import validate_table


def test_silver_deduplicates_amended_filings(warehouse):
    spark = get_spark()
    s = get_settings()
    bronze = spark.read.parquet(str(s.warehouse / "bronze" / "financials")).count()
    fact = read_gold(spark, s.warehouse, "fact_bank_quarter")
    assert bronze > fact.count() == 120 * 8
    assert fact.select("cert", "report_date").distinct().count() == fact.count()


def test_amended_value_wins(warehouse):
    import csv

    s = get_settings()
    amended = next(csv.DictReader((s.raw_dir / "financials_batch2_amendments.csv").open()))
    spark = get_spark()
    row = (
        read_gold(spark, s.warehouse, "fact_bank_quarter")
        .where(f"cert = {amended['CERT']} AND quarter = '{amended['REPDTE'][:4]}Q{(int(amended['REPDTE'][4:6]) - 1) // 3 + 1}'")
        .collect()[0]
    )
    assert row.net_income == float(amended["NETINC"])


def test_thousands_separators_are_parsed(warehouse):
    s = get_settings()
    fact = read_gold(get_spark(), s.warehouse, "fact_bank_quarter")
    assert fact.where("total_assets IS NULL").count() == 0


def test_gold_passes_quality_suite(warehouse):
    s = get_settings()
    spark = get_spark()
    for table in ("dim_bank", "fact_bank_quarter"):
        report = validate_table(table, read_gold(spark, s.warehouse, table).toPandas())
        assert report.success, report.failed


def test_quality_suite_catches_bad_data(warehouse):
    s = get_settings()
    df = read_gold(get_spark(), s.warehouse, "fact_bank_quarter").toPandas()
    df.loc[0, "total_deposits"] = df.loc[0, "total_assets"] * 2  # deposits larger than assets
    report = validate_table("fact_bank_quarter", df)
    assert not report.success
    assert "expect_column_pair_values_a_to_be_greater_than_b" in report.failed
