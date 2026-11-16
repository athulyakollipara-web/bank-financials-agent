"""AWS Glue job entry point: runs the same medallion code used locally, against S3."""
import sys
from pathlib import PurePosixPath

from awsglue.utils import getResolvedOptions  # provided by the Glue runtime
from pyspark.sql import SparkSession

import medallion  # shipped via --extra-py-files


class S3Path(PurePosixPath):
    """Lets medallion.py build paths with `/` while keeping the s3:// scheme."""

    def __new__(cls, uri: str):
        obj = super().__new__(cls, uri.replace("s3://", "/s3-scheme/", 1))
        return obj

    def __str__(self) -> str:
        return super().__str__().replace("/s3-scheme/", "s3://", 1)

    def __truediv__(self, other):
        return S3Path(str(self).rstrip("/") + "/" + str(other))

    def glob(self, pattern: str):
        import boto3

        bucket, _, prefix = str(self)[5:].partition("/")
        stem = pattern.split("*")[0]
        keys = boto3.client("s3").list_objects_v2(Bucket=bucket, Prefix=prefix + stem).get("Contents", [])
        return [S3Path(f"s3://{bucket}/{k['Key']}") for k in keys if k["Key"].endswith(".csv")]


args = getResolvedOptions(sys.argv, ["RAW_PATH", "WAREHOUSE"])
spark = SparkSession.builder.getOrCreate()
print(medallion.run_all(spark, S3Path(args["RAW_PATH"]), S3Path(args["WAREHOUSE"])))
