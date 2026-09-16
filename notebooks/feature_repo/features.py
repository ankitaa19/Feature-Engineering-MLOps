"""
Feast feature definitions for PrepEdge's customer features.

This file is what `feast apply` reads to register entities and feature views --
the same "single source of truth" concept explained in the notebook: instead of
three teams each writing their own version of avg_transaction_30d, everyone points
at THIS one definition.
"""
from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64

# The entity: what is a "row" of features attached to? Here, one customer.
customer = Entity(name="customer_id", join_keys=["customer_id"])

# Where the raw feature values live (a parquet file for this local demo;
# in production this would typically point at a data warehouse table).
customer_features_source = FileSource(
    path="data/customer_features.parquet",
    timestamp_field="event_timestamp",
)

# The feature view: which columns are "features," what their types are,
# and how long a value stays valid (ttl) before it's considered stale.
customer_features_view = FeatureView(
    name="customer_features",
    entities=[customer],
    ttl=timedelta(days=365),
    schema=[
        Field(name="avg_transaction_30d", dtype=Float32),
        Field(name="days_since_last_login", dtype=Int64),
        Field(name="login_count_30d", dtype=Int64),
    ],
    source=customer_features_source,
)
