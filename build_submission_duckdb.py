#!/usr/bin/env python3
"""Build Kaggle-format submissions from model-row prediction parquets.

This script uses DuckDB window functions to align per-node rollout rows with
sample_submission ordering deterministically.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb


def _quote(path: str) -> str:
    # Minimal quoting helper for SQL string literals.
    return path.replace("'", "''")


def _validate_output(con: duckdb.DuckDBPyConnection, sample_path: str, output_path: str) -> None:
    required_cols = ["row_id", "model_id", "event_id", "node_type", "node_id", "water_level"]

    output_cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM parquet_scan('{_quote(output_path)}')").fetchall()]
    if output_cols != required_cols:
        raise RuntimeError(f"Output columns mismatch: {output_cols} != {required_cols}")

    sample_rows = con.execute(f"SELECT COUNT(*) FROM parquet_scan('{_quote(sample_path)}')").fetchone()[0]
    output_rows = con.execute(f"SELECT COUNT(*) FROM parquet_scan('{_quote(output_path)}')").fetchone()[0]
    if sample_rows != output_rows:
        raise RuntimeError(f"Row count mismatch: output={output_rows}, sample={sample_rows}")

    null_wl = con.execute(
        f"SELECT COUNT(*) FROM parquet_scan('{_quote(output_path)}') WHERE water_level IS NULL"
    ).fetchone()[0]
    if null_wl != 0:
        raise RuntimeError(f"Output has {null_wl} NULL water_level values")

    key_mismatch = con.execute(
        f"""
        SELECT COUNT(*)
        FROM parquet_scan('{_quote(output_path)}') o
        JOIN parquet_scan('{_quote(sample_path)}') s USING (row_id)
        WHERE o.model_id != s.model_id
           OR o.event_id != s.event_id
           OR o.node_type != s.node_type
           OR o.node_id != s.node_id
        """
    ).fetchone()[0]
    if key_mismatch != 0:
        raise RuntimeError(f"Output has {key_mismatch} key mismatches against sample")

    min_row, max_row = con.execute(
        f"SELECT MIN(row_id), MAX(row_id) FROM parquet_scan('{_quote(output_path)}')"
    ).fetchone()
    print(f"Validation OK: rows={output_rows:,}, row_id=[{min_row}, {max_row}], null_water_level=0")


def build_full_from_rows(
    con: duckdb.DuckDBPyConnection,
    sample_path: str,
    pred_model1_path: str,
    pred_model2_path: str,
    output_path: str,
) -> None:
    print("Building full submission from Model 1 + Model 2 row predictions...")

    con.execute(
        f"""
        COPY (
            WITH sample_seq AS (
                SELECT
                    row_id,
                    model_id,
                    event_id,
                    node_type,
                    node_id,
                    ROW_NUMBER() OVER (
                        PARTITION BY model_id, event_id, node_type, node_id
                        ORDER BY row_id
                    ) - 1 AS step_idx
                FROM parquet_scan('{_quote(sample_path)}')
            ),
            pred_rows AS (
                SELECT
                    model_id,
                    event_id,
                    CASE node_type WHEN '1d' THEN 1 WHEN '2d' THEN 2 END AS node_type,
                    node_idx AS node_id,
                    timestep,
                    water_level
                FROM parquet_scan('{_quote(pred_model1_path)}')
                UNION ALL
                SELECT
                    model_id,
                    event_id,
                    CASE node_type WHEN '1d' THEN 1 WHEN '2d' THEN 2 END AS node_type,
                    node_idx AS node_id,
                    timestep,
                    water_level
                FROM parquet_scan('{_quote(pred_model2_path)}')
            ),
            pred_seq AS (
                SELECT
                    model_id,
                    event_id,
                    node_type,
                    node_id,
                    ROW_NUMBER() OVER (
                        PARTITION BY model_id, event_id, node_type, node_id
                        ORDER BY timestep
                    ) - 1 AS step_idx,
                    water_level
                FROM pred_rows
            )
            SELECT
                s.row_id,
                s.model_id,
                s.event_id,
                s.node_type,
                s.node_id,
                p.water_level
            FROM sample_seq s
            LEFT JOIN pred_seq p
              ON p.model_id = s.model_id
             AND p.event_id = s.event_id
             AND p.node_type = s.node_type
             AND p.node_id = s.node_id
             AND p.step_idx = s.step_idx
            ORDER BY s.row_id
        ) TO '{_quote(output_path)}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

    missing = con.execute(
        f"SELECT COUNT(*) FROM parquet_scan('{_quote(output_path)}') WHERE water_level IS NULL"
    ).fetchone()[0]
    if missing:
        raise RuntimeError(f"Aligned full submission has {missing} missing water_level entries")


def build_replace_model2_on_base(
    con: duckdb.DuckDBPyConnection,
    sample_path: str,
    base_path: str,
    pred_model2_path: str,
    output_path: str,
) -> None:
    print("Building submission by replacing Model 2 rows on base submission...")

    con.execute(
        f"""
        COPY (
            WITH sample_m2_seq AS (
                SELECT
                    row_id,
                    model_id,
                    event_id,
                    node_type,
                    node_id,
                    ROW_NUMBER() OVER (
                        PARTITION BY model_id, event_id, node_type, node_id
                        ORDER BY row_id
                    ) - 1 AS step_idx
                FROM parquet_scan('{_quote(sample_path)}')
                WHERE model_id = 2
            ),
            pred_m2_seq AS (
                SELECT
                    model_id,
                    event_id,
                    CASE node_type WHEN '1d' THEN 1 WHEN '2d' THEN 2 END AS node_type,
                    node_idx AS node_id,
                    ROW_NUMBER() OVER (
                        PARTITION BY model_id, event_id,
                                     CASE node_type WHEN '1d' THEN 1 WHEN '2d' THEN 2 END,
                                     node_idx
                        ORDER BY timestep
                    ) - 1 AS step_idx,
                    water_level
                FROM parquet_scan('{_quote(pred_model2_path)}')
            ),
            m2_aligned AS (
                SELECT
                    s.row_id,
                    p.water_level
                FROM sample_m2_seq s
                JOIN pred_m2_seq p
                  ON p.model_id = s.model_id
                 AND p.event_id = s.event_id
                 AND p.node_type = s.node_type
                 AND p.node_id = s.node_id
                 AND p.step_idx = s.step_idx
            )
            SELECT
                b.row_id,
                b.model_id,
                b.event_id,
                b.node_type,
                b.node_id,
                COALESCE(m2.water_level, b.water_level) AS water_level
            FROM parquet_scan('{_quote(base_path)}') b
            LEFT JOIN m2_aligned m2 USING (row_id)
            ORDER BY b.row_id
        ) TO '{_quote(output_path)}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

    expected_m2 = con.execute(
        f"SELECT COUNT(*) FROM parquet_scan('{_quote(sample_path)}') WHERE model_id = 2"
    ).fetchone()[0]
    replaced_m2 = con.execute(
        f"""
        SELECT COUNT(*)
        FROM parquet_scan('{_quote(output_path)}') out
        JOIN parquet_scan('{_quote(base_path)}') base USING (row_id)
        WHERE out.model_id = 2 AND out.water_level != base.water_level
        """
    ).fetchone()[0]
    print(f"Model 2 rows changed vs base: {replaced_m2:,} / {expected_m2:,}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", required=True, help="Path to sample_submission.parquet")
    parser.add_argument("--pred-model1", required=True, help="Path to model1 row predictions parquet")
    parser.add_argument("--pred-model2", required=True, help="Path to model2 row predictions parquet")
    parser.add_argument("--output-full", required=True, help="Output path for full M1+M2 submission")
    parser.add_argument("--base", required=False, help="Base full submission parquet for M2 replacement")
    parser.add_argument("--output-m2-on-base", required=False, help="Output path for base-with-M2-replaced")
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="DuckDB execution threads (default=1 for deterministic, corruption-safe parquet writes).",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Enable DuckDB progress bar.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    sample = str(Path(args.sample).resolve())
    pred_m1 = str(Path(args.pred_model1).resolve())
    pred_m2 = str(Path(args.pred_model2).resolve())
    output_full = str(Path(args.output_full).resolve())

    con = duckdb.connect(database=":memory:")
    con.execute(f"PRAGMA threads={max(1, int(args.threads))}")
    con.execute(f"PRAGMA enable_progress_bar={'true' if args.progress else 'false'}")

    try:
        build_full_from_rows(con, sample, pred_m1, pred_m2, output_full)
        _validate_output(con, sample, output_full)

        if args.base or args.output_m2_on_base:
            if not (args.base and args.output_m2_on_base):
                raise RuntimeError("Both --base and --output-m2-on-base are required together")
            base = str(Path(args.base).resolve())
            output_m2_base = str(Path(args.output_m2_on_base).resolve())
            build_replace_model2_on_base(con, sample, base, pred_m2, output_m2_base)
            _validate_output(con, sample, output_m2_base)

    finally:
        con.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
