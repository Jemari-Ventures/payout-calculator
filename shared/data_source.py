"""
Data source management for loading data from various sources.
"""
import io
import os
import re
from typing import Optional, Tuple
from urllib.parse import urlparse, parse_qs

import pandas as pd
import requests
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError


class DataSource:
    """Handle data loading from Google Sheets and other sources."""

    @staticmethod
    def _extract_gsheet_id_and_gid(url_or_id: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract spreadsheet ID and GID from Google Sheets URL or ID.

        Args:
            url_or_id: Google Sheets URL or spreadsheet ID

        Returns:
            Tuple of (spreadsheet_id, gid) or (None, None) if invalid
        """
        if not url_or_id:
            return None, None

        # If it's just an ID (alphanumeric, 20+ chars)
        if re.fullmatch(r"[A-Za-z0-9_-]{20,}", url_or_id):
            return url_or_id, None

        try:
            parsed = urlparse(url_or_id)
            path_parts = [p for p in parsed.path.split('/') if p]
            spreadsheet_id = None

            if 'spreadsheets' in path_parts and 'd' in path_parts:
                try:
                    idx = path_parts.index('d')
                    if idx + 1 < len(path_parts):
                        spreadsheet_id = path_parts[idx + 1]
                except (ValueError, IndexError):
                    spreadsheet_id = None

            query_gid = parse_qs(parsed.query).get('gid', [None])[0]
            frag_gid_match = re.search(r"gid=(\d+)", parsed.fragment or "")
            frag_gid = frag_gid_match.group(1) if frag_gid_match else None
            gid = query_gid or frag_gid

            return spreadsheet_id, gid
        except Exception:
            return None, None

    @staticmethod
    def _build_gsheet_csv_url(spreadsheet_id: str, sheet_name: Optional[str], gid: Optional[str]) -> str:
        """
        Construct CSV export URL for Google Sheets.

        Args:
            spreadsheet_id: Google Sheets spreadsheet ID
            sheet_name: Optional sheet name
            gid: Optional sheet GID

        Returns:
            CSV export URL
        """
        base = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}"
        if sheet_name:
            return f"{base}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
        if gid:
            return f"{base}/export?format=csv&gid={gid}"
        return f"{base}/export?format=csv"

    @staticmethod
    @st.cache_data(ttl=300)
    def read_google_sheet(url_or_id: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch a Google Sheet as CSV and return a DataFrame.

        Args:
            url_or_id: Google Sheets URL or spreadsheet ID
            sheet_name: Optional sheet name

        Returns:
            DataFrame containing sheet data

        Raises:
            ValueError: If URL/ID is invalid
            requests.RequestException: If request fails
        """
        spreadsheet_id, gid = DataSource._extract_gsheet_id_and_gid(url_or_id)
        if not spreadsheet_id:
            raise ValueError("Invalid Google Sheet URL or ID.")

        csv_url = DataSource._build_gsheet_csv_url(spreadsheet_id, sheet_name, gid)

        try:
            resp = requests.get(csv_url, timeout=30)
            resp.raise_for_status()
            return pd.read_csv(io.BytesIO(resp.content))
        except requests.Timeout:
            raise requests.RequestException("Request to Google Sheets timed out. Please try again.")
        except requests.RequestException as exc:
            raise requests.RequestException(f"Failed to fetch Google Sheet: {exc}")

    @staticmethod
    def _get_postgres_connection_string(data_source: dict) -> str:
        """
        Resolve the Postgres connection string from config, secrets, or environment.

        Args:
            data_source: Data source configuration dictionary

        Returns:
            Connection string

        Raises:
            ValueError: If no connection string is available
        """
        # Allow explicit config override for local/dev usage
        config_value = data_source.get("postgres_connection_string")
        if config_value:
            return config_value

        # Prefer Streamlit secrets if available
        connection_string = None
        try:
            secrets = st.secrets if st else None
        except Exception:
            secrets = None

        if secrets:
            try:
                if "postgres_connection_string" in secrets:
                    connection_string = secrets["postgres_connection_string"]
                elif "postgres" in secrets and "connection_string" in secrets["postgres"]:
                    connection_string = secrets["postgres"]["connection_string"]
            except Exception:
                connection_string = None

        if not connection_string:
            # Fall back to environment variable for CLI or batch jobs
            connection_string = os.getenv("POSTGRES_CONNECTION_STRING")

        if not connection_string:
            raise ValueError(
                "Postgres connection string not configured. "
                "Set 'postgres_connection_string' in data_source, "
                "Streamlit secrets `[postgres].connection_string`, "
                "or POSTGRES_CONNECTION_STRING env var."
            )
        return connection_string

    @staticmethod
    def _read_postgres_data(connection_string: str, data_source: dict) -> pd.DataFrame:
        """
        Execute a Postgres query or fetch an entire table into a DataFrame.

        Args:
            connection_string: SQLAlchemy connection string
            data_source: Data source configuration dictionary

        Returns:
            DataFrame containing the requested data

        Raises:
            ValueError | SQLAlchemyError | pandas errors
        """
        query = data_source.get("postgres_query")
        table_name = data_source.get("postgres_table") or data_source.get("table_name")

        if not query and not table_name:
            raise ValueError(
                "Postgres data source requires either 'postgres_query' or 'postgres_table' in config."
            )

        engine = create_engine(connection_string, pool_pre_ping=True)

        try:
            with engine.connect() as conn:
                if query:
                    return pd.read_sql_query(text(query), conn)
                return pd.read_sql_table(table_name, conn)
        finally:
            engine.dispose()

    @staticmethod
    def load_data(config: dict) -> Optional[pd.DataFrame]:
        """
        Load data based on configuration.

        Args:
            config: Configuration dictionary with data_source settings

        Returns:
            DataFrame if successful, None otherwise
        """
        data_source = config.get("data_source", {})
        source_type = data_source.get("type", "gsheet")

        if source_type == "gsheet":
            gsheet_url = data_source.get("gsheet_url")
            if not gsheet_url:
                if st:
                    st.error("Google Sheet URL not configured")
                return None

            try:
                return DataSource.read_google_sheet(
                    gsheet_url,
                    data_source.get("sheet_name")
                )
            except Exception as exc:
                if st:
                    st.error(f"Error reading Google Sheet: {exc}")
                return None

        if source_type == "postgres":
            try:
                connection_string = DataSource._get_postgres_connection_string(data_source)
                return DataSource._read_postgres_data(connection_string, data_source)
            except (ValueError, SQLAlchemyError, requests.RequestException) as exc:
                if st:
                    st.error(f"Error loading data from Postgres: {exc}")
                return None
            except Exception as exc:
                if st:
                    st.error(f"Unexpected Postgres error: {exc}")
                return None

        # Add other data source types here (database, CSV, etc.)
        if st:
            st.error(f"Unsupported data source type: {source_type}")
        return None

    @staticmethod
    def load_penalty_data(config: dict, sheet_name: str = "Sheet2") -> Optional[pd.DataFrame]:
        """
        Load penalty data from a specific sheet.

        Args:
            config: Configuration dictionary
            sheet_name: Name of the sheet containing penalty data

        Returns:
            DataFrame if successful, None otherwise
        """
        data_source = config.get("data_source", {})
        source_type = data_source.get("type", "gsheet")

        if source_type == "gsheet":
            gsheet_url = data_source.get("gsheet_url")
            if not gsheet_url:
                return None

            try:
                return DataSource.read_google_sheet(gsheet_url, sheet_name)
            except Exception as exc:
                if st:
                    st.warning(f"Could not load penalty data from {sheet_name}: {exc}")
                return None

        return None
