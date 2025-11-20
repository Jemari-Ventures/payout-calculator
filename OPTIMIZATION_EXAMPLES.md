# Optimization Examples

This document shows practical examples of the recommended optimizations.

## 1. Using Shared Modules

### Before (app.py)
```python
class ColorScheme:
    PRIMARY = "#4f46e5"
    # ... 40+ lines of color definitions
```

### After (using shared module)
```python
from shared import ColorScheme

# Use ColorScheme directly
st.markdown(f"color: {ColorScheme.PRIMARY}")
```

## 2. Improved Error Handling

### Before
```python
try:
    df = DataSource.read_google_sheet(url)
except Exception as exc:
    st.error(f"Error: {exc}")
    return None
```

### After
```python
import logging

logger = logging.getLogger(__name__)

try:
    df = DataSource.read_google_sheet(url)
except requests.Timeout:
    logger.error("Google Sheets request timed out")
    st.error("Request timed out. Please check your connection and try again.")
    return None
except requests.RequestException as e:
    logger.error(f"Network error: {e}", exc_info=True)
    st.error("Unable to connect to Google Sheets. Please check the URL.")
    return None
except ValueError as e:
    logger.error(f"Invalid URL: {e}")
    st.error("Invalid Google Sheet URL format.")
    return None
```

## 3. Configuration with Validation

### Before
```python
config = json.load(f)
# No validation - could have wrong types, missing keys, etc.
```

### After (using pydantic)
```python
from pydantic import BaseModel, Field, validator
from typing import Optional, List

class TierConfig(BaseModel):
    tier: str
    min_parcels: int = Field(alias="Min Parcels")
    max_parcels: Optional[int] = Field(alias="Max Parcels")
    rate: float = Field(alias="Rate (RM)", gt=0)

    @validator('max_parcels')
    def validate_max(cls, v, values):
        if v is not None and 'min_parcels' in values:
            if v <= values['min_parcels']:
                raise ValueError('max_parcels must be greater than min_parcels')
        return v

class AppConfig(BaseModel):
    tiers: List[TierConfig]
    currency_symbol: str = "RM"

    @validator('currency_symbol')
    def validate_currency(cls, v):
        if len(v) > 5:
            raise ValueError('Currency symbol too long')
        return v

# Usage
try:
    config = AppConfig.parse_file("config.json")
except ValidationError as e:
    st.error(f"Configuration error: {e}")
```

## 4. Optimized Data Processing

### Before
```python
# Multiple passes over dataframe
work = work.sort_values(by=["__date", "__waybill"])
work = work.drop_duplicates(subset=["__date", "__waybill"])
per_day = work.groupby(["__date"])["__waybill"].nunique()
```

### After
```python
# Single pass with vectorized operations
work = (
    work
    .sort_values(by=["__date", "__waybill", "Delivery Signature"])
    .drop_duplicates(subset=["__date", "__waybill"], keep="last")
)

per_day = (
    work
    .groupby(["__date"], as_index=False)
    .agg(daily_parcels=("__waybill", "nunique"))
)
```

## 5. Cached Column Mapping

### Before
```python
# Called multiple times for same dataframe
waybill_col = find_column(df, 'waybill')
date_col = find_column(df, 'date')
dispatcher_col = find_column(df, 'dispatcher_id')
```

### After
```python
class ColumnMapper:
    def __init__(self, df: pd.DataFrame):
        self._mapping = {}
        self._build_mapping(df)

    def _build_mapping(self, df: pd.DataFrame):
        for standard_name, possible_names in COLUMN_MAPPINGS.items():
            for col in df.columns:
                if col in possible_names:
                    self._mapping[standard_name] = col
                    break

    def get(self, standard_name: str) -> Optional[str]:
        return self._mapping.get(standard_name)

# Usage - build once, use many times
mapper = ColumnMapper(df)
waybill_col = mapper.get('waybill')
date_col = mapper.get('date')
```

## 6. Logging Setup

### Add to app.py
```python
import logging
import sys
from pathlib import Path

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "app.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Usage
logger.info("Loading data from Google Sheets")
logger.error("Failed to load data", exc_info=True)
```

## 7. Environment Variables

### Create .env file
```
GOOGLE_SHEET_URL=https://docs.google.com/spreadsheets/d/...
PENALTY_RATE=100.0
CURRENCY_SYMBOL=RM
```

### Use in code
```python
from dotenv import load_dotenv
import os

load_dotenv()

config = {
    "gsheet_url": os.getenv("GOOGLE_SHEET_URL", ""),
    "penalty_rate": float(os.getenv("PENALTY_RATE", "100.0")),
    "currency_symbol": os.getenv("CURRENCY_SYMBOL", "RM"),
}
```

## 8. Type Hints

### Before
```python
def calculate_payout(df, config):
    # ...
    return display_df, total_payout
```

### After
```python
from typing import Tuple, Dict, Any
import pandas as pd

def calculate_payout(
    df: pd.DataFrame,
    config: Dict[str, Any]
) -> Tuple[pd.DataFrame, float]:
    """
    Calculate payout for dispatchers.

    Args:
        df: DataFrame with delivery data
        config: Configuration dictionary

    Returns:
        Tuple of (display_dataframe, total_payout)
    """
    # ...
    return display_df, total_payout
```

## 9. Unit Test Example

### Create tests/test_calculator.py
```python
import pytest
import pandas as pd
from shared.utils import clean_dispatcher_name, normalize_weight

def test_clean_dispatcher_name():
    assert clean_dispatcher_name("JMR John Doe") == "John Doe"
    assert clean_dispatcher_name("ECP-Jane Smith") == "Jane Smith"
    assert clean_dispatcher_name("No Prefix") == "No Prefix"

def test_normalize_weight():
    series = pd.Series(["1.5", "2,3", "3.5 kg", "invalid"])
    result = normalize_weight(series)
    assert result.iloc[0] == 1.5
    assert result.iloc[1] == 2.3
    assert result.iloc[2] == 3.5
    assert pd.isna(result.iloc[3])
```

## 10. Requirements with Versions

### Before (requirements.txt)
```
streamlit
pandas
altair
requests
```

### After
```
streamlit>=1.28.0,<2.0.0
pandas>=2.0.0,<3.0.0
altair>=5.0.0,<6.0.0
requests>=2.31.0,<3.0.0
python-dotenv>=1.0.0
```
