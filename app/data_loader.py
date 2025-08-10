import pandas as pd
import requests
from bs4 import BeautifulSoup
from io import StringIO
import re
import duckdb
import time

def load_csv(path):
    """Load CSV file"""
    return pd.read_csv(path)

def extract_url(task: str) -> str:
    """Extract the first URL from a task string, if present."""
    url_match = re.search(r'(https?://[^\s]+)', task)
    return url_match.group(1) if url_match else None

def scrape_wikipedia(task):
    """Scrape Wikipedia table data"""
    # Extract URL from task string
    url = extract_url(task)
    if not url:
        raise ValueError("No URL found in task.")
    
    # Add headers to avoid being blocked
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    
    soup = BeautifulSoup(resp.text, "html.parser")
    
    # Find candidate tables
    tables = soup.find_all("table")
    if not tables:
        raise ValueError("No table found on the page.")

    def normalize_cols(df):
        try:
            if isinstance(df.columns, pd.MultiIndex):
                new_cols = []
                for tup in df.columns.tolist():
                    parts = [str(part).strip() for part in tup if part is not None and str(part).strip() != ""]
                    new_cols.append(" ".join(parts))
                df.columns = new_cols
            else:
                df.columns = [str(col).strip() for col in df.columns]
        except Exception:
            df.columns = [str(col) for col in df.columns]
        return df

    def score_columns(cols):
        keys = [
            ("revenue", 3), ("net income", 2), ("employee", 2),
            ("fy", 2), ("year", 2)
        ]
        s = 0
        for c in cols:
            cl = str(c).lower()
            for k, w in keys:
                if k in cl:
                    s += w
        return s

    best_df = None
    best_score = -1
    fallback_df = None
    
    for t in tables:
        try:
            dfl = pd.read_html(StringIO(str(t)))
            if not dfl:
                continue
            df0 = normalize_cols(dfl[0])
            if fallback_df is None:
                fallback_df = df0
            score = score_columns(df0.columns)
            if score > best_score:
                best_score = score
                best_df = df0
        except Exception:
            continue

    df = best_df if best_df is not None else fallback_df
    if df is None:
        raise ValueError("No readable table found on the page.")

    # Columns already normalized above
    
    return df

def get_wikipedia_summary(task: str) -> str | None:
    """Fetch a brief summary (first meaningful paragraph) from a Wikipedia article."""
    url = extract_url(task)
    if not url:
        return None
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Find the content area and first non-empty paragraph
        content = soup.find(id="mw-content-text") or soup
        paragraphs = content.find_all("p", recursive=True)
        for p in paragraphs:
            text = p.get_text(" ", strip=True)
            if not text:
                continue
            # Skip disambiguation-like or navigation texts
            if "may refer to:" in text.lower():
                continue
            # Remove citation markers like [1], [2]
            text = re.sub(r"\[\d+\]", "", text)
            # Trim overly long text to a reasonable size
            if len(text) > 1200:
                text = text[:1200].rsplit(" ", 1)[0] + "…"
            return text
    except Exception:
        return None
    return None

def query_s3_parquet(s3_url: str, query: str):
    """Query S3 parquet files using DuckDB"""
    try:
        con = duckdb.connect()
        con.execute("INSTALL httpfs; LOAD httpfs;")
        con.execute("INSTALL parquet; LOAD parquet;")
        
        # Set S3 region
        con.execute("SET s3_region='ap-south-1';")
        
        result = con.execute(query).fetchdf()
        con.close()
        return result
    except Exception as e:
        print(f"Error querying S3: {e}")
        # Return empty DataFrame with expected columns
        return pd.DataFrame()

def clean_scraped_df(df):
    """Clean scraped Wikipedia data"""
    try:
        # Clean 'Year' column: extract the first 4-digit number
        if 'Year' in df.columns:
            df['Year'] = df['Year'].astype(str).str.extract(r'(\d{4})').astype(float)
        
        # Clean 'Worldwide gross': remove $ and commas, convert to float
        if 'Worldwide gross' in df.columns:
            df['Worldwide gross'] = (
                df['Worldwide gross']
                .astype(str)
                .str.replace(r'[\$,]', '', regex=True)
                .str.extract(r'(\d+\.?\d*)')[0]
                .astype(float)
            )
        
        # Clean 'Rank' and 'Peak'
        if 'Rank' in df.columns:
            df['Rank'] = pd.to_numeric(df['Rank'], errors='coerce')
        if 'Peak' in df.columns:
            df['Peak'] = pd.to_numeric(df['Peak'], errors='coerce')
        
        # Clean 'Title' column
        if 'Title' in df.columns:
            df['Title'] = df['Title'].astype(str).str.strip()
        
        return df
    except Exception as e:
        print(f"Error cleaning data: {e}")
        return df

def get_sample_high_court_data():
    """Get sample high court data for testing when S3 is not accessible"""
    # Create sample data for testing
    sample_data = {
        'court': ['33_10', '33_10', '33_10', '33_10', '33_10'],
        'year': [2019, 2020, 2021, 2022, 2023],
        'avg_delay': [30, 35, 40, 45, 50]
    }
    return pd.DataFrame(sample_data)