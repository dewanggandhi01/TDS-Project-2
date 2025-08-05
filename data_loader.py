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

def scrape_wikipedia(task):
    """Scrape Wikipedia table data"""
    # Extract URL from task string
    url_match = re.search(r'(https?://[^\s]+)', task)
    url = url_match.group(1) if url_match else None
    if not url:
        raise ValueError("No URL found in task.")
    
    # Add headers to avoid being blocked
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    
    soup = BeautifulSoup(resp.text, "html.parser")
    
    # Find the first wikitable (can be adjusted for specific tables)
    table = soup.find("table", {"class": "wikitable"})
    if table is None:
        # Try alternative table classes
        table = soup.find("table", {"class": "sortable"})
        if table is None:
            table = soup.find("table")
    
    if table is None:
        raise ValueError("No table found on the page.")
    
    # Read the table
    df = pd.read_html(StringIO(str(table)))[0]
    
    # Clean column names
    df.columns = [col.strip() for col in df.columns]
    
    return df

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