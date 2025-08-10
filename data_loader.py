"""Deprecated top-level module.

Please import from the app package instead:
    from app.data_loader import scrape_wikipedia, clean_scraped_df, load_csv
"""

from app.data_loader import *  # re-export for backward compatibility