from openbb import obb
import pandas as pd
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()
obb.user.credentials.fmp_api_key = os.getenv('FMP_API_KEY')


SPX = obb.equity.price.historical(symbol='^GSPC', start_date='1986-01-01', end_date='2025-12-30', interval = '1d', provider = 'fmp').to_df()



