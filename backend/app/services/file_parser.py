import pandas as pd
import io
from typing import List, Dict, Any
from datetime import datetime

from app.models.portfolio import Portfolio, PortfolioItem


class FileParserService:
    """
    Service for parsing portfolio files (CSV, Excel) and converting them to Portfolio objects.
    """
    
    # Common column name mappings
    COLUMN_MAPPINGS = {
        'ticker': ['ticker', 'symbol', 'stock', 'security', 'instrument'],
        'quantity': ['quantity', 'shares', 'units', 'amount', 'holding'],
        'avg_price': ['avg_price', 'average_price', 'price', 'cost', 'purchase_price', 'avg_cost'],
        'current_price': ['current_price', 'market_price', 'last_price', 'quote'],
    }
    
    async def parse_file(self, file_content: bytes, filename: str) -> Portfolio:
        """
        Parse portfolio file and return Portfolio object.
        
        Args:
            file_content: Raw file content
            filename: Name of the uploaded file
            
        Returns:
            Portfolio object with parsed data
        """
        try:
            # Determine file type and parse
            if filename.lower().endswith('.csv'):
                df = self._parse_csv(file_content)
            elif filename.lower().endswith(('.xlsx', '.xls')):
                df = self._parse_excel(file_content)
            else:
                raise ValueError(f"Unsupported file type: {filename}")
            
            # Validate and clean data
            df = self._clean_dataframe(df)
            
            # Convert to Portfolio
            portfolio = self._dataframe_to_portfolio(df)
            
            return portfolio
            
        except Exception as e:
            raise Exception(f"Failed to parse file {filename}: {str(e)}")
    
    def _parse_csv(self, file_content: bytes) -> pd.DataFrame:
        """Parse CSV file content."""
        try:
            # Try different encodings
            for encoding in ['utf-8', 'iso-8859-1', 'cp1252']:
                try:
                    content_str = file_content.decode(encoding)
                    df = pd.read_csv(io.StringIO(content_str))
                    return df
                except UnicodeDecodeError:
                    continue
            
            raise ValueError("Unable to decode CSV file with supported encodings")
            
        except Exception as e:
            raise Exception(f"Failed to parse CSV: {str(e)}")
    
    def _parse_excel(self, file_content: bytes) -> pd.DataFrame:
        """Parse Excel file content."""
        try:
            df = pd.read_excel(io.BytesIO(file_content))
            return df
        except Exception as e:
            raise Exception(f"Failed to parse Excel: {str(e)}")
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate DataFrame."""
        if df.empty:
            raise ValueError("File is empty or contains no data")
        
        # Normalize column names
        df.columns = df.columns.str.lower().str.strip()
        
        # Map columns to standard names
        column_mapping = {}
        for standard_name, variations in self.COLUMN_MAPPINGS.items():
            for col in df.columns:
                if any(variation in col for variation in variations):
                    column_mapping[col] = standard_name
                    break
        
        df = df.rename(columns=column_mapping)
        
        # Check required columns
        required_columns = ['ticker', 'quantity', 'avg_price']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Remove rows with missing critical data
        df = df.dropna(subset=required_columns)
        
        # Clean ticker symbols
        if 'ticker' in df.columns:
            df['ticker'] = df['ticker'].astype(str).str.upper().str.strip()
            df = df[df['ticker'] != '']
        
        # Ensure numeric columns are numeric
        numeric_columns = ['quantity', 'avg_price', 'current_price']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with invalid numeric data
        df = df.dropna(subset=['quantity', 'avg_price'])
        df = df[df['quantity'] > 0]
        df = df[df['avg_price'] > 0]
        
        # Remove duplicates based on ticker
        df = df.drop_duplicates(subset=['ticker'], keep='first')
        
        return df
    
    def _dataframe_to_portfolio(self, df: pd.DataFrame) -> Portfolio:
        """Convert DataFrame to Portfolio object."""
        items = []
        
        for _, row in df.iterrows():
            # Calculate current price if not provided
            current_price = row.get('current_price', None)
            if pd.isna(current_price):
                current_price = None
            
            # Create portfolio item
            item = PortfolioItem(
                ticker=row['ticker'],
                quantity=float(row['quantity']),
                avg_price=float(row['avg_price']),
                current_price=current_price
            )
            
            # Calculate total value
            price_for_calculation = current_price if current_price else item.avg_price
            item.total_value = item.quantity * price_for_calculation
            
            items.append(item)
        
        # Create portfolio
        portfolio = Portfolio(items=items)
        portfolio.calculate_total_value()
        portfolio.calculate_allocations()
        
        return portfolio
    
    def get_sample_format(self) -> Dict[str, Any]:
        """Return sample file format for user reference."""
        return {
            "csv_example": {
                "headers": ["ticker", "quantity", "avg_price"],
                "sample_rows": [
                    ["AAPL", "100", "150.00"],
                    ["GOOGL", "50", "2500.00"],
                    ["MSFT", "75", "300.00"]
                ]
            },
            "column_variations": {
                "ticker": "ticker, symbol, stock, security",
                "quantity": "quantity, shares, units, amount",
                "avg_price": "avg_price, average_price, price, cost"
            },
            "notes": [
                "CSV and Excel files are supported",
                "File size limit: 10MB",
                "Required columns: ticker, quantity, avg_price",
                "Optional columns: current_price",
                "Duplicate tickers will be merged"
            ]
        } 