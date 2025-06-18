"""
Enhanced File Parser Service

This service provides comprehensive file parsing for portfolio uploads:
- CSV and Excel file support with multiple encodings
- Robust column mapping and data cleaning
- Integration with new input schemas
- Detailed error reporting and validation
- Support for Indian stock market formats

Key features:
- Intelligent column detection and mapping
- Multiple encoding support for CSV files
- Data validation and cleaning
- Detailed parsing reports with row-level feedback
- Integration with portfolio validation service

Supported formats:
- CSV files (UTF-8, ISO-8859-1, CP1252 encodings)
- Excel files (.xlsx, .xls)
- TSV (Tab-separated values)

Integration:
- Uses app.schemas.input models for type safety
- Integrates with portfolio validation service
- Provides detailed parsing reports
- Supports batch file processing
"""

import pandas as pd
import io
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, date
from dataclasses import dataclass
from enum import Enum
import logging

from app.schemas.input import (
    PortfolioHolding, PortfolioInput, FileParseResponse,
    FileUploadRequest
)

# Configure logger
logger = logging.getLogger(__name__)


class ParseError(Exception):
    """Custom exception for parsing errors"""
    pass


@dataclass
class ParseWarning:
    """Represents a parsing warning"""
    row: int
    column: str
    message: str
    original_value: Any
    corrected_value: Any = None


class FileParserService:
    """
    Enhanced service for parsing portfolio files and converting them to PortfolioInput objects.
    """
    
    # Enhanced column name mappings for Indian market context
    COLUMN_MAPPINGS = {
        'ticker': [
            'ticker', 'symbol', 'stock', 'security', 'instrument', 'scrip',
            'stock_symbol', 'stock_code', 'isin', 'nse_symbol', 'bse_code'
        ],
        'quantity': [
            'quantity', 'shares', 'units', 'amount', 'holding', 'qty',
            'no_of_shares', 'share_quantity', 'holdings', 'volume'
        ],
        'avg_buy_price': [
            'avg_price', 'average_price', 'price', 'cost', 'purchase_price', 
            'avg_cost', 'buy_price', 'cost_price', 'avg_buy_price',
            'average_cost', 'unit_cost', 'acquisition_price'
        ],
        'current_price': [
            'current_price', 'market_price', 'last_price', 'quote',
            'ltp', 'last_traded_price', 'closing_price', 'close_price'
        ],
        'buy_date': [
            'buy_date', 'purchase_date', 'date', 'transaction_date',
            'acquisition_date', 'bought_on', 'date_of_purchase'
        ]
    }
    
    # Date formats commonly used in Indian financial files
    DATE_FORMATS = [
        '%Y-%m-%d',      # 2023-12-31
        '%d/%m/%Y',      # 31/12/2023
        '%d-%m-%Y',      # 31-12-2023
        '%d/%m/%y',      # 31/12/23
        '%d-%m-%y',      # 31-12-23
        '%Y/%m/%d',      # 2023/12/31
        '%m/%d/%Y',      # 12/31/2023
        '%d %b %Y',      # 31 Dec 2023
        '%d-%b-%Y',      # 31-Dec-2023
    ]
    
    def __init__(self):
        """Initialize the file parser service"""
        self.warnings: List[ParseWarning] = []
        self.rows_processed = 0
        self.rows_skipped = 0
    
    async def parse_file(
        self, 
        file_content: bytes, 
        file_request: FileUploadRequest,
        validate_data: bool = True
    ) -> FileParseResponse:
        """
        Parse portfolio file and return detailed parsing response.
        
        Args:
            file_content: Raw file content
            file_request: File upload request with metadata
            validate_data: Whether to perform data validation
            
        Returns:
            Detailed file parse response with portfolio data and errors
        """
        logger.info(f"🔄 Starting parse_file for: {file_request.filename}")
        
        self.warnings = []
        self.rows_processed = 0
        self.rows_skipped = 0
        
        try:
            # Determine file type and parse
            filename = file_request.filename
            logger.info(f"📁 Processing file: {filename}")
            
            if filename.lower().endswith('.csv'):
                logger.info("📊 Parsing as CSV")
                df = self._parse_csv(file_content)
            elif filename.lower().endswith(('.xlsx', '.xls')):
                logger.info("📈 Parsing as Excel")
                df = self._parse_excel(file_content)
            elif filename.lower().endswith('.tsv'):
                logger.info("📋 Parsing as TSV")
                df = self._parse_tsv(file_content)
            else:
                logger.error(f"❌ Unsupported file type: {filename}")
                return FileParseResponse(
                    success=False,
                    portfolio=None,
                    errors=[f"Unsupported file type: {filename}. Supported formats: CSV, Excel (.xlsx, .xls), TSV"],
                    warnings=[],
                    rows_processed=0,
                    rows_skipped=0
                )
            
            logger.info(f"📊 DataFrame created - shape: {df.shape}, columns: {list(df.columns)}")
            logger.debug(f"📄 DataFrame head:\n{df.head()}")
            
            # Validate minimum data requirements
            if df.empty:
                logger.error("📭 DataFrame is empty")
                return FileParseResponse(
                    success=False,
                    portfolio=None,
                    errors=["File is empty or contains no readable data"],
                    warnings=[],
                    rows_processed=0,
                    rows_skipped=0
                )
            
            # Clean and validate data
            logger.info("🧹 Starting data cleaning...")
            df_cleaned, cleaning_errors = self._clean_dataframe(df)
            logger.info(f"✅ Cleaning complete - cleaned shape: {df_cleaned.shape}, errors: {len(cleaning_errors)}")
            
            if cleaning_errors:
                logger.warning(f"⚠️ Cleaning errors: {cleaning_errors}")
            
            if df_cleaned.empty:
                logger.error("📭 No valid data after cleaning")
                return FileParseResponse(
                    success=False,
                    portfolio=None,
                    errors=["No valid data rows found after cleaning"] + cleaning_errors,
                    warnings=[w.message for w in self.warnings],
                    rows_processed=self.rows_processed,
                    rows_skipped=self.rows_skipped
                )
            
            # Convert to PortfolioInput
            logger.info("🔄 Converting to PortfolioInput...")
            portfolio_input, conversion_errors = self._dataframe_to_portfolio_input(df_cleaned)
            logger.info(f"✅ Conversion complete - holdings: {len(portfolio_input.holdings) if portfolio_input else 0}")
            
            if conversion_errors:
                logger.warning(f"⚠️ Conversion errors: {conversion_errors}")
            
            # Final result
            all_errors = cleaning_errors + conversion_errors
            success = portfolio_input and portfolio_input.holdings
            
            logger.info(f"🎯 Final result - success: {success}, total errors: {len(all_errors)}")
            
            if portfolio_input and portfolio_input.holdings:
                return FileParseResponse(
                    success=True,
                    portfolio=portfolio_input,
                    errors=all_errors,
                    warnings=[w.message for w in self.warnings],
                    rows_processed=self.rows_processed,
                    rows_skipped=self.rows_skipped
                )
            else:
                return FileParseResponse(
                    success=False,
                    portfolio=None,
                    errors=["Failed to create valid portfolio from file data"] + all_errors,
                    warnings=[w.message for w in self.warnings],
                    rows_processed=self.rows_processed,
                    rows_skipped=self.rows_skipped
                )
                
        except Exception as e:
            logger.error(f"❌ Parse failed with exception: {str(e)}", exc_info=True)
            return FileParseResponse(
                success=False,
                portfolio=None,
                errors=[f"Failed to parse file {filename}: {str(e)}"],
                warnings=[],
                rows_processed=self.rows_processed,
                rows_skipped=self.rows_skipped
            )
    
    def _parse_csv(self, file_content: bytes) -> pd.DataFrame:
        """Parse CSV file content with multiple encoding support."""
        logger.info(f"📊 Parsing CSV - content length: {len(file_content)} bytes")
        
        # Show raw content sample
        try:
            sample_content = file_content[:500].decode('utf-8', errors='ignore')
            logger.debug(f"📄 Raw CSV sample:\n{sample_content}")
        except Exception as e:
            logger.warning(f"⚠️ Could not decode sample content: {e}")
        
        encodings = ['utf-8', 'utf-8-sig', 'iso-8859-1', 'cp1252', 'latin1']
        separators = [',', ';', '\t']
        
        for encoding in encodings:
            for separator in separators:
                try:
                    logger.debug(f"🔄 Trying encoding: {encoding}, separator: '{separator}'")
                    content_str = file_content.decode(encoding)
                    
                    # Show decoded content sample
                    logger.debug(f"📄 Decoded sample (first 300 chars):\n{content_str[:300]}")
                    
                    df = pd.read_csv(
                        io.StringIO(content_str),
                        sep=separator,
                        skipinitialspace=True,
                        na_values=['', 'NA', 'N/A', 'null', 'NULL', '-']
                    )
                    
                    logger.info(f"✅ Successfully parsed with {encoding}/{separator} - shape: {df.shape}")
                    logger.debug(f"📊 Parsed columns: {list(df.columns)}")
                    
                    # Check if parsing was successful (has multiple columns)
                    if len(df.columns) >= 3:
                        return df
                    else:
                        logger.debug(f"⚠️ Not enough columns: {len(df.columns)}")
                        
                except (UnicodeDecodeError, pd.errors.Error) as e:
                    logger.debug(f"❌ Failed with {encoding}/{separator}: {str(e)}")
                    continue
        
        logger.error("❌ Unable to parse CSV with any supported encoding/separator")
        raise ParseError("Unable to parse CSV file with supported encodings and separators")
    
    def _parse_excel(self, file_content: bytes) -> pd.DataFrame:
        """Parse Excel file content."""
        try:
            # Try reading the first sheet
            df = pd.read_excel(
                io.BytesIO(file_content),
                na_values=['', 'NA', 'N/A', 'null', 'NULL', '-']
            )
            return df
            
        except Exception as e:
            # Try reading with different engines
            try:
                df = pd.read_excel(
                    io.BytesIO(file_content),
                    engine='openpyxl',
                    na_values=['', 'NA', 'N/A', 'null', 'NULL', '-']
                )
                return df
            except Exception:
                raise ParseError(f"Failed to parse Excel file: {str(e)}")
    
    def _parse_tsv(self, file_content: bytes) -> pd.DataFrame:
        """Parse TSV (Tab-separated values) file content."""
        try:
            content_str = file_content.decode('utf-8')
            df = pd.read_csv(
                io.StringIO(content_str),
                sep='\t',
                skipinitialspace=True,
                na_values=['', 'NA', 'N/A', 'null', 'NULL', '-']
            )
            return df
        except Exception as e:
            raise ParseError(f"Failed to parse TSV file: {str(e)}")
    
    def _clean_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Clean and validate DataFrame with detailed error reporting."""
        logger.info(f"🧹 Starting dataframe cleaning - input shape: {df.shape}")
        logger.debug(f"📊 Original columns: {list(df.columns)}")
        logger.debug(f"📄 First few rows:\n{df.head()}")
        
        # Show raw data for first few rows
        for idx, row in df.head(3).iterrows():
            logger.debug(f"📄 Row {idx} raw data: {dict(row)}")
        
        errors = []
        original_rows = len(df)
        
        # Remove completely empty rows
        df = df.dropna(how='all')
        logger.info(f"🧹 Removed empty rows: {original_rows} -> {len(df)}")
        if len(df) < original_rows:
            self.rows_skipped += (original_rows - len(df))
        
        # Normalize column names
        original_columns = list(df.columns)
        df.columns = df.columns.astype(str).str.lower().str.strip().str.replace(' ', '_')
        logger.info(f"🏷️ Column mapping: {dict(zip(original_columns, df.columns))}")
        
        # Map columns to standard names
        column_mapping = self._detect_columns(df.columns)
        logger.info(f"🗺️ Detected column mapping: {column_mapping}")
        
        if not column_mapping:
            logger.error("❌ Could not identify required columns")
            errors.append("Could not identify required columns (ticker, quantity, avg_buy_price)")
            return pd.DataFrame(), errors
        
        df = df.rename(columns=column_mapping)
        logger.info(f"🏷️ Final columns after mapping: {list(df.columns)}")
        
        # Show data after column mapping
        logger.debug(f"📄 Data after column mapping:\n{df.head()}")
        
        # Check required columns
        required_columns = ['ticker', 'quantity', 'avg_buy_price']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {', '.join(missing_columns)}")
            return pd.DataFrame(), errors
        
        # Clean data row by row
        cleaned_rows = []
        for idx, row in df.iterrows():
            try:
                cleaned_row = self._clean_row(row, idx)
                if cleaned_row is not None:
                    cleaned_rows.append(cleaned_row)
                    self.rows_processed += 1
                else:
                    self.rows_skipped += 1
            except Exception as e:
                self.warnings.append(ParseWarning(
                    row=idx,
                    column="all",
                    message=f"Skipped row {idx}: {str(e)}",
                    original_value=row.to_dict()
                ))
                self.rows_skipped += 1
        
        if not cleaned_rows:
            errors.append("No valid data rows found after cleaning")
            return pd.DataFrame(), errors
        
        # Create cleaned DataFrame
        df_cleaned = pd.DataFrame(cleaned_rows)
        
        # Remove duplicates based on ticker
        initial_count = len(df_cleaned)
        df_cleaned = df_cleaned.drop_duplicates(subset=['ticker'], keep='first')
        duplicates_removed = initial_count - len(df_cleaned)
        
        if duplicates_removed > 0:
            self.warnings.append(ParseWarning(
                row=-1,
                column="ticker",
                message=f"Removed {duplicates_removed} duplicate ticker entries",
                original_value=duplicates_removed
            ))
        
        return df_cleaned, errors
    
    def _detect_columns(self, columns: List[str]) -> Dict[str, str]:
        """Intelligently detect and map column names."""
        column_mapping = {}
        
        for standard_name, variations in self.COLUMN_MAPPINGS.items():
            for col in columns:
                col_clean = str(col).lower().strip().replace(' ', '_')
                
                # Exact match
                if col_clean in [v.lower().replace(' ', '_') for v in variations]:
                    column_mapping[col] = standard_name
                    break
                
                # Partial match
                for variation in variations:
                    variation_clean = variation.lower().replace(' ', '_')
                    if variation_clean in col_clean or col_clean in variation_clean:
                        column_mapping[col] = standard_name
                        break
                
                if col in column_mapping:
                    continue
        
        return column_mapping
    
    def _clean_row(self, row: pd.Series, row_idx: int) -> Optional[Dict[str, Any]]:
        """Clean individual row and return cleaned data or None if invalid."""
        cleaned_row = {}
        
        # Clean ticker
        ticker = self._clean_ticker(row.get('ticker'), row_idx)
        if not ticker:
            return None
        cleaned_row['ticker'] = ticker
        
        # Clean quantity
        quantity = self._clean_quantity(row.get('quantity'), row_idx)
        if quantity is None:
            return None
        cleaned_row['quantity'] = quantity
        
        # Clean average buy price
        avg_buy_price = self._clean_price(row.get('avg_buy_price'), row_idx, 'avg_buy_price')
        if avg_buy_price is None:
            return None
        cleaned_row['avg_buy_price'] = avg_buy_price
        
        # Clean current price (optional)
        if 'current_price' in row and pd.notna(row['current_price']):
            current_price = self._clean_price(row.get('current_price'), row_idx, 'current_price')
            if current_price is not None:
                cleaned_row['current_price'] = current_price
        
        # Clean buy date (optional)
        if 'buy_date' in row and pd.notna(row['buy_date']):
            buy_date = self._clean_date(row.get('buy_date'), row_idx)
            if buy_date:
                cleaned_row['buy_date'] = buy_date
        
        return cleaned_row
    
    def _clean_ticker(self, ticker_value: Any, row_idx: int) -> Optional[str]:
        """Clean and validate ticker symbol."""
        if pd.isna(ticker_value) or ticker_value == '':
            return None
        
        ticker = str(ticker_value).upper().strip()
        
        # Remove common prefixes/suffixes from broker reports
        ticker = re.sub(r'^(NSE:|BSE:|EQ:|BE:)', '', ticker)
        ticker = re.sub(r'(-EQ|-BE)$', '', ticker)
        
        # Basic validation
        if not ticker or len(ticker) > 20:
            self.warnings.append(ParseWarning(
                row=row_idx,
                column='ticker',
                message=f"Invalid ticker format: {ticker_value}",
                original_value=ticker_value
            ))
            return None
        
        # Check for valid characters
        if not re.match(r'^[A-Z0-9\.\-&]+$', ticker):
            self.warnings.append(ParseWarning(
                row=row_idx,
                column='ticker',
                message=f"Ticker contains invalid characters: {ticker}",
                original_value=ticker_value,
                corrected_value=ticker
            ))
        
        return ticker
    
    def _clean_quantity(self, quantity_value: Any, row_idx: int) -> Optional[int]:
        """Clean and validate quantity."""
        if pd.isna(quantity_value):
            return None
        
        try:
            # Handle string quantities with commas
            if isinstance(quantity_value, str):
                quantity_value = quantity_value.replace(',', '').strip()
            
            quantity = float(quantity_value)
            
            # Convert to integer (Indian markets typically trade in whole shares)
            quantity_int = int(round(quantity))
            
            if quantity_int <= 0:
                self.warnings.append(ParseWarning(
                    row=row_idx,
                    column='quantity',
                    message=f"Invalid quantity: {quantity_value} (must be positive)",
                    original_value=quantity_value
                ))
                return None
            
            if quantity_int > 1000000:  # Reasonable upper limit
                self.warnings.append(ParseWarning(
                    row=row_idx,
                    column='quantity',
                    message=f"Very large quantity: {quantity_int:,} - please verify",
                    original_value=quantity_value,
                    corrected_value=quantity_int
                ))
            
            return quantity_int
            
        except (ValueError, TypeError):
            self.warnings.append(ParseWarning(
                row=row_idx,
                column='quantity',
                message=f"Could not parse quantity: {quantity_value}",
                original_value=quantity_value
            ))
            return None
    
    def _clean_price(self, price_value: Any, row_idx: int, column: str) -> Optional[float]:
        """Clean and validate price values."""
        logger.debug(f"🔢 Cleaning price - row {row_idx}, column {column}, value: '{price_value}' (type: {type(price_value)})")
        
        if pd.isna(price_value):
            logger.debug(f"🔢 Price is NaN for row {row_idx}")
            return None
        
        try:
            original_value = price_value
            
            # Handle string prices with currency symbols and commas
            if isinstance(price_value, str):
                logger.debug(f"🔢 Processing string price: '{price_value}'")
                # Remove currency symbols, commas, and underscores
                price_clean = re.sub(r'[₹$,\s_]', '', price_value.strip())
                logger.debug(f"🔢 After regex cleaning: '{price_clean}'")
                price_value = price_clean
            
            price = float(price_value)
            logger.debug(f"🔢 Converted to float: {price}")
            
            if price <= 0:
                logger.warning(f"🔢 Invalid price <= 0: {original_value} -> {price}")
                self.warnings.append(ParseWarning(
                    row=row_idx,
                    column=column,
                    message=f"Invalid price: {original_value} (must be positive)",
                    original_value=original_value
                ))
                return None
            
            # Reasonable price range check (₹0.01 to ₹1,00,000)
            if price < 0.01 or price > 100000:
                logger.warning(f"🔢 Price outside expected range: {price}")
                self.warnings.append(ParseWarning(
                    row=row_idx,
                    column=column,
                    message=f"Price seems unusual: ₹{price:,.2f} - please verify",
                    original_value=original_value,
                    corrected_value=price
                ))
            
            result = round(price, 2)
            logger.debug(f"🔢 Final price result: {result}")
            return result
            
        except (ValueError, TypeError) as e:
            logger.error(f"🔢 Failed to parse price '{price_value}': {str(e)}")
            self.warnings.append(ParseWarning(
                row=row_idx,
                column=column,
                message=f"Could not parse price: {price_value}",
                original_value=price_value
            ))
            return None
    
    def _clean_date(self, date_value: Any, row_idx: int) -> Optional[date]:
        """Clean and validate date values."""
        if pd.isna(date_value):
            return None
        
        # If already a date object
        if isinstance(date_value, (date, datetime)):
            return date_value.date() if isinstance(date_value, datetime) else date_value
        
        # Try parsing string dates
        if isinstance(date_value, str):
            date_str = date_value.strip()
            
            for date_format in self.DATE_FORMATS:
                try:
                    parsed_date = datetime.strptime(date_str, date_format).date()
                    
                    # Validate date range
                    if parsed_date > date.today():
                        self.warnings.append(ParseWarning(
                            row=row_idx,
                            column='buy_date',
                            message=f"Future date detected: {parsed_date}",
                            original_value=date_value
                        ))
                        return None
                    
                    if parsed_date < date(1990, 1, 1):
                        self.warnings.append(ParseWarning(
                            row=row_idx,
                            column='buy_date',
                            message=f"Very old date: {parsed_date} - please verify",
                            original_value=date_value,
                            corrected_value=parsed_date
                        ))
                    
                    return parsed_date
                    
                except ValueError:
                    continue
        
        self.warnings.append(ParseWarning(
            row=row_idx,
            column='buy_date',
            message=f"Could not parse date: {date_value}",
            original_value=date_value
        ))
        return None
    
    def _dataframe_to_portfolio_input(self, df: pd.DataFrame) -> Tuple[Optional[PortfolioInput], List[str]]:
        """Convert cleaned DataFrame to PortfolioInput object."""
        errors = []
        holdings = []
        
        for _, row in df.iterrows():
            try:
                # Create PortfolioHolding
                holding_data = {
                    'ticker': row['ticker'],
                    'quantity': int(row['quantity']),
                    'avg_buy_price': float(row['avg_buy_price'])
                }
                
                # Add optional fields
                if 'current_price' in row and pd.notna(row['current_price']):
                    holding_data['current_price'] = float(row['current_price'])
                
                if 'buy_date' in row and pd.notna(row['buy_date']):
                    holding_data['buy_date'] = row['buy_date']
                
                # Create and validate holding
                holding = PortfolioHolding(**holding_data)
                holdings.append(holding)
                
            except Exception as e:
                errors.append(f"Failed to create holding for {row.get('ticker', 'unknown')}: {str(e)}")
        
        if not holdings:
            errors.append("No valid holdings could be created from the file")
            return None, errors
        
        try:
            # Create PortfolioInput
            portfolio_input = PortfolioInput(holdings=holdings)
            return portfolio_input, errors
            
        except Exception as e:
            errors.append(f"Failed to create portfolio: {str(e)}")
            return None, errors
    
    def get_sample_format(self) -> Dict[str, Any]:
        """Return sample file format for user reference."""
        return {
            "csv_example": {
                "headers": ["ticker", "quantity", "avg_buy_price", "current_price", "buy_date"],
                "sample_rows": [
                    ["RELIANCE", "50", "2450.00", "2520.00", "2023-01-15"],
                    ["TCS", "25", "3200.00", "3450.00", "2023-02-10"],
                    ["HDFCBANK", "75", "1650.00", "1720.00", "2023-03-05"],
                    ["INFY", "100", "1250.00", "1380.00", "2023-01-20"]
                ]
            },
            "supported_formats": [
                "CSV (.csv)",
                "Excel (.xlsx, .xls)",
                "Tab-separated values (.tsv)"
            ],
            "column_variations": {
                "ticker": "ticker, symbol, stock, security, scrip, nse_symbol",
                "quantity": "quantity, shares, units, qty, no_of_shares",
                "avg_buy_price": "avg_price, average_price, price, cost, buy_price",
                "current_price": "current_price, market_price, ltp, last_price",
                "buy_date": "buy_date, purchase_date, date, transaction_date"
            },
            "data_formats": {
                "ticker": "NSE/BSE symbol (e.g., RELIANCE, TCS)",
                "quantity": "Positive integer (whole shares)",
                "prices": "Positive number (₹ symbol optional)",
                "dates": "DD/MM/YYYY, DD-MM-YYYY, YYYY-MM-DD"
            },
            "notes": [
                "Required columns: ticker, quantity, avg_buy_price",
                "Optional columns: current_price, buy_date",
                "File size limit: 10MB",
                "Maximum 100 holdings per portfolio",
                "Duplicate tickers will be merged (first entry kept)",
                "Currency symbols (₹) will be automatically removed",
                "Dates can be in multiple formats",
                "Empty rows and invalid data will be skipped"
            ]
        }
    
    async def validate_file_format(self, file_request: FileUploadRequest) -> Tuple[bool, List[str]]:
        """Validate file format before processing."""
        errors = []
        
        # Check file extension
        valid_extensions = ['.csv', '.xlsx', '.xls', '.tsv']
        if not any(file_request.filename.lower().endswith(ext) for ext in valid_extensions):
            errors.append(f"Unsupported file format. Supported: {', '.join(valid_extensions)}")
        
        # Check file size
        max_size = 10 * 1024 * 1024  # 10MB
        if file_request.file_size > max_size:
            errors.append(f"File too large ({file_request.file_size / (1024*1024):.1f}MB). Maximum size: 10MB")
        
        return len(errors) == 0, errors
