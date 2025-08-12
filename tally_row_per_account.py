import polars as pl
import duckdb
import re
from pathlib import Path
from typing import Dict, List, Optional
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TallyProcessor:
    """
    Memory-efficient processor for account and transaction data using modern data libraries.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.chunk_size = config.get('chunk_size', 10000)
        self.output_dir = Path(config.get('output_dir', 'output'))
        self.output_dir.mkdir(exist_ok=True)

        # Initialize DuckDB connection for efficient querying
        self.conn = duckdb.connect()

    def preprocess_accounts(self, accounts_file: str) -> None:
        """
        Preprocess and index customer details data using Polars for speed.
        New format: CustomerName | AccountNumber | Amount
        """
        logger.info("Preprocessing customer details data...")

        # Read with Polars for efficient processing
        if accounts_file.endswith('.xlsx'):
            accounts_df = pl.read_excel(accounts_file)
        else:
            accounts_df = pl.read_csv(accounts_file)

        # Clean and standardize data for new format
        accounts_df = accounts_df.with_columns([
            pl.col("AccountNumber").cast(pl.Utf8).alias("account_number"),
            pl.col("CustomerName").cast(pl.Utf8).alias("customer_name"),
            pl.col("Amount").cast(pl.Float64, strict=False).alias("customer_amount")
        ]).select(["account_number", "customer_name", "customer_amount"])

        # Create indexed table in DuckDB for fast lookups
        self.conn.execute("DROP TABLE IF EXISTS accounts")
        self.conn.execute("""
            CREATE TABLE accounts AS 
            SELECT * FROM accounts_df
        """)

        # Create hash index on account number for O(1) lookups
        self.conn.execute("CREATE INDEX idx_account_number ON accounts(account_number)")

        logger.info(f"Processed {len(accounts_df)} customer accounts with indexing")

    def extract_transaction_number(self, narration: str) -> Optional[str]:
        """
        Extract transaction number from narration using regex.
        """
        if not narration or narration is None or str(narration).lower() == 'nan':
            return None

        # Pattern to match "Txn No: TXN..." format
        pattern = r'Txn No:\s*(TXN\w+)'
        match = re.search(pattern, str(narration), re.IGNORECASE)
        return match.group(1) if match else None

    def process_transactions_chunk(self, chunk_df: pl.DataFrame) -> pl.DataFrame:
        """
        Process a chunk of transactions with efficient matching logic.
        New format for ledger.xlsx: AccountNumber | Date | Type | Amount | TransactionNumber | OpeningBalance | ClosingBalance
        """
        # Clean and prepare transaction data for new ledger format
        processed_chunk = chunk_df.with_columns([
            pl.col("AccountNumber").cast(pl.Utf8).alias("account_number"),
            pl.col("Date").alias("date"),
            pl.col("Type").cast(pl.Utf8).alias("transaction_type"),
            pl.col("Amount").cast(pl.Float64, strict=False).alias("amount"),
            pl.col("TransactionNumber").cast(pl.Utf8).alias("ledger_txn_no"),
            pl.col("OpeningBalance").cast(pl.Float64, strict=False).alias("opening_balance"),
            pl.col("ClosingBalance").cast(pl.Float64, strict=False).alias("closing_balance")
        ]).select(["account_number", "date", "transaction_type", "amount", "ledger_txn_no", "opening_balance", "closing_balance"])

        # Fill null values
        processed_chunk = processed_chunk.with_columns([
            pl.col("amount").fill_null(0.0),
            pl.col("opening_balance").fill_null(0.0),
            pl.col("closing_balance").fill_null(0.0)
        ])

        # Convert Type (DR/CR) to debit_amount and credit_amount
        processed_chunk = processed_chunk.with_columns([
            pl.when(pl.col("transaction_type") == "DR")
            .then(pl.col("amount"))
            .otherwise(0.0)
            .alias("debit_amount"),

            pl.when(pl.col("transaction_type") == "CR")
            .then(pl.col("amount"))
            .otherwise(0.0)
            .alias("credit_amount")
        ])

        return processed_chunk

    def match_and_tally(self, transactions_df: pl.DataFrame) -> pl.DataFrame:
        """
        Perform efficient matching and tallying using DuckDB SQL operations.
        Now produces one row per transaction instead of aggregating by account.
        """
        # Register transactions DataFrame with DuckDB
        self.conn.execute("DROP VIEW IF EXISTS transactions")
        self.conn.execute("CREATE VIEW transactions AS SELECT * FROM transactions_df")

        query = f"""
        WITH numbered_transactions AS (
            -- First, add row numbers to preserve order
            SELECT 
                t.*,
                ROW_NUMBER() OVER () as original_order
            FROM transactions t
        ),
        transaction_pairs AS (
            -- Identify DR-CR pairs by looking at consecutive transactions with same amount
            SELECT 
                nt.*,
                -- For each transaction, find the previous transaction (to match CR with preceding DR)
                LAG(nt.debit_amount) OVER (PARTITION BY nt.account_number ORDER BY nt.original_order) as prev_debit,
                LAG(nt.ledger_txn_no) OVER (PARTITION BY nt.account_number ORDER BY nt.original_order) as prev_txn_no,
                LAG(nt.date) OVER (PARTITION BY nt.account_number ORDER BY nt.original_order) as prev_date
            FROM numbered_transactions nt
        ),
        classified_pairs AS (
            -- Classify transactions into pair types after LAG functions are calculated
            SELECT 
                tp.*,
                -- Identify if this is part of a DR-CR pair
                CASE 
                    WHEN tp.debit_amount > 0 AND tp.ledger_txn_no IS NOT NULL AND tp.ledger_txn_no != '' THEN 'DR_with_txn'
                    WHEN tp.credit_amount > 0 AND tp.prev_debit = tp.credit_amount AND tp.prev_txn_no IS NOT NULL AND tp.prev_txn_no != '' THEN 'CR_paired'
                    ELSE 'other'
                END as pair_type
            FROM transaction_pairs tp
        ),
        consolidated_pairs AS (
            -- Group DR-CR pairs together, keeping only one entry per pair (DR transaction represents the pair)
            SELECT 
                cp.account_number,
                a.customer_name,
                a.customer_amount,
                -- For DR transactions or unpaired transactions, use their own data
                CAST(cp.date AS DATE) as transaction_date,
                cp.opening_balance,
                cp.debit_amount,
                cp.credit_amount,
                cp.closing_balance,
                -- Get transaction number without prefix
                CASE 
                    WHEN cp.pair_type = 'DR_with_txn' THEN cp.ledger_txn_no
                    WHEN cp.ledger_txn_no IS NOT NULL AND cp.ledger_txn_no != '' THEN cp.ledger_txn_no
                    ELSE ''
                END as clean_txn_no,
                -- For paired transactions, use the transaction date from transactions.xlsx if available
                CASE 
                    WHEN cp.pair_type = 'DR_with_txn' AND tm.txn_date IS NOT NULL THEN CAST(tm.txn_date AS DATE)
                    ELSE CAST(cp.date AS DATE)
                END as final_date,
                cp.original_order,
                cp.pair_type
            FROM classified_pairs cp
            LEFT JOIN accounts a ON cp.account_number = a.account_number
            -- Match transaction mappings for getting dates
            LEFT JOIN transaction_mappings tm ON cp.account_number = tm.account_number 
                AND cp.ledger_txn_no = tm.transaction_number
            -- Only include DR transactions and unpaired transactions, skip paired CR transactions
            WHERE cp.pair_type != 'CR_paired'
        ),
        -- Now we need to add the credit amounts to the corresponding DR transactions
        paired_transactions AS (
            SELECT 
                cp.*,
                -- For DR transactions, find the following CR transaction amount and date
                LEAD(tp2.credit_amount) OVER (PARTITION BY cp.account_number ORDER BY cp.original_order) as next_credit,
                LEAD(tp2.date) OVER (PARTITION BY cp.account_number ORDER BY cp.original_order) as next_cr_date,
                LEAD(tm2.txn_date) OVER (PARTITION BY cp.account_number ORDER BY cp.original_order) as next_cr_txn_date
            FROM consolidated_pairs cp
            LEFT JOIN classified_pairs tp2 ON cp.account_number = tp2.account_number 
                AND tp2.original_order = cp.original_order + 1
                AND tp2.pair_type = 'CR_paired'
            LEFT JOIN transaction_mappings tm2 ON tp2.account_number = tm2.account_number 
                AND tp2.prev_txn_no = tm2.transaction_number
        ),
        final_transactions AS (
            SELECT 
                pt.account_number,
                pt.customer_name,
                pt.customer_amount,
                pt.opening_balance,
                pt.closing_balance,
                pt.clean_txn_no,
                -- For DR transactions with paired CR, we have both amounts in the pair
                pt.debit_amount,
                COALESCE(pt.next_credit, pt.credit_amount, 0.0) as credit_amount,
                -- Prioritize credit date from transactions.xlsx, then credit date from ledger, then DR date
                CASE 
                    WHEN pt.pair_type = 'DR_with_txn' AND pt.next_cr_txn_date IS NOT NULL THEN CAST(pt.next_cr_txn_date AS DATE)
                    WHEN pt.pair_type = 'DR_with_txn' AND pt.next_cr_date IS NOT NULL THEN CAST(pt.next_cr_date AS DATE)
                    ELSE pt.final_date
                END as final_date,
                pt.original_order
            FROM paired_transactions pt
        ),
        -- NEW: Instead of aggregating, return one row per transaction
        transaction_results AS (
            SELECT 
                ft.account_number,
                ft.customer_name,
                -- Single transaction number (no aggregation)
                CASE WHEN ft.clean_txn_no != '' THEN ft.clean_txn_no ELSE NULL END as transaction_nos,
                -- Single date (no aggregation)
                ft.final_date::VARCHAR as dates,
                -- Individual transaction amounts
                ROUND(ft.debit_amount, 2) as total_debit,
                ROUND(ft.credit_amount, 2) as total_credit,
                -- Individual transaction opening balance
                ROUND(ft.opening_balance, 2) as opening_balance,
                -- For individual transactions, count is always 1 if transaction exists, 0 otherwise
                CAST(CASE WHEN ft.clean_txn_no != '' THEN 1 ELSE 0 END AS INTEGER) as debit_count,
                CAST(CASE WHEN ft.clean_txn_no != '' THEN 1 ELSE 0 END AS INTEGER) as credit_count,
                -- Individual transaction tally: Opening Balance + Credit - Debit
                ROUND(ft.opening_balance + ft.credit_amount - ft.debit_amount, 2) as tally_value,
                -- Include customer amount for reference
                ft.customer_amount,
                -- Keep original order for sorting
                ft.original_order
            FROM final_transactions ft
            -- Only include transactions that have transaction numbers
            WHERE ft.clean_txn_no IS NOT NULL AND ft.clean_txn_no != ''
        )
        SELECT 
            account_number,
            customer_name,
            transaction_nos,
            dates,
            total_debit,
            total_credit,
            opening_balance,
            debit_count,
            credit_count,
            tally_value,
            customer_amount
        FROM transaction_results
        ORDER BY account_number, original_order
        """

        result_df = self.conn.execute(query).pl()
        return result_df

    def process_transactions_in_chunks(self, transactions_file: str) -> List[Path]:
        """
        Process transactions file in memory-efficient chunks.
        """
        logger.info("Processing transactions in chunks...")

        # Use Polars lazy reading for memory efficiency
        if transactions_file.endswith('.xlsx'):
            # For Excel files, we need to read in chunks manually
            df_reader = pl.read_excel(transactions_file)
            total_rows = len(df_reader)
        else:
            df_reader = pl.scan_csv(transactions_file)
            total_rows = df_reader.select(pl.count()).collect().item()

        chunk_results = []
        chunks_processed = 0

        # Process in chunks to manage memory
        for i in range(0, total_rows, self.chunk_size):
            logger.info(f"Processing chunk {chunks_processed + 1}, rows {i} to {min(i + self.chunk_size, total_rows)}")

            if transactions_file.endswith('.xlsx'):
                chunk_df = df_reader.slice(i, self.chunk_size)
            else:
                chunk_df = df_reader.slice(i, self.chunk_size).collect()

            # Process the chunk
            processed_chunk = self.process_transactions_chunk(chunk_df)

            # Match and tally for this chunk
            tallied_chunk = self.match_and_tally(processed_chunk)

            # Save intermediate results as Parquet for memory efficiency
            chunk_file = self.output_dir / f"chunk_{chunks_processed:04d}.parquet"
            tallied_chunk.write_parquet(chunk_file)

            chunk_results.append(chunk_file)
            chunks_processed += 1

            logger.info(f"Completed chunk {chunks_processed}, saved to {chunk_file}")

        return chunk_results

    def consolidate_results(self, chunk_files: List[Path]) -> pl.DataFrame:
        """
        Consolidate results from all chunks efficiently.
        Now properly handles balance-aware tally calculation with opening balance.
        """
        logger.info("Consolidating results from all chunks...")

        # Use DuckDB to efficiently merge all chunk files
        self.conn.execute("DROP TABLE IF EXISTS all_chunks")

        # Create a union of all parquet files
        parquet_files = "', '".join([str(f) for f in chunk_files])
        query = f"""
        CREATE TABLE all_chunks AS
        SELECT * FROM read_parquet(['{parquet_files}'])
        """
        self.conn.execute(query)

        # Final aggregation across all chunks - maintain order and pairing
        # Use balance-aware tally calculation consistent with chunk processing
        final_query = """
        SELECT 
            account_number,
            customer_name,
            -- Concatenate all transaction_nos strings, preserving order
            STRING_AGG(transaction_nos, chr(10)) as transaction_nos,
            -- Concatenate all dates strings, preserving order  
            STRING_AGG(dates, chr(10)) as dates,
            ROUND(SUM(total_debit), 2) as total_debit,
            ROUND(SUM(total_credit), 2) as total_credit,
            CAST(SUM(debit_count) AS INTEGER) as debit_count,
            CAST(SUM(credit_count) AS INTEGER) as credit_count,
            -- Use balance-aware tally: Opening Balance + Credits - Debits
            ROUND(MIN(account_opening_balance) + SUM(total_credit) - SUM(total_debit), 2) as tally_value,
            -- Include balance information for optional output
            ROUND(MIN(account_opening_balance), 2) as opening_balance,
            ROUND(MAX(account_closing_balance), 2) as closing_balance,
            -- Include customer amount for reference
            MAX(customer_amount) as customer_amount
        FROM all_chunks
        GROUP BY account_number, customer_name
        ORDER BY account_number
        """

        final_df = self.conn.execute(final_query).pl()

        # Clean up intermediate files
        for chunk_file in chunk_files:
            chunk_file.unlink()

        return final_df

    def apply_output_config(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Apply output column configuration based on settings.
        Now includes optional opening and closing balance columns and customer amount.
        """
        # Build final column list maintaining the exact order you specified
        final_columns = ["account_number", "customer_name", "transaction_nos"]

        # Optional columns that can be enabled via configuration
        if self.config.get('include_dates', False):
            final_columns.append("dates")
        if self.config.get('include_debit', False):
            final_columns.append("total_debit")
        if self.config.get('include_credit', False):
            final_columns.append("total_credit")
        if self.config.get('include_opening_balance', False):
            final_columns.append("opening_balance")
        if self.config.get('include_closing_balance', False):
            final_columns.append("closing_balance")
        if self.config.get('include_customer_amount', False):
            final_columns.append("customer_amount")

        # Always include these core columns at the end
        final_columns.extend(["debit_count", "credit_count", "tally_value"])

        return df.select(final_columns)

    def preprocess_transaction_mappings(self, transactions_mapping_file: str) -> None:
        """
        Preprocess transaction mappings from transactions.xlsx file.
        Format: AccountNumber | Date | TransactionNumber
        """
        logger.info("Preprocessing transaction mappings...")

        # Read transaction mappings
        if transactions_mapping_file.endswith('.xlsx'):
            txn_mappings_df = pl.read_excel(transactions_mapping_file)
        else:
            txn_mappings_df = pl.read_csv(transactions_mapping_file)

        # Clean and standardize transaction mappings
        txn_mappings_df = txn_mappings_df.with_columns([
            pl.col("AccountNumber").cast(pl.Utf8).alias("account_number"),
            pl.col("Date").alias("txn_date"),
            pl.col("TransactionNumber").cast(pl.Utf8).alias("transaction_number")
        ]).select(["account_number", "txn_date", "transaction_number"])

        # Create indexed table in DuckDB for fast lookups
        self.conn.execute("DROP TABLE IF EXISTS transaction_mappings")
        self.conn.execute("""
            CREATE TABLE transaction_mappings AS 
            SELECT * FROM txn_mappings_df
        """)

        # Create composite index on account number and transaction number for fast lookups
        self.conn.execute("CREATE INDEX idx_txn_mapping ON transaction_mappings(account_number, transaction_number)")

        logger.info(f"Processed {len(txn_mappings_df)} transaction mappings with indexing")

    def pipeline(self, accounts_file: str, ledger_file: str, transactions_mapping_file: str = None) -> str:
        """
        Main processing pipeline for the new data structure.
        Now supports: customer_details.xlsx + ledger.xlsx + transactions.xlsx (optional)
        """
        try:
            # Step 1: Preprocess and index customer accounts
            self.preprocess_accounts(accounts_file)

            # Step 2: Preprocess transaction mappings if provided
            if transactions_mapping_file:
                self.preprocess_transaction_mappings(transactions_mapping_file)

            # Step 3: Process ledger transactions in chunks
            chunk_files = self.process_transactions_in_chunks(ledger_file)

            # Step 4: Consolidate results
            final_df = self.consolidate_results(chunk_files)

            # Step 5: Apply output configuration
            output_df = self.apply_output_config(final_df)

            # Step 6: Save final results
            output_file = self.output_dir / self.config.get('output_filename', 'tally_results.csv')
            output_df.write_csv(output_file)

            logger.info(f"Processing completed. Results saved to {output_file}")
            logger.info(f"Total accounts processed: {len(output_df)}")

            return str(output_file)

        except Exception as e:
            logger.error(f"Error during processing: {str(e)}")
            raise
        finally:
            self.conn.close()


def main():
    """
    Main function with configurable parameters.
    Updated to use the new data structure: customer_details.xlsx + ledger.xlsx + transactions.xlsx
    """
    # Configuration - All tweakable parameters
    config = {
        # File paths - Updated to use the new data structure
        'customer_details_file': 'data/customer_details.xlsx',
        'ledger_file': 'data/ledger.xlsx',
        'transactions_mapping_file': 'data/transactions.xlsx',  # Optional transaction mappings

        # Performance settings
        'chunk_size': 50000,  # Adjust based on available memory

        # Output settings
        'output_dir': 'output',
        'output_filename': f'tally_results_new_format_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',

        # Column inclusion flags (configurable output)
        'include_dates': True,                # Set to True to include dates column
        'include_debit': True,                # Set to True to include total debit column
        'include_credit': True,               # Set to True to include total credit column
        'include_opening_balance': True,      # Set to True to include opening balance column
        'include_closing_balance': False,      # Set to True to include closing balance column
        'include_customer_amount': False,      # Set to True to include customer amount column

        # Processing options
        'sort_by_date': True,                 # Set to True to sort transactions by date
        'log_level': 'INFO'
    }

    # Set up logging level
    logging.getLogger().setLevel(config['log_level'])

    # Create processor and run
    processor = TallyProcessor(config)

    try:
        output_file = processor.pipeline(
            config['customer_details_file'],
            config['ledger_file'],
            config['transactions_mapping_file']
        )
        print(f"✅ Processing completed successfully!")
        print(f"📁 Output file: {output_file}")
        print(f"\n📊 New Data Structure Details:")
        print(f"   • Customer Details: {config['customer_details_file']}")
        print(f"   • Ledger Data: {config['ledger_file']}")
        print(f"   • Transaction Mappings: {config['transactions_mapping_file']}")
        print(f"   • Both DR and CR transactions can now have transaction numbers")
        print(f"   • Transaction numbers are matched from separate transactions.xlsx file")
        print(f"   • Tally Formula: Opening Balance + Credits - Debits")
        print(f"   • Output includes customer names and amounts alongside account data")

    except Exception as e:
        print(f"❌ Processing failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
