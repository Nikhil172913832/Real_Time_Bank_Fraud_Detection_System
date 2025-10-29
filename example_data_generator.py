"""
Example usage of refactored data generator.

This shows how to use the modular data_generator package
to create batch fraud datasets.
"""

from datetime import datetime
from data_generator import DataGenerator


def main():
    # Initialize data generator
    print("Initializing DataGenerator...\n")
    generator = DataGenerator(
        n_records=100_000,      # Generate 100k records
        fraud_ratio=0.012,      # 1.2% fraud rate
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 12, 31)
    )
    
    # Generate dataset
    print("\nGenerating dataset...")
    df = generator.generate()
    
    # Show statistics
    print("\n" + "="*50)
    print("Dataset Statistics")
    print("="*50)
    print(f"Total records: {len(df):,}")
    print(f"Fraud records: {df['fraud_bool'].sum():,}")
    print(f"Fraud rate: {df['fraud_bool'].mean():.2%}")
    print(f"\nDate range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Unique users: {df['sender_id'].nunique():,}")
    
    # Show pattern distribution
    print("\nTop 10 Patterns:")
    print(df['pattern'].value_counts().head(10))
    
    # Save to file
    output_file = "generated_fraud_data.csv"
    print(f"\nSaving to {output_file}...")
    df.to_csv(output_file, index=False)
    print("✓ Done!")


if __name__ == "__main__":
    main()
