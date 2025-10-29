"""
Database Initialization Script
==============================

Initializes PostgreSQL database for fraud detection system.
Creates tables, indexes, and initial data.

Usage:
    python scripts/init_db.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from config import config

def init_database():
    """Initialize the fraud detection database."""
    print("="*70)
    print("DATABASE INITIALIZATION")
    print("="*70)
    
    try:
        # Connect to PostgreSQL server (not specific database)
        print(f"Connecting to PostgreSQL at {config.DB_HOST}:{config.DB_PORT}...")
        conn = psycopg2.connect(
            host=config.DB_HOST,
            port=config.DB_PORT,
            user=config.DB_USER,
            password=config.DB_PASSWORD,
            database='postgres'  # Connect to default database first
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Create database if it doesn't exist
        cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{config.DB_NAME}'")
        if not cursor.fetchone():
            print(f"Creating database: {config.DB_NAME}...")
            cursor.execute(f"CREATE DATABASE {config.DB_NAME}")
            print(f"✓ Database '{config.DB_NAME}' created successfully")
        else:
            print(f"✓ Database '{config.DB_NAME}' already exists")
        
        cursor.close()
        conn.close()
        
        # Connect to the fraud detection database
        print(f"\nConnecting to database: {config.DB_NAME}...")
        conn = psycopg2.connect(config.DATABASE_URL)
        cursor = conn.cursor()
        
        # Read and execute SQL schema
        sql_file = os.path.join(os.path.dirname(__file__), 'init_db.sql')
        print(f"Executing SQL from: {sql_file}...")
        
        with open(sql_file, 'r') as f:
            sql_script = f.read()
        
        cursor.execute(sql_script)
        conn.commit()
        
        # Verify tables were created
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_type = 'BASE TABLE'
            ORDER BY table_name
        """)
        
        tables = cursor.fetchall()
        print("\n✓ Database schema initialized successfully!")
        print("\nCreated tables:")
        for table in tables:
            print(f"  - {table[0]}")
        
        # Verify views
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.views 
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)
        
        views = cursor.fetchall()
        if views:
            print("\nCreated views:")
            for view in views:
                print(f"  - {view[0]}")
        
        cursor.close()
        conn.close()
        
        print("\n" + "="*70)
        print("DATABASE INITIALIZATION COMPLETE")
        print("="*70)
        print(f"Database URL: {config.DATABASE_URL.replace(config.DB_PASSWORD, '***')}")
        print("="*70)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error initializing database: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = init_database()
    sys.exit(0 if success else 1)
