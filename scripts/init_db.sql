-- Fraud Detection Database Schema
-- ==================================
-- Optimized PostgreSQL schema for high-throughput transaction storage

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Drop existing tables (for re-initialization)
DROP TABLE IF EXISTS fraud_alerts CASCADE;
DROP TABLE IF EXISTS transactions CASCADE;
DROP TABLE IF EXISTS consumers CASCADE;
DROP TABLE IF EXISTS performance_metrics CASCADE;

-- Consumers table
CREATE TABLE consumers (
    user_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) NOT NULL,
    phone VARCHAR(20),
    zip_code VARCHAR(10),
    account_balance DECIMAL(15, 2) DEFAULT 0.00,
    account_created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_profile VARCHAR(20) DEFAULT 'moderate',
    is_active BOOLEAN DEFAULT true,
    last_transaction_at TIMESTAMP,
    total_transactions INTEGER DEFAULT 0,
    zip_history TEXT[],
    ip_history TEXT[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Transactions table (partitioned by date for performance)
CREATE TABLE transactions (
    transaction_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    sender_id UUID REFERENCES consumers(user_id),
    receiver_id UUID,
    amount DECIMAL(15, 2) NOT NULL,
    merchant_category VARCHAR(50),
    source VARCHAR(20),
    device_os VARCHAR(20),
    browser VARCHAR(20),
    zip_code VARCHAR(10),
    ip_address INET,
    country_code VARCHAR(5),
    is_international BOOLEAN DEFAULT false,
    session_id VARCHAR(100),
    device_fingerprint VARCHAR(255),
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    fraud_probability DECIMAL(5, 4),
    is_fraud BOOLEAN DEFAULT false,
    fraud_pattern VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) PARTITION BY RANGE (timestamp);

-- Create partitions for current and next month
CREATE TABLE transactions_2025_01 PARTITION OF transactions
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');
CREATE TABLE transactions_2025_02 PARTITION OF transactions
    FOR VALUES FROM ('2025-02-01') TO ('2025-03-01');
CREATE TABLE transactions_2025_03 PARTITION OF transactions
    FOR VALUES FROM ('2025-03-01') TO ('2025-04-01');
CREATE TABLE transactions_2025_04 PARTITION OF transactions
    FOR VALUES FROM ('2025-04-01') TO ('2025-05-01');

-- Fraud alerts table (for detected fraud)
CREATE TABLE fraud_alerts (
    alert_id SERIAL PRIMARY KEY,
    transaction_id UUID REFERENCES transactions(transaction_id),
    sender_id UUID REFERENCES consumers(user_id),
    amount DECIMAL(15, 2) NOT NULL,
    merchant_category VARCHAR(50),
    fraud_probability DECIMAL(5, 4) NOT NULL,
    alert_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    investigation_status VARCHAR(20) DEFAULT 'pending',
    investigated_by VARCHAR(100),
    investigation_notes TEXT,
    is_confirmed_fraud BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Performance metrics table
CREATE TABLE performance_metrics (
    metric_id SERIAL PRIMARY KEY,
    metric_type VARCHAR(50) NOT NULL,
    metric_value DECIMAL(15, 4),
    metric_unit VARCHAR(20),
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- Indexes for performance optimization
CREATE INDEX idx_transactions_timestamp ON transactions(timestamp DESC);
CREATE INDEX idx_transactions_sender ON transactions(sender_id);
CREATE INDEX idx_transactions_fraud ON transactions(is_fraud) WHERE is_fraud = true;
CREATE INDEX idx_transactions_amount ON transactions(amount);
CREATE INDEX idx_fraud_alerts_timestamp ON fraud_alerts(alert_timestamp DESC);
CREATE INDEX idx_fraud_alerts_sender ON fraud_alerts(sender_id);
CREATE INDEX idx_fraud_alerts_status ON fraud_alerts(investigation_status);
CREATE INDEX idx_consumers_profile ON consumers(user_profile);
CREATE INDEX idx_performance_metrics_type ON performance_metrics(metric_type, recorded_at DESC);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers for automatic timestamp updates
CREATE TRIGGER update_consumers_updated_at BEFORE UPDATE ON consumers
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_fraud_alerts_updated_at BEFORE UPDATE ON fraud_alerts
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create views for analytics

-- Fraud summary by hour
CREATE VIEW fraud_summary_hourly AS
SELECT 
    DATE_TRUNC('hour', timestamp) as hour,
    COUNT(*) as total_transactions,
    SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) as fraud_count,
    AVG(CASE WHEN is_fraud THEN fraud_probability ELSE NULL END) as avg_fraud_prob,
    SUM(CASE WHEN is_fraud THEN amount ELSE 0 END) as fraud_amount
FROM transactions
GROUP BY DATE_TRUNC('hour', timestamp)
ORDER BY hour DESC;

-- Top fraud merchants
CREATE VIEW top_fraud_merchants AS
SELECT 
    merchant_category,
    COUNT(*) as fraud_count,
    SUM(amount) as total_fraud_amount,
    AVG(fraud_probability) as avg_fraud_probability
FROM transactions
WHERE is_fraud = true
GROUP BY merchant_category
ORDER BY fraud_count DESC;

-- User fraud summary
CREATE VIEW user_fraud_summary AS
SELECT 
    c.user_id,
    c.email,
    c.user_profile,
    COUNT(t.transaction_id) as total_transactions,
    SUM(CASE WHEN t.is_fraud THEN 1 ELSE 0 END) as fraud_count,
    ROUND(SUM(CASE WHEN t.is_fraud THEN 1 ELSE 0 END)::NUMERIC / 
          NULLIF(COUNT(t.transaction_id), 0) * 100, 2) as fraud_rate
FROM consumers c
LEFT JOIN transactions t ON c.user_id = t.sender_id
GROUP BY c.user_id, c.email, c.user_profile;

-- Grant permissions (adjust as needed)
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO fraud_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO fraud_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO fraud_user;

-- Insert sample data for testing (optional)
INSERT INTO consumers (user_id, email, phone, zip_code, account_balance, user_profile)
VALUES 
    (uuid_generate_v4(), 'test1@example.com', '+1234567890', '12345', 10000.00, 'active'),
    (uuid_generate_v4(), 'test2@example.com', '+1234567891', '12346', 5000.00, 'moderate'),
    (uuid_generate_v4(), 'test3@example.com', '+1234567892', '12347', 2000.00, 'low');

-- Log completion
DO $$
BEGIN
    RAISE NOTICE 'Database schema initialized successfully!';
    RAISE NOTICE 'Tables created: consumers, transactions, fraud_alerts, performance_metrics';
    RAISE NOTICE 'Views created: fraud_summary_hourly, top_fraud_merchants, user_fraud_summary';
END $$;
