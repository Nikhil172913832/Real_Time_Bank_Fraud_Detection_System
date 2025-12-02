# System Architecture

## Overview

The Real-Time Bank Fraud Detection System is a production-grade MLOps platform designed to detect fraudulent banking transactions in real-time with sub-150ms latency and 1,000+ transactions per second throughput.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          CLIENT LAYER                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│  │   Web App    │  │  Mobile App  │  │  Admin Panel │                  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                  │
└─────────┼──────────────────┼──────────────────┼──────────────────────────┘
          │                  │                  │
          └──────────────────┼──────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────────────┐
│                          API GATEWAY LAYER                               │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  FastAPI / Flask REST API                                         │  │
│  │  - Rate Limiting (100 req/min)                                    │  │
│  │  - Authentication (JWT)                                           │  │
│  │  - Input Validation (Pydantic)                                    │  │
│  │  - OpenAPI Documentation                                          │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└────────────────────────────┬─────────────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────────────┐
│                      STREAMING LAYER                                     │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Apache Kafka Cluster (3 Brokers)                                │   │
│  │  - Topic: transactions (partitions: 6, replication: 3)           │   │
│  │  - Topic: fraud-alerts (partitions: 3, replication: 3)           │   │
│  │  - Consumer Group: fraud-detector                                │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└────────────────────────────┬─────────────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────────────┐
│                    PROCESSING LAYER                                      │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐      │
│  │  Consumer        │  │  Feature         │  │  ML Inference    │      │
│  │  Service         │─▶│  Engineering     │─▶│  Engine          │      │
│  │  (Async I/O)     │  │  (70+ features)  │  │  (XGBoost)       │      │
│  └──────────────────┘  └──────────────────┘  └──────┬───────────┘      │
│                                                       │                   │
│  ┌──────────────────────────────────────────────────▼───────────────┐   │
│  │  Model Explainability (SHAP)                                     │   │
│  │  - Feature importance                                            │   │
│  │  - Waterfall plots                                               │   │
│  │  - Decision explanations                                         │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└────────────────────────────┬─────────────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────────────┐
│                      DATA LAYER                                          │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐      │
│  │  PostgreSQL      │  │  Redis Cache     │  │  Model Registry  │      │
│  │  - Transactions  │  │  - Features      │  │  (MLflow)        │      │
│  │  - Predictions   │  │  - Sessions      │  │  - Versions      │      │
│  │  - Alerts        │  │  - Rate Limits   │  │  - Metrics       │      │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘      │
└──────────────────────────────────────────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────────────┐
│                   MONITORING & OBSERVABILITY LAYER                       │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐      │
│  │  Prometheus      │  │  Grafana         │  │  ELK Stack       │      │
│  │  - Metrics       │─▶│  - Dashboards    │  │  - Logs          │      │
│  │  - Alerts        │  │  - Visualizations│  │  - Search        │      │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘      │
│                                                                           │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Streamlit Dashboard                                             │   │
│  │  - Real-time metrics                                             │   │
│  │  - Geographic heatmap                                            │   │
│  │  - SHAP explanations                                             │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Transaction Producer
- **Technology**: Python with Kafka-Python
- **Responsibility**: Generate realistic transaction data with fraud patterns
- **Patterns**: Account takeover, burst fraud, money laundering
- **Throughput**: 1,000+ TPS
- **Data Format**: JSON over Kafka

### 2. Kafka Streaming Platform
- **Brokers**: 3 (development), 5+ (production)
- **Partitioning Strategy**: Hash by user_id for ordering
- **Retention**: 7 days
- **Replication Factor**: 3
- **Consumer Group**: fraud-detector with 6 consumers

### 3. Feature Engineering Pipeline
- **Real-time Features**: 
  - Transaction amount, timestamp, location
  - Device fingerprint, IP address, session data
- **Aggregated Features**:
  - 24h transaction count/amount
  - 7d/30d velocity metrics
  - Historical behavior patterns
- **Risk Indicators**:
  - Merchant risk score
  - Geographic anomaly
  - Device mismatch
- **Total Features**: 70+

### 4. ML Inference Engine
- **Primary Model**: XGBoost Classifier
- **Model Size**: 15MB
- **Inference Time**: <50ms (p95)
- **Batch Size**: 1-100 transactions
- **Explainability**: SHAP TreeExplainer
- **Fallback**: Rule-based system

### 5. Model Registry (MLflow)
- **Versioning**: Semantic versioning (v1.x.x)
- **A/B Testing**: Traffic splitting
- **Rollback**: Instant version switching
- **Metrics Tracking**: ROC-AUC, Precision, Recall
- **Experiment Tracking**: Hyperparameters, artifacts

### 6. Data Persistence
#### PostgreSQL
- **Schema**: 
  - `transactions` (indexed on timestamp, user_id)
  - `predictions` (indexed on transaction_id, timestamp)
  - `alerts` (indexed on severity, timestamp)
- **Connection Pool**: 20 connections
- **Partitioning**: Monthly partitions
- **Backup**: Daily snapshots

#### Redis Cache
- **Use Cases**:
  - Session management
  - Rate limiting counters
  - Feature cache (30s TTL)
  - Hot user profiles
- **Memory**: 4GB
- **Persistence**: AOF + RDB

### 7. REST API
- **Framework**: FastAPI (async) + Flask (fallback)
- **Endpoints**:
  - `POST /predict` - Single prediction
  - `POST /predict/batch` - Batch predictions
  - `GET /health` - Health check
  - `GET /metrics` - Prometheus metrics
  - `GET /model/info` - Model metadata
- **Rate Limiting**: 100 requests/minute per IP
- **Authentication**: JWT tokens
- **Documentation**: OpenAPI/Swagger UI

### 8. Monitoring Stack
#### Prometheus
- **Metrics**:
  - Request rate, latency, errors (RED)
  - Model accuracy, drift
  - System resources (CPU, memory, disk)
  - Kafka lag
- **Scrape Interval**: 15s
- **Retention**: 30 days

#### Grafana
- **Dashboards**:
  - System overview
  - Model performance
  - Business metrics
  - Alerting rules

#### Streamlit Dashboard
- **Real-time Metrics**: TPS, fraud rate, latency
- **Visualizations**: Geographic heatmap, time series
- **SHAP Explanations**: Feature importance, waterfall
- **Alert Management**: View, acknowledge, escalate

## Data Flow

### Transaction Processing Flow
```
1. Transaction Generation
   ↓
2. Kafka Producer (serialize to JSON)
   ↓
3. Kafka Topic: transactions (partition by user_id)
   ↓
4. Consumer Service (poll with 500ms timeout)
   ↓
5. Feature Engineering
   - Extract from transaction
   - Fetch from cache/DB
   - Compute aggregations
   - One-hot encoding
   ↓
6. ML Inference
   - Preprocess features
   - XGBoost prediction
   - SHAP explanation
   ↓
7. Decision Logic
   - Compare to threshold (0.2)
   - Apply business rules
   - Generate alert if fraud
   ↓
8. Persistence
   - Save to PostgreSQL
   - Update Redis cache
   - Publish to fraud-alerts topic (if fraud)
   ↓
9. Response/Alert
   - Return to API caller
   - Send email/SMS alert
   - Update dashboard
```

## Scalability Considerations

### Horizontal Scaling
- **API Layer**: Load balanced across multiple instances
- **Kafka Consumers**: Scale to match partition count (6)
- **Database**: Read replicas for queries
- **Redis**: Cluster mode with sharding

### Vertical Scaling
- **Model Optimization**: Quantization, pruning
- **Caching**: Multi-level cache (Redis → in-memory)
- **Connection Pooling**: Reuse DB connections
- **Async I/O**: Non-blocking operations

### Performance Targets
| Metric | Target | Current |
|--------|--------|---------|
| Throughput | 10,000 TPS | 1,000+ TPS |
| P50 Latency | 30ms | 45ms |
| P95 Latency | 150ms | 120ms |
| P99 Latency | 300ms | 180ms |
| Availability | 99.95% | 99.9% |

## Disaster Recovery

### Backup Strategy
- **Database**: Daily full + hourly incremental
- **Models**: Versioned in S3 with 90-day retention
- **Kafka**: Replication factor 3
- **Configuration**: Version controlled in Git

### Recovery Procedures
- **Database Failure**: Promote read replica
- **Kafka Failure**: Consumer rebalance
- **Model Failure**: Rollback to previous version
- **API Failure**: Load balancer redirects

### Monitoring & Alerting
- **Health Checks**: Every 30s
- **Circuit Breaker**: Open after 5 failures
- **Auto-restart**: systemd/Kubernetes
- **PagerDuty Integration**: Critical alerts

## Security Architecture

### Authentication & Authorization
- **API**: JWT with 1-hour expiration
- **Database**: Role-based access control
- **Kafka**: SASL/SCRAM authentication
- **TLS**: Encryption in transit

### Data Protection
- **PII**: Hashed/encrypted user IDs
- **Credentials**: Environment variables
- **Secrets**: HashiCorp Vault
- **Audit Log**: All access logged

### Compliance
- **GDPR**: Right to be forgotten
- **PCI-DSS**: Secure card data handling
- **SOC 2**: Audit trails
- **Model Explainability**: SHAP for regulatory compliance

## Technology Choices

### Why XGBoost?
- Superior performance on tabular data
- Built-in feature importance
- Fast inference (<50ms)
- Handles missing values
- Regularization prevents overfitting

### Why Kafka?
- High throughput (millions TPS)
- Fault tolerant (replication)
- Scalable (partitioning)
- Persistent (log-based)
- Decouples producers/consumers

### Why PostgreSQL?
- ACID compliance
- Complex queries
- Indexes for fast lookups
- Partitioning for large tables
- JSON support for flexibility

### Why FastAPI?
- Async support
- Auto-generated OpenAPI docs
- Type validation with Pydantic
- High performance
- Modern Python 3.10+ features

## Deployment Architecture

### Development
- Docker Compose on local machine
- Single instance of each service
- SQLite for quick testing

### Staging
- Kubernetes cluster (3 nodes)
- Scaled-down replicas
- Synthetic data
- Full monitoring stack

### Production
- Kubernetes cluster (10+ nodes)
- Auto-scaling based on CPU/memory
- Multi-region deployment
- Blue-green deployment strategy
- Canary releases for models

## Future Enhancements

1. **Graph Neural Networks**: Detect fraud rings
2. **Feature Store**: Centralized feature management (Feast)
3. **AutoML**: Automated model selection
4. **Edge Computing**: On-device fraud detection
5. **Blockchain**: Immutable audit trail
6. **Real-time Retraining**: Continuous learning
