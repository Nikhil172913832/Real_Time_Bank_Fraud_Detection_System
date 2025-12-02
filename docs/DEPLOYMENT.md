# Deployment Guide

## Table of Contents
- [Prerequisites](#prerequisites)
- [Local Development](#local-development)
- [Production Deployment](#production-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Monitoring & Observability](#monitoring--observability)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements
- **CPU**: 4+ cores recommended
- **RAM**: 8GB minimum, 16GB recommended
- **Disk**: 50GB+ for data and models
- **OS**: Linux (Ubuntu 20.04+), macOS, or Windows with WSL2

### Software Requirements
- Python 3.10+
- Docker 20.10+
- Docker Compose 2.0+
- Git 2.30+
- PostgreSQL 15+ (or Docker)
- Apache Kafka 3.0+ (or Docker)

---

## Local Development

### 1. Clone Repository
```bash
git clone https://github.com/Nikhil172913832/Real_Time_Bank_Fraud_Detection_System.git
cd Real_Time_Bank_Fraud_Detection_System
```

### 2. Create Virtual Environment
```bash
python3.10 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Set Up Environment Variables
```bash
cp .env.example .env
# Edit .env with your configuration
nano .env
```

Required environment variables:
```env
# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=fraud_detection
DB_USER=fraud_user
DB_PASSWORD=your_secure_password

# Kafka
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_TOPIC=transactions

# Model
MODEL_PATH=models/xgb_final.pkl
FRAUD_THRESHOLD=0.2

# API
API_HOST=0.0.0.0
API_PORT=5000
```

### 4. Start Infrastructure with Docker
```bash
docker-compose up -d postgres redis kafka
```

Wait for services to be healthy:
```bash
docker-compose ps
```

### 5. Initialize Database
```bash
python scripts/init_db.py
```

### 6. Generate Training Data
```bash
python data.py
```

### 7. Train Model
```bash
python training.py
```

### 8. Start Services

**Terminal 1 - API Server:**
```bash
python app.py
```

**Terminal 2 - Inference Service:**
```bash
python inference.py
```

**Terminal 3 - Transaction Producer (Optional):**
```bash
python transactions.py
```

**Terminal 4 - Dashboard (Optional):**
```bash
streamlit run dashboard.py
```

### 9. Verify Installation
```bash
# Check API health
curl http://localhost:5000/health

# Test prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"amount": 1500, "source": "online"}'
```

---

## Production Deployment

### Option 1: Docker Compose (Recommended for Small-Medium Scale)

#### 1. Configure Production Environment
```bash
cp .env.example .env.prod
# Edit production settings
nano .env.prod
```

Production settings:
```env
# Use strong passwords!
DB_PASSWORD=CHANGE_THIS_SECURE_PASSWORD
REDIS_PASSWORD=CHANGE_THIS_SECURE_PASSWORD

# Production database (separate from dev)
DB_HOST=postgres
DB_NAME=fraud_detection_prod

# MLflow
MLFLOW_TRACKING_URI=http://mlflow:5001

# Kafka
KAFKA_BOOTSTRAP_SERVERS=kafka:9092

# Security
API_WORKERS=4
ENABLE_API_AUTH=true
JWT_SECRET_KEY=CHANGE_THIS_SECRET_KEY
```

#### 2. Build and Deploy
```bash
# Build images
docker-compose -f docker-compose.prod.yml build

# Start all services
docker-compose -f docker-compose.prod.yml up -d

# Check status
docker-compose -f docker-compose.prod.yml ps

# View logs
docker-compose -f docker-compose.prod.yml logs -f api
```

#### 3. Initialize Production Database
```bash
docker-compose -f docker-compose.prod.yml exec api python scripts/init_db.py
```

#### 4. Train Production Model
```bash
docker-compose -f docker-compose.prod.yml exec api python training.py --production
```

### Option 2: Kubernetes (Recommended for Large Scale)

#### 1. Prerequisites
- Kubernetes cluster (1.24+)
- kubectl configured
- Helm 3.0+

#### 2. Create Namespace
```bash
kubectl create namespace fraud-detection
```

#### 3. Deploy PostgreSQL (using Helm)
```bash
helm repo add bitnami https://charts.bitnami.com/bitnami

helm install fraud-db bitnami/postgresql \
  --namespace fraud-detection \
  --set auth.username=fraud_user \
  --set auth.password=secure_password \
  --set auth.database=fraud_detection \
  --set primary.persistence.size=50Gi
```

#### 4. Deploy Redis
```bash
helm install fraud-redis bitnami/redis \
  --namespace fraud-detection \
  --set auth.password=redis_password \
  --set master.persistence.size=10Gi
```

#### 5. Deploy Kafka
```bash
helm install fraud-kafka bitnami/kafka \
  --namespace fraud-detection \
  --set replicaCount=3 \
  --set persistence.size=20Gi
```

#### 6. Deploy Application
```bash
# Create ConfigMap
kubectl create configmap fraud-config \
  --from-env-file=.env.prod \
  -n fraud-detection

# Create Secrets
kubectl create secret generic fraud-secrets \
  --from-literal=db-password=secure_password \
  --from-literal=redis-password=redis_password \
  -n fraud-detection

# Apply Kubernetes manifests
kubectl apply -f infrastructure/kubernetes/ -n fraud-detection
```

#### 7. Expose Service
```bash
# For LoadBalancer
kubectl expose deployment fraud-api \
  --type=LoadBalancer \
  --port=80 \
  --target-port=5000 \
  -n fraud-detection

# Get external IP
kubectl get svc fraud-api -n fraud-detection
```

#### 8. Enable Auto-scaling
```bash
kubectl autoscale deployment fraud-api \
  --cpu-percent=70 \
  --min=2 \
  --max=10 \
  -n fraud-detection
```

---

## Monitoring & Observability

### Prometheus Metrics

Access Prometheus at `http://localhost:9090`

Key metrics to monitor:
- `fraud_predictions_total` - Total predictions
- `fraud_prediction_latency_seconds` - Prediction latency
- `api_requests_total` - API request count
- `model_accuracy` - Current model accuracy
- `data_drift_score` - Data drift detection

### Grafana Dashboards

Access Grafana at `http://localhost:3000` (default: admin/admin)

Pre-configured dashboards:
1. **System Overview** - CPU, memory, disk usage
2. **API Performance** - Request rate, latency, errors
3. **Model Performance** - Accuracy, precision, recall
4. **Business Metrics** - Fraud rate, transaction volume

### Log Aggregation

View logs:
```bash
# Docker Compose
docker-compose logs -f api inference

# Kubernetes
kubectl logs -f deployment/fraud-api -n fraud-detection
```

Aggregate logs with ELK stack (optional):
```bash
docker-compose -f docker-compose.elk.yml up -d
```

### Alerts

Configure alerting in `infrastructure/prometheus/alerts.yml`:
```yaml
groups:
  - name: fraud_detection
    rules:
      - alert: HighLatency
        expr: histogram_quantile(0.95, fraud_prediction_latency_seconds) > 0.2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High prediction latency detected"

      - alert: DataDrift
        expr: data_drift_score > 0.3
        for: 10m
        labels:
          severity: critical
        annotations:
          summary: "Significant data drift detected"
```

---

## SSL/TLS Configuration

### Using Let's Encrypt with Nginx

#### 1. Install Certbot
```bash
sudo apt-get install certbot python3-certbot-nginx
```

#### 2. Obtain Certificate
```bash
sudo certbot --nginx -d fraud-detection.yourdomain.com
```

#### 3. Configure Nginx
```nginx
server {
    listen 443 ssl http2;
    server_name fraud-detection.yourdomain.com;
    
    ssl_certificate /etc/letsencrypt/live/fraud-detection.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/fraud-detection.yourdomain.com/privkey.pem;
    
    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## Backup & Recovery

### Database Backup

```bash
# Automated daily backup
docker-compose exec postgres pg_dump -U fraud_user fraud_detection > backup_$(date +%Y%m%d).sql

# Restore from backup
docker-compose exec -T postgres psql -U fraud_user fraud_detection < backup_20231201.sql
```

### Model Versioning

Models are automatically versioned in MLflow:
```bash
# List model versions
curl http://localhost:5001/api/2.0/mlflow/registered-models/list

# Download specific version
mlflow artifacts download --run-id <run_id> --dst-path ./backup_models/
```

---

## Performance Optimization

### Database Optimization

```sql
-- Create indexes for faster queries
CREATE INDEX idx_transactions_timestamp ON transactions(timestamp);
CREATE INDEX idx_predictions_fraud_prob ON predictions(fraud_probability DESC);

-- Partition large tables by date
CREATE TABLE transactions_2024_01 PARTITION OF transactions
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
```

### API Optimization

```python
# Use connection pooling
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True
)
```

### Redis Caching

```python
# Cache feature calculations
cache_key = f"features:{user_id}:{transaction_id}"
cached_features = redis.get(cache_key)

if cached_features:
    features = json.loads(cached_features)
else:
    features = calculate_features(transaction)
    redis.setex(cache_key, 300, json.dumps(features))  # 5 min TTL
```

---

## Troubleshooting

### Common Issues

#### 1. Kafka Connection Errors
```bash
# Check Kafka health
docker-compose exec kafka kafka-topics --list --bootstrap-server localhost:9092

# Recreate topic
docker-compose exec kafka kafka-topics --delete --topic transactions --bootstrap-server localhost:9092
docker-compose exec kafka kafka-topics --create --topic transactions --partitions 6 --replication-factor 1 --bootstrap-server localhost:9092
```

#### 2. Database Connection Issues
```bash
# Check PostgreSQL logs
docker-compose logs postgres

# Test connection
psql -h localhost -U fraud_user -d fraud_detection -c "SELECT 1;"
```

#### 3. Model Loading Errors
```bash
# Verify model exists
ls -lh models/xgb_final.pkl

# Retrain if corrupted
python training.py --force-retrain
```

#### 4. High Memory Usage
```bash
# Check memory usage
docker stats

# Reduce batch size in config
BATCH_SIZE=50  # Default: 100
```

### Debug Mode

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
python app.py
```

---

## Scaling Guide

### Horizontal Scaling

#### API Layer
```bash
# Docker Compose
docker-compose -f docker-compose.prod.yml up -d --scale api=4

# Kubernetes
kubectl scale deployment fraud-api --replicas=4 -n fraud-detection
```

#### Inference Service
```bash
# Scale based on Kafka partition count
kubectl scale deployment fraud-inference --replicas=6 -n fraud-detection
```

### Vertical Scaling

Adjust resource limits:
```yaml
# docker-compose.prod.yml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
```

---

## Security Checklist

- [ ] Change all default passwords
- [ ] Enable SSL/TLS
- [ ] Configure firewall rules
- [ ] Enable API authentication
- [ ] Set up VPN for database access
- [ ] Regular security updates
- [ ] Enable audit logging
- [ ] Implement rate limiting
- [ ] Use secrets management (Vault)
- [ ] Regular backups
- [ ] Disaster recovery plan

---

## Support

For issues and questions:
- GitHub Issues: https://github.com/Nikhil172913832/Real_Time_Bank_Fraud_Detection_System/issues
- Documentation: https://github.com/Nikhil172913832/Real_Time_Bank_Fraud_Detection_System/docs
- Email: nikhil.dev@example.com
