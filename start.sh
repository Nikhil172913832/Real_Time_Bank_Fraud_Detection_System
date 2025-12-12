#!/bin/bash

set -e

# Check for Docker
if ! command -v docker &> /dev/null; then
  echo "Docker is not installed. Please install Docker first."
  exit 1
fi

# Check for Docker Compose (compose v2 syntax)
if ! docker compose version &> /dev/null; then
  echo "Docker Compose is not available. Please install Docker Compose v2 (comes with recent Docker versions)."
  exit 1
fi

# Start the project using compose.yaml
COMPOSE_FILE="compose.yaml"
if [ ! -f "$COMPOSE_FILE" ]; then
  echo "$COMPOSE_FILE not found in project root."
  exit 1
fi

echo "Building and starting all services with Docker Compose..."
docker compose -f "$COMPOSE_FILE" up --build -d

echo ""
echo "✓ Project started successfully!"
echo ""
echo "Services:"
echo "  - Fraud Detection API:    http://localhost:5000"
echo "  - MLflow Tracking:        http://localhost:5001"
echo "  - Kafka UI:               http://localhost:8080"
echo "  - PgAdmin:                http://localhost:5050"
echo "  - Prometheus:             http://localhost:9090 (if enabled)"
echo "  - Grafana:                http://localhost:3000 (if enabled)"
echo ""
echo "Infrastructure:"
echo "  - Kafka Broker:           localhost:9092"
echo "  - PostgreSQL:             localhost:5432"
echo ""
echo "To view logs:             docker compose logs -f"
echo "To stop services:         docker compose down"
echo "To stop and clean:        docker compose down -v"
echo ""
echo "Documentation: See docs/ folder for more info."
