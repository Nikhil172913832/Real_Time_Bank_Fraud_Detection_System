"""
Unit Tests for API Endpoints
=============================

Tests for Flask API endpoints to ensure correct behavior.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def mock_model():
    """Mock XGBoost model for testing."""
    model = Mock()
    model.predict_proba = Mock(return_value=[[0.8, 0.2]])
    model.predict = Mock(return_value=[0])
    return model


@pytest.fixture
def mock_feature_columns():
    """Mock feature columns."""
    return ['amount', 'hour_of_day', 'is_weekend', 'merchant_risk_level']


@pytest.fixture
def app_client(mock_model, mock_feature_columns):
    """Create Flask test client with mocked model."""
    with patch('app.model', mock_model):
        with patch('app.expected_columns', mock_feature_columns):
            import app as flask_app
            flask_app.app.config['TESTING'] = True
            client = flask_app.app.test_client()
            yield client


class TestHealthEndpoint:
    """Test /health endpoint."""
    
    def test_health_check_success(self, app_client):
        """Test health check returns 200 when model is loaded."""
        response = app_client.get('/health')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert 'model_version' in data
        assert 'uptime_seconds' in data
    
    def test_health_check_model_not_loaded(self):
        """Test health check returns 503 when model is not loaded."""
        with patch('app.model', None):
            import app as flask_app
            flask_app.app.config['TESTING'] = True
            client = flask_app.app.test_client()
            
            response = client.get('/health')
            assert response.status_code == 503
            
            data = json.loads(response.data)
            assert data['status'] == 'unhealthy'


class TestPredictEndpoint:
    """Test /predict endpoint."""
    
    def test_predict_success(self, app_client):
        """Test successful prediction."""
        transaction = {
            'transaction_id': 'tx_001',
            'amount': 100.0,
            'source': 'online',
            'device_os': 'iOS',
            'merchant_category': 'Restaurants',
            'is_international': False,
            'hour_of_day': 14
        }
        
        response = app_client.post(
            '/predict',
            data=json.dumps(transaction),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert 'fraud_probability' in data
        assert 'is_fraud' in data
        assert 'threshold' in data
        assert 'prediction_time_ms' in data
        assert data['transaction_id'] == 'tx_001'
    
    def test_predict_no_data(self, app_client):
        """Test prediction with no data returns 400."""
        response = app_client.post('/predict')
        assert response.status_code == 400
        
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_predict_invalid_json(self, app_client):
        """Test prediction with invalid JSON returns 400."""
        response = app_client.post(
            '/predict',
            data='invalid json',
            content_type='application/json'
        )
        assert response.status_code == 400


class TestBatchPredictEndpoint:
    """Test /predict/batch endpoint."""
    
    def test_batch_predict_success(self, app_client):
        """Test successful batch prediction."""
        transactions = [
            {
                'transaction_id': 'tx_001',
                'amount': 100.0,
                'source': 'online',
                'merchant_category': 'Restaurants'
            },
            {
                'transaction_id': 'tx_002',
                'amount': 500.0,
                'source': 'atm',
                'merchant_category': 'Gambling'
            }
        ]
        
        response = app_client.post(
            '/predict/batch',
            data=json.dumps(transactions),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert 'predictions' in data
        assert 'batch_size' in data
        assert 'total_time_ms' in data
        assert 'avg_time_ms' in data
        assert data['batch_size'] == 2
        assert len(data['predictions']) == 2
    
    def test_batch_predict_not_array(self, app_client):
        """Test batch prediction with non-array returns 400."""
        response = app_client.post(
            '/predict/batch',
            data=json.dumps({'foo': 'bar'}),
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_batch_predict_empty_array(self, app_client):
        """Test batch prediction with empty array returns 400."""
        response = app_client.post(
            '/predict/batch',
            data=json.dumps([]),
            content_type='application/json'
        )
        
        assert response.status_code == 400


class TestMetricsEndpoint:
    """Test /metrics endpoint."""
    
    def test_metrics_endpoint(self, app_client):
        """Test metrics endpoint returns Prometheus format."""
        response = app_client.get('/metrics')
        
        assert response.status_code == 200
        assert response.content_type == 'text/plain; version=0.0.4; charset=utf-8'
        
        # Check for some expected metrics
        data = response.data.decode('utf-8')
        assert 'fraud_predictions_total' in data or 'api_requests_total' in data


class TestModelInfoEndpoint:
    """Test /model/info endpoint."""
    
    def test_model_info_success(self, app_client, mock_feature_columns):
        """Test model info endpoint returns correct information."""
        response = app_client.get('/model/info')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert 'model_version' in data
        assert 'model_type' in data
        assert 'num_features' in data
        assert data['num_features'] == len(mock_feature_columns)


class TestRootEndpoint:
    """Test / endpoint."""
    
    def test_root_endpoint(self, app_client):
        """Test root endpoint returns API documentation."""
        response = app_client.get('/')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert 'service' in data
        assert 'version' in data
        assert 'endpoints' in data


class TestErrorHandlers:
    """Test error handlers."""
    
    def test_404_handler(self, app_client):
        """Test 404 error handler."""
        response = app_client.get('/nonexistent')
        
        assert response.status_code == 404
        data = json.loads(response.data)
        assert 'error' in data


class TestRequestTracking:
    """Test request tracking with Prometheus."""
    
    def test_request_counter_increments(self, app_client):
        """Test that request counter increments on API calls."""
        # Make a request
        app_client.get('/health')
        
        # Get metrics
        response = app_client.get('/metrics')
        data = response.data.decode('utf-8')
        
        # Check that metrics were recorded
        assert 'api_requests_total' in data or 'fraud_predictions_total' in data
