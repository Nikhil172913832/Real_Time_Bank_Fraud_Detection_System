# Contributing to Fraud Detection System

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the project.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)

---

## Code of Conduct

### Our Pledge
We are committed to providing a welcoming and inclusive environment for all contributors.

### Expected Behavior
- Be respectful and constructive
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other community members

---

## Getting Started

### 1. Fork and Clone
```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/YOUR_USERNAME/Real_Time_Bank_Fraud_Detection_System.git
cd Real_Time_Bank_Fraud_Detection_System

# Add upstream remote
git remote add upstream https://github.com/Nikhil172913832/Real_Time_Bank_Fraud_Detection_System.git
```

### 2. Set Up Development Environment
```bash
# Create virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# Install dependencies including dev tools
pip install -r requirements.txt
pip install -e .

# Install pre-commit hooks
pre-commit install
```

### 3. Create a Branch
```bash
# Update main branch
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name
```

---

## Development Workflow

### Branch Naming Convention
- `feature/` - New features
- `bugfix/` - Bug fixes
- `hotfix/` - Urgent fixes
- `docs/` - Documentation changes
- `refactor/` - Code refactoring
- `test/` - Test additions or modifications

Examples:
- `feature/add-lstm-model`
- `bugfix/fix-kafka-connection`
- `docs/update-api-documentation`

### Commit Message Format
Follow conventional commits:
```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Build process or auxiliary tool changes
- `perf`: Performance improvements

Examples:
```bash
git commit -m "feat(models): Add LightGBM ensemble model

Implemented LightGBM as an alternative to XGBoost
with comparable performance and faster training time.

Closes #123"
```

---

## Code Style Guidelines

### Python Code Style

We follow PEP 8 with some modifications:
- Line length: 120 characters
- Use type hints for function parameters and return values
- Use docstrings (Google style) for all public functions/classes

#### Example:
```python
from typing import Dict, List, Optional
import pandas as pd


def predict_fraud(
    transaction: Dict[str, any],
    model_version: str = "latest"
) -> Dict[str, float]:
    """
    Predict fraud probability for a transaction.
    
    Args:
        transaction: Dictionary containing transaction features
        model_version: Model version to use for prediction
        
    Returns:
        Dictionary with fraud probability and confidence score
        
    Raises:
        ValueError: If transaction data is invalid
        
    Example:
        >>> transaction = {"amount": 1500, "source": "online"}
        >>> result = predict_fraud(transaction)
        >>> print(result['fraud_probability'])
        0.0234
    """
    # Implementation
    pass
```

### Code Formatting

We use automated formatters:
```bash
# Format code
black --line-length 120 .

# Sort imports
isort --profile black .

# Lint
flake8 .
pylint src/

# Type check
mypy src/
```

Pre-commit hooks will automatically run these checks.

---

## Testing

### Writing Tests

All new features must include tests:

```python
# tests/unit/test_new_feature.py
import pytest
from src.models.new_feature import NewFeature


class TestNewFeature:
    """Test cases for NewFeature."""
    
    def test_initialization(self):
        """Test that NewFeature initializes correctly."""
        feature = NewFeature(param=42)
        assert feature.param == 42
    
    def test_process_data(self):
        """Test data processing."""
        feature = NewFeature()
        result = feature.process(input_data)
        assert result is not None
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_new_feature.py

# Run with coverage
pytest --cov=src --cov-report=html

# Run integration tests
pytest tests/integration/

# Run load tests
locust -f tests/load/locustfile.py
```

### Test Coverage

- Aim for >80% code coverage
- All public APIs must be tested
- Include edge cases and error conditions

---

## Pull Request Process

### 1. Prepare Your Changes

```bash
# Ensure you're on your feature branch
git checkout feature/your-feature-name

# Make your changes
# ...

# Run tests
pytest

# Run linters
black .
flake8 .

# Commit changes
git add .
git commit -m "feat: Add new feature"
```

### 2. Update Documentation

- Update README.md if needed
- Add/update docstrings
- Update API documentation
- Add examples if applicable

### 3. Push to Your Fork

```bash
# Push to your fork
git push origin feature/your-feature-name
```

### 4. Create Pull Request

On GitHub:
1. Navigate to your fork
2. Click "New Pull Request"
3. Select your feature branch
4. Fill out the PR template:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] All tests pass
- [ ] Added new tests
- [ ] Updated existing tests

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-reviewed code
- [ ] Commented complex code
- [ ] Updated documentation
- [ ] No new warnings
```

### 5. Code Review Process

- Maintainers will review your PR
- Address feedback and requested changes
- Update your branch if needed:
  ```bash
  git add .
  git commit -m "fix: Address review comments"
  git push origin feature/your-feature-name
  ```

### 6. Merge

Once approved:
- PR will be merged by maintainers
- Your branch will be deleted
- Congratulations! 🎉

---

## Project Structure

```
Real_Time_Bank_Fraud_Detection_System/
├── src/                    # Source code
│   ├── api/               # API layer
│   ├── models/            # ML models
│   ├── monitoring/        # Monitoring utilities
│   └── utils/             # Utilities
├── tests/                 # Test suite
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   └── load/             # Load tests
├── docs/                  # Documentation
├── infrastructure/        # Deployment configs
├── scripts/              # Utility scripts
└── notebooks/            # Jupyter notebooks
```

---

## Issue Guidelines

### Reporting Bugs

Use the bug report template:
```markdown
**Describe the bug**
A clear description of the bug

**To Reproduce**
Steps to reproduce:
1. ...
2. ...

**Expected behavior**
What should happen

**Actual behavior**
What actually happens

**Environment**
- OS: [e.g., Ubuntu 20.04]
- Python version: [e.g., 3.10.5]
- Version: [e.g., 1.2.0]

**Additional context**
Any other relevant information
```

### Requesting Features

Use the feature request template:
```markdown
**Is your feature request related to a problem?**
Description of the problem

**Describe the solution you'd like**
What you want to happen

**Describe alternatives you've considered**
Alternative solutions

**Additional context**
Any other context
```

---

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Given credit in documentation

---

## Questions?

- Open a GitHub Discussion
- Join our Slack channel (if applicable)
- Email: nikhil.dev@example.com

Thank you for contributing! 🙏
