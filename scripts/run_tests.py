"""
Test runner script for verification
"""
import subprocess
import sys

def run_tests():
    """Run all tests and generate coverage report."""
    print("=" * 80)
    print("Running Test Suite")
    print("=" * 80)
    
    # Run tests with coverage
    result = subprocess.run([
        'pytest',
        'tests/',
        '-v',
        '--cov=src',
        '--cov=training',
        '--cov=inference',
        '--cov=app',
        '--cov-report=term-missing',
        '--cov-report=html',
        '--cov-fail-under=70'
    ])
    
    if result.returncode == 0:
        print("\n✅ All tests passed!")
        print("📊 Coverage report generated in htmlcov/index.html")
    else:
        print("\n❌ Some tests failed")
        sys.exit(1)

if __name__ == '__main__':
    run_tests()
