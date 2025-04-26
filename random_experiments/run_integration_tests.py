import os
import sys
import subprocess
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define colors for output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD} {text} {Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}")

def run_test(test_path, test_name=None):
    if test_name is None:
        test_name = os.path.basename(test_path)
    
    print(f"\n{Colors.BOLD}Running {test_name}...{Colors.ENDC}")
    
    try:
        result = subprocess.run(
            [sys.executable, test_path],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode == 0:
            print(f"{Colors.OKGREEN}✓ {test_name} passed{Colors.ENDC}")
            return True
        else:
            print(f"{Colors.FAIL}✗ {test_name} failed{Colors.ENDC}")
            print(f"{Colors.FAIL}Error:{Colors.ENDC}\n{result.stderr}")
            return False
    except Exception as e:
        print(f"{Colors.FAIL}✗ {test_name} failed with exception{Colors.ENDC}")
        print(f"{Colors.FAIL}Exception:{Colors.ENDC} {str(e)}")
        return False

def run_pytest(test_path, test_name=None):
    if test_name is None:
        test_name = os.path.basename(test_path)
    
    print(f"\n{Colors.BOLD}Running {test_name}...{Colors.ENDC}")
    
    try:
        result = subprocess.run(
            ["pytest", test_path, "-v"],
            capture_output=True,
            text=True,
            check=False
        )
        
        print(result.stdout)
        
        if result.returncode == 0:
            print(f"{Colors.OKGREEN}✓ {test_name} passed{Colors.ENDC}")
            return True
        else:
            print(f"{Colors.FAIL}✗ {test_name} failed{Colors.ENDC}")
            if result.stderr:
                print(f"{Colors.FAIL}Error:{Colors.ENDC}\n{result.stderr}")
            return False
    except Exception as e:
        print(f"{Colors.FAIL}✗ {test_name} failed with exception{Colors.ENDC}")
        print(f"{Colors.FAIL}Exception:{Colors.ENDC} {str(e)}")
        return False

def main():
    print_header("LangChain ASI Integration Tests")
    
    # Verify API keys are set
    if not os.environ.get("ASI_API_KEY"):
        print(f"{Colors.FAIL}Error: ASI_API_KEY environment variable is not set{Colors.ENDC}")
        return False
    
    # Track test results
    results = {}
    
    # Run basic functionality tests
    print_header("Basic Functionality Tests")
    results["test_asi_integration.py"] = run_test("test_asi_integration.py", "Basic ASI Integration Test")
    
    # Run integration tests
    print_header("Integration Tests")
    results["tests/integration_tests/test_chat_models.py"] = run_pytest(
        "tests/integration_tests/test_chat_models.py", 
        "LangChain Integration Tests"
    )
    
    # Run tool calling tests
    print_header("Tool Calling Tests")
    results["test_tool_calling.py"] = run_test("test_tool_calling.py", "Tool Calling Test")
    
    # Summary
    print_header("Test Summary")
    passed = sum(1 for result in results.values() if result)
    failed = sum(1 for result in results.values() if not result)
    
    print(f"{Colors.BOLD}Total tests: {len(results)}{Colors.ENDC}")
    print(f"{Colors.OKGREEN}Passed: {passed}{Colors.ENDC}")
    print(f"{Colors.FAIL}Failed: {failed}{Colors.ENDC}")
    
    for test_name, result in results.items():
        status = f"{Colors.OKGREEN}✓ PASS{Colors.ENDC}" if result else f"{Colors.FAIL}✗ FAIL{Colors.ENDC}"
        print(f"{status} - {test_name}")
    
    return all(results.values())

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
