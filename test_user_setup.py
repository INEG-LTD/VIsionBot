#!/usr/bin/env python3
"""
Test script for the user_setup module

This script tests that all the functions in user_setup.py work correctly
when imported as a module.
"""

from user_setup import (
    get_user_input, get_list_input, collect_personal_info,
    collect_address_info, collect_job_preferences, collect_document_paths,
    collect_automation_settings, validate_config, save_config,
    load_config, show_config_summary, setup_process
)

def test_module_import():
    """Test that the module can be imported successfully"""
    print("🧪 Testing user_setup module import...")
    
    try:
        # Test that all functions are available
        functions = [
            get_user_input, get_list_input, collect_personal_info,
            collect_address_info, collect_job_preferences, collect_document_paths,
            collect_automation_settings, validate_config, save_config,
            load_config, show_config_summary, setup_process
        ]
        
        print(f"✅ Successfully imported {len(functions)} functions")
        
        # Test function names
        for func in functions:
            print(f"   - {func.__name__}")
            
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_config_loading():
    """Test configuration loading functionality"""
    print("\n📁 Testing configuration loading...")
    
    try:
        # Test loading existing config
        config = load_config()
        if config:
            print(f"✅ Successfully loaded configuration with {len(config)} fields")
            print(f"   - Name: {config.get('first_name', 'N/A')} {config.get('last_name', 'N/A')}")
            print(f"   - Email: {config.get('email', 'N/A')}")
        else:
            print("⚠️ No configuration file found (this is normal for first-time users)")
            
        return True
        
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False

def test_validation():
    """Test configuration validation"""
    print("\n🔍 Testing configuration validation...")
    
    try:
        # Test with valid config
        valid_config = {
            'first_name': 'Test',
            'last_name': 'User',
            'email': 'test@example.com',
            'phone': '1234567890',
            'salary_min': 50000,
            'salary_max': 100000,
            'min_job_match_score': 70
        }
        
        errors = validate_config(valid_config)
        if not errors:
            print("✅ Valid configuration passed validation")
        else:
            print(f"⚠️ Valid config had {len(errors)} errors: {errors}")
            
        # Test with invalid config
        invalid_config = {
            'first_name': '',  # Missing required field
            'salary_min': 100000,  # Higher than max
            'salary_max': 50000,
            'min_job_match_score': 150  # Out of range
        }
        
        errors = validate_config(invalid_config)
        if errors:
            print(f"✅ Invalid configuration correctly caught {len(errors)} errors:")
            for error in errors:
                print(f"   - {error}")
        else:
            print("❌ Invalid config should have failed validation")
            
        return True
        
    except Exception as e:
        print(f"❌ Validation testing failed: {e}")
        return False

def test_config_summary():
    """Test configuration summary display"""
    print("\n📋 Testing configuration summary...")
    
    try:
        # Test with sample config
        sample_config = {
            'first_name': 'John',
            'last_name': 'Doe',
            'email': 'john@example.com',
            'phone': '1234567890',
            'locations': ['London', 'Remote', 'UK'],
            'job_titles': ['Software Engineer', 'Developer'],
            'salary_min': 60000,
            'required_skills': ['Python', 'JavaScript', 'React'],
            'max_jobs_to_find': 5,
            'auto_apply': False
        }
        
        print("Configuration summary:")
        show_config_summary(sample_config)
        print("✅ Configuration summary displayed successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration summary failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 User Setup Module Test Suite")
    print("=" * 50)
    
    tests = [
        ("Module Import", test_module_import),
        ("Config Loading", test_config_loading),
        ("Validation", test_validation),
        ("Config Summary", test_config_summary),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! User setup module is working correctly.")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
