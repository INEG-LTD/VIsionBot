#!/usr/bin/env python3
"""
Simple test script for the configuration file

This script tests the config.py functionality without requiring
the full automation engine dependencies.
"""

from config import get_all_preferences, validate_config, print_config_summary

def test_config_basics():
    """Test basic configuration functionality"""
    print("🧪 Testing Configuration Basics")
    print("=" * 50)
    
    # Test 1: Load preferences
    print("\n1️⃣ Testing preference loading...")
    try:
        preferences = get_all_preferences()
        print(f"✅ Successfully loaded {len(preferences)} preferences")
        print(f"   - Personal info: {len([k for k in preferences.keys() if 'name' in k.lower()])} name fields")
        print(f"   - Job titles: {len(preferences.get('job_titles', []))} titles")
        print(f"   - Skills: {len(preferences.get('required_skills', []))} skills")
        print(f"   - Locations: {len(preferences.get('locations', []))} locations")
    except Exception as e:
        print(f"❌ Failed to load preferences: {e}")
        return False
    
    # Test 2: Validate configuration
    print("\n2️⃣ Testing configuration validation...")
    try:
        errors = validate_config()
        if errors:
            print(f"⚠️ Found {len(errors)} validation errors:")
            for error in errors:
                print(f"   - {error}")
        else:
            print("✅ Configuration validation passed")
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False
    
    # Test 3: Test specific preference access
    print("\n3️⃣ Testing specific preference access...")
    try:
        # Test personal info
        first_name = preferences.get('first_name')
        last_name = preferences.get('last_name')
        email = preferences.get('email')
        print(f"✅ Personal info: {first_name} {last_name} ({email})")
        
        # Test job preferences
        job_titles = preferences.get('job_titles', [])
        locations = preferences.get('locations', [])
        salary_min = preferences.get('salary_min')
        print(f"✅ Job preferences: {len(job_titles)} titles, {len(locations)} locations, £{salary_min:,} min")
        
        # Test skills
        skills = preferences.get('required_skills', [])
        print(f"✅ Skills: {len(skills)} skills listed")
        
    except Exception as e:
        print(f"❌ Preference access failed: {e}")
        return False
    
    return True

def test_config_modification():
    """Test modifying configuration programmatically"""
    print("\n🔧 Testing Configuration Modification")
    print("=" * 50)
    
    try:
        # Load preferences
        preferences = get_all_preferences()
        
        # Test 1: Modify job titles
        print("\n1️⃣ Testing job title modification...")
        original_titles = preferences['job_titles'].copy()
        preferences['job_titles'].append('Test Engineer')
        print(f"✅ Added 'Test Engineer' to job titles")
        print(f"   - Before: {len(original_titles)} titles")
        print(f"   - After: {len(preferences['job_titles'])} titles")
        
        # Test 2: Modify salary
        print("\n2️⃣ Testing salary modification...")
        original_salary = preferences['salary_min']
        preferences['salary_min'] = 90000
        print(f"✅ Modified salary from £{original_salary:,} to £{preferences['salary_min']:,}")
        
        # Test 3: Add new skills
        print("\n3️⃣ Testing skills modification...")
        original_skills = preferences['required_skills'].copy()
        preferences['required_skills'].extend(['TestSkill1', 'TestSkill2'])
        print(f"✅ Added 2 new skills")
        print(f"   - Before: {len(original_skills)} skills")
        print(f"   - After: {len(preferences['required_skills'])} skills")
        
        # Test 4: Validate modified config
        print("\n4️⃣ Testing modified configuration validation...")
        errors = validate_config()
        if errors:
            print(f"⚠️ Modified config has {len(errors)} validation errors:")
            for error in errors:
                print(f"   - {error}")
        else:
            print("✅ Modified configuration is still valid")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration modification failed: {e}")
        return False

def test_config_structure():
    """Test the structure of the configuration"""
    print("\n📋 Testing Configuration Structure")
    print("=" * 50)
    
    try:
        preferences = get_all_preferences()
        
        # Check for required sections
        required_sections = [
            'first_name', 'last_name', 'email', 'phone',  # Personal
            'job_titles', 'locations', 'salary_min',       # Job search
            'required_skills', 'experience_levels',        # Skills
            'max_jobs_to_find', 'headless'                 # Settings
        ]
        
        missing_fields = []
        for field in required_sections:
            if field not in preferences:
                missing_fields.append(field)
        
        if missing_fields:
            print(f"❌ Missing required fields: {missing_fields}")
            return False
        else:
            print("✅ All required fields present")
        
        # Check data types
        print("\n🔍 Checking data types...")
        
        # Job titles should be a list
        if isinstance(preferences['job_titles'], list):
            print("✅ job_titles is a list")
        else:
            print("❌ job_titles is not a list")
            return False
        
        # Salary should be a number
        if isinstance(preferences['salary_min'], (int, float)):
            print("✅ salary_min is a number")
        else:
            print("❌ salary_min is not a number")
            return False
        
        # Skills should be a list
        if isinstance(preferences['required_skills'], list):
            print("✅ required_skills is a list")
        else:
            print("❌ required_skills is not a list")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Structure testing failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Configuration File Test Suite")
    print("=" * 60)
    
    tests = [
        ("Basic Configuration", test_config_basics),
        ("Configuration Modification", test_config_modification),
        ("Configuration Structure", test_config_structure),
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
    print(f"\n{'='*60}")
    print("📊 Test Results Summary")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Configuration is working correctly.")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
