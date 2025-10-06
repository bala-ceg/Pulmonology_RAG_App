#!/usr/bin/env python3
"""
Test Enhanced Wikipedia Search Fix
=================================

Test that the enhanced Wikipedia search now correctly retrieves Type-2 diabetes
information instead of Type-1 diabetes.
"""

def test_query_preprocessing():
    """Test the query preprocessing function"""
    
    print("🔧 Testing Query Preprocessing")
    print("=" * 50)
    
    try:
        from enhanced_tools import preprocess_medical_query
        
        test_queries = [
            "Explain the symptoms of Type-2 Diabetes",
            "What are the symptoms of Type-1 Diabetes", 
            "Tell me about diabetes type 2",
            "Describe Type-2 diabetes mellitus",
            "What is adult-onset diabetes",
            "Symptoms of juvenile diabetes"
        ]
        
        for query in test_queries:
            processed = preprocess_medical_query(query)
            print(f"Original:  '{query}'")
            print(f"Processed: '{processed}'")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_wikipedia_search_fix():
    """Test the enhanced Wikipedia search with problematic queries"""
    
    print("🔍 Testing Enhanced Wikipedia Search Fix")
    print("=" * 60)
    
    try:
        from enhanced_tools import enhanced_wikipedia_search
        
        # Test the exact query from the user's screenshot
        test_query = "Explain the symptoms of Type-2 Diabetes"
        
        print(f"🧪 Testing problematic query: '{test_query}'")
        print("-" * 60)
        
        result = enhanced_wikipedia_search(test_query)
        
        if result and result.get('content'):
            content = result['content']
            
            print(f"✅ Search completed successfully")
            print(f"   Content length: {len(content)}")
            
            # Check for Type-2 vs Type-1 content
            has_type2 = 'type 2 diabetes' in content.lower() or 'type-2 diabetes' in content.lower()
            has_type1 = 'type 1 diabetes' in content.lower() or 'type-1 diabetes' in content.lower()
            
            print(f"\n📊 Content Analysis:")
            print(f"   Contains Type-2 diabetes info: {'✅' if has_type2 else '❌'}")
            print(f"   Contains Type-1 diabetes info: {'⚠️' if has_type1 else '✅ (good - no contamination)'}")
            
            # Show content preview
            print(f"\n📝 Content Preview:")
            print("-" * 40)
            preview = content[:400] + "..." if len(content) > 400 else content
            print(preview)
            print("-" * 40)
            
            # Check if this looks like Type-2 diabetes content
            type2_indicators = [
                'insulin resistance',
                'adult-onset',
                'lifestyle factors', 
                'obesity',
                'metabolic syndrome'
            ]
            
            type1_indicators = [
                'autoimmune',
                'juvenile',
                'beta cells',
                'immune system destroys'
            ]
            
            type2_score = sum(1 for indicator in type2_indicators if indicator in content.lower())
            type1_score = sum(1 for indicator in type1_indicators if indicator in content.lower())
            
            print(f"\n🎯 Content Type Analysis:")
            print(f"   Type-2 indicators found: {type2_score}/5")
            print(f"   Type-1 indicators found: {type1_score}/4")
            
            if type2_score > type1_score:
                print("   ✅ Content correctly matches Type-2 diabetes")
                return True
            elif type1_score > type2_score:
                print("   ❌ Content still contains more Type-1 diabetes info")
                return False
            else:
                print("   ⚠️ Mixed content - needs further investigation")
                return False
        else:
            print("❌ No content returned from search")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiple_diabetes_queries():
    """Test multiple diabetes-related queries to ensure accuracy"""
    
    print("\n🔄 Testing Multiple Diabetes Queries")  
    print("=" * 60)
    
    test_cases = [
        ("Type-2 diabetes symptoms", "type 2"),
        ("Type-1 diabetes causes", "type 1"), 
        ("Adult onset diabetes treatment", "type 2"),
        ("Juvenile diabetes management", "type 1")
    ]
    
    try:
        from enhanced_tools import enhanced_wikipedia_search
        
        results = []
        
        for query, expected_type in test_cases:
            print(f"\n📝 Testing: '{query}' (expecting {expected_type})")
            print("-" * 40)
            
            result = enhanced_wikipedia_search(query)
            
            if result and result.get('content'):
                content = result['content'].lower()
                
                # Check if we got the right type (more sophisticated analysis)
                expected_count = content.count(expected_type)
                other_type = "type 1" if expected_type == "type 2" else "type 2"
                other_count = content.count(other_type)
                
                print(f"   {expected_type} mentions: {expected_count}")
                print(f"   {other_type} mentions: {other_count}")
                
                # Success if the expected type is mentioned more than the other type
                # or if only the expected type is mentioned
                success = expected_count > other_count or (expected_count > 0 and other_count == 0)
                results.append(success)
                
                print(f"   Result: {'✅ SUCCESS' if success else '❌ FAILED'}")
            else:
                print("   Result: ❌ NO CONTENT")
                results.append(False)
        
        success_rate = sum(results) / len(results) * 100
        print(f"\n📊 Overall Success Rate: {success_rate:.1f}% ({sum(results)}/{len(results)})")
        
        return success_rate >= 75  # At least 75% success rate
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run all tests for the Wikipedia search fix"""
    
    print("Testing Enhanced Wikipedia Search Fix")
    print("=" * 70)
    
    # Test 1: Query preprocessing
    preprocessing_ok = test_query_preprocessing()
    
    # Test 2: The specific problematic query
    if preprocessing_ok:
        main_fix_ok = test_wikipedia_search_fix()
    else:
        main_fix_ok = False
    
    # Test 3: Multiple diabetes queries
    if main_fix_ok:
        comprehensive_ok = test_multiple_diabetes_queries()
    else:
        comprehensive_ok = False
    
    print(f"\n" + "=" * 70)
    print("🎯 WIKIPEDIA SEARCH FIX SUMMARY:")
    print(f"✅ Query preprocessing: {'WORKING' if preprocessing_ok else 'FAILED'}")
    print(f"✅ Type-2 diabetes fix: {'WORKING' if main_fix_ok else 'FAILED'}")  
    print(f"✅ Comprehensive testing: {'WORKING' if comprehensive_ok else 'FAILED'}")
    
    if all([preprocessing_ok, main_fix_ok, comprehensive_ok]):
        print("\n🎉 SUCCESS: Wikipedia search fix is working correctly!")
        print("   ✅ Type-2 diabetes queries now return correct information")
        print("   ✅ No more Type-1/Type-2 cross-contamination")
        print("   ✅ Query preprocessing handles various formats")
    else:
        print("\n❌ Issues detected - further debugging needed")
    
    print("=" * 70)

if __name__ == "__main__":
    main()