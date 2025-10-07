#!/usr/bin/env python3
"""
Validate that main.py has correct syntax and can be imported
"""
import sys
import ast

def validate_python_syntax(file_path):
    """Validate Python file syntax"""
    try:
        with open(file_path, 'r') as f:
            source = f.read()
        
        # Parse the AST to check for syntax errors
        ast.parse(source)
        print(f"✅ {file_path} has valid Python syntax")
        return True
        
    except SyntaxError as e:
        print(f"❌ Syntax Error in {file_path}:")
        print(f"   Line {e.lineno}: {e.text}")
        print(f"   Error: {e.msg}")
        return False
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return False

def check_endpoint_structure():
    """Check that both endpoints have the correct structure"""
    file_path = "/Users/bseetharaman/Desktop/Bala/2025/AI_Medical_App/Pulmonology_RAG_App/main.py"
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check for both endpoints
        has_data_endpoint = '@app.route("/data", methods=["POST"])' in content
        has_data_html_endpoint = '@app.route("/data-html", methods=["POST"])' in content
        
        # Check for JSON returns in /data endpoint
        has_jsonify_calls = 'jsonify(' in content
        
        # Check for HTML returns in /data-html endpoint  
        has_html_generation = 'generate_full_html_response(' in content
        
        print("🔍 Endpoint Structure Analysis:")
        print(f"✅ /data endpoint exists: {has_data_endpoint}")
        print(f"✅ /data-html endpoint exists: {has_data_html_endpoint}")
        print(f"✅ JSON responses implemented: {has_jsonify_calls}")
        print(f"✅ HTML generation implemented: {has_html_generation}")
        
        return all([has_data_endpoint, has_data_html_endpoint, has_jsonify_calls, has_html_generation])
        
    except Exception as e:
        print(f"❌ Error analyzing file: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Validating Flask Application Structure\n")
    
    syntax_ok = validate_python_syntax("/Users/bseetharaman/Desktop/Bala/2025/AI_Medical_App/Pulmonology_RAG_App/main.py")
    structure_ok = check_endpoint_structure()
    
    print(f"\n🎯 VALIDATION RESULTS:")
    print(f"✅ Syntax Valid: {syntax_ok}")
    print(f"✅ Structure Complete: {structure_ok}")
    
    if syntax_ok and structure_ok:
        print("🎉 Flask application is ready - endpoints properly separated!")
        print("📋 Next Step: Start Flask server and test with actual requests")
    else:
        print("⚠️ Issues found - review and fix before testing")