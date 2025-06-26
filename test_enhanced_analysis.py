#!/usr/bin/env python3
"""
Test script for Enhanced Stock Analysis functionality
"""
import sys
import os
import asyncio
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "backend"))

async def test_enhanced_analysis():
    print("🧪 Testing Enhanced Stock Analysis...")
    
    try:
        # Import the enhanced stock analyzer
        from backend.app.services.enhanced_stock_analyzer import EnhancedStockAnalyzer
        print("✅ Enhanced Stock Analyzer imported successfully")
        
        # Create analyzer instance
        analyzer = EnhancedStockAnalyzer()
        print("✅ Enhanced Stock Analyzer instance created")
        
        # Test with a well-known Indian stock
        test_ticker = "RELIANCE"  # This should work for Indian markets
        print(f"🔍 Analyzing {test_ticker}...")
        
        # Get enhanced analysis
        result = await analyzer.analyze_stock(test_ticker)
        
        if result:
            print(f"✅ Enhanced analysis completed for {test_ticker}")
            print(f"📊 Technical Score: {result.technical_score:.1f}/100")
            print(f"📈 Fundamental Score: {result.fundamental_score:.1f}/100")
            print(f"⭐ Overall Score: {result.overall_score:.1f}/100")
            print(f"💡 Investment Thesis: {result.investment_thesis[:100]}...")
            print(f"🎯 Target Price: ${result.target_price:.2f}")
            print(f"📈 Upside Potential: {result.upside_potential:.1f}%")
            print(f"🏢 Company: {result.company_name}")
            print(f"📈 Recommendation: {result.recommendation}")
            
            # Print available attributes for debugging
            print(f"📋 Available attributes: {[attr for attr in dir(result) if not attr.startswith('_')]}")
            
            return True
        else:
            print("❌ Enhanced analysis returned None")
            return False
            
    except Exception as e:
        print(f"❌ Error during enhanced analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_api_endpoints():
    print("\n🌐 Testing API Endpoints...")
    
    try:
        import requests
        import json
        
        # Test health endpoint first
        print("Testing health endpoint...")
        health_response = requests.get("http://localhost:8000/health", timeout=5)
        print(f"✅ Health check: {health_response.status_code}")
        
        # Test enhanced stock analysis endpoint
        print(f"Testing enhanced stock analysis endpoint...")
        test_ticker = "RELIANCE"
        analysis_response = requests.get(
            f"http://localhost:8000/api/stock/enhanced/{test_ticker}",
            timeout=30
        )
        
        if analysis_response.status_code == 200:
            data = analysis_response.json()
            print(f"✅ Enhanced analysis API works!")
            print(f"📊 Overall Score: {data.get('overall_score', 'N/A')}")
            return True
        else:
            print(f"❌ API returned status: {analysis_response.status_code}")
            print(f"Response: {analysis_response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing API: {e}")
        return False

if __name__ == "__main__":
    print("🚀 TAI-Roaster Enhanced Analysis Test Suite")
    print("=" * 50)
    
    # Test 1: Direct enhanced analysis
    asyncio.run(test_enhanced_analysis())
    
    # Test 2: API endpoints (if backend is running)
    print("\n" + "=" * 50)
    asyncio.run(test_api_endpoints())
    
    print("\n✅ Test suite completed!") 