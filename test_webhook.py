#!/usr/bin/env python3
"""
Simple test script to verify webhook logic without running Flask server
"""

import asyncio
import sys
from src.rag_faq_bot import RAGFAQBot
from api.whatsapp_webhook import run_async_query, format_whatsapp_response

async def test_query():
    """Test the async query functionality"""
    print("🧪 Testing RAG Bot Query...")
    
    try:
        # Initialize bot
        bot = RAGFAQBot(use_cache=True)
        
        # Test query
        test_question = "When do admissions open?"
        print(f"\n📝 Test Question: {test_question}")
        
        # Test async query directly
        print("\n1️⃣ Testing direct async query...")
        result = await bot.query(test_question, return_sources=True)
        print(f"✅ Direct async query successful!")
        print(f"   Intent: {result['response'].intent}")
        print(f"   Answer: {result['response'].answer[:100]}...")
        print(f"   Sources: {len(result.get('sources', []))} documents")
        
        # Test sync wrapper
        print("\n2️⃣ Testing sync wrapper (run_async_query)...")
        result2 = run_async_query(bot, test_question)
        print(f"✅ Sync wrapper successful!")
        print(f"   Intent: {result2['response'].intent}")
        print(f"   Answer: {result2['response'].answer[:100]}...")
        
        # Test formatting
        print("\n3️⃣ Testing WhatsApp formatting...")
        formatted = format_whatsapp_response(result2)
        print(f"✅ Formatting successful!")
        print(f"\n📱 Formatted Message:\n{formatted}\n")
        
        print("✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_query())
    sys.exit(0 if success else 1)

