#!/usr/bin/env python3
"""
OpenAI API Key Debugging Script

This script helps debug issues with the OpenAI API key by testing
different aspects of the initialization and request process.
"""

import os
import asyncio
from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI
from thinharness import Harness, HarnessConfig

# Load environment variables
load_dotenv(override=True)

def print_section(title):
    """Print a section title"""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80)

async def main():
    """Main function to debug API key issues"""
    print_section("Environment Variables")
    
    # Check if API key is set in environment
    api_key = os.getenv("OPENAI_API_KEY")
    masked_key = f"{api_key[:4]}...{api_key[-4:]}" if api_key and len(api_key) > 8 else "None"
    print(f"OPENAI_API_KEY from environment: {masked_key}")
    
    # Check for malformed placeholders
    if api_key in ["YOUR_OPENAI_API_KEY", "sk-...", "YOUR_OPE***_API"] or not api_key:
        print("WARNING: API key appears to be a placeholder or is missing!")
        
        # Prompt for key
        print("Enter your OpenAI API key for testing:")
        api_key = input("> ").strip()
        os.environ["OPENAI_API_KEY"] = api_key
        masked_key = f"{api_key[:4]}...{api_key[-4:]}" if api_key and len(api_key) > 8 else "None"
        print(f"Using API key: {masked_key}")
    
    # Test standard OpenAI client
    print_section("Standard OpenAI Client Test")
    try:
        with OpenAI(api_key=api_key) as client:
            models = client.models.list()
        print("✅ Standard OpenAI client works!")
        print(f"Found {len(models.data)} models")
    except Exception as e:
        print(f"❌ Standard OpenAI client failed: {e}")
    
    # Test AsyncOpenAI client
    print_section("Async OpenAI Client Test")
    try:
        async with AsyncOpenAI(api_key=api_key) as async_client:
            models = await async_client.models.list()
        print("✅ Async OpenAI client works!")
        print(f"Found {len(models.data)} models")
    except Exception as e:
        print(f"❌ Async OpenAI client failed: {e}")
    
    # Test a minimal thinharness agent without database dependencies
    print_section("thinharness Test")
    try:
        harness = Harness(
            HarnessConfig(
                root=".",
                model="openai:gpt-4o",
                system_prompt="Reply briefly.",
                builtin_tools=[],
            )
        )
        print("✅ Harness created successfully!")

        async with harness:
            try:
                result = await harness.run("Reply with: Hello, world!")
                print("✅ Harness request successful!")
                print(f"Response: {result.text[:50]}...")
            except Exception as e:
                print(f"❌ Harness request failed: {e}")

                error_str = str(e).lower()
                if "api key" in error_str or "openai" in error_str:
                    print("\nDetected potential API key issue in the harness request.")
                    print("Check that OPENAI_API_KEY is set to a valid key.")
    except Exception as e:
        print(f"❌ Harness creation failed: {e}")
    
    print_section("Summary")
    print("This debug information should help identify where the API key issue is occurring.")
    print("Look for any failures above and focus debugging on those components.")
    print("\nNext steps:")
    print("1. If all tests pass, the issue is likely in how your agent is using the API key")
    print("2. If some tests fail, focus on those specific components")
    print("3. Check that OPENAI_API_KEY is available to thinharness")

if __name__ == "__main__":
    asyncio.run(main())
