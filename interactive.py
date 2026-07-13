#!/usr/bin/env python3
"""
Interactive shell for the Pine Script Expert Agent.

Run this script to have an interactive conversation with the agent.
"""

import asyncio
import sys
from dotenv import load_dotenv
from openai import AsyncOpenAI

from agent import (
    PineScriptResult,
    build_harness,
    database_connect,
    get_openai_api_key,
    resolve_model,
)

# Load environment variables
load_dotenv(override=True)

# Make sure we have a valid API key before starting
def ensure_api_key():
    """Make sure we have a valid API key"""
    try:
        api_key = get_openai_api_key()
        print(f"Using OpenAI API key: {api_key[:4]}...{api_key[-4:]}")
        return api_key
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

class InteractiveAgent:
    """Interactive shell for the Pine Script Expert Agent"""

    def __init__(self, harness, preset: str | None = None):
        self.harness = harness
        self.resume_state = None
        self.running = True
        self.preset = preset
        self.print_welcome()
        
    def print_welcome(self):
        """Print welcome message"""
        print("\n" + "=" * 80)
        print("Pine Script Expert Agent - Interactive Shell".center(80))
        print("=" * 80)
        print("Ask any question about Pine Script v6 or type 'exit' to quit.")
        print("Type 'clear' to clear the conversation history.")
        if self.preset:
            print(f"Model preset: {self.preset}")
        print("=" * 80 + "\n")
        
    async def process_input(self, user_input: str) -> bool:
        """Process user input.
        
        Args:
            user_input: The user's input
            
        Returns:
            bool: Whether to continue the conversation
        """
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("\nExiting interactive shell...")
            return False
            
        if user_input.lower() in ['clear', 'cls']:
            self.resume_state = None
            print("\nConversation history cleared.")
            return True
            
        print("\nProcessing your question...")
        try:
            result = await self.harness.run(
                user_input,
                resume_from=self.resume_state,
            )
            
            if result and isinstance(result.output, PineScriptResult):
                print("\n" + "=" * 80)
                print(result.output.response)
                print("=" * 80)
                self.resume_state = result.resume_state
            else:
                print("\nError: Failed to get a response from the agent.")
        except Exception as e:
            print(f"\nError running agent: {e}")
            
        return True
        
    async def run(self):
        """Run the interactive shell"""
        while self.running:
            try:
                # Get user input
                user_input = input("\n> ")
                
                # Skip empty input
                if not user_input.strip():
                    continue
                    
                # Process input
                self.running = await self.process_input(user_input)
                
            except KeyboardInterrupt:
                print("\nExiting interactive shell...")
                self.running = False
            except Exception as e:
                print(f"\nError: {e}")

async def main(preset: str | None = None):
    """Main function to run the interactive shell"""
    # Ensure we have a valid API key before starting
    api_key = ensure_api_key()
    model, temperature, max_tokens = resolve_model(preset)

    async with database_connect(False) as pool:
        async with AsyncOpenAI(api_key=api_key) as openai_client:
            harness = build_harness(
                pool,
                openai_client,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            async with harness:
                interactive = InteractiveAgent(harness, preset=preset)
                await interactive.run()

if __name__ == "__main__":
    asyncio.run(main())
