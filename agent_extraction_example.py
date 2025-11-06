"""
Example usage of agentic mode with extraction functionality.
"""
from vision_bot import BrowserVisionBot
from agent import AgentResult

# Initialize bot
bot = BrowserVisionBot()
bot.start()
bot.page.goto("https://google.com/")

# Example 1: Basic extraction
print("\n=== Example 1: Basic Extraction ===")
result = bot.agentic_mode(
    "navigate to yahoo finance, search for 'AAPL', extract the stock price"
)

if result.success:
    print("✅ Task completed!")
    print(f"Reasoning: {result.reasoning}")
    print(f"Confidence: {result.confidence:.2f}")
    
    # Access extracted data
    if result.extracted_data:
        print("\n📊 Extracted Data:")
        for prompt, data in result.extracted_data.items():
            print(f"  {prompt}: {data}")
        
        # Get specific extracted value
        price = result.get("stock price")
        if price:
            print(f"\n💰 Stock Price: {price}")

# Example 2: Multiple extractions
print("\n=== Example 2: Multiple Extractions ===")
result2 = bot.agentic_mode(
    "go to a news website, extract the top headline, then extract the current date"
)

if result2.success:
    print("✅ Task completed!")
    print("\n📊 Extracted Data:")
    for prompt, data in result2.extracted_data.items():
        print(f"  {prompt}: {data}")
    
    # Access specific values
    headline = result2.get("top headline")
    date = result2.get("current date")
    print(f"\n📰 Headline: {headline}")
    print(f"📅 Date: {date}")

# Example 3: Using dict-like access
print("\n=== Example 3: Dict-like Access ===")
result3 = bot.agentic_mode(
    "extract product name and price from amazon product page"
)

if result3.success and result3.extracted_data:
    # Dict-like access
    if "product name" in result3:
        print(f"Product: {result3['product name']}")
    
    if "price" in result3:
        print(f"Price: {result3['price']}")

# Example 4: Save extracted data
print("\n=== Example 4: Save Extracted Data ===")
result4 = bot.agentic_mode(
    "collect information: product name, price, and rating"
)

if result4.success and result4.extracted_data:
    import json
    with open("extracted_data.json", "w") as f:
        json.dump(result4.extracted_data, f, indent=2)
    print("✅ Extracted data saved to extracted_data.json")

# Example 5: Natural language extraction (no explicit "extract:")
print("\n=== Example 5: Natural Language Extraction ===")
result5 = bot.agentic_mode(
    "go to twitter, find the most liked tweet and get its text"
)

if result5.success:
    print("✅ Task completed!")
    if result5.extracted_data:
        for key, value in result5.items():
            print(f"{key}: {value}")

input("\nPress Enter to continue...")


