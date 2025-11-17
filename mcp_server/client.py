import asyncio
import os
import httpx
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# FastAPI base URL - Direct connection to main FastAPI server for data operations
API_BASE_URL = "http://127.0.0.1:8000"

# MCP Server URL - For AI tool integration (if needed later)
MCP_SERVER_URL = "http://127.0.0.1:8001"

def display_menu():
    """Display the main menu options to the user."""
    print("\n" + "="*50)
    print("Student Recommendation Chatbot")
    print("="*50)
    print("Please select a category:")
    print("1.Forums")
    print("2.Learning")
    print("3.Wellness")
    print("4.Opportunities")
    print("5.Events")
    print("6.Exit")
    print("="*50)

async def get_recommendations(category: str, query: str, student_id: int = None):
    """Make POST request to FastAPI recommendation endpoint."""
    try:
        async with httpx.AsyncClient() as client:
            url = f"{API_BASE_URL}/recommend/{category}"
            
            # Prepare parameters
            params = {"query": query}
            if student_id:
                params["student_id"] = student_id
                
            response = await client.post(url, params=params)
            response.raise_for_status()
            
            return response.json()
            
    except httpx.HTTPStatusError as e:
        print(f"❌ HTTP Error {e.response.status_code}: {e.response.text}")
        return None
    except httpx.RequestError as e:
        print(f"❌ Request Error: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        return None

async def process_recommendation(category: str, user_input: str):
    """Process user input and get formatted recommendations using Gemini."""
    print(f"\n🔍 Searching for {category} recommendations...")
    
    # Get recommendations from FastAPI
    recommendations_data = await get_recommendations(category, user_input)
    
    if not recommendations_data:
        print("❌ Sorry, I couldn't get recommendations at this time. Please try again later.")
        return
    
    try:
        # Use Gemini to format the response conversationally
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Create a prompt for Gemini to format the recommendations
        prompt = f"""
        You are a helpful student assistant chatbot. A student asked for {category} recommendations with the query: "{user_input}"
        
        Here's the API response data:
        {recommendations_data}
        
        Please format this information in a friendly, conversational way. Include:
        1. A brief acknowledgment of their request with an emoji
        2. The top recommendations with titles, descriptions, and any relevant details
        3. Use relevant emojis for each recommendation (📚 for learning, 💪 for wellness, 💼 for opportunities, 📅 for events, 💬 for forums)
        4. Encourage them to explore the options
        
        Make it engaging and helpful, as if you're talking to a student friend.
        Use emojis generously and keep the tone warm and supportive.
        """
        
        response = model.generate_content(prompt)
        print("\n" + "="*60)
        print(response.text)
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error formatting response with Gemini: {e}")
        # Fallback to enhanced display with emojis
        print("\n" + "="*60)
        
        # Category-specific emojis
        emoji_map = {
            "forums": "💬",
            "learning": "📚",
            "wellness": "💪",
            "opportunities": "💼",
            "events": "📅"
        }
        category_emoji = emoji_map.get(category.lower(), "🎯")
        
        print(f"{category_emoji} Here are your {category} recommendations:\n")
        
        if 'recommendations' in recommendations_data:
            for i, rec in enumerate(recommendations_data['recommendations'][:5], 1):
                title = rec.get('title', 'No title')
                description = rec.get('description', rec.get('content', 'No description'))
                similarity = rec.get('similarity_score', 0)
                
                print(f"{i}. ✨ {title}")
                print(f"   📝 {description[:200]}...")
                if similarity > 0:
                    print(f"   🎯 Match Score: {similarity:.2%}")
                print()
        
        print("="*60)

async def main():
    """Main chatbot loop."""
    print("👋 Welcome to the Student Recommendation Chatbot!")
    print("I'm here to help you find forums, learning resources, wellness content, opportunities, and events!")
    
    # Category mapping
    categories = {
        "1": "forums",
        "2": "learning", 
        "3": "wellness",
        "4": "opportunities",
        "5": "events"
    }
    
    while True:
        try:
            display_menu()
            choice = input("Enter your choice (1-6): ").strip()
            
            if choice == "6":
                print("\n👋 Thank you for using the Student Recommendation Chatbot!")
                print("Have a great day and happy learning! 🎓")
                break
                
            elif choice in categories:
                category = categories[choice]
                category_name = category.title()
                
                print(f"\n🎯 You selected: {category_name}")
                user_query = input(f"What kind of {category_name.lower()} are you looking for? ").strip()
                
                if user_query:
                    await process_recommendation(category, user_query)
                else:
                    print("❌ Please enter a valid search query.")
                    
                # Ask if they want to continue
                continue_choice = input("\nWould you like to search for something else? (y/n): ").strip().lower()
                if continue_choice not in ['y', 'yes']:
                    print("\n👋 Thank you for using the Student Recommendation Chatbot!")
                    print("Have a great day and happy learning! 🎓")
                    break
                    
            else:
                print("❌ Invalid choice. Please enter a number between 1-6.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Thanks for using the Student Recommendation Chatbot!")
            break
        except Exception as e:
            print(f"❌ An unexpected error occurred: {e}")
            print("Let's try again!")

if __name__ == "__main__":
    asyncio.run(main())
