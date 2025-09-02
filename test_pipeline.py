from news_pipeline import NewsNarrator
import openai
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize the single-class narrator
narrator = NewsNarrator(config_path="config.json")

print("🚀 Testing Enhanced News Pipeline")
print("=" * 50)

# Fetch and summarize news in one step
print("\n1️⃣ Fetching and summarizing news in one step...")
articles = narrator.fetch_news()

if articles:
    print(f"✅ Retrieved {len(articles)} articles with summaries")
    
    # Save comprehensive news content
    print("\n2️⃣ Saving comprehensive news content...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"test_{timestamp}"
    summary_file = narrator.save_article_summaries(articles, filename)
    
    # Extract speech summary from pre-generated content
    print("\n3️⃣ Creating speech summary from articles...")
    # Build the speech text from summaries
    speech_summary = narrator.build_speech_script(articles)
    
    if speech_summary:
        # Save the speech summary to the text subdirectory for consistency
        summary_file = narrator.text_dir / f"{filename}_speech_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(speech_summary)
        
        print(f"✅ Speech summary saved to: {summary_file}")
        print(f"📊 Summary length: {len(speech_summary.split())} words")
        
        # Show first 300 characters of summary
        print("\n📝 Summary preview:")
        print("-" * 40)
        print(speech_summary[:300] + "..." if len(speech_summary) > 300 else speech_summary)
        
    else:
        print("❌ Failed to create speech summary")
        
else:
    print("❌ No articles retrieved")
