from news_pipeline import NewsNarrator

# Initialize the NewsNarrator
narrator = NewsNarrator(config_path="config.json")

# Fetch news (with web_search via Responses API)
articles = narrator.fetch_news()

# Print the fetched news articles
if articles:
    print(f"Retrieved {len(articles)} articles:")
    for item in articles:
        topic = item.get('topic', 'Unknown')
        title = item.get('title', 'Untitled')
        print(f"- {topic}: {title}")
else:
    print("No articles retrieved.")

# Save the fetched articles to a comprehensive text file and speech script
if articles:
    filename = "test_news_summary"
    narrator.save_article_summaries(articles, filename)
    speech_text = narrator.build_speech_script(articles)
    with open(narrator.text_dir / f"{filename}_speech_summary.txt", 'w', encoding='utf-8') as f:
        f.write(speech_text)
    print(f"Summaries saved to {narrator.text_dir / (filename + '_comprehensive_news.txt')}")
    print(f"Speech summary saved to {narrator.text_dir / (filename + '_speech_summary.txt')}")
