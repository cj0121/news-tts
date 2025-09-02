# News Pipeline 🎙️

A Python pipeline that uses ChatGPT to fetch, summarize, and read news to you. Perfect for staying informed while multitasking!

## Features

- 🤖 **AI-Powered News**: Uses ChatGPT to simulate and select important news scenarios
- 📊 **Smart Summarization**: Intelligent article selection and summarization
- 🎯 **Topic-Based Filtering**: Customizable topics with story count allocation
- ⏱️ **Time Management**: Configurable audio time limits and story counts
- 🔊 **Text-to-Speech**: High-quality audio output using OpenAI TTS
- 💾 **Storage**: Saves summaries and audio files locally
- ⚙️ **Flexible Configuration**: JSON-based config for easy customization

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up API Keys

Create a `.env` file in the project root (or copy the example):

```bash
# Option A: copy the example file
cp env_example.txt .env

# Option B: create it manually
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env

# Then edit .env with your API keys
```

**Required API Keys:**
- **OpenAI API Key**: Get from [OpenAI Platform](https://platform.openai.com/api-keys)
- **Azure Speech Key**: Get from [Azure Portal](https://portal.azure.com) (Speech Service)

### 3. Customize Configuration

Edit `config.json` to match your preferences:

```json
{
  "audio_settings": {
    "time_limit_minutes": 15
  },
  "content_settings": {
    "total_stories": 10
  },
  "topics": [
    {
      "name": "AI News",
      "keywords": ["AI", "artificial intelligence"],
      "story_count": 2
    }
  ]
}
```

### 4. Run the Pipeline

```bash
python news_pipeline.py
```

## Configuration

### Audio Settings
- `time_limit_minutes`: Maximum audio duration (default: 15)
- `speech_rate`: Speech speed multiplier (default: 1.0)

### Content Settings
- `total_stories`: Total number of stories to fetch (default: 10)
- `time_window_hours`: How recent the news should be (default: 24)
- `max_summary_length`: Maximum characters per summary (default: 200)

### Topics
Each topic can have:
- `name`: Display name
- `keywords`: Search terms for ChatGPT
- `story_count`: Number of stories (can be a range like `{"min": 1, "max": 2}`)
- `priority`: "high", "medium", or "low"

## Output

The pipeline creates:

1. **Audio File**: MP3 file with the news summary
2. **Text File**: Plain text version of the summary
3. **Local Files**: Text summaries and audio files
4. **Directory Structure**:
   ```
   news_output/
   ├── daily_news_20241201_143022.mp3
   ├── daily_news_20241201_143022.txt
   ├── daily_news_20241201_143022_article_summaries.txt
   └── daily_news_20241201_143022_speech_summary.txt
   ```

## Usage Examples

### Basic Usage
```bash
python news_pipeline.py
```

### Custom Config
```bash
python news_pipeline.py --config my_config.json
```

### Programmatic Usage
```python
from news_pipeline import NewsPipeline

pipeline = NewsPipeline("config.json")
success = pipeline.generate_news_summary()
```

## Troubleshooting

### Common Issues

1. **"Config file not found"**
   - Ensure `config.json` exists in the project directory

2. **"OpenAI API key not found"**
   - Check your `.env` file has the correct `OPENAI_API_KEY`

3. **"Web search not available"**
   - Ensure you're using GPT-4o model for web search capability

4. **"Invalid JSON response from ChatGPT"**
   - This is normal occasionally - the pipeline will retry or continue

### Performance Tips

- Reduce `total_stories` for faster processing
- Lower `max_summary_length` for shorter audio
- Use fewer topics for more focused summaries

## Future Enhancements

- [ ] Real-time news via ChatGPT web search (when available)
- [ ] News API integration for breaking news
- [ ] Web interface for easy topic management
- [ ] Email/SMS notifications
- [ ] Podcast-style RSS feed generation
- [ ] Multiple voice options
- [ ] Scheduled runs with cron jobs

## Contributing

Feel free to submit issues and enhancement requests!

## License

MIT License - feel free to use this for personal or commercial projects.
