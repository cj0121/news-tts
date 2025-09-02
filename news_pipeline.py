#!/usr/bin/env python3
"""
News Narrator - Fetch real-time news and generate a speech script/audio with OpenAI
"""

import json
import os

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse
import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
import openai
# import pyttsx3  # Replaced with OpenAI TTS
from dotenv import load_dotenv
import logging
from logging.handlers import RotatingFileHandler

# Load environment variables
load_dotenv()

console = Console()
logger = logging.getLogger("news_pipeline")

def setup_logging(level_str: str = "INFO") -> None:
    """Configure console logging with the given level.

    This sets up a basic console handler. A file handler can be added later
    once the output directory is known.
    """
    level = getattr(logging, level_str.upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Avoid duplicate handlers on reconfiguration
    if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

def add_file_logging(log_file_path: str) -> None:
    """Attach a rotating file handler to the root logger."""
    root_logger = logging.getLogger()
    # Ensure parent directory exists
    try:
        parent_dir = os.path.dirname(os.path.abspath(log_file_path))
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
    except Exception:
        pass
    # Don't add duplicate file handlers for the same path
    for h in root_logger.handlers:
        if isinstance(h, RotatingFileHandler):
            try:
                if getattr(h, 'baseFilename', None) == os.path.abspath(log_file_path):
                    return
            except Exception:
                pass
    file_handler = RotatingFileHandler(log_file_path, maxBytes=1_000_000, backupCount=3, encoding='utf-8')
    file_handler.setLevel(root_logger.level)
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

class NewsNarrator:
    def __init__(self, config_path: str = "config.json"):
        """Initialize with configuration, OpenAI client, and output directory."""
        logger.debug("Initializing NewsNarrator with config path: %s", config_path)
        self.config = self._load_config(config_path)
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.output_dir = Path(self.config["output_settings"]["output_directory"])
        self.output_dir.mkdir(exist_ok=True)
        # Subfolders for organized outputs
        self.text_dir = self.output_dir / "text"
        self.audio_dir = self.output_dir / "audio"
        self.text_dir.mkdir(exist_ok=True)
        self.audio_dir.mkdir(exist_ok=True)
        # Audio config
        self.speech_rate = float(self.config.get("audio_settings", {}).get("speech_rate", 1.0))
        self.time_limit_minutes = int(self.config.get("audio_settings", {}).get("time_limit_minutes", 15))
        self.words_per_minute = int(self.config.get("audio_settings", {}).get("words_per_minute", 150))
        # Content config
        self.max_summary_len = int(self.config.get("content_settings", {}).get("max_summary_length", 1000))
        self.total_stories_cap = int(self.config.get("content_settings", {}).get("total_stories", 10))
        self.time_window_hours = int(self.config.get("content_settings", {}).get("time_window_hours", 24))
        self.min_words_per_story = int(self.config.get("content_settings", {}).get("min_words_per_story", 150))
        self.min_sources_per_story = int(self.config.get("content_settings", {}).get("min_sources_per_story", 2))
        self.allow_single_source_fallback = bool(self.config.get("content_settings", {}).get("allow_single_source_fallback", True))
        # Per-topic allowlist policy (default True)
        self.topic_allowlist_enforcement = {
            t.get("name"): bool(t.get("enforce_allowlist", True))
            for t in self.config.get("topics", [])
            if t.get("name")
        }
        # Allowed US-based, established outlets (configurable)
        self.allowed_domains = (
            self.config.get("content_settings", {}).get(
                "allowed_domains",
                [
                    "wsj.com",
                    "nytimes.com",
                    "washingtonpost.com",
                    "apnews.com",
                    "reuters.com",
                    "bloomberg.com",
                    "cnbc.com",
                    "npr.org",
                    "axios.com",
                    "politico.com",
                    "cnn.com",
                    "abcnews.go.com",
                    "nbcnews.com",
                    "cbsnews.com",
                    "usatoday.com",
                    "latimes.com",
                    "marketwatch.com",
                    "foxbusiness.com",
                ],
            )
        )
        logger.info(
            "NewsNarrator ready | output_dir=%s | text_dir=%s | audio_dir=%s | time_limit=%s min | total_stories_cap=%s",
            str(self.output_dir), str(self.text_dir), str(self.audio_dir), self.time_limit_minutes, self.total_stories_cap,
        )

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file."""
        try:
            logger.debug("Loading configuration file: %s", config_path)
            with open(config_path, 'r') as f:
                data = json.load(f)
                logger.info("Configuration loaded | topics=%d | output_dir=%s", len(data.get("topics", [])), data.get("output_settings", {}).get("output_directory"))
                return data
        except FileNotFoundError:
            console.print(f"[red]Config file {config_path} not found![/red]")
            logger.exception("Config file not found: %s", config_path)
            raise
        except json.JSONDecodeError:
            console.print(f"[red]Invalid JSON in config file {config_path}[/red]")
            logger.exception("Invalid JSON in config file: %s", config_path)
            raise

    def fetch_news(self) -> List[Dict]:
        """Fetch news using ChatGPT with web search capability."""
        topics = self.config["topics"]
        prompt = self._generate_enhanced_news_prompt(topics)
        logger.info("Fetching news via Responses API | topics=%d | time_window_hours=%d", len(topics), self.time_window_hours)

        try:
            # Enforce real-time news via Responses API web_search ONLY
            response = self.client.responses.create(
                model=self.config["api_settings"]["openai_model"],
                input=prompt,
                tools=[{"type": "web_search"}],
                temperature=self.config["api_settings"]["temperature"],
                max_output_tokens=self.config["api_settings"]["max_tokens"]
            )
            content = getattr(response, "output_text", None)
            if not content and hasattr(response, "choices"):
                content = response.choices[0].message.content

            if not content:
                raise RuntimeError("No content returned from Responses API.")

            # Log token usage if available
            usage = getattr(response, "usage", None)
            def _get_u(key: str) -> int:
                try:
                    if hasattr(usage, key):
                        return int(getattr(usage, key))
                    if isinstance(usage, dict):
                        return int(usage.get(key) or 0)
                except Exception:
                    return 0
                return 0
            if usage is not None:
                in_tok = _get_u("input_tokens") or _get_u("input_text_tokens")
                out_tok = _get_u("output_tokens") or _get_u("output_text_tokens")
                tot_tok = _get_u("total_tokens") or (in_tok + out_tok)
                cfg_max = int(self.config.get("api_settings", {}).get("max_tokens", 0))
                logger.info(
                    "Token usage | input=%d output=%d total=%d | configured_max_output_tokens=%d",
                    in_tok, out_tok, tot_tok, cfg_max,
                )

            console.print(f"[yellow]Raw response from ChatGPT (web): {str(content)[:500]}...[/yellow]")
            logger.debug("Responses API output length=%d", len(str(content)))
            
            # Try to extract JSON from the response
            try:
                # Find JSON array in the response
                start = content.find('[')
                end = content.rfind(']') + 1
                if start != -1 and end != 0:
                    json_str = content[start:end]
                    articles = json.loads(json_str)
                    # Enforce allowed US-based sources
                    filtered = self._filter_articles_by_allowed_sources(articles)
                    console.print(f"[green]Successfully parsed {len(filtered)} articles[/green]")
                    logger.info("Parsed articles | raw=%d | allowed_sources=%d", len(articles), len(filtered))
                    return filtered
                else:
                    console.print("[yellow]Could not parse JSON from ChatGPT response[/yellow]")
                    logger.warning("Could not locate JSON array in Responses output")
                    return []
            except json.JSONDecodeError as e:
                console.print(f"[yellow]Invalid JSON response from ChatGPT: {e}[/yellow]")
                logger.warning("Invalid JSON from Responses output: %s", e)
                return []

        except Exception as e:
            console.print(f"[red]Error fetching news from ChatGPT: {e}[/red]")
            logger.exception("Error fetching news from ChatGPT")
            return []

    def _filter_articles_by_allowed_sources(self, articles: List[Dict]) -> List[Dict]:
        """Keep only stories that cite at least one allowed US-based source; filter sources array to allowlist."""
        before_count = len(articles)
        filtered: List[Dict] = []
        dropped_insufficient_sources = 0
        for article in articles:
            sources = article.get("sources") or []
            topic_name = article.get("topic")
            enforce = self.topic_allowlist_enforcement.get(topic_name, True)
            # Build the source list respecting allowlist policy
            selected_sources: List[Dict] = []
            if enforce:
                for s in sources:
                    url = (s or {}).get("url", "")
                    try:
                        netloc = urlparse(url).netloc.lower()
                    except Exception:
                        netloc = ""
                    if any(netloc.endswith(dom) for dom in self.allowed_domains):
                        selected_sources.append(s)
            else:
                selected_sources = list(sources)

            # Enforce minimum sources per story with single-source fallback if configured
            if len(selected_sources) >= self.min_sources_per_story:
                article["sources"] = selected_sources
                filtered.append(article)
            elif len(selected_sources) == 1 and self.allow_single_source_fallback:
                article["sources"] = selected_sources
                filtered.append(article)
            else:
                dropped_insufficient_sources += 1
                continue
        logger.debug("Allowed-sources filter applied | before=%d | after=%d", before_count, len(filtered))
        if dropped_insufficient_sources:
            logger.info("Dropped %d articles due to insufficient sources (min=%d, allow_single=%s)", dropped_insufficient_sources, self.min_sources_per_story, self.allow_single_source_fallback)
        return filtered

    # Note: legacy `_generate_news_prompt` removed. The narrator uses `_generate_enhanced_news_prompt` only.

    def _generate_enhanced_news_prompt(self, topics: List[Dict]) -> str:
        """Generate a single prompt for ChatGPT to fetch, summarize, and create speech-ready content."""
        topic_descriptions = []
        for topic in topics:
            target_count = self._get_topic_story_count(topic)
            keywords = ", ".join(topic.get("keywords", []))
            priority = topic.get("priority", "medium")
            enforce_allowlist = bool(topic.get("enforce_allowlist", True))
            topic_descriptions.append(
                f"- {topic['name']} | required_stories: {target_count} | priority: {priority} | enforce_allowlist: {enforce_allowlist} | keywords: {keywords}"
            )

        allowlist_str = ", ".join(self.allowed_domains)
        wpm = self.words_per_minute
        total_words_target = self.time_limit_minutes * self.words_per_minute
        # Per-topic policy guidance
        per_topic_policy_lines = []
        for t in topics:
            name = t.get("name", "General")
            enforce = bool(t.get("enforce_allowlist", True))
            if enforce:
                per_topic_policy_lines.append(f"- {name}: STRICT — cite only sources from allowlist")
            else:
                per_topic_policy_lines.append(f"- {name}: RELAXED — prefer allowlist, but acceptable to cite reputable US outlets beyond allowlist (no invented links)")
        per_topic_policy = "\n".join(per_topic_policy_lines)

        # Describe the minimum sources policy based on config
        sources_policy_line = (
            f"Aim for AT LEAST {self.min_sources_per_story} distinct sources per story whenever possible; "
            f"if fewer than {self.min_sources_per_story} reputable sources are available within the time window, include the single best source only."
            if self.allow_single_source_fallback
            else
            f"AT LEAST {self.min_sources_per_story} distinct sources per story are required. If fewer than {self.min_sources_per_story} are available, omit that story."
        )

        prompt = f"""
You are an expert news aggregator and editor. Use the web_search tool to fetch the most recent and important articles for each topic. You MUST ground all links in the web_search tool results. Do not invent URLs.

IMPORTANT: Respond ONLY with a valid JSON array. No explanations, no narrative text. Every story MUST include a non-empty sources array with real URLs returned by web_search.

For each topic, search for recent news from the last {self.time_window_hours} hours and provide a speech-ready summary per story (write in natural, flowing, spoken style). If multiple articles cover the same story, count them as one story. If they cover different stories within the same topic, count them separately.

Return JSON in this structure (exact keys):

[
  {{
    "topic": "Federal Reserve & Interest Rates",
    "title": "Fed Signals Potential Rate Changes",
    "summary": "{self.min_words_per_story}+ word speech-ready summary grounded in sources (<= {self.max_summary_len} characters)",
    "importance_score": 9,
    "story_id": "slug_like_identifier",
    "sources": [
      {{"title": "WSJ article title", "url": "https://www.wsj.com/...", "site": "wsj.com", "published_at": "2025-09-01"}},
      {{"title": "Reuters article title", "url": "https://www.reuters.com/...", "site": "reuters.com", "published_at": "2025-09-01"}}
    ]
  }}
]

Topics and keyword hints (use these terms in your web_search queries):
{chr(10).join(topic_descriptions)}

IMPORTANT REQUIREMENTS:
1. Provide speech-ready summaries (minimum {self.min_words_per_story} words per story) that can be read aloud as-is
2. Each story MUST include a sources array with real URLs returned by web_search (no invented links). {sources_policy_line}
4. Include story_id for deduplication (same story = same story_id)
5. If multiple sources cover the same story, use the same story_id
6. Different stories within the same topic should have different story_ids

GLOBAL LENGTH TARGET:
- Aim for a total script length around {total_words_target} words across all stories.

SOURCE QUALITY & ORIGIN CONSTRAINTS:
- Allowlist for STRICT topics (below): cite only US-based, established outlets from this allowlist: {allowlist_str}
- For RELAXED topics (below): prefer allowlist sources, but you MAY cite other reputable US-based outlets not on the allowlist if needed (still must be real results from web_search)

PER-TOPIC SOURCE POLICY:
{per_topic_policy}

NON-EMPTY OUTPUT:
- Do NOT return an empty array. If strict rules would lead to zero stories overall, relax constraints on RELAXED topics to ensure at least one story is returned.

Create engaging, realistic news content. Respond with ONLY the JSON array.
"""
        logger.debug("Generated prompt | length=%d chars | topics=%d | total_words_target=%d", len(prompt), len(topics), total_words_target)
        return prompt

    def _get_topic_story_count(self, topic: Dict) -> int:
        """Get the number of stories for a topic."""
        story_count = topic.get("story_count", 1)
        if isinstance(story_count, dict):
            return story_count.get("min", 1)
        return story_count

    def save_article_summaries(self, articles: List[Dict], filename: str):
        """Save comprehensive news content including summaries, meta info, and speech summaries."""
        summary_file = self.text_dir / f"{filename}_comprehensive_news.txt"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"COMPREHENSIVE NEWS CONTENT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
            
            # Group articles by topic
            articles_by_topic = {}
            for article in articles:
                topic = article.get("topic", "Unknown")
                if topic not in articles_by_topic:
                    articles_by_topic[topic] = []
                articles_by_topic[topic].append(article)
            
            # Write comprehensive content by topic
            for topic, topic_articles in articles_by_topic.items():
                f.write(f"TOPIC: {topic.upper()}\n")
                f.write("-" * 40 + "\n\n")
                
                for i, article in enumerate(topic_articles, 1):
                    f.write(f"ARTICLE {i}:\n")
                    f.write(f"Title: {article.get('title', 'No title')}\n")
                    # Write sources block (array of citations)
                    sources = article.get('sources') or []
                    if sources:
                        f.write("Sources:\n")
                        for s in sources:
                            title = s.get('title', 'Untitled')
                            url = s.get('url', 'No URL')
                            site = s.get('site', '')
                            published = s.get('published_at', '')
                            meta = f" ({site})" if site else ""
                            meta += f" [{published}]" if published else ""
                            f.write(f"  - {title}{meta}: {url}\n")
                    f.write(f"Story ID: {article.get('story_id', 'N/A')}\n")
                    f.write(f"Importance Score: {article.get('importance_score', 'N/A')}\n")
                    f.write(f"Speech Text: {article.get('summary', 'No summary available')}\n")
                    f.write("\n" + "=" * 40 + "\n\n")
        
        console.print(f"[green]Comprehensive news content saved to: {summary_file}[/green]")
        return summary_file

    def build_speech_script(self, articles: List[Dict], greeting: bool = True) -> str:
        """Deduplicate by story_id and combine speech-ready summaries into a single script."""
        unique: Dict[str, Dict] = {}
        for art in articles:
            sid = art.get("story_id") or f"story_{len(unique)}"
            if sid not in unique:
                unique[sid] = art
            else:
                # prefer higher importance score if numeric prefix exists
                def score(x: Dict) -> int:
                    raw = x.get("importance_score", 0)
                    if isinstance(raw, int):
                        return raw
                    try:
                        return int(str(raw).split()[0])
                    except Exception:
                        return 0
                if score(art) > score(unique[sid]):
                    unique[sid] = art

        parts: List[str] = []
        if greeting:
            parts.append("Hello, here's today's news summary.\n")
        # Priority-aware ordering and cap total stories
        def prio_val(a: Dict) -> int:
            # high -> 0, medium -> 1, low -> 2
            p = str(a.get("priority", "medium")).lower()
            return {"high": 0, "medium": 1, "low": 2}.get(p, 1)
        def imp_score(a: Dict) -> int:
            val = a.get("importance_score", 0)
            if isinstance(val, int):
                return val
            try:
                head = str(val).split()[0]
                return int(head) if head.isdigit() else 0
            except Exception:
                return 0
        ordered = sorted(unique.values(), key=lambda a: (prio_val(a), -imp_score(a)))

        # Enforce per-topic story_count
        per_topic_counts: Dict[str, int] = {}
        selected: List[Dict] = []
        for art in ordered:
            topic = art.get("topic", "General")
            # find desired count from config topics
            desired = 1
            for t in self.config.get("topics", []):
                if t.get("name") == topic:
                    sc = t.get("story_count", 1)
                    desired = sc.get("min", 1) if isinstance(sc, dict) else int(sc)
                    break
            current = per_topic_counts.get(topic, 0)
            if current < desired:
                per_topic_counts[topic] = current + 1
                selected.append(art)
            if len(selected) >= self.total_stories_cap:
                break

        for art in selected:
            topic = art.get("topic", "News")
            text = art.get("summary", "")
            section = f"{topic}: {text}"
            parts.append(section.strip())
        parts.append("\nThat concludes today's news summary. Thank you for listening.")
        script = "\n\n".join(parts)
        # Hard word cap based on time limit (approx 150 wpm)
        max_words = self.time_limit_minutes * 150
        words = script.split()
        if len(words) > max_words:
            script = " ".join(words[:max_words]) + "..."
        logger.info(
            "Built speech script | unique_stories=%d | selected=%d | words=%d (cap=%d)",
            len(unique), len(selected), len(script.split()), max_words,
        )
        return script

    def convert_text_to_speech(self, text: str, filename: str) -> Optional[str]:
        """Convert text to speech using OpenAI TTS and save MP3; returns path or None."""
        try:
            audio_file = self.audio_dir / f"{filename}.mp3"
            console.print(f"[yellow]Generating audio using OpenAI TTS...[/yellow]")
            logger.info("Starting TTS generation | file=%s | length_chars=%d | rate=%.2f", str(audio_file), len(text or ""), self.speech_rate)
            with self.client.audio.speech.with_streaming_response.create(
                model="gpt-4o-mini-tts",
                voice="ash",
                input=text,
                instructions="Speak in a neutral and formal tone.",
                speed=self.speech_rate
            ) as response:
                response.stream_to_file(str(audio_file))
            console.print(f"[green]Audio saved to: {audio_file}[/green]")
            logger.info("Audio saved | path=%s", str(audio_file))
            return str(audio_file)
        except Exception as e:
            console.print(f"[red]Error in text-to-speech: {e}[/red]")
            logger.exception("Error during text-to-speech generation")
            return None

class TextToSpeech:
    def __init__(self, output_dir: Path, openai_client, speech_rate: float = 1.0):
        self.output_dir = output_dir
        self.client = openai_client
        self.speech_rate = speech_rate

    def convert_text_to_speech(self, text: str, filename: str) -> Optional[str]:
        """Convert text to speech using OpenAI TTS."""
        try:
            # Create audio file path
            audio_file = self.output_dir / f"{filename}.mp3"
            
            console.print(f"[yellow]Generating audio using OpenAI TTS...[/yellow]")
            logger.info("[TTS helper] Starting TTS generation | file=%s | length_chars=%d | rate=%.2f", str(audio_file), len(text or ""), self.speech_rate)
            
            # Use OpenAI TTS
            with self.client.audio.speech.with_streaming_response.create(
                model="gpt-4o-mini-tts",
                voice="ash",
                input=text,
                instructions="Speak in a neutral and formal tone.",
                speed=self.speech_rate
            ) as response:
                response.stream_to_file(str(audio_file))
            
            console.print(f"[green]Audio saved to: {audio_file}[/green]")
            logger.info("[TTS helper] Audio saved | path=%s", str(audio_file))
            return str(audio_file)

        except Exception as e:
            console.print(f"[red]Error in text-to-speech: {e}[/red]")
            logger.exception("[TTS helper] Error during text-to-speech generation")
            return None


class NewsSummarizer:
    def __init__(self, openai_client):
        """Initialize the news summarizer."""
        self.client = openai_client

    def create_speech_summary(self, articles: List[Dict], target_duration_minutes: int = 15) -> str:
        """Combine story summaries (already speech-ready) while ignoring metadata."""
        logger.debug("Creating speech summary | articles_in=%d | target_minutes=%d", len(articles or []), target_duration_minutes)
        # Deduplicate articles by story_id
        unique_stories = {}
        for article in articles:
            story_id = article.get("story_id", f"story_{len(unique_stories)}")
            if story_id not in unique_stories:
                unique_stories[story_id] = article
            else:
                # If multiple sources cover the same story, keep the one with higher importance
                def parse_score(value) -> int:
                    try:
                        if isinstance(value, int):
                            return value
                        text = str(value)
                        head = text.split()[0]
                        return int(head) if head.isdigit() else 0
                    except Exception:
                        return 0
                current_score = parse_score(unique_stories[story_id].get("importance_score", 0))
                new_score = parse_score(article.get("importance_score", 0))
                if new_score > current_score:
                    unique_stories[story_id] = article
        
        if not unique_stories:
            logger.warning("No unique stories found for speech generation")
            return "No articles were found for speech generation."
        
        # Combine speech-ready summaries from all articles
        speech_summaries = []
        for article in unique_stories.values():
            summary_text = article.get("summary")
            if summary_text:
                speech_summaries.append(f"{article.get('topic', 'Unknown')}: {summary_text}")
        
        if not speech_summaries:
            logger.warning("No speech-ready summaries found in articles")
            return "No speech-ready summaries found in articles."
        
        # Create a combined speech summary
        combined_summary = "Here's today's news summary.\n\n" + "\n\n".join(speech_summaries)
        combined_summary += "\n\nThat concludes today's news summary. Thank you for listening."
        
        console.print(f"[green]Created combined speech summary: {len(combined_summary.split())} words[/green]")
        logger.info("Combined speech summary built | words=%d | stories=%d", len(combined_summary.split()), len(speech_summaries))
        return combined_summary
        



@click.command()
@click.option('--config', default='config.json', help='Path to configuration file')
@click.option('--no-audio', is_flag=True, default=False, help='Skip audio generation')
@click.option('--log-level', default='INFO', type=click.Choice(['DEBUG','INFO','WARNING','ERROR','CRITICAL'], case_sensitive=False), help='Logging level (default: INFO)')
@click.option('--log-file', default=None, help='Optional log file path (defaults to news_output/news_pipeline.log)')
def main(config, no_audio, log_level, log_file):
    """CLI: fetch news, save comprehensive text, build speech, optionally TTS."""
    try:
        # Configure console logging early
        setup_logging(log_level)
        logger.info("News Narrator starting...")

        narrator = NewsNarrator(config)
        # Attach file logging now that we know the output directory
        logs_dir = narrator.output_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        if not log_file:
            log_file = str(logs_dir / "news_pipeline.log")
        else:
            # Ensure custom path directory exists
            try:
                os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
            except Exception:
                pass
        add_file_logging(log_file)
        logger.info("File logging enabled | %s", log_file)
        console.print(Panel.fit("🎙️ News Narrator Starting...", style="blue"))

        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            task = progress.add_task("Fetching and preparing news...", total=None)
            articles = narrator.fetch_news()
            progress.update(task, completed=True)

            if not articles:
                console.print("[red]No news articles retrieved. Exiting.[/red]")
                logger.warning("No news articles retrieved; exiting")
                return

            filename = narrator.config["output_settings"]["filename_format"].format(
                date=datetime.now().strftime("%Y%m%d"),
                time=datetime.now().strftime("%H%M%S")
            )

            if narrator.config.get("output_settings", {}).get("save_summaries", True):
                path = narrator.save_article_summaries(articles, filename)
                logger.info("Saved comprehensive summaries | %s", str(path))

            speech_text = narrator.build_speech_script(articles)
            speech_txt = narrator.text_dir / f"{filename}_speech_summary.txt"
            with open(speech_txt, 'w', encoding='utf-8') as f:
                f.write(speech_text)
            console.print(f"[green]Speech script saved to: {speech_txt}[/green]")
            logger.info("Speech script saved | %s", str(speech_txt))

            if not no_audio and narrator.config["output_settings"].get("save_audio", True):
                audio_path = narrator.convert_text_to_speech(speech_text, filename)
                logger.info("Audio generation attempted | path=%s", audio_path)

        console.print(Panel.fit("✅ News Narrator Complete!", style="green"))
        logger.info("News Narrator completed successfully")
    except Exception as e:
        console.print(f"\n[bold red]Error: {e}[/bold red]")
        logger.exception("Fatal error in News Narrator execution")

if __name__ == "__main__":
    main()
