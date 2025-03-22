import os
import json
import argparse
import hashlib
import time
from datetime import datetime
import anthropic
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

class MeetingTranscriptAnalyzer:
    """
    A class to analyze meeting transcripts using Anthropic's Claude API.
    The analyzer extracts key information, identifies speakers, generates summaries,
    and compiles action items.
    """
    
    def __init__(self, api_key: Optional[str] = None, cache_dir: Optional[str] = None, 
                 use_cache: bool = True, dry_run: bool = False, model: Optional[str] = None):
        """
        Initialize the MeetingTranscriptAnalyzer.
        
        Args:
            api_key: Anthropic API key (optional, will use environment variable if not provided)
            cache_dir: Directory to store cache files (default: ~/.meeting_analyzer_cache)
            use_cache: Whether to use cached API responses
            dry_run: Whether to run in dry-run mode (no actual API calls)
            model: Anthropic model to use (default: claude-3-opus-20240229)
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key and not dry_run:
            raise ValueError("Anthropic API key is required. Set ANTHROPIC_API_KEY environment variable or provide it directly.")
        
        self.client = anthropic.Anthropic(api_key=self.api_key) if not dry_run else None
        self.model = model or "claude-3-opus-20240229"  # Using Opus for best analysis capabilities
        
        # Cache setup
        self.use_cache = use_cache
        self.dry_run = dry_run
        
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.home() / ".meeting_analyzer_cache"
            
        if self.use_cache or self.dry_run:
            self.cache_dir.mkdir(exist_ok=True, parents=True)
            
        # Rate limiting setup
        self.last_api_call = 0
        self.min_call_interval = 1.0  # Minimum seconds between API calls
    
    def _get_cache_key(self, prompt: str, model: str) -> str:
        """
        Generate a cache key for an API call based on prompt and model.
        
        Args:
            prompt: The prompt text
            model: The model name
            
        Returns:
            A hash string to use as cache key
        """
        # Create a hash of the prompt and model
        hash_input = f"{prompt}|{model}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a cached API response if available.
        
        Args:
            cache_key: The cache key for the response
            
        Returns:
            The cached response or None if not found
        """
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            with open(cache_file, 'r', encoding='utf-8') as f:
                print(f"🔄 Found cached response for {cache_key[:8]}...")
                return json.load(f)
        return None
    
    def _save_to_cache(self, cache_key: str, response: Dict[str, Any]) -> None:
        """
        Save an API response to the cache.
        
        Args:
            cache_key: The cache key for the response
            response: The response to cache
        """
        cache_file = self.cache_dir / f"{cache_key}.json"
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(response, f, indent=2)
            print(f"💾 Cached response with key {cache_key[:8]} for future use")
    
    def _generate_mock_response(self, prompt: str) -> Dict[str, Any]:
        """
        Generate a mock response for dry-run mode.
        
        Args:
            prompt: The prompt that would be sent to the API
            
        Returns:
            A mock response object
        """
        # Create a simple mock response based on what the prompt is asking for
        if "identify all speakers" in prompt:
            return {"content": [{"text": '{"Speaker 1": ["Hello everyone."], "Speaker 2": ["Hi there."]}'}, {"type": "text"}]}
        elif "executive summary" in prompt:
            return {"content": [{"text": "This is a mock executive summary of the meeting."}, {"type": "text"}]}
        elif "action items" in prompt:
            return {"content": [{"text": '[{"task": "Mock task", "assignee": "Mock Person", "deadline": "Next week", "priority": "High"}]'}, {"type": "text"}]}
        elif "sentiment and emotional tone" in prompt:
            return {"content": [{"text": '{"overall_sentiment": "positive", "emotional_dynamics": ["collaborative"], "tone_shifts": [], "engagement_level": "high"}'}, {"type": "text"}]}
        elif "key topics" in prompt:
            return {"content": [{"text": '[{"topic": "Mock Topic", "summary": "This is a mock summary.", "duration": "10 minutes", "key_participants": ["Speaker 1", "Speaker 2"]}]'}, {"type": "text"}]}
        elif "detailed meeting minutes" in prompt:
            return {"content": [{"text": "# Mock Meeting Minutes\n\nAttendees: Mock Person 1, Mock Person 2\n\n## Agenda\n\nDiscussed mock topics."}, {"type": "text"}]}
        else:
            return {"content": [{"text": "This is a mock response for dry-run mode."}, {"type": "text"}]}
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((anthropic.APIError, anthropic.APITimeoutError, anthropic.RateLimitError))
    )
    def _call_claude_api(self, prompt: str, model: str, temperature: float = 0, 
                        max_tokens: int = 2000, system: str = None) -> Dict[str, Any]:
        """
        Call the Claude API with built-in retry logic and rate limiting.
        
        Args:
            prompt: The prompt to send
            model: The model to use
            temperature: The temperature setting
            max_tokens: Maximum tokens to generate
            system: Optional system message
            
        Returns:
            The API response
        """
        # Rate limiting
        current_time = time.time()
        elapsed = current_time - self.last_api_call
        if elapsed < self.min_call_interval:
            time.sleep(self.min_call_interval - elapsed)
            
        # Make the API call
        response = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system or "You are an expert at analyzing meeting transcripts.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        self.last_api_call = time.time()
        return response
        
    def _process_api_request(self, prompt: str, model: Optional[str] = None, 
                           temperature: float = 0, max_tokens: int = 2000, 
                           system: Optional[str] = None, use_cache: Optional[bool] = None,
                           operation_name: str = "Processing") -> Dict[str, Any]:
        """
        Process an API request with caching and dry-run support.
        
        Args:
            prompt: The prompt to send
            model: The model to use (defaults to self.model)
            temperature: The temperature setting
            max_tokens: Maximum tokens to generate
            system: Optional system message
            use_cache: Whether to use cache for this request (overrides instance setting)
            operation_name: Name of the operation for progress messages
            
        Returns:
            The API response
        """
        model = model or self.model
        use_cache_for_request = self.use_cache if use_cache is None else use_cache
        
        # Generate cache key
        cache_key = self._get_cache_key(prompt, model)
        
        # Check cache if enabled
        if use_cache_for_request:
            cached_response = self._get_cached_response(cache_key)
            if cached_response:
                return cached_response
                
        # Handle dry-run mode
        if self.dry_run:
            print(f"🧪 Dry run: Would call Claude API for {operation_name}")
            response = self._generate_mock_response(prompt)
            if use_cache_for_request:
                self._save_to_cache(cache_key, response)
            return response
            
        # Call the API
        print(f"🔄 {operation_name}...")
        response = self._call_claude_api(prompt, model, temperature, max_tokens, system)
        
        # Cache the response if caching is enabled
        if use_cache_for_request:
            self._save_to_cache(cache_key, response)
            
        return response
    
    def load_transcript(self, file_path: str) -> str:
        """
        Load the meeting transcript from a file.
        
        Args:
            file_path: Path to the transcript file
            
        Returns:
            The transcript text
        """
        print(f"📄 Reading transcript from {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def identify_speakers(self, transcript: str) -> Dict[str, List[str]]:
        """
        Identify all speakers in the transcript and their contributions.
        
        Args:
            transcript: The meeting transcript
            
        Returns:
            Dictionary mapping speakers to their statements
        """
        prompt = f"""
        Please analyze this meeting transcript and identify all speakers and their contributions.
        Format the response as a JSON object where each key is a speaker's name and 
        the value is an array of all their statements.
        
        Transcript:
        {transcript}
        
        Provide ONLY the JSON response without any additional text.
        """
        
        system_msg = "You are an expert at analyzing meeting transcripts and extracting structured information. Return only JSON as requested."
        
        print("🧠 Claude is pondering deeply about who said what...")
        response = self._process_api_request(
            prompt=prompt,
            max_tokens=4000,
            temperature=0,
            system=system_msg,
            operation_name="Identifying speakers"
        )
        
        try:
            # Extract JSON from the response
            content = response.get("content", [{}])[0].get("text", "{}")
            # If the response includes markdown code blocks, extract the JSON part
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            parsed_content = json.loads(content)
            print(f"✅ Identified {len(parsed_content)} speakers in the transcript")
            return parsed_content
        except json.JSONDecodeError:
            # If parsing fails, make a second attempt with a more explicit request
            retry_prompt = f"""
            The previous response couldn't be parsed as JSON. Please analyze this meeting transcript 
            and return ONLY a valid JSON object where each key is a speaker's name and 
            the value is an array of all their statements. No markdown, no explanations.
            
            Transcript:
            {transcript}
            """
            
            print("🔄 First attempt couldn't be parsed, retrying with a clearer prompt...")
            retry_response = self._process_api_request(
                prompt=retry_prompt,
                max_tokens=4000,
                temperature=0,
                system="Return only valid JSON with no additional text.",
                operation_name="Retrying speaker identification",
                use_cache=False  # Don't use cache for retry attempts
            )
            
            content = retry_response.get("content", [{}])[0].get("text", "{}")
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
                
            parsed_content = json.loads(content)
            print(f"✅ Successfully identified {len(parsed_content)} speakers on retry")
            return parsed_content
    
    def generate_executive_summary(self, transcript: str) -> str:
        """
        Generate a concise executive summary of the meeting.
        
        Args:
            transcript: The meeting transcript
            
        Returns:
            Executive summary of the meeting
        """
        prompt = f"""
        Please create a concise executive summary of this meeting transcript.
        Focus on the main topics discussed, key decisions made, and overall purpose.
        Keep it to around 150-200 words, clear and informative for busy executives.
        
        Transcript:
        {transcript}
        """
        
        print("✍️ Distilling meeting essence into executive haiku...")
        response = self._process_api_request(
            prompt=prompt,
            max_tokens=1000,
            temperature=0.3,
            system="You are an expert at creating clear, concise executive summaries from meeting transcripts.",
            operation_name="Generating executive summary"
        )
        
        result = response.get("content", [{}])[0].get("text", "").strip()
        print("📝 Executive summary created successfully")
        return result
    
    def extract_action_items(self, transcript: str) -> List[Dict[str, str]]:
        """
        Extract action items from the transcript with assignees and deadlines.
        
        Args:
            transcript: The meeting transcript
            
        Returns:
            List of action items with details
        """
        prompt = f"""
        Please extract all action items from this meeting transcript.
        For each action item, identify:
        1. The specific task to be done
        2. Who is assigned to do it
        3. Any mentioned deadline or timeframe
        4. The priority level (if mentioned)
        
        Format the response as a JSON array where each item has keys:
        "task", "assignee", "deadline", "priority"
        
        If any information is not specified in the transcript, use null for that field.
        
        Transcript:
        {transcript}
        
        Provide ONLY the JSON response without any additional text.
        """
        
        print("📋 Hunting for those to-dos and action items...")
        response = self._process_api_request(
            prompt=prompt,
            max_tokens=3000,
            temperature=0,
            system="You are an expert at identifying action items in meeting transcripts. Return only JSON as requested.",
            operation_name="Extracting action items"
        )
        
        try:
            content = response.get("content", [{}])[0].get("text", "[]")
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
                
            parsed_content = json.loads(content)
            print(f"✅ Found {len(parsed_content)} action items")
            return parsed_content
        except json.JSONDecodeError:
            retry_prompt = f"""
            The previous response couldn't be parsed as JSON. Please extract all action items from 
            this meeting transcript and return ONLY a valid JSON array. Each item should have keys:
            "task", "assignee", "deadline", "priority". No markdown, no explanations.
            
            Transcript:
            {transcript}
            """
            
            print("🔄 First attempt couldn't be parsed, retrying with a clearer prompt...")
            retry_response = self._process_api_request(
                prompt=retry_prompt,
                max_tokens=3000,
                temperature=0,
                system="Return only valid JSON with no additional text.",
                operation_name="Retrying action item extraction",
                use_cache=False  # Don't use cache for retry attempts
            )
            
            content = retry_response.get("content", [{}])[0].get("text", "[]")
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
                
            parsed_content = json.loads(content)
            print(f"✅ Successfully found {len(parsed_content)} action items on retry")
            return parsed_content
    
    def analyze_sentiment(self, transcript: str) -> Dict[str, Any]:
        """
        Analyze the overall sentiment and emotional tone of the meeting.
        
        Args:
            transcript: The meeting transcript
            
        Returns:
            Dictionary with sentiment analysis results
        """
        prompt = f"""
        Please analyze the sentiment and emotional tone of this meeting transcript.
        Include:
        1. Overall meeting sentiment (positive, negative, neutral, mixed)
        2. Emotional dynamics (e.g., conflict, enthusiasm, hesitation)
        3. Any notable shifts in tone throughout the meeting
        4. Level of engagement from participants
        
        Format the response as a JSON object with keys:
        "overall_sentiment", "emotional_dynamics", "tone_shifts", "engagement_level"
        
        Transcript:
        {transcript}
        
        Provide ONLY the JSON response without any additional text.
        """
        
        print("🎭 Reading the room and sensing the vibes...")
        response = self._process_api_request(
            prompt=prompt,
            max_tokens=2000,
            temperature=0.2,
            system="You are an expert at analyzing sentiment and emotional dynamics in conversations. Return only JSON as requested.",
            operation_name="Analyzing sentiment"
        )
        
        try:
            content = response.get("content", [{}])[0].get("text", "{}")
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
                
            parsed_content = json.loads(content)
            print(f"✅ Mood analysis complete: overall sentiment is {parsed_content.get('overall_sentiment', 'unknown')}")
            return parsed_content
        except json.JSONDecodeError:
            print("⚠️ Could not parse sentiment analysis response, using fallback")
            # Fallback to a more explicit format if JSON parsing fails
            return {
                "overall_sentiment": "Unable to parse sentiment analysis",
                "emotional_dynamics": [],
                "tone_shifts": [],
                "engagement_level": "unknown"
            }
    
 = response.content[0].text
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
                
            return json.loads(content)
        except json.JSONDecodeError:
            retry_prompt = f"""
            The previous response couldn't be parsed as JSON. Please identify the key topics discussed 
            in this meeting transcript and return ONLY a valid JSON array. Each item should have keys:
            "topic", "summary", "duration", "key_participants". No markdown, no explanations.
            
            Transcript:
            {transcript}
            """
            
            retry_response = self.client.messages.create(
                model=self.model,
                max_tokens=3000,
                temperature=0.1,
                system="Return only valid JSON with no additional text.",
                messages=[
                    {"role": "user", "content": retry_prompt}
                ]
            )
            
            content = retry_response.content[0].text
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
                
            return json.loads(content)
    
    def generate_detailed_minutes(self, transcript: str, speakers: Dict[str, List[str]], 
                                 topics: List[Dict[str, Any]], action_items: List[Dict[str, str]]) -> str:
        """
        Generate detailed meeting minutes based on all analyzed components.
        
        Args:
            transcript: The meeting transcript
            speakers: Dictionary of speakers and their statements
            topics: List of key topics discussed
            action_items: List of action items
            
        Returns:
            Formatted detailed meeting minutes
        """
        # Create a structured context for Claude to generate comprehensive minutes
        speakers_list = list(speakers.keys())
        topics_json = json.dumps(topics, indent=2)
        action_items_json = json.dumps(action_items, indent=2)
        
        prompt = f"""
        Please generate detailed meeting minutes based on this transcript and analysis.
        
        The meeting included these participants: {', '.join(speakers_list)}
        
        The key topics discussed were:
        {topics_json}
        
        The action items identified were:
        {action_items_json}
        
        Create professional, well-structured meeting minutes that include:
        1. An introduction with meeting date, time, and purpose
        2. A list of attendees
        3. Detailed discussion points organized by topic
        4. A clear section for decisions made
        5. A formatted list of action items with assignees
        6. Any notable questions or concerns raised
        
        Use professional language and formatting appropriate for business documentation.
        
        Transcript:
        {transcript}
        """
        
        print("📝 Crafting detailed meeting minutes from all the analytical ingredients...")
        response = self._process_api_request(
            prompt=prompt,
            max_tokens=4000,
            temperature=0.3,
            system="You are an expert at creating comprehensive, professional meeting minutes. Format your response with clear headings and structure.",
            operation_name="Generating detailed minutes"
        )
        
        result = response.get("content", [{}])[0].get("text", "").strip()
        print("✅ Detailed meeting minutes created successfully")
        return result
    
    def generate_participation_metrics(self, speakers: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Generate metrics about participant engagement and contributions.
        
        Args:
            speakers: Dictionary of speakers and their statements
            
        Returns:
            Dictionary with participation metrics
        """
        print("📊 Calculating participation metrics...")
        metrics = {}
        total_contributions = 0
        
        # Count contributions and calculate word counts
        for speaker, statements in speakers.items():
            speaker_word_count = sum(len(statement.split()) for statement in statements)
            
            metrics[speaker] = {
                "contribution_count": len(statements),
                "word_count": speaker_word_count,
                "average_statement_length": speaker_word_count / len(statements) if statements else 0
            }
            
            total_contributions += len(statements)
        
        # Calculate participation percentages
        for speaker in metrics:
            metrics[speaker]["participation_percentage"] = round(
                (metrics[speaker]["contribution_count"] / total_contributions) * 100, 2
            ) if total_contributions > 0 else 0
        
        result = {
            "individual_metrics": metrics,
            "total_contributions": total_contributions,
            "unique_participants": len(metrics)
        }
        
        print(f"✅ Participation metrics calculated for {len(metrics)} participants")
        return result
    
    def analyze_full_transcript(self, transcript_path: str, analysis_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform a complete analysis of the meeting transcript.
        
        Args:
            transcript_path: Path to the transcript file
            analysis_types: List of analysis types to perform (default: all)
                Options: 'speakers', 'summary', 'action_items', 'sentiment', 
                        'topics', 'participation', 'minutes'
            
        Returns:
            Dictionary with all analysis results
        """
        # Define all available analysis types
        all_analysis_types = [
            'speakers', 'summary', 'action_items', 'sentiment', 
            'topics', 'participation', 'minutes'
        ]
        
        # If no specific types are provided, run all analyses
        analysis_types = analysis_types or all_analysis_types
        
        print(f"🚀 Starting analysis of transcript: {transcript_path}")
        print(f"📋 Requested analysis types: {', '.join(analysis_types)}")
        
        # Load the transcript
        transcript = self.load_transcript(transcript_path)
        
        # Initialize results dictionary
        results = {
            "meta": {
                "transcript_file": transcript_path,
                "analysis_timestamp": datetime.now().isoformat(),
                "model_used": self.model,
                "analysis_types": analysis_types
            }
        }
        
        # Always identify speakers first as it's needed for other analyses
        if 'speakers' in analysis_types or 'participation' in analysis_types or 'minutes' in analysis_types:
            speakers = self.identify_speakers(transcript)
            results["speakers"] = speakers
        else:
            speakers = {}
            
        # Run requested analyses
        if 'summary' in analysis_types:
            results["executive_summary"] = self.generate_executive_summary(transcript)
            
        if 'action_items' in analysis_types:
            results["action_items"] = self.extract_action_items(transcript)
        else:
            results["action_items"] = []
            
        if 'sentiment' in analysis_types:
            results["sentiment_analysis"] = self.analyze_sentiment(transcript)
            
        if 'topics' in analysis_types:
            results["key_topics"] = self.identify_key_topics(transcript)
        else:
            results["key_topics"] = []
            
        if 'participation' in analysis_types and speakers:
            results["participation_metrics"] = self.generate_participation_metrics(speakers)
            
        if 'minutes' in analysis_types:
            # Ensure we have the required components for minutes
            if "speakers" not in results:
                speakers = self.identify_speakers(transcript)
                results["speakers"] = speakers
                
            if "key_topics" not in results:
                results["key_topics"] = self.identify_key_topics(transcript)
                
            if "action_items" not in results:
                results["action_items"] = self.extract_action_items(transcript)
                
            results["detailed_minutes"] = self.generate_detailed_minutes(
                transcript, results["speakers"], results["key_topics"], results["action_items"]
            )
        
        print("✅ Analysis complete!")
        return results
    
    def save_results(self, results: Dict[str, Any], output_path: str) -> None:
        """
        Save the analysis results to JSON and markdown files.
        
        Args:
            results: Dictionary with all analysis results
            output_path: Base path for output files (without extension)
        """
        print(f"💾 Saving analysis results to {output_path}...")
        
        # Save complete results as JSON
        with open(f"{output_path}.json", 'w', encoding='utf-8') as json_file:
            json.dump(results, json_file, indent=2, ensure_ascii=False)
            print(f"✅ JSON data saved to {output_path}.json")
        
        # Create a markdown summary, only including sections that we analyzed
        markdown = f"""# Meeting Analysis Summary
Generated on: {results['meta']['analysis_timestamp']}
"""
        
        if 'executive_summary' in results:
            markdown += f"""
## Executive Summary
{results['executive_summary']}
"""
        
        if 'speakers' in results:
            markdown += f"""
## Participants
{', '.join(results['speakers'].keys())}
"""
        
        if 'key_topics' in results and results['key_topics']:
            markdown += """
## Key Topics
"""
            
            for topic in results['key_topics']:
                markdown += f"""
### {topic['topic']}
{topic['summary']}

**Key participants:** {', '.join(topic['key_participants'])}
**Duration:** {topic['duration']}
"""

        if 'action_items' in results and results['action_items']:
            markdown += """
## Action Items
| Task | Assignee | Deadline | Priority |
|------|----------|----------|----------|
"""
            
            for item in results['action_items']:
                task = item.get('task', '')
                assignee = item.get('assignee', '')
                deadline = item.get('deadline', '')
                priority = item.get('priority', '')
                markdown += f"| {task} | {assignee} | {deadline} | {priority} |\n"
        
        if 'sentiment_analysis' in results:
            markdown += f"""
## Sentiment Analysis
**Overall sentiment:** {results['sentiment_analysis']['overall_sentiment']}
"""
        
        if 'participation_metrics' in results:
            markdown += """
## Participation Metrics
"""
            
            for speaker, metrics in results['participation_metrics']['individual_metrics'].items():
                markdown += f"**{speaker}:** {metrics['contribution_count']} contributions ({metrics['participation_percentage']}% of discussion)\n\n"
        
        if 'detailed_minutes' in results:
            markdown += """
## Detailed Minutes
"""
            markdown += results['detailed_minutes']
        
        # Save markdown summary
        with open(f"{output_path}.md", 'w', encoding='utf-8') as md_file:
            md_file.write(markdown)
            print(f"✅ Markdown summary saved to {output_path}.md")
        
        # Also create HTML version if minutes are available
        if 'detailed_minutes' in results:
            try:
                import markdown as md_converter
                html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Meeting Analysis</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }}
        h1, h2, h3 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
    </style>
</head>
<body>
    {md_converter.markdown(markdown)}
</body>
</html>
"""
                with open(f"{output_path}.html", 'w', encoding='utf-8') as html_file:
                    html_file.write(html)
                    print(f"✅ HTML report saved to {output_path}.html")
            except ImportError:
                print("⚠️ Python markdown package not available, skipping HTML export")
                pass


def main():
    """
    Main function to run the meeting transcript analyzer from command line.
    """
    parser = argparse.ArgumentParser(description='Analyze meeting transcripts using Anthropic API')
    parser.add_argument('transcript_file', help='Path to the meeting transcript file')
    parser.add_argument('--output', '-o', help='Base path for output files (without extension)')
    parser.add_argument('--api-key', help='Anthropic API key (optional, can use ANTHROPIC_API_KEY env var)')
    parser.add_argument('--cache-dir', help='Directory to store cache files')
    parser.add_argument('--no-cache', action='store_true', help='Disable caching of API responses')
    parser.add_argument('--dry-run', action='store_true', help='Run in dry-run mode (no actual API calls)')
    parser.add_argument('--model', help='Anthropic model to use (default: claude-3-opus-20240229)')
    
    # Add options to run specific analysis types
    analysis_group = parser.add_argument_group('Analysis Types')
    analysis_group.add_argument('--summary-only', action='store_true', help='Generate only the executive summary')
    analysis_group.add_argument('--action-items-only', action='store_true', help='Extract only action items')
    analysis_group.add_argument('--sentiment-only', action='store_true', help='Perform only sentiment analysis')
    analysis_group.add_argument('--topics-only', action='store_true', help='Identify only key topics')
    analysis_group.add_argument('--participation-only', action='store_true', help='Calculate only participation metrics')
    analysis_group.add_argument('--minutes-only', action='store_true', help='Generate only detailed minutes')
    
    args = parser.parse_args()
    
    # Determine output path (default to transcript filename base)
    output_path = args.output or os.path.splitext(args.transcript_file)[0] + "_analysis"
    
    # Determine which analysis types to run
    analysis_types = []
    
    # Check if any specific analysis types were requested
    specific_analyses = [
        (args.summary_only, 'summary'),
        (args.action_items_only, 'action_items'),
        (args.sentiment_only, 'sentiment'),
        (args.topics_only, 'topics'),
        (args.participation_only, 'participation'),
        (args.minutes_only, 'minutes')
    ]
    
    # If any specific analyses are requested, only run those
    for flag, analysis_type in specific_analyses:
        if flag:
            analysis_types.append(analysis_type)
            
    # If no specific analyses are requested, run all of them
    if not analysis_types:
        analysis_types = None  # This will run all analyses
        
    try:
        # Show a fun welcome message
        print("""
✨ Meeting Transcript Analyzer ✨
Powered by Anthropic's Claude API
---------------------------------------
Let's distill those meeting ramblings into something useful!
        """)
        
        # Initialize and run analyzer
        analyzer = MeetingTranscriptAnalyzer(
            api_key=args.api_key,
            cache_dir=args.cache_dir,
            use_cache=not args.no_cache,
            dry_run=args.dry_run,
            model=args.model
        )
        
        print("🔮 Claude is rolling up its sleeves and getting to work...")
        results = analyzer.analyze_full_transcript(args.transcript_file, analysis_types)
        analyzer.save_results(results, output_path)
        
        print(f"""
🎉 Analysis complete! 🎉
Results saved to:
- {output_path}.json (complete data)
- {output_path}.md (formatted report)
        """)
        
        if args.dry_run:
            print("Note: This was a dry run. No actual API calls were made.")
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nDetailed traceback:")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
