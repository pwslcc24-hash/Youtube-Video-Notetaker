import os
import tempfile
from typing import List, Tuple
from urllib.parse import urlparse, parse_qs

from flask import Flask, render_template, request
from pytube import YouTube
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
)
import openai


app = Flask(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")

SUPPORTED_LANG_CODES: List[str] = [
    "en",
    "en-US",
    "en-GB",
    "en-AU",
    "en-CA",
    "en-IN",
]


def extract_video_id(youtube_url: str) -> str:
    """
    Support:
      - https://www.youtube.com/watch?v=VIDEOID
      - https://youtu.be/VIDEOID
      - https://www.youtube.com/shorts/VIDEOID
    Return the video ID string, or "" if it cannot be parsed.
    """
    if not youtube_url:
        return ""

    parsed = urlparse(youtube_url.strip())
    hostname = (parsed.hostname or "").lower()

    if "youtube.com" in hostname:
        query = parse_qs(parsed.query)
        if "v" in query and query["v"]:
            return query["v"][0]

        path_parts = [part for part in parsed.path.split("/") if part]
        if path_parts:
            if path_parts[0] == "shorts" and len(path_parts) >= 2:
                return path_parts[1]
            if path_parts[0] == "embed" and len(path_parts) >= 2:
                return path_parts[1]

    if hostname == "youtu.be":
        path = parsed.path.lstrip("/")
        if path:
            return path.split("?")[0]

    return ""


def fetch_transcript_text(video_id: str) -> str:
    """Fetch a transcript via the YouTube transcript API, then fall back to Whisper."""
    if not video_id:
        return ""

    transcript_text = fetch_youtube_transcript(video_id)
    if transcript_text:
        return transcript_text

    return fetch_transcript_via_whisper(video_id)


def fetch_youtube_transcript(video_id: str) -> str:
    try:
        transcript = YouTubeTranscriptApi.get_transcript(
            video_id, languages=SUPPORTED_LANG_CODES
        )
    except (TranscriptsDisabled, NoTranscriptFound):
        return ""
    except Exception:
        return ""

    parts = []
    for chunk in transcript:
        text = chunk.get("text", "").replace("\n", " ").strip()
        if text:
            parts.append(text)
    return " ".join(parts).strip()


def fetch_transcript_via_whisper(video_id: str) -> str:
    video_url = f"https://www.youtube.com/watch?v={video_id}"

    try:
        yt = YouTube(video_url)
        stream = (
            yt.streams.filter(only_audio=True, mime_type="audio/mp4")
            .order_by("abr")
            .desc()
            .first()
        )
        if not stream:
            stream = (
                yt.streams.filter(only_audio=True)
                .order_by("abr")
                .desc()
                .first()
            )
        if not stream:
            return ""

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_filename = "audio.mp4"
            stream.download(output_path=temp_dir, filename=temp_filename)
            temp_file_path = os.path.join(temp_dir, temp_filename)
            with open(temp_file_path, "rb") as audio_file:
                transcription = openai.Audio.transcribe("whisper-1", audio_file)
            text = (transcription or {}).get("text", "").strip()
            return text
    except Exception:
        return ""


def summarize_transcript(transcript_text: str) -> Tuple[bool, str]:
    if not openai.api_key:
        return False, "OpenAI API key is not set. Please configure OPENAI_API_KEY in your environment."

    truncated = transcript_text[:8000]

    prompt = (
        "You are a helpful note-taking assistant. The following is a transcript of a YouTube video. "
        "First, give a concise 3–6 sentence summary. Second, give 5–10 bullet points of key ideas. "
        "Third, give 3–5 key takeaways. Use clear headings: Summary, Notes, Key Takeaways.\n\n"
        f"Transcript:\n{truncated}"
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are a precise and concise assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        text = response["choices"][0]["message"]["content"].strip()
        return True, text
    except Exception as exc:
        return False, f"There was an error contacting OpenAI: {exc}"


@app.route("/", methods=["GET", "POST"])
def index():
    error_message = ""
    result_text = ""
    youtube_url_value = ""

    if request.method == "POST":
        youtube_url_value = request.form.get("youtube_url", "").strip()
        video_id = extract_video_id(youtube_url_value)

        if not video_id:
            error_message = "Please provide a valid YouTube URL."
        else:
            transcript_text = fetch_transcript_text(video_id)
            if not transcript_text:
                error_message = "Could not get a transcript for this video (audio transcription failed). Make sure the video is public and try again."
            else:
                success, response_text = summarize_transcript(transcript_text)
                if success:
                    result_text = response_text
                else:
                    error_message = response_text

    return render_template(
        "index.html",
        error_message=error_message,
        result_text=result_text,
        youtube_url_value=youtube_url_value,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
