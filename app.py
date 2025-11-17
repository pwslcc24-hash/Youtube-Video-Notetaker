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
    """Try YouTube transcripts first, then fall back to Whisper."""
    if not video_id:
        return ""

    # STEP A — Try YouTubeTranscriptAPI first
    try:
        transcripts = YouTubeTranscriptApi.list_transcripts(video_id)

        for transcript in transcripts:
            language_code = (getattr(transcript, "language_code", "") or "").lower()
            if language_code.startswith("en"):
                try:
                    fetched = transcript.fetch()
                    combined = " ".join(entry.get("text", "") for entry in fetched).strip()
                    if combined:
                        return combined
                except Exception:
                    continue

        for transcript in transcripts:
            if not getattr(transcript, "is_translatable", False):
                continue
            try:
                translated = transcript.translate("en").fetch()
                combined = " ".join(entry.get("text", "") for entry in translated).strip()
                if combined:
                    return combined
            except Exception:
                continue
    except Exception:
        pass

    # STEP B — Whisper fallback
    temp_file_path = ""
    try:
        yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
        stream = (
            yt.streams.filter(only_audio=True)
            .order_by("abr")
            .desc()
            .first()
        )
        if not stream:
            return ""

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_file_path = temp_file.name
        temp_file.close()

        stream.download(
            output_path=os.path.dirname(temp_file_path) or ".",
            filename=os.path.basename(temp_file_path),
        )

        with open(temp_file_path, "rb") as audio_file:
            transcription = openai.Audio.transcribe("whisper-1", audio_file)
            text = (transcription or {}).get("text", "").strip()
            if text:
                return text
    except Exception:
        pass
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass

    # STEP C — If everything fails, return empty string
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
                error_message = "This video doesn’t expose an English transcript through the API. Try another video."
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
