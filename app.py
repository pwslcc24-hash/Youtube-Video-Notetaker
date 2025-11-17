import os
import re
from urllib.parse import parse_qs, urlparse

import requests
from flask import Flask, render_template, request
from openai import OpenAI
from youtube_transcript_api import (NoTranscriptFound, TranscriptsDisabled,
                                    YouTubeTranscriptApi)

app = Flask(__name__)


def extract_video_id(url: str) -> str:
    if not url:
        raise ValueError("Please provide a YouTube URL.")

    parsed = urlparse(url.strip())
    if parsed.hostname in {"youtu.be"} and parsed.path:
        return parsed.path.lstrip("/")

    if parsed.hostname and "youtube" in parsed.hostname:
        if parsed.path.startswith("/watch"):
            query = parse_qs(parsed.query)
            if "v" in query:
                return query["v"][0]
        path_parts = parsed.path.split("/")
        for part in path_parts:
            if re.fullmatch(r"[A-Za-z0-9_-]{11}", part):
                return part

    match = re.search(r"(?:(?:v=)|(?:be/)|(?:embed/)|(?:shorts/))([A-Za-z0-9_-]{11})", url)
    if match:
        return match.group(1)

    raise ValueError("Could not extract a video ID from the provided URL.")


def fetch_transcript_text(video_id: str) -> str:
    """
    Try to download an English transcript for the given video_id.
    Prefer human-created transcripts, then auto-generated.
    Return the transcript text as one big string.
    Return an empty string if no transcript is available.
    """

    try:
        transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
        try:
            transcript = transcripts.find_manually_created_transcript(['en'])
        except Exception:  # noqa: BLE001
            transcript = transcripts.find_generated_transcript(['en'])
        entries = transcript.fetch()
        return " ".join(entry["text"] for entry in entries)
    except (TranscriptsDisabled, NoTranscriptFound, KeyError, ValueError):
        return ""


def fetch_video_title(video_url: str) -> str:
    try:
        response = requests.get(
            "https://www.youtube.com/oembed",
            params={"url": video_url, "format": "json"},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("title", "YouTube Video")
    except Exception:
        return "YouTube Video"


def call_openai(summary_prompt: str) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

    client = OpenAI(api_key=api_key)

    try:
        response = client.responses.create(
            model="gpt-4o-mini",
            input=summary_prompt,
            max_output_tokens=600,
            temperature=0.4,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Failed to generate notes using OpenAI.") from exc

    contents = []
    for item in response.output:
        if item.type == "output_text":
            contents.append(item.text)
        elif hasattr(item, "content"):
            for block in item.content:
                if getattr(block, "type", None) == "output_text":
                    contents.append(block.text)
    return "\n".join(contents).strip()


def parse_sections(text: str):
    sections = {"summary": [], "notes": [], "takeaways": []}
    current = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        lowered = line.lower()
        if lowered.startswith("summary"):
            current = "summary"
            remainder = line.split(":", 1)
            if len(remainder) > 1 and remainder[1].strip():
                sections[current].append(remainder[1].strip())
            continue
        if lowered.startswith("notes"):
            current = "notes"
            continue
        if "takeaway" in lowered:
            current = "takeaways"
            continue

        if current is None:
            continue

        if current == "summary":
            sections[current].append(line)
        else:
            cleaned = line.lstrip("-•*0123456789. ").strip()
            sections[current].append(cleaned)

    summary_text = " ".join(sections["summary"]) if sections["summary"] else ""
    return summary_text, sections["notes"], sections["takeaways"]


@app.route("/", methods=["GET", "POST"])
def index():
    context = {
        "video_title": None,
        "summary": None,
        "notes": None,
        "takeaways": None,
        "error": None,
        "url": "",
    }

    if request.method == "POST":
        video_url = request.form.get("video_url", "").strip()
        context["url"] = video_url

        try:
            video_id = extract_video_id(video_url)
            transcript_text = fetch_transcript_text(video_id)
            video_title = fetch_video_title(video_url)
        except ValueError as exc:
            context["error"] = str(exc)
            return render_template("index.html", **context)

        if not transcript_text:
            context["error"] = (
                "This video doesn’t have an available English transcript. "
                "Try another video."
            )
            return render_template("index.html", **context)

        prompt = (
            "You are an assistant that summarizes YouTube transcripts. "
            "Given the transcript below, write a concise response with the following exact sections:\n"
            "Summary: Provide 3-6 sentences summarizing the video.\n"
            "Notes: Provide 5-10 bullet points, each on a new line starting with '-'.\n"
            "Key takeaways: Provide 3-5 bullet points, each on a new line starting with '-'.\n"
            "Transcript:\n" + transcript_text
        )

        try:
            ai_output = call_openai(prompt)
            summary_text, notes, takeaways = parse_sections(ai_output)
        except RuntimeError as exc:
            context["error"] = str(exc)
            return render_template("index.html", **context)

        context.update(
            {
                "video_title": video_title,
                "summary": summary_text or ai_output,
                "notes": notes,
                "takeaways": takeaways,
            }
        )

    return render_template("index.html", **context)


if __name__ == "__main__":
    app.run(debug=True)
