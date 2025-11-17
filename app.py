import json
from typing import Optional
from urllib.parse import parse_qs, urlparse
from urllib.request import urlopen

from flask import Flask, render_template, request
from openai import OpenAI
from youtube_transcript_api import (NoTranscriptFound,
                                    TranscriptsDisabled,
                                    YouTubeTranscriptApi)


app = Flask(__name__)
client = OpenAI()


def extract_video_id(url: str) -> Optional[str]:
    """Return the YouTube video ID for the given URL string."""
    if not url:
        return None

    parsed = urlparse(url.strip())

    if parsed.netloc in {"youtu.be"} and parsed.path:
        return parsed.path.lstrip("/").split("/")[0]

    if "youtube" in parsed.netloc:
        if parsed.path == "/watch":
            return parse_qs(parsed.query).get("v", [None])[0]

        if parsed.path.startswith("/shorts/"):
            return parsed.path.split("/shorts/")[-1].split("/")[0]

        if parsed.path.startswith("/embed/"):
            return parsed.path.split("/embed/")[-1].split("/")[0]

    return None


def fetch_video_title(video_id: str) -> Optional[str]:
    """Fetch the video title via YouTube's oEmbed endpoint."""
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    oembed_url = f"https://www.youtube.com/oembed?url={video_url}&format=json"

    try:
        with urlopen(oembed_url) as response:  # noqa: S310 (trusted domain)
            return json.load(response).get("title")
    except Exception:
        return None


def build_notes(transcript_text: str) -> str:
    system_prompt = (
        "You are an expert note taker. Create clear, well-structured notes with "
        "concise headings, bullet points, and highlight the key takeaways."
    )
    user_prompt = (
        "Create structured notes for the following transcript. The notes should "
        "include informative section headings, nested bullet points when "
        "appropriate, and an overall summary at the end.\n\n"
        f"Transcript:\n{transcript_text}"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
    except Exception as exc:
        raise RuntimeError("We had trouble connecting to OpenAI. Please try again.") from exc

    return response.choices[0].message.content.strip()


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/summarize", methods=["POST"])
def summarize():
    video_url = request.form.get("video_url", "").strip()

    video_id = extract_video_id(video_url)
    if not video_id:
        return render_template(
            "index.html",
            error="Please provide a valid YouTube URL (watch, shorts, or youtu.be).",
            previous_url=video_url,
        )

    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
    except TranscriptsDisabled:
        return render_template(
            "index.html",
            error="This video has transcripts disabled. Please try another video.",
            previous_url=video_url,
        )
    except NoTranscriptFound:
        return render_template(
            "index.html",
            error="No English transcript is available for this video.",
            previous_url=video_url,
        )
    except Exception:
        return render_template(
            "index.html",
            error="We couldn't fetch the transcript. Double-check the URL and try again.",
            previous_url=video_url,
        )

    transcript_text = " ".join(
        segment["text"].strip()
        for segment in transcript
        if segment.get("text", "").strip()
    ).strip()

    if not transcript_text:
        return render_template(
            "index.html",
            error="The transcript appears to be empty. Please try a different video.",
            previous_url=video_url,
        )

    try:
        notes = build_notes(transcript_text)
    except RuntimeError as err:
        return render_template("index.html", error=str(err), previous_url=video_url)

    title = fetch_video_title(video_id) or "Unknown Title"
    canonical_url = f"https://www.youtube.com/watch?v={video_id}"

    return render_template(
        "result.html",
        notes=notes,
        video_title=title,
        video_url=canonical_url,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
