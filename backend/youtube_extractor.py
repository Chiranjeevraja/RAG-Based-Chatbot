import re
import os
import tempfile
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
from googleapiclient.discovery import build
from openai import OpenAI

_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Whisper API limit is 25 MB
WHISPER_MAX_BYTES = 25 * 1024 * 1024


def extract_video_id(url: str) -> str:
    patterns = [
        r"(?:youtube\.com\/watch\?v=)([^&\n?#]+)",
        r"(?:youtu\.be\/)([^&\n?#]+)",
        r"(?:youtube\.com\/embed\/)([^&\n?#]+)",
        r"(?:youtube\.com\/shorts\/)([^&\n?#]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(f"Could not extract video ID from URL: {url}")


def get_transcript(video_id: str) -> dict:
    """Returns transcript text and metadata."""
    try:
        transcript_entries = YouTubeTranscriptApi.get_transcript(video_id)
        full_text = " ".join(entry["text"] for entry in transcript_entries)
        return {
            "success": True,
            "text": full_text,
            "entries": transcript_entries,
        }
    except TranscriptsDisabled:
        return {"success": False, "text": "", "error": "Transcripts are disabled for this video."}
    except NoTranscriptFound:
        return {"success": False, "text": "", "error": "No transcript found for this video."}
    except Exception as e:
        return {"success": False, "text": "", "error": str(e)}


def get_transcript_whisper(video_id: str) -> dict:
    """
    Download YouTube audio with pytubefix and transcribe using OpenAI Whisper.
    Used as a fallback when youtube-transcript-api finds no captions.
    Whisper API limit: 25 MB.
    """
    from pytubefix import YouTube

    url = f"https://www.youtube.com/watch?v={video_id}"

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            yt = YouTube(url)
            stream = yt.streams.filter(only_audio=True).order_by("abr").last()
            if stream is None:
                return {"success": False, "text": "", "method": "whisper", "error": "No audio stream found."}

            print(f"[whisper] downloading: {stream.mime_type} @ {stream.abr}")
            # Let pytubefix keep its default filename so the extension is preserved
            audio_path = stream.download(output_path=tmpdir)
        except Exception as e:
            import traceback
            print(f"[whisper] download ERROR:\n{traceback.format_exc()}")
            return {"success": False, "text": "", "method": "whisper", "error": f"Download failed: {e}"}

        file_size = os.path.getsize(audio_path)
        print(f"[whisper] file: {os.path.basename(audio_path)} — {file_size / 1024 / 1024:.1f} MB")

        if file_size > WHISPER_MAX_BYTES:
            return {
                "success": False,
                "text": "",
                "method": "whisper",
                "error": f"Audio too large for Whisper API ({file_size / 1024 / 1024:.1f} MB > 25 MB). Try a shorter video.",
            }

        try:
            # Pass as tuple (filename, bytes) so Whisper can detect format from the extension
            with open(audio_path, "rb") as f:
                response = _openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=(os.path.basename(audio_path), f),
                    response_format="text",
                )
            text = response if isinstance(response, str) else response.text
            print(f"[whisper] transcription complete — {len(text)} chars")
            return {"success": True, "text": text, "method": "whisper"}
        except Exception as e:
            return {"success": False, "text": "", "method": "whisper", "error": f"Whisper API error: {e}"}


def get_comments_with_replies(video_id: str, max_threads: int = 100) -> dict:
    """
    Fetch comment threads including replies using YouTube Data API v3.
    Returns structured threads: [{text, author, likes, replies: [{text, author, likes}]}]
    Each thread is one comment + all its replies in order.
    """
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        return {"success": False, "comment_threads": [], "error": "YOUTUBE_API_KEY not set."}

    try:
        youtube = build("youtube", "v3", developerKey=api_key)
        threads = []
        page_token = None

        while len(threads) < max_threads:
            request = youtube.commentThreads().list(
                part="snippet,replies",          # replies gives up to 20 replies inline
                videoId=video_id,
                maxResults=min(100, max_threads - len(threads)),
                order="relevance",
                pageToken=page_token,
            )
            response = request.execute()

            for item in response.get("items", []):
                top = item["snippet"]["topLevelComment"]["snippet"]
                thread = {
                    "text": top["textDisplay"],
                    "author": top.get("authorDisplayName", "Unknown"),
                    "likes": top.get("likeCount", 0),
                    "replies": [],
                }

                total_replies = item["snippet"].get("totalReplyCount", 0)
                inline_replies = item.get("replies", {}).get("comments", [])

                # Use inline replies (up to 20 come free with the thread request)
                for r in inline_replies:
                    rs = r["snippet"]
                    thread["replies"].append({
                        "text": rs["textDisplay"],
                        "author": rs.get("authorDisplayName", "Unknown"),
                        "likes": rs.get("likeCount", 0),
                    })

                # If there are more replies than inline gave us, fetch the rest
                if total_replies > len(inline_replies):
                    try:
                        reply_req = youtube.comments().list(
                            part="snippet",
                            parentId=item["snippet"]["topLevelComment"]["id"],
                            maxResults=100,
                        )
                        reply_resp = reply_req.execute()
                        seen = {r["text"] for r in thread["replies"]}
                        for r in reply_resp.get("items", []):
                            rs = r["snippet"]
                            if rs["textDisplay"] not in seen:
                                thread["replies"].append({
                                    "text": rs["textDisplay"],
                                    "author": rs.get("authorDisplayName", "Unknown"),
                                    "likes": rs.get("likeCount", 0),
                                })
                    except Exception:
                        pass  # partial replies are fine

                threads.append(thread)

            page_token = response.get("nextPageToken")
            if not page_token:
                break

        print(f"[comments] fetched {len(threads)} threads")
        return {"success": True, "comment_threads": threads}

    except Exception as e:
        return {"success": False, "comment_threads": [], "error": str(e)}


def get_comments(video_id: str, max_results: int = 200) -> dict:
    """Fetch top-level comments using YouTube Data API v3."""
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        return {"success": False, "comments": [], "error": "YOUTUBE_API_KEY not set."}

    try:
        youtube = build("youtube", "v3", developerKey=api_key)
        comments = []
        page_token = None

        while len(comments) < max_results:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(100, max_results - len(comments)),
                order="relevance",
                pageToken=page_token,
            )
            response = request.execute()

            for item in response.get("items", []):
                snippet = item["snippet"]["topLevelComment"]["snippet"]
                comments.append({
                    "text": snippet["textDisplay"],
                    "author": snippet.get("authorDisplayName", "Unknown"),
                    "likes": snippet.get("likeCount", 0),
                })

            page_token = response.get("nextPageToken")
            if not page_token:
                break

        return {"success": True, "comments": comments}

    except Exception as e:
        return {"success": False, "comments": [], "error": str(e)}


def get_video_metadata(video_id: str) -> dict:
    """Fetch basic video metadata."""
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        return {"title": f"Video {video_id}", "channel": "Unknown", "description": ""}

    try:
        youtube = build("youtube", "v3", developerKey=api_key)
        response = youtube.videos().list(
            part="snippet",
            id=video_id,
        ).execute()

        if not response.get("items"):
            return {"title": f"Video {video_id}", "channel": "Unknown", "description": ""}

        snippet = response["items"][0]["snippet"]
        return {
            "title": snippet.get("title", ""),
            "channel": snippet.get("channelTitle", ""),
            "description": snippet.get("description", ""),
        }
    except Exception:
        return {"title": f"Video {video_id}", "channel": "Unknown", "description": ""}
