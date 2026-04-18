import re
import os
import tempfile
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
from googleapiclient.discovery import build
from openai import OpenAI

_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

WHISPER_DEPLOYMENT = "whisper-1"

# Whisper API limit is 25 MB
WHISPER_MAX_BYTES = 25 * 1024 * 1024
# Split audio into 5-minute segments when file exceeds limit
CHUNK_DURATION_MS = 5 * 60 * 1000


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


def _transcribe_single(audio_path: str) -> dict:
    """Send a single audio file to Whisper and return the transcript dict."""
    try:
        with open(audio_path, "rb") as f:
            response = _openai.audio.transcriptions.create(
                model=WHISPER_DEPLOYMENT,
                file=(os.path.basename(audio_path), f),
                response_format="text",
            )
        text = response if isinstance(response, str) else response.text
        print(f"[whisper] transcription complete — {len(text)} chars")
        return {"success": True, "text": text, "method": "whisper"}
    except Exception as e:
        return {"success": False, "text": "", "method": "whisper", "error": f"Whisper API error: {e}"}


def _transcribe_chunked(audio_path: str) -> dict:
    """
    Split audio into 5-minute chunks, transcribe all in parallel, then join.
    Uses pydub for splitting; falls back to mp3 export so Whisper always gets
    a recognised format regardless of the original container.
    """
    try:
        from pydub import AudioSegment
    except ImportError:
        return {
            "success": False, "text": "", "method": "whisper",
            "error": "pydub is required for large audio files. Install it: pip install pydub",
        }

    try:
        audio = AudioSegment.from_file(audio_path)
    except Exception as e:
        return {"success": False, "text": "", "method": "whisper", "error": f"Audio load error: {e}"}

    duration_ms = len(audio)
    n_chunks = math.ceil(duration_ms / CHUNK_DURATION_MS)
    print(f"[whisper] splitting {duration_ms / 60000:.1f} min audio into {n_chunks} chunks of 5 min")

    with tempfile.TemporaryDirectory() as chunk_dir:
        # Export each 5-min slice as mp3 (universally accepted by Whisper)
        chunk_paths: list[tuple[int, str]] = []
        for i in range(n_chunks):
            start = i * CHUNK_DURATION_MS
            end = min(start + CHUNK_DURATION_MS, duration_ms)
            chunk = audio[start:end]
            chunk_path = os.path.join(chunk_dir, f"chunk_{i:03d}.mp3")
            chunk.export(chunk_path, format="mp3")
            chunk_paths.append((i, chunk_path))
            print(f"[whisper] chunk {i+1}/{n_chunks}: {(end - start) / 1000:.0f}s → {os.path.getsize(chunk_path) / 1024 / 1024:.1f} MB")

        # Transcribe all chunks in parallel
        results: dict[int, str] = {}
        errors: list[str] = []

        def _transcribe_chunk(idx: int, path: str) -> tuple[int, str]:
            with open(path, "rb") as f:
                response = _openai.audio.transcriptions.create(
                    model=WHISPER_DEPLOYMENT,
                    file=(os.path.basename(path), f),
                    response_format="text",
                )
            text = response if isinstance(response, str) else response.text
            print(f"[whisper] chunk {idx+1}/{n_chunks} done — {len(text)} chars")
            return idx, text

        with ThreadPoolExecutor(max_workers=min(n_chunks, 8)) as pool:
            futures = {pool.submit(_transcribe_chunk, idx, path): idx for idx, path in chunk_paths}
            for future in as_completed(futures):
                try:
                    idx, text = future.result()
                    results[idx] = text
                except Exception as e:
                    errors.append(str(e))
                    print(f"[whisper] chunk error: {e}")

        if not results:
            return {"success": False, "text": "", "method": "whisper",
                    "error": f"All chunks failed: {'; '.join(errors)}"}

        # Join in original order
        full_text = " ".join(results[i] for i in sorted(results))
        if errors:
            print(f"[whisper] {len(errors)} chunk(s) failed but continuing with partial transcript")

        print(f"[whisper] combined transcript — {len(full_text)} chars from {len(results)}/{n_chunks} chunks")
        return {"success": True, "text": full_text, "method": "whisper"}


def _preprocess_audio(input_path: str, output_dir: str) -> str:
    """
    Use ffmpeg to reduce noise and amplify audio before sending to Whisper.
    Pipeline:
      - afftdn=nf=-25   : adaptive FFT denoiser (noise floor -25 dB)
      - dynaudnorm       : dynamic loudness normalisation / amplification
    Output is mono 16 kHz WAV — Whisper's native format, so it needs no
    further conversion.
    Falls back to the original file if ffmpeg is unavailable or fails.
    """
    import subprocess

    output_path = os.path.join(output_dir, "cleaned.wav")
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-af", "afftdn=nf=-25,dynaudnorm=p=0.95",
        "-ac", "1",      # mono
        "-ar", "16000",  # 16 kHz — Whisper's native sample rate
        output_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=300)
        if result.returncode == 0 and os.path.exists(output_path):
            cleaned_mb = os.path.getsize(output_path) / 1024 / 1024
            print(f"[whisper] audio cleaned & amplified → {cleaned_mb:.1f} MB (mono 16 kHz WAV)")
            return output_path
        print(f"[whisper] ffmpeg preprocessing failed — using original audio\n"
              f"          stderr: {result.stderr.decode(errors='replace')[:300]}")
    except FileNotFoundError:
        print("[whisper] ffmpeg not found — using original audio without preprocessing")
    except Exception as e:
        print(f"[whisper] ffmpeg error: {e} — using original audio")
    return input_path


def get_transcript_whisper(video_id: str) -> dict:
    """
    Download YouTube audio with pytubefix and transcribe using OpenAI Whisper.
    Used as a fallback when youtube-transcript-api finds no captions.
    Audio is preprocessed with ffmpeg (noise reduction + amplification) before
    being sent to Whisper. Whisper API limit: 25 MB.
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

        raw_mb = os.path.getsize(audio_path) / 1024 / 1024
        print(f"[whisper] downloaded: {os.path.basename(audio_path)} — {raw_mb:.1f} MB")

        # Clean and amplify audio with ffmpeg before transcription
        audio_path = _preprocess_audio(audio_path, tmpdir)

        file_size = os.path.getsize(audio_path)
        print(f"[whisper] sending to Whisper: {os.path.basename(audio_path)} — {file_size / 1024 / 1024:.1f} MB")

        if file_size > WHISPER_MAX_BYTES:
            print(f"[whisper] file too large — splitting into 5-min chunks and transcribing in parallel")
            return _transcribe_chunked(audio_path)

        return _transcribe_single(audio_path)


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


def transcribe_local_audio(audio_path: str) -> dict:
    """
    Preprocess and transcribe a local audio file using Whisper.
    The original file is not modified; preprocessing runs in a temp directory.
    Handles files larger than 25 MB by splitting into 5-minute chunks.
    Supports any format accepted by ffmpeg: mp3, wav, m4a, ogg, flac, webm, etc.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        processed_path = _preprocess_audio(audio_path, tmpdir)
        file_size = os.path.getsize(processed_path)
        print(f"[whisper] local audio ready: {os.path.basename(processed_path)} — {file_size / 1024 / 1024:.1f} MB")
        if file_size > WHISPER_MAX_BYTES:
            print("[whisper] file too large — splitting into 5-min chunks and transcribing in parallel")
            return _transcribe_chunked(processed_path)
        return _transcribe_single(processed_path)


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
