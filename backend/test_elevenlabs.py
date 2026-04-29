"""
Standalone verification script for ElevenLabs.

Run from the `backend/` folder:
    python test_elevenlabs.py

Prints:
  1. Whether the API key is valid (via /v1/voices/<id>)
  2. Subscription tier + remaining character quota
  3. Performs a real TTS synthesis and saves /tmp/elevenlabs_test.mp3
  4. Calls the same speak_with_tts code path the server uses

Use this to confirm ElevenLabs is reachable, the key is valid, the voice
exists, you have quota left, and the audio gets generated.
"""

import json
import os
import sys
import urllib.error
import urllib.request


def _load_env_file(path: str = ".env") -> None:
    """Tiny .env parser so this script has zero pip dependencies."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip().lstrip("﻿")  # strip BOM if present
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            # Honor existing real env vars; only fill in missing ones.
            os.environ.setdefault(k, v)


# Try python-dotenv if installed (handles edge cases better); fall back otherwise.
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(".env")
except ImportError:
    _load_env_file(".env")

KEY = (os.getenv("ELEVENLABS_API_KEY") or "").strip()
VOICE_ID = (os.getenv("ELEVENLABS_VOICE_ID") or "").strip()
MODEL = (os.getenv("ELEVENLABS_MODEL") or "eleven_turbo_v2").strip()

if not KEY or not VOICE_ID:
    print("FAIL: ELEVENLABS_API_KEY or ELEVENLABS_VOICE_ID not set in .env")
    sys.exit(1)

print(f"Key:      {KEY[:8]}...{KEY[-4:]}  (len={len(KEY)})")
print(f"Voice ID: {VOICE_ID}")
print(f"Model:    {MODEL}")
print(f"Enabled:  ENABLE_ELEVENLABS_TTS={os.getenv('ENABLE_ELEVENLABS_TTS', '<unset>')}")
print(f"BASE_URL: {os.getenv('BASE_URL', '<unset>')}")
print()


def http(method, path, body=None, accept="application/json"):
    req = urllib.request.Request(
        f"https://api.elevenlabs.io{path}",
        data=body,
        headers={
            "xi-api-key": KEY,
            "Content-Type": "application/json",
            "Accept": accept,
        },
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


# --- 1. Voice exists / key valid -------------------------------------------
print("=== Test 1: Voice lookup ===")
code, data = http("GET", f"/v1/voices/{VOICE_ID}")
if code == 200:
    info = json.loads(data)
    print(f"  OK  HTTP 200")
    print(f"  Voice name: {info.get('name')}")
    print(f"  Category:   {info.get('category')}")
elif code == 401:
    print("  FAIL: invalid API key (HTTP 401). Regenerate at elevenlabs.io.")
    sys.exit(2)
elif code in (400, 404) and b"voice_not_found" in data:
    print(f"  FAIL: voice '{VOICE_ID}' is not in this account.")
    print("  Listing voices your key CAN use, so you can pick one:")
    code2, data2 = http("GET", "/v1/voices")
    if code2 == 200:
        voices = json.loads(data2).get("voices", [])
        if not voices:
            print("    (no voices found — visit elevenlabs.io/app/voice-library to add one)")
        for v in voices:
            print(f"    {v.get('voice_id')}  {v.get('name'):<25}  ({v.get('category')})")
        print("\n  Pick one and put it in backend/.env as ELEVENLABS_VOICE_ID=<id>, then re-run.")
    else:
        print(f"    (could not list voices: HTTP {code2})")
    sys.exit(2)
else:
    print(f"  FAIL: HTTP {code}: {data[:300]}")
    sys.exit(2)
print()

# --- 2. Subscription / quota -----------------------------------------------
print("=== Test 2: Subscription / quota ===")
code, data = http("GET", "/v1/user/subscription")
if code == 200:
    sub = json.loads(data)
    used = sub.get("character_count", 0)
    limit = sub.get("character_limit", 0)
    remaining = limit - used
    print(f"  Tier:      {sub.get('tier')}")
    print(f"  Status:    {sub.get('status')}")
    print(f"  Used:      {used:,} / {limit:,}")
    print(f"  Remaining: {remaining:,}")
    if remaining < 500:
        print("  WARN: less than 500 characters remaining — TTS may fail.")
else:
    print(f"  WARN: HTTP {code}: {data[:200]}")
print()

# --- 3. Real synthesis ------------------------------------------------------
print("=== Test 3: TTS synthesis ===")
body = json.dumps({
    "text": "Hello, this is a test from the credentialing assistant.",
    "model_id": MODEL,
    "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
}).encode()
code, audio = http("POST", f"/v1/text-to-speech/{VOICE_ID}", body=body, accept="audio/mpeg")
if code == 200 and audio[:3] == b"ID3":
    out = "/tmp/elevenlabs_test.mp3" if os.name != "nt" else os.path.join(os.getcwd(), "elevenlabs_test.mp3")
    with open(out, "wb") as f:
        f.write(audio)
    print(f"  OK  HTTP 200, {len(audio):,} bytes (valid MP3)")
    print(f"  Saved: {out}")
elif code == 401:
    print("  FAIL: 401 unauthorized — key probably has TTS permission revoked.")
    sys.exit(3)
elif code == 402 or code == 429:
    print(f"  FAIL: HTTP {code} — payment / tier / rate limit issue.")
    print(f"  Server said: {audio[:500]!r}")
    print()
    print("  Listing voices that ARE allowed on your tier:")
    code2, data2 = http("GET", "/v1/voices")
    if code2 == 200:
        voices = json.loads(data2).get("voices", [])
        # On Free tier, you can only use 'premade' voices.
        for v in voices:
            cat = v.get("category", "")
            ok = "OK " if cat == "premade" else "(needs paid tier)"
            print(f"    {ok}  {v.get('voice_id')}  {v.get('name'):<25}  category={cat}")
        print()
        print("  Pick a voice marked OK (category=premade) for the Free tier.")
        print("  Common premade IDs that work on Free:")
        print("    21m00Tcm4TlvDq8ikWAM  Rachel    (only if NOT cloned in your workspace)")
        print("    EXAVITQu4vr4xnSDxMaL  Bella")
        print("    ErXwobaYiN019PkySvjV  Antoni")
        print("    VR6AewLTigWG4xSOukaG  Arnold")
        print("    pNInz6obpgDQGcFmaJgB  Adam")
        print("    yoZ06aMxZJJ28mfd3POQ  Sam")
    else:
        print(f"    (could not list voices: HTTP {code2})")
    sys.exit(3)
else:
    print(f"  FAIL: HTTP {code}: {audio[:300]}")
    sys.exit(3)
print()

# --- 4. Run the actual server code path ------------------------------------
print("=== Test 4: server's speak_with_tts() path ===")
try:
    sys.path.insert(0, os.getcwd())
    from api_server import generate_elevenlabs_audio_url, ENABLE_ELEVENLABS_TTS
    print(f"  ENABLE_ELEVENLABS_TTS = {ENABLE_ELEVENLABS_TTS}")
    if not ENABLE_ELEVENLABS_TTS:
        print("  FAIL: flag is False — TTS will fall back to Polly silently.")
        sys.exit(4)
    url = generate_elevenlabs_audio_url("Server-side path verification.")
    if url:
        print(f"  OK  audio URL: {url}")
        print(f"      (Twilio fetches this URL; BASE_URL must be public.)")
    else:
        print("  FAIL: function returned None — check server logs.")
        sys.exit(4)
except Exception as e:
    print(f"  WARN: could not import server module: {e}")
    print("  (Run this from the backend/ folder.)")
print()
print("All ElevenLabs checks passed. If you still hear Polly during a call,")
print("look in the server logs for 'Falling back to Polly TTS' or")
print("'BASE_URL is not public'.")
