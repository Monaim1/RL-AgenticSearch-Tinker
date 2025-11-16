import json
import os
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from dotenv import load_dotenv
import dspy

import dspy
lm = dspy.LM('gemini/gemini-2.5-pro-preview-03-25', api_key='GEMINI_API_KEY')
dspy.configure(lm=lm)

def get_LLM_client():

    load_dotenv()
    base_url = os.getenv("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise Exception("Missing GEMINI_API_KEY (or GOOGLE_API_KEY) in environment")
    return {"base_url": base_url, "api_key": api_key}

DEFAULT_SYSTEM_PROMPT = (
    "you are gonna be given an abtract of a patent and you need to generate 3 queries that"
    "can be used to search for prior art patents related to the given patent abstract. "
    "The queries should be concise and relevant to the key aspects of the patent abstract."
    "only return the queries as a json array of strings with no other text."
)



def get_LLM_response(
    prompt: str,
    model: str = "gemini-2.0-flash",
    max_tokens: int = 30000,
    temperature: float = 0.99,
) -> str:
    try:
        cfg = get_LLM_client()

        if not str(model).startswith("gemini"):
            model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

        url = f"{cfg['base_url']}/models/{model}:generateContent?{urlencode({'key': cfg['api_key']})}"

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "temperature": float(temperature),
                "maxOutputTokens": int(max_tokens),
            },
            "systemInstruction": {
                "parts": [{"text": DEFAULT_SYSTEM_PROMPT}],
            },
        }

        req = urlrequest.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlrequest.urlopen(req) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except HTTPError as he:
            try:
                err_body = he.read().decode("utf-8")
            except Exception:
                err_body = str(he)
            raise Exception(f"Gemini HTTPError {he.code}: {err_body}")
        except URLError as ue:
            raise Exception(f"Gemini URLError: {ue.reason}")

        # Extract text from the first candidate
        try:
            candidates = data.get("candidates", [])
            if not candidates:
                raise Exception(f"No candidates in response: {data}")
            parts = candidates[0].get("content", {}).get("parts", [])
            text_segments = [p.get("text", "") for p in parts if isinstance(p, dict)]
            return "".join(text_segments).strip()
        except Exception as parse_err:
            raise Exception(f"Failed to parse Gemini response: {parse_err}")
    except Exception as e:
        raise Exception(f"Error getting AI response: {str(e)}")
