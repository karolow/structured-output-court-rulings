import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from prompts import get_system_prompt
from schema import OrzeczenieSN

load_dotenv()


def process_ruling(ruling: str, case_signature: str, output_dir: Path) -> None:
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    # Replace invalid characters in filename
    safe_filename = case_signature.replace("/", "_")
    output_path = output_dir / f"{safe_filename}.json"

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Calculate text length to use for dynamic summary length
    text_length = len(ruling)
    system_prompt = get_system_prompt(text_length)

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=ruling,
        config={
            "response_mime_type": "application/json",
            "response_schema": OrzeczenieSN,
            "system_instruction": system_prompt,
        },
    )

    # Parse the JSON response
    try:
        if not response.text:
            raise ValueError("Empty response from API")
        response_data = json.loads(response.text)
        # Add case signature to the response
        response_data = {"case_signature": case_signature, **response_data}
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Failed to parse JSON response: {e}")
        return

    with open(output_path, "w") as f:
        json.dump(response_data, f, ensure_ascii=False, indent=2)


def main() -> None:
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    with open("input_data_sample.json", "r") as f:
        data = json.load(f)

    # Process all items with delay
    for item in data:
        ruling = item["content"]["raw_markdown"]
        case_signature = item["case_signature"]
        process_ruling(ruling, case_signature, output_dir)
        time.sleep(0.1)


if __name__ == "__main__":
    main()
