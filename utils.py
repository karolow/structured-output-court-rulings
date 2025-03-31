import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, TypeVar

from google.cloud import storage

T = TypeVar("T")  # Type variable for generic items


def calculate_summary_length(text_length: int) -> int:
    """
    Calculates the optimal summary length based on text length.

    Parameters:
    - text_length: length of the text in characters (with spaces)

    Returns:
    - summary_length: recommended summary length in words
    """
    # Character to word ratio for Polish legal texts
    chars_per_word = 7.5

    # Minimum summary length even for short texts (500 chars ≈ 67 words)
    min_summary_words = int(500 / chars_per_word)  # ≈ 67 words

    # Logarithmic scale - summary length grows slower than text length
    if text_length <= 1500:
        summary_words = min_summary_words
    else:
        # Logarithmic function: summary grows slower for longer texts
        summary_chars = min(2000, int(500 + 300 * math.log(text_length / 1500, 2)))
        summary_words = int(summary_chars / chars_per_word)

    return summary_words


def process_result_files(
    result_files: Sequence[Path],
    output_dir: Path,
    job_name: str = "",
    timestamp: str = "",
) -> None:
    """
    Process downloaded result files and save them in a usable format.

    Args:
        result_files: List of downloaded result files
        output_dir: Directory to save processed results
        job_name: Optional batch job name for better file naming
        timestamp: Optional timestamp for better file naming
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Processing {len(result_files)} result files")

    # Make sure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Combined results with case signatures
    all_results: List[Dict[str, Any]] = []

    # Extract a run ID from the first file path (typically containing the timestamp)
    run_id = "run"
    if result_files:
        # Try to extract a timestamp or batch job ID from the path
        file_path_str = str(result_files[0])
        # Look for patterns like batch_extract/output/20230325_123456 or batch job ID
        import re

        timestamp_match = re.search(r"(\d{8}_\d{6})", file_path_str)
        job_id_match = re.search(r"(?:jobs?[/-])(\d+)", file_path_str)

        if timestamp_match:
            run_id = f"{timestamp_match.group(1)}"
        elif job_id_match:
            run_id = f"{job_id_match.group(1)}"

    # Override run_id if timestamp is provided
    if timestamp:
        run_id = timestamp

    # Process each file
    for result_file in result_files:
        logger.info(f"Processing result file: {result_file}")
        with open(result_file, "r") as f:
            try:
                # Results are typically in JSONL format
                line_count = 0
                result_count = 0
                for line in f:
                    line_count += 1
                    if not line.strip():
                        continue

                    try:
                        result_data = json.loads(line)

                        # Debug info about the structure
                        if line_count == 1:
                            logger.info(
                                f"Result file structure keys: {list(result_data.keys())}"
                            )

                        # Extract the prediction content - handle different formats
                        prediction: Any = None
                        case_signature = "unknown"

                        # Format 1: Vertex AI standard format
                        if "instance" in result_data and "predictions" in result_data:
                            # The instance contains our original request
                            instance = result_data.get("instance", {})
                            # Extract case_signature from the instance metadata
                            if "case_signature" in instance:
                                case_signature = instance["case_signature"]
                            elif isinstance(instance, dict) and "request" in instance:
                                # It might be nested in the request field
                                case_signature = instance.get(
                                    "case_signature", "unknown"
                                )

                            # Get the prediction content
                            prediction_data = result_data.get("predictions", {})

                            # Extract JSON from Gemini's response if needed
                            if (
                                isinstance(prediction_data, dict)
                                and "candidates" in prediction_data
                            ):
                                candidates = prediction_data.get("candidates", [])
                                if candidates and len(candidates) > 0:
                                    candidate = candidates[0]
                                    if (
                                        "content" in candidate
                                        and "parts" in candidate["content"]
                                    ):
                                        parts = candidate["content"]["parts"]
                                        for part in parts:
                                            if "text" in part:
                                                try:
                                                    prediction = json.loads(
                                                        part["text"]
                                                    )
                                                except json.JSONDecodeError:
                                                    prediction = {
                                                        "text_response": part["text"]
                                                    }
                            else:
                                prediction = prediction_data

                        # Format 2: Vertex AI incremental format
                        elif "prediction" in result_data:
                            prediction_data = result_data.get("prediction", {})

                            # Extract JSON from Gemini's response if needed
                            if (
                                isinstance(prediction_data, dict)
                                and "candidates" in prediction_data
                            ):
                                candidates = prediction_data.get("candidates", [])
                                if candidates and len(candidates) > 0:
                                    candidate = candidates[0]
                                    if (
                                        "content" in candidate
                                        and "parts" in candidate["content"]
                                    ):
                                        parts = candidate["content"]["parts"]
                                        for part in parts:
                                            if "text" in part:
                                                try:
                                                    prediction = json.loads(
                                                        part["text"]
                                                    )
                                                except json.JSONDecodeError:
                                                    prediction = {
                                                        "text_response": part["text"]
                                                    }
                            else:
                                prediction = prediction_data

                            # Try to extract case_signature from input
                            if "input" in result_data:
                                input_data = result_data.get("input", {})
                                if (
                                    isinstance(input_data, dict)
                                    and "case_signature" in input_data
                                ):
                                    case_signature = input_data["case_signature"]
                                elif (
                                    isinstance(input_data, dict)
                                    and "request" in input_data
                                ):
                                    # It might be nested
                                    case_signature = input_data.get("request", {}).get(
                                        "case_signature", "unknown"
                                    )

                        # Format 3: Raw prediction output
                        elif isinstance(result_data, dict) and "request" in result_data:
                            # Direct format with our payload structure
                            case_signature = result_data.get(
                                "case_signature", "unknown"
                            )
                            if "response" in result_data:
                                prediction_data = result_data["response"]

                                # Extract JSON from Gemini's response if needed
                                if (
                                    isinstance(prediction_data, dict)
                                    and "candidates" in prediction_data
                                ):
                                    candidates = prediction_data.get("candidates", [])
                                    if candidates and len(candidates) > 0:
                                        candidate = candidates[0]
                                        if (
                                            "content" in candidate
                                            and "parts" in candidate["content"]
                                        ):
                                            parts = candidate["content"]["parts"]
                                            for part in parts:
                                                if "text" in part:
                                                    try:
                                                        prediction = json.loads(
                                                            part["text"]
                                                        )
                                                    except json.JSONDecodeError:
                                                        prediction = {
                                                            "text_response": part[
                                                                "text"
                                                            ]
                                                        }
                                else:
                                    prediction = prediction_data
                            else:
                                prediction = result_data

                        # Format 4: Gemini response with candidates
                        elif "candidates" in result_data:
                            # Direct Gemini API response
                            candidates = result_data.get("candidates", [])
                            if candidates and len(candidates) > 0:
                                candidate = candidates[0]
                                # Check if content exists and has parts
                                if (
                                    "content" in candidate
                                    and "parts" in candidate["content"]
                                ):
                                    parts = candidate["content"]["parts"]
                                    for part in parts:
                                        if "text" in part:
                                            # Try to parse the text as JSON
                                            try:
                                                prediction = json.loads(part["text"])
                                            except json.JSONDecodeError:
                                                # If it's not valid JSON, store raw text
                                                prediction = {
                                                    "text_response": part["text"]
                                                }

                            # Try to find case_signature in result_data
                            if "case_signature" in result_data:
                                case_signature = result_data["case_signature"]

                            # Also check if there's a case signature in the request
                            if (
                                prediction
                                and isinstance(prediction, dict)
                                and "case_signature" in prediction
                            ):
                                case_signature = prediction["case_signature"]

                        # If we couldn't find a structured prediction, use the whole object
                        if prediction is None:
                            prediction = result_data

                        # Only process if we have valid content
                        if prediction:
                            # If prediction is a string (might be JSON), try to parse it
                            if isinstance(prediction, str):
                                try:
                                    prediction = json.loads(prediction)
                                except json.JSONDecodeError:
                                    # Keep as is if not valid JSON
                                    pass

                            # Create processed result object with standard format
                            from datetime import datetime

                            # Format result in the requested structure
                            processed_result = {
                                "case_signature": case_signature,
                                "processed_time": datetime.now().isoformat(),
                            }

                            # Copy all fields from prediction to the processed result
                            if isinstance(prediction, dict):
                                for key, value in prediction.items():
                                    if (
                                        key != "case_signature"
                                        and key != "processed_time"
                                    ):
                                        processed_result[key] = value

                            all_results.append(processed_result)
                            result_count += 1
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON on line {line_count}: {e}")

                logger.info(
                    f"Processed {result_count} results from {line_count} lines in {result_file}"
                )
            except Exception as e:
                logger.error(
                    f"Error processing result file {result_file}: {e}", exc_info=True
                )

    # Save all results to a single file
    if all_results:
        # Create a descriptive filename
        output_filename = "batch_prediction_results"

        # Add job name if available
        if job_name:
            # Extract just the last part of the job name if it's a full path
            short_job_name = job_name.split("/")[-1]
            output_filename += f"_{short_job_name}"

        # Add timestamp
        if run_id != "run":
            output_filename += f"_{run_id}"

        # Add case count
        output_filename += f"_{len(all_results)}_cases"

        # Add extension
        result_path = output_dir / f"{output_filename}.json"
        with open(result_path, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(all_results)} processed results to {result_path}")

        # Also save individual results based on case signature
        for result in all_results:
            case_sig = result["case_signature"]
            # Sanitize filename by replacing invalid characters
            safe_filename = "".join(
                c if c.isalnum() or c in "._- " else "_" for c in case_sig
            )
            # Save with case signature as filename
            case_file = output_dir / f"{safe_filename}.json"
            with open(case_file, "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
    else:
        logger.warning("No valid results found in the downloaded files")

        # Debug: Save sample content from the first file
        if result_files:
            try:
                sample_path = output_dir / "debug_sample.txt"
                with open(result_files[0], "r") as f_in:
                    sample_content = f_in.read(10000)  # First 10k chars

                with open(sample_path, "w") as f_out:
                    f_out.write(f"Sample content from {result_files[0]}:\n\n")
                    f_out.write(sample_content)

                logger.info(f"Saved debug sample to {sample_path} for troubleshooting")
            except Exception as e:
                logger.error(f"Could not save debug sample: {e}")


def download_blobs_from_gcs(output_uri: str, output_dir: Path) -> List[Path]:
    """
    Download result blobs from GCS URI.

    Args:
        output_uri: The GCS URI where results are stored
        output_dir: Local directory to save downloaded files

    Returns:
        List of local paths to downloaded files
    """
    logger = logging.getLogger(__name__)
    downloaded_files: List[Path] = []

    if not output_uri.startswith("gs://"):
        logger.error(f"Invalid output URI format: {output_uri}")
        return downloaded_files

    # Extract bucket name and path from gs:// URL
    parts = output_uri[5:].split("/", 1)
    if len(parts) != 2:
        logger.error(f"Invalid GCS path format: {output_uri}")
        return downloaded_files

    bucket_name, output_path = parts

    # Create storage client and get bucket
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # List all files (recursively)
    logger.info(f"Listing files in {output_uri}")
    blobs = list(bucket.list_blobs(prefix=output_path, delimiter="/"))
    logger.info(f"Found {len(blobs)} files/folders in {output_uri}")

    # Make sure output dir exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if we have batch_XXXX directories
    batch_dirs = []
    for blob in blobs:
        if blob.name.endswith("/") and "batch_" in blob.name:
            batch_dirs.append(blob.name)

    if batch_dirs:
        logger.info(f"Found {len(batch_dirs)} batch directories")
        # Process each batch directory
        for batch_dir in batch_dirs:
            # List all prediction files recursively in this batch directory
            batch_blobs = list(bucket.list_blobs(prefix=batch_dir))
            logger.info(f"Checking {len(batch_blobs)} files in {batch_dir}")

            for blob in batch_blobs:
                if blob.name.endswith("/"):  # Skip directories
                    continue

                # Get the filename
                blob_name = os.path.basename(blob.name)

                # Check if this is a prediction file
                if (
                    "predictions.jsonl" in blob.name
                    or "predictions-chunked" in blob.name
                ):
                    # Create a more descriptive local filename that includes batch info
                    batch_id = os.path.basename(batch_dir.rstrip("/"))
                    local_file = output_dir / f"{batch_id}_{blob_name}"

                    logger.info(f"Downloading {blob.name} to {local_file}")
                    blob.download_to_filename(str(local_file))
                    downloaded_files.append(local_file)
    else:
        # If no batch directories, try the direct approach
        for blob in blobs:
            if not blob.name.endswith("/"):  # Skip directories
                blob_name = os.path.basename(blob.name)

                # Check if it's a result file we want
                if is_result_file(blob_name):
                    local_file = output_dir / blob_name
                    logger.info(f"Downloading {blob_name} to {local_file}")
                    blob.download_to_filename(str(local_file))
                    downloaded_files.append(local_file)

    # If still no files, try full recursive listing
    if not downloaded_files:
        logger.info("No files found in direct folders. Trying recursive search...")
        all_blobs = list(bucket.list_blobs(prefix=output_path))

        for blob in all_blobs:
            if blob.name.endswith("/"):  # Skip directories
                continue

            if "predictions.jsonl" in blob.name or "predictions-chunked" in blob.name:
                # Create a simpler filename
                blob_name = os.path.basename(blob.name)
                path_parts = blob.name.split("/")

                # Get parent folder name for clarity
                parent_folder = ""
                if len(path_parts) > 1:
                    parent_folder = path_parts[-2] + "_"

                local_file = output_dir / f"{parent_folder}{blob_name}"

                logger.info(f"Downloading {blob.name} to {local_file}")
                blob.download_to_filename(str(local_file))
                downloaded_files.append(local_file)

    if not downloaded_files:
        logger.warning(f"No result files found in {output_uri}")

    return downloaded_files


def is_result_file(filename: str) -> bool:
    """
    Check if the filename is likely a batch prediction result file.

    Args:
        filename: Name of the file

    Returns:
        True if the file is likely a result file
    """
    # Skip files named or starting with "predictions"
    if filename == "predictions" or filename.startswith("predictions"):
        return False

    # Common patterns in batch prediction result files
    patterns = [
        "results",
        ".jsonl",
        ".ndjson",
        "batch_predict",
        "predictions.jsonl",
        "predictions-chunked",
    ]

    return any(pattern in filename.lower() for pattern in patterns)


def find_output_uri(job: Any, bucket_name: Optional[str] = None) -> Optional[str]:
    """
    Find the output URI from a job object using various methods.

    Args:
        job: The batch job object
        bucket_name: Optional bucket name to use as fallback

    Returns:
        Output URI if found, None otherwise
    """
    logger = logging.getLogger(__name__)

    # Try to get job as dictionary
    job_dict = {}
    if hasattr(job, "model_dump"):
        job_dict = job.model_dump()
    elif hasattr(job, "__dict__"):
        job_dict = job.__dict__

    # Check common locations for output URI in job dictionary
    if isinstance(job_dict, dict):
        # Check destination URIs
        if "destinationUris" in job_dict and isinstance(
            job_dict["destinationUris"], dict
        ):
            if "output" in job_dict["destinationUris"]:
                logger.info(
                    f"Found output URI in destinationUris: {job_dict['destinationUris']['output']}"
                )
                return job_dict["destinationUris"]["output"]

        # Check config.dest
        if "config" in job_dict and isinstance(job_dict["config"], dict):
            if "dest" in job_dict["config"]:
                logger.info(
                    f"Found output URI in config.dest: {job_dict['config']['dest']}"
                )
                return job_dict["config"]["dest"]

    # Try regex approach for extracting GCS URIs
    job_str = str(job)
    gs_uri_pattern = re.compile(r"gs://[a-zA-Z0-9-_.]+/[a-zA-Z0-9-_./]+")
    gs_uri_matches = gs_uri_pattern.findall(job_str)

    # Find output URIs
    if gs_uri_matches:
        # Prefer URIs with 'output' in them
        output_candidates = [uri for uri in gs_uri_matches if "output" in uri.lower()]
        if output_candidates:
            logger.info(f"Found output URI using regex: {output_candidates[0]}")
            return output_candidates[0]
        # Otherwise, take first URI
        logger.info(f"Using first found URI: {gs_uri_matches[0]}")
        return gs_uri_matches[0]

    # Fallback: construct from bucket name and job ID
    if bucket_name and hasattr(job, "name"):
        job_id = job.name.split("/")[-1]
        constructed_uri = f"gs://{bucket_name}/batch_extract/output/{job_id}"
        logger.info(f"Constructed fallback URI: {constructed_uri}")
        return constructed_uri

    logger.warning("Could not find output URI")
    return None


def upload_to_gcs(
    local_file_paths: List[str], bucket_name: str, gcs_folder: str
) -> List[str]:
    """
    Upload local JSONL files to Google Cloud Storage.

    Args:
        local_file_paths: List of local file paths to upload
        bucket_name: GCS bucket name
        gcs_folder: Folder in the bucket to upload files to

    Returns:
        List of GCS URIs for uploaded files
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Uploading {len(local_file_paths)} files to GCS bucket {bucket_name}")

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    gcs_uris: List[str] = []
    for local_path in local_file_paths:
        file_name = os.path.basename(local_path)
        blob_name = f"{gcs_folder}/{file_name}"
        blob = bucket.blob(blob_name)

        logger.info(f"Uploading {local_path} to gs://{bucket_name}/{blob_name}")
        blob.upload_from_filename(local_path)

        gcs_uris.append(f"gs://{bucket_name}/{blob_name}")

    logger.info(f"Uploaded {len(gcs_uris)} files to GCS")
    return gcs_uris


def split_into_batches(items: Sequence[T], batch_size: int) -> List[List[T]]:
    """
    Split a list of items into batches of specified size.

    Args:
        items: List of items to split
        batch_size: Maximum number of items per batch

    Returns:
        List of batches, where each batch is a list of items
    """
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def convert_schema_for_gemini(
    schema: Dict[str, Any],
    root_schema: Optional[Dict[str, Any]] = None,
    processed_refs: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """
    Convert a JSON schema to be compatible with Vertex AI/Gemini.
    Creates a properly structured schema with all fields defined appropriately.
    Preserves all property constraints from the original schema.

    Args:
        schema: The original JSON schema
        root_schema: The original root schema (used for resolving refs)
        processed_refs: Set of already processed references to avoid circular references

    Returns:
        A modified schema compatible with Gemini and BigQuery
    """
    # Initialize tracking set if not provided
    if processed_refs is None:
        processed_refs = set()

    # Use schema as root_schema if not provided
    if root_schema is None:
        root_schema = schema

    # Handle simple case (non-dict values)
    if not isinstance(schema, dict):
        return schema

    # Special handling for anyOf and oneOf - they need to be the only property in the object
    # according to Vertex AI/Gemini requirements
    if any(key in schema for key in ("anyOf", "oneOf")):
        for key in ("anyOf", "oneOf"):
            if key in schema:
                # Extract the anyOf/oneOf value
                variants = schema[key]
                if not isinstance(variants, list):
                    continue

                # Get description to propagate to variants
                description = schema.get("description", "")

                # Filter out any entries with type: "null" and merge properties
                filtered_values = []
                for item in variants:
                    if isinstance(item, dict) and item.get("type") != "null":
                        # Clone the item to avoid modifying the original
                        processed_item = convert_schema_for_gemini(
                            item, root_schema, processed_refs
                        )

                        # Add description if missing
                        if description and "description" not in processed_item:
                            processed_item["description"] = description

                        # Remove title field
                        if "title" in processed_item:
                            del processed_item["title"]

                        filtered_values.append(processed_item)

                # Only add if we have items left
                if filtered_values:
                    # Create clean result with ONLY the anyOf/oneOf field
                    return {key: filtered_values}
                else:
                    # If no valid options remain, return a simple string type
                    return {
                        "type": "string",
                        "description": description
                        or "Default type when no valid schema options remain",
                    }

    # Create new schema with same structure
    result = {}

    # Process each property
    for key, value in schema.items():
        # Skip null defaults - Vertex AI doesn't allow these
        if key == "default" and value is None:
            continue

        # Skip title fields
        elif key == "title":
            continue

        # Handle $ref specifically
        elif key == "$ref" and isinstance(value, str):
            # Already processed this reference?
            if value in processed_refs:
                continue

            # Mark as processed to avoid recursive loops
            processed_refs.add(value)

            # Only handle #/$defs/ references for now
            if value.startswith("#/$defs/"):
                # Get the definition name
                def_name = value.split("/")[-1]

                # Try to find it in the original schema
                if def_name in root_schema.get("$defs", {}):
                    # Get the definition
                    definition = root_schema["$defs"][def_name].copy()

                    # Process it recursively
                    processed_def = convert_schema_for_gemini(
                        definition, root_schema, processed_refs
                    )

                    # Merge any other properties from the schema with the definition
                    for other_key, other_value in schema.items():
                        if other_key != "$ref" and other_key != "title":
                            processed_def[other_key] = other_value

                    # Return the processed definition
                    return processed_def
        # Skip anyOf and oneOf here since we handle them at the top of the function
        elif key in ("anyOf", "oneOf"):
            continue
        # Handle type arrays like ["string", "null"] - remove "null"
        elif key == "type" and isinstance(value, list):
            # Filter out "null" from type array
            filtered_types = [t for t in value if t != "null"]
            if filtered_types:
                if len(filtered_types) == 1:
                    # If only one type remains, use it directly
                    result[key] = filtered_types[0]
                else:
                    # Keep as array if multiple types remain
                    result[key] = filtered_types
        # Skip other $-prefixed properties
        elif key.startswith("$"):
            continue
        # Process nested objects
        elif isinstance(value, dict):
            result[key] = convert_schema_for_gemini(value, root_schema, processed_refs)
        # Process arrays
        elif isinstance(value, list):
            result[key] = [
                convert_schema_for_gemini(item, root_schema, processed_refs)
                if isinstance(item, dict)
                else item
                for item in value
            ]
        # Keep other values as is
        else:
            result[key] = value

    # Add description if it's missing
    if "type" in result and result.get("properties") and "description" not in result:
        result["description"] = f"A {result['type']} value"

    # Add descriptions to properties
    if "properties" in result:
        for prop_name, prop_value in result["properties"].items():
            if isinstance(prop_value, dict) and "description" not in prop_value:
                prop_value["description"] = f"The {prop_name} value"

            # Add default values to required properties
            if "required" in result and prop_name in result.get("required", []):
                if isinstance(prop_value, dict) and "default" not in prop_value:
                    if prop_value.get("type") == "string":
                        prop_value["default"] = ""
                    elif prop_value.get("type") == "array":
                        prop_value["default"] = []
                    # Don't add empty objects as defaults - Claude can't handle these
                    # Instead, only add populated objects with required fields
                    elif prop_value.get("type") == "object" and prop_value.get(
                        "properties"
                    ):
                        # Skip adding default value for object types
                        # This avoids the "Cannot store struct with no fields" error
                        pass
                    elif prop_value.get("type") == "integer":
                        prop_value["default"] = 0
                    elif prop_value.get("type") == "number":
                        prop_value["default"] = 0.0
                    elif prop_value.get("type") == "boolean":
                        prop_value["default"] = False

    return result


def download_results(
    output_dir: Path,
    output_uris: List[str],
    bucket_name: Optional[str] = None,
    job_name: str = "",
    timestamp: str = "",
) -> None:
    """
    Download and process batch prediction results from GCS.

    Args:
        output_dir: Directory to save downloaded results
        output_uris: List of GCS URIs for job outputs
        bucket_name: Optional GCS bucket name, used as fallback if needed
        job_name: Optional job name for better file naming
        timestamp: Optional timestamp for better file naming
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Downloading batch results to {output_dir}")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download files from each output URI
    downloaded_files = []
    for output_uri in output_uris:
        logger.info(f"Processing output from: {output_uri}")

        # Download blobs from this output URI
        job_files = download_blobs_from_gcs(output_uri, output_dir)
        downloaded_files.extend(job_files)

    # Process downloaded files
    if downloaded_files:
        process_result_files(downloaded_files, output_dir, job_name, timestamp)
    else:
        logger.warning("No result files found to download")
