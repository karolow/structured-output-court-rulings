#!/usr/bin/env python
import argparse
import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, List

import google.cloud.aiplatform as aiplatform
from dotenv import load_dotenv
from google.cloud import storage
from google.cloud.aiplatform.jobs import BatchPredictionJob

from prompts import get_system_prompt
from schema import OrzeczenieSN

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()


def split_into_batches(items: List[Any], batch_size: int) -> List[List[Any]]:
    """
    Split a list of items into batches of the specified size.

    Args:
        items: List of items to split
        batch_size: Maximum size of each batch

    Returns:
        List of batches, where each batch is a list of items
    """
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def prepare_input_data(
    input_file: str, output_jsonl_path: str, batch_size: int = 1000
) -> List[str]:
    """
    Process input cases file and prepare JSONL files for batch prediction.

    Args:
        input_file: Path to the input JSON file with cases
        output_jsonl_path: Path to store generated JSONL files
        batch_size: Number of cases per batch file

    Returns:
        List of created JSONL file paths
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_jsonl_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read input data
    logger.info(f"Reading input data from {input_file}")
    with open(input_file, "r") as f:
        data = json.load(f)

    total_cases = len(data)
    logger.info(f"Found {total_cases} cases to process")

    # Calculate number of batches
    num_batches = (total_cases + batch_size - 1) // batch_size
    batch_files: List[str] = []

    # Process cases in batches
    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, total_cases)

        batch_file = output_dir / f"batch_{batch_num:04d}.jsonl"
        batch_files.append(str(batch_file))

        logger.info(
            f"Processing batch {batch_num + 1}/{num_batches} with {end_idx - start_idx} cases"
        )

        with open(batch_file, "w") as f:
            for i in range(start_idx, end_idx):
                item = data[i]
                ruling = item["content"]["raw_markdown"]
                case_signature = item["case_signature"]

                # Calculate text length for dynamic prompt
                text_length = len(ruling)
                system_prompt = get_system_prompt(text_length)

                # Create request in the format expected by Vertex AI Batch Prediction
                request = {
                    "instances": [
                        {
                            "content": ruling,
                            "case_signature": case_signature,
                        }
                    ],
                    "parameters": {
                        "system": system_prompt,
                        "response_mime_type": "application/json",
                        "response_schema": OrzeczenieSN.model_json_schema(),
                        "temperature": 0.2,
                        "max_output_tokens": 8192,
                    },
                }

                f.write(json.dumps(request) + "\n")

    logger.info(f"Created {len(batch_files)} batch files at {output_jsonl_path}")
    return batch_files


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


def submit_batch_prediction_jobs(
    input_files: List[Path],
    bucket_name: str,
    batch_size: int,
    model_name: str,
    region: str,
) -> List[BatchPredictionJob]:
    """
    Upload batch data to GCS and submit batch prediction jobs.

    Args:
        input_files: List of JSONL files containing data for batch prediction
        bucket_name: GCS bucket to store input and output data
        batch_size: Maximum number of cases per batch job
        model_name: Vertex AI model name
        region: Vertex AI region

    Returns:
        List of submitted batch prediction jobs
    """
    # Create storage client
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # Upload input files to GCS
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    gcs_input_prefix = f"batch_extract/input/{timestamp}"
    gcs_output_prefix = f"batch_extract/output/{timestamp}"

    input_uris = []
    for input_file in input_files:
        # Create a unique blob name
        blob_name = f"{gcs_input_prefix}/{input_file.name}"
        blob = bucket.blob(blob_name)

        # Upload the file
        logger.info(f"Uploading {input_file} to gs://{bucket_name}/{blob_name}")
        blob.upload_from_filename(str(input_file))

        # Store the GCS URI
        input_uris.append(f"gs://{bucket_name}/{blob_name}")

    # Get the project ID from environment variable
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        raise ValueError("GOOGLE_CLOUD_PROJECT environment variable is not set")

    # Construct the full model path with publisher for Vertex AI
    full_model_path = f"projects/{project_id}/locations/{region}/publishers/google/models/{model_name}"

    # Initialize Vertex AI client
    aiplatform.init(project=project_id, location=region)

    # Submit batch prediction jobs
    jobs = []
    for input_uri in input_uris:
        logger.info(
            f"Submitting batch job for {input_uri} using model {full_model_path}"
        )
        job = BatchPredictionJob.submit(
            model_name=full_model_path,
            job_display_name=f"extract-batch-{timestamp}",
            gcs_source=input_uri,
            gcs_destination_prefix=f"gs://{bucket_name}/{gcs_output_prefix}",
            instances_format="jsonl",
            predictions_format="jsonl",
            batch_size=batch_size,
        )
        jobs.append(job)
        logger.info(f"Submitted job {job.resource_name}")

    return jobs


def download_and_process_results(
    jobs: List[BatchPredictionJob],
    output_dir: Path,
    bucket_name: str,
) -> None:
    """
    Monitor jobs, download results when complete, and process them.

    Args:
        jobs: List of batch prediction jobs
        output_dir: Local directory to save processed results
        bucket_name: GCS bucket name containing the results
    """
    logger.info(f"Monitoring {len(jobs)} batch prediction jobs")

    # Create storage client
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Monitor job completion
    for i, job in enumerate(jobs):
        logger.info(f"Monitoring job {i + 1}/{len(jobs)}: {job.resource_name}")

        # Wait for job completion - monitor state until completed
        job_id = job.resource_name
        completed = False
        last_state = None
        max_retries = 60  # Limit how long we'll wait (60 minutes)
        retries = 0

        while not completed and retries < max_retries:
            # Refresh job status - instantiate a new job object with the resource name
            job = aiplatform.BatchPredictionJob(job_id)

            # Only log status changes to reduce noise
            if job.state != last_state:
                logger.info(f"Job {i + 1}/{len(jobs)} status: {job.state}")
                last_state = job.state

            # Check if job is completed
            if job.state in [
                "JOB_STATE_SUCCEEDED",
                "JOB_STATE_FAILED",
                "JOB_STATE_CANCELLED",
            ]:
                completed = True
            else:
                time.sleep(60)  # Check status every minute
                retries += 1

        # Handle max retries reached
        if retries >= max_retries and not completed:
            logger.error(
                f"Exceeded maximum monitoring time for job {job_id}. Current state: {job.state}"
            )
            continue

        if job.state == "JOB_STATE_SUCCEEDED":
            logger.info(f"Job {i + 1}/{len(jobs)} completed successfully")

            # Get output location from the job
            try:
                output_info = job._gca_resource.output_info
                if not output_info:
                    logger.error(
                        f"No output info available for job {job.resource_name}"
                    )
                    continue

                # Get GCS output path
                gcs_output_info = output_info.gcs_output_directory
                if not gcs_output_info:
                    logger.error(
                        f"No GCS output info available for job {job.resource_name}"
                    )
                    continue

                output_location = gcs_output_info
                logger.info(f"Job output location: {output_location}")

                if (
                    output_location
                    and isinstance(output_location, str)
                    and output_location.startswith("gs://")
                ):
                    # Strip "gs://bucket_name/" prefix if it contains our bucket
                    bucket_prefix = f"gs://{bucket_name}/"
                    if output_location.startswith(bucket_prefix):
                        output_path = output_location[len(bucket_prefix) :]
                    else:
                        # Handle case when output is in different bucket
                        logger.warning(
                            f"Output is in different bucket: {output_location}"
                        )
                        logger.warning(
                            "Will attempt to process from the specified location"
                        )
                        # Extract bucket name and path from gs:// URL
                        parts = output_location[5:].split("/", 1)
                        if len(parts) == 2:
                            other_bucket_name, output_path = parts
                            try:
                                other_bucket = storage_client.bucket(other_bucket_name)
                                bucket = other_bucket  # Switch to the other bucket
                            except Exception as e:
                                logger.error(
                                    f"Failed to access bucket {other_bucket_name}: {e}"
                                )
                                continue
                        else:
                            logger.error(f"Invalid GCS path format: {output_location}")
                            continue

                    # List blobs in the output location
                    logger.info(f"Listing files in {output_location}")
                    try:
                        blobs = list(bucket.list_blobs(prefix=output_path))
                        logger.info(f"Found {len(blobs)} files in {output_location}")
                    except Exception as e:
                        logger.error(f"Error listing blobs: {e}")
                        continue

                    # Download and process each result file
                    results_found = False
                    for blob in blobs:
                        if blob.name.endswith(".jsonl"):
                            results_found = True
                            # Download result file
                            local_file = f"/tmp/{os.path.basename(blob.name)}"
                            logger.info(f"Downloading {blob.name} to {local_file}")
                            try:
                                blob.download_to_filename(local_file)
                            except Exception as e:
                                logger.error(f"Error downloading {blob.name}: {e}")
                                continue

                            # Process results
                            with open(local_file, "r") as f:
                                for line in f:
                                    try:
                                        result = json.loads(line)
                                    except json.JSONDecodeError as e:
                                        logger.error(f"Error parsing JSON: {e}")
                                        continue

                                    # Extract metadata and case_signature
                                    case_signature = result.get(
                                        "case_signature", "unknown"
                                    )
                                    if isinstance(result.get("instance"), dict):
                                        case_signature = result.get("instance", {}).get(
                                            "case_signature", case_signature
                                        )

                                    # Initialize prediction with case_signature
                                    prediction = {"case_signature": case_signature}

                                    # Record processing time
                                    prediction["processed_time"] = result.get(
                                        "processed_time",
                                        datetime.datetime.now().isoformat(),
                                    )

                                    # Extract the actual prediction content from various places it might be
                                    if (
                                        "response" in result
                                        and "candidates" in result["response"]
                                    ):
                                        # New format with structured response
                                        candidates = result["response"]["candidates"]
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
                                                        text = part["text"]
                                                        try:
                                                            # This is likely JSON text content from the model
                                                            content_data = json.loads(
                                                                text
                                                            )
                                                            # Update our prediction with this data
                                                            prediction.update(
                                                                content_data
                                                            )
                                                            logger.info(
                                                                f"Successfully extracted structured content for {case_signature}"
                                                            )
                                                        except json.JSONDecodeError:
                                                            # If it's not valid JSON, store raw text
                                                            prediction[
                                                                "text_response"
                                                            ] = text
                                                            logger.warning(
                                                                f"Failed to parse JSON content for {case_signature}"
                                                            )

                                    # Fallback to legacy format handling
                                    elif "predictions" in result:
                                        raw_prediction = result["predictions"]

                                        # The response might be in different formats depending on Gemini's output
                                        if isinstance(raw_prediction, dict):
                                            # Direct JSON response
                                            prediction.update(raw_prediction)
                                        elif isinstance(raw_prediction, str):
                                            # Text response - try to parse as JSON if possible
                                            try:
                                                content_data = json.loads(
                                                    raw_prediction
                                                )
                                                prediction.update(content_data)
                                            except json.JSONDecodeError:
                                                # If it's not valid JSON, store as is
                                                prediction["text_response"] = (
                                                    raw_prediction
                                                )
                                        else:
                                            # Handle other potential formats
                                            prediction["raw_response"] = str(
                                                raw_prediction
                                            )

                                    # Handle empty or incomplete responses
                                    if (
                                        len(prediction) <= 3
                                    ):  # Only has case_signature, processed_time, and maybe status
                                        prediction["status"] = "incomplete_response"
                                        logger.warning(
                                            f"Incomplete or empty model response for {case_signature}"
                                        )

                                    # Create safe filename
                                    safe_filename = case_signature.replace(
                                        "/", "_"
                                    ).replace("\\", "_")
                                    output_file_path = (
                                        output_dir / f"{safe_filename}.json"
                                    )

                                    # Save raw result for debugging if there were issues with the prediction
                                    if (
                                        prediction.get("status")
                                        == "incomplete_response"
                                    ):
                                        raw_output_path = (
                                            output_dir / f"{safe_filename}_raw.json"
                                        )
                                        with open(raw_output_path, "w") as raw_f:
                                            json.dump(
                                                result,
                                                raw_f,
                                                ensure_ascii=False,
                                                indent=2,
                                            )
                                            logger.info(
                                                f"Saved raw response for debugging to {raw_output_path}"
                                            )

                                    # Save processed result
                                    with open(output_file_path, "w") as out_f:
                                        json.dump(
                                            prediction,
                                            out_f,
                                            ensure_ascii=False,
                                            indent=2,
                                        )
                                        logger.info(
                                            f"Saved processed result to {output_file_path}"
                                        )

                            # Clean up temporary file
                            os.remove(local_file)

                    if not results_found:
                        logger.warning(
                            f"No JSONL result files found in {output_location}"
                        )
                else:
                    logger.error(f"Invalid output location format: {output_location}")
            except Exception as e:
                logger.error(f"Error processing job results: {e}")
                import traceback

                logger.error(traceback.format_exc())
        else:
            logger.error(f"Job {i + 1}/{len(jobs)} failed with state: {job.state}")
            if hasattr(job, "_gca_resource") and hasattr(job._gca_resource, "error"):
                logger.error(f"Error details: {job._gca_resource.error}")
            else:
                logger.error("No detailed error information available")

    logger.info("All batch processing jobs completed")


def main():
    # Load environment variables from .env file if present
    load_dotenv()

    # Parse arguments
    parser = argparse.ArgumentParser(description="Batch extract data using Vertex AI")
    parser.add_argument(
        "--input-file", required=True, help="JSON or JSONL input file with cases"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory to save extracted data"
    )
    parser.add_argument(
        "--batch-size", type=int, default=100, help="Number of cases per batch"
    )
    parser.add_argument("--gcs-bucket", required=True, help="GCS bucket for batch jobs")
    parser.add_argument(
        "--model", default="gemini-2.0-flash-001", help="Vertex AI model name"
    )
    parser.add_argument("--region", default="us-central1", help="Vertex AI region")
    args = parser.parse_args()

    # Convert paths
    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir)

    # Check if input file exists
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return 1

    # Prepare output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load input data
    logger.info(f"Reading input data from {input_file}")
    if input_file.suffix == ".json":
        with open(input_file, "r") as f:
            cases = json.load(f)
            if not isinstance(cases, list):
                cases = [cases]
    elif input_file.suffix == ".jsonl":
        cases = []
        with open(input_file, "r") as f:
            for line in f:
                if line.strip():
                    cases.append(json.loads(line))
    else:
        logger.error(f"Unsupported input file format: {input_file.suffix}")
        return 1

    logger.info(f"Loaded {len(cases)} cases from input file")

    # Split cases into batches for processing
    logger.info(f"Splitting into batches of max {args.batch_size} cases")
    batches = split_into_batches(cases, args.batch_size)
    logger.info(f"Created {len(batches)} batches")

    # Get the schema and convert it to a Gemini-compatible format
    schema = OrzeczenieSN.model_json_schema()
    gemini_schema = convert_schema_for_gemini(schema)
    logger.info("Prepared Gemini-compatible schema")

    # Create batch input files
    batch_files = []
    temp_dir = Path("/tmp/batch_extract")
    temp_dir.mkdir(parents=True, exist_ok=True)

    for i, batch in enumerate(batches):
        batch_file = temp_dir / f"batch_{i:04d}.jsonl"
        batch_files.append(batch_file)

        with open(batch_file, "w") as f:
            for case in batch:
                # Format each case entry correctly for Gemini batch prediction
                # Wrap the content in a "request" property as required by Vertex AI
                text_content = case.get("content", {}).get("raw_markdown", "")
                case_signature = case.get("case_signature", "unknown")

                # Calculate text length for dynamic prompt
                text_length = len(text_content)
                system_prompt = get_system_prompt(text_length)

                payload = {
                    "request": {
                        "contents": [
                            {"role": "user", "parts": [{"text": text_content}]}
                        ],
                        "system_instruction": {"parts": [{"text": system_prompt}]},
                        "generation_config": {
                            "temperature": 0.2,
                            "max_output_tokens": 8192,
                            "response_mime_type": "application/json",
                            "response_schema": gemini_schema,
                        },
                    },
                    "case_signature": case_signature,  # Include metadata for result processing
                }

                f.write(json.dumps(payload) + "\n")

        logger.info(f"Created batch file {batch_file} with {len(batch)} cases")

    # Submit batch prediction jobs
    logger.info("Submitting batch prediction jobs to Vertex AI")

    jobs = submit_batch_prediction_jobs(
        batch_files, args.gcs_bucket, args.batch_size, args.model, args.region
    )

    # Process the results
    download_and_process_results(jobs, output_dir, args.gcs_bucket)

    # Clean up temporary files
    for batch_file in batch_files:
        batch_file.unlink()
    temp_dir.rmdir()

    logger.info(f"Processing complete. Results saved to {output_dir}")
    return 0


def convert_schema_for_gemini(schema: dict, processed_models=None) -> dict:
    """
    Convert a JSON schema to be compatible with Vertex AI/Gemini.
    Creates a properly structured schema with all fields defined appropriately.

    Args:
        schema: The original JSON schema
        processed_models: Set of already processed model paths to avoid circular references

    Returns:
        A modified schema compatible with Gemini and BigQuery
    """
    # Initialize tracking set if not provided
    if processed_models is None:
        processed_models = set()

    # Handle None or empty schema
    if not schema:
        return {"type": "object"}

    # Create a deep copy to avoid modifying the original
    result = {}

    # Check for reference
    if "$ref" in schema:
        ref_path = schema["$ref"]
        if ref_path in processed_models:
            # Avoid circular reference
            return {"type": "object"}

        # Track this reference to avoid circular dependencies
        processed_models.add(ref_path)

        # Try to resolve reference if we have $defs
        if schema.get("$defs") and ref_path.startswith("#/$defs/"):
            ref_name = ref_path.split("/")[-1]
            if ref_name in schema.get("$defs", {}):
                # Merge the referenced schema with any additional properties
                ref_schema = schema["$defs"][ref_name].copy()
                for key, value in schema.items():
                    if key not in ("$ref", "$defs"):
                        ref_schema[key] = value

                # Process the merged schema
                return convert_schema_for_gemini(ref_schema, processed_models)

    # Process schema properties - remove problematic elements
    for key, value in schema.items():
        # Skip all fields starting with $ (like $ref, $defs) which cause BigQuery issues
        if key.startswith("$"):
            continue

        # Handle nested structures
        if isinstance(value, dict):
            # Process nested dictionary
            result[key] = convert_schema_for_gemini(value, processed_models)
        elif isinstance(value, list):
            # Process list items
            result[key] = [
                convert_schema_for_gemini(item, processed_models)
                if isinstance(item, dict)
                else item
                for item in value
            ]
        else:
            # Keep primitive values as is
            result[key] = value

    # Ensure type is specified for object schemas
    if "properties" in result and "type" not in result:
        result["type"] = "object"

    # Special handling for object properties - add inner structure if missing
    if result.get("type") == "object" and "properties" in result:
        # Process each property
        for prop_name, prop_value in result["properties"].items():
            # Ensure each property has a type
            if not isinstance(prop_value, dict):
                # Convert simple values to proper schema objects
                result["properties"][prop_name] = {
                    "type": "string",
                    "default": str(prop_value),
                }
            elif "type" not in prop_value:
                # Infer type from other properties
                if "properties" in prop_value:
                    prop_value["type"] = "object"
                elif "items" in prop_value:
                    prop_value["type"] = "array"
                else:
                    # Default to string if can't infer
                    prop_value["type"] = "string"

    # Handle required fields - add default values to avoid empty responses
    if "properties" in result and "required" in result:
        for field in result.get("required", []):
            if field in result["properties"]:
                prop = result["properties"][field]
                if prop.get("type") == "object" and "default" not in prop:
                    # For objects that are required but don't have properties,
                    # add explicit empty properties
                    if "properties" not in prop:
                        prop["properties"] = {}

                    # Add default empty object
                    prop["default"] = {}
                elif prop.get("type") == "array" and "default" not in prop:
                    prop["default"] = []
                elif prop.get("type") == "string" and "default" not in prop:
                    prop["default"] = ""

    # For the root OrzeczenieSN schema, ensure each field has the right structure
    if "title" in result and result.get("title") == "OrzeczenieSN":
        # Root schema
        root_schema = result

        # Define proper object structures for main fields
        field_structure = {
            "klasyfikacja": {
                "type": "object",
                "properties": {
                    "galaz_prawa": {"type": "string", "default": ""},
                    "glowna_kategoria_prawna": {"type": "string", "default": ""},
                    "podkategoria_prawna": {"type": "string", "default": ""},
                    "instytucje_prawne": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": [],
                    },
                },
            },
            "slowa_kluczowe_frazy": {
                "type": "object",
                "properties": {
                    "slowa_kluczowe": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": [],
                    },
                    "kluczowe_frazy": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": [],
                    },
                },
            },
            "podstawy_prawne_powiazania": {
                "type": "object",
                "properties": {
                    "podstawy_prawne": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": [],
                    },
                    "przywolane_akty_prawne": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": [],
                    },
                    "relacje_z_innymi_orzeczeniami": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": [],
                    },
                    "status_linii_orzeczniczej": {"type": "string", "default": ""},
                    "metody_wykładni": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": [],
                    },
                },
            },
            "analiza_prawna": {
                "type": "object",
                "properties": {
                    "dylematy_prawne": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": [],
                    },
                    "kluczowe_cytaty": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": [],
                    },
                    "kontrowersje_i_krytyka": {"type": "string", "default": ""},
                },
            },
            "wyszukiwanie_zastosowanie": {
                "type": "object",
                "properties": {
                    "potencjalne_pytania": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": [],
                    },
                    "praktyczne_zastosowanie": {"type": "string", "default": ""},
                    "waga_precedensowa": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5,
                        "default": 1,
                    },
                },
            },
            "streszczenie": {
                "type": "object",
                "properties": {"pelne_streszczenie": {"type": "string", "default": ""}},
            },
        }

        # Apply structure to each field
        for field_name, structure in field_structure.items():
            if field_name in root_schema.get("properties", {}):
                field_value = root_schema["properties"][field_name]

                # If field is a simple string type, replace with proper object structure
                if field_value.get("type") == "string" or not isinstance(
                    field_value, dict
                ):
                    root_schema["properties"][field_name] = structure

    return result


if __name__ == "__main__":
    sys.exit(main())
