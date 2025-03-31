#!/usr/bin/env python
import argparse
import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import google.genai as genai
from dotenv import load_dotenv
from google.cloud import storage
from google.genai.types import CreateBatchJobConfig, HttpOptions, JobState

from prompts import get_system_prompt
from schema import OrzeczenieSN
from utils import (
    convert_schema_for_gemini,
    download_blobs_from_gcs,
    process_result_files,
    split_into_batches,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Set environment variables for Vertex AI integration
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

load_dotenv()


def submit_batch_jobs(
    input_files: List[Path],
    bucket_name: str,
    batch_size: int,
    model_name: str,
    region: str,
) -> List[Dict[str, Any]]:
    """
    Upload batch data to GCS and submit batch jobs using Gemini API.

    Args:
        input_files: List of JSONL files containing data for batch prediction
        bucket_name: GCS bucket to store input and output data
        batch_size: Maximum number of cases per batch job
        model_name: Model name (e.g., gemini-2.0-flash-001)
        region: GCP region

    Returns:
        List of submitted batch jobs information
    """
    # Create storage client
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # Upload input files to GCS
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    gcs_input_prefix = f"batch_extract/input/{timestamp}"
    gcs_output_prefix = f"batch_extract/output/{timestamp}"

    # Get the project ID from environment variable
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        raise ValueError("GOOGLE_CLOUD_PROJECT environment variable is not set")

    # Set up the Gemini client
    client = genai.Client(
        project=project_id, location=region, http_options=HttpOptions(api_version="v1")
    )

    # Upload input files to GCS and submit batch jobs
    jobs = []
    for input_file in input_files:
        # Create a unique blob name
        blob_name = f"{gcs_input_prefix}/{input_file.name}"
        blob = bucket.blob(blob_name)

        # Upload the file
        logger.info(f"Uploading {input_file} to gs://{bucket_name}/{blob_name}")
        blob.upload_from_filename(str(input_file))

        # Prepare input and output URIs
        input_uri = f"gs://{bucket_name}/{blob_name}"
        output_uri = f"gs://{bucket_name}/{gcs_output_prefix}/{input_file.stem}"

        # Submit the batch job
        logger.info(f"Submitting batch job for {input_uri} using model {model_name}")
        job = client.batches.create(
            model=model_name,
            src=input_uri,
            config=CreateBatchJobConfig(dest=output_uri),
        )

        jobs.append(
            {
                "name": job.name,
                "input_uri": input_uri,
                "output_uri": output_uri,
                "job_obj": job,
            }
        )

        logger.info(f"Submitted job {job.name}")

    return jobs


def download_results(
    output_dir: Path,
    output_uris: List[str],
    bucket_name: str,
    job_name: str = "",
    timestamp: str = "",
) -> None:
    """
    Download and process batch prediction results from GCS.

    Args:
        output_dir: Local directory to save the downloaded results
        output_uris: List of GCS URIs containing batch job results
        bucket_name: GCS bucket name
        job_name: Optional job name for better file naming
        timestamp: Optional timestamp for better file naming
    """
    logger.info(f"Downloading batch results to {output_dir}")

    # Download results for each job
    downloaded_files = []

    for output_uri in output_uris:
        # Process the URI to get the path
        if not output_uri.startswith("gs://"):
            logger.warning(f"Unexpected URI format: {output_uri}, skipping")
            continue

        # Download blobs from this output URI
        job_files = download_blobs_from_gcs(output_uri, output_dir)
        downloaded_files.extend(job_files)

    # Process downloaded files
    if downloaded_files:
        process_result_files(downloaded_files, output_dir, job_name, timestamp)
    else:
        logger.warning("No result files found to download")


def main():
    # Load environment variables from .env file if present
    load_dotenv()

    # Parse arguments
    parser = argparse.ArgumentParser(description="Batch extract data using Gemini API")
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
        "--model", default="gemini-2.0-flash-001", help="Generative AI model name"
    )
    parser.add_argument(
        "--region", default="europe-central2", help="Google Cloud region"
    )
    parser.add_argument(
        "--wait", action="store_true", help="Wait for batch jobs to complete"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Output debug schema to file"
    )
    parser.add_argument(
        "--download-results",
        action="store_true",
        help="Download and process results after jobs complete",
    )
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

    # Add to main() to debug the schema:
    if args.debug:
        debug_schema_path = output_dir / "schema_debug.json"
        with open(debug_schema_path, "w") as f:
            json.dump(gemini_schema, f, indent=2)
        logger.info(f"Debug schema written to {debug_schema_path}")

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
                text_content = case.get("content", {}).get("raw_markdown", "")
                case_signature = case.get("case_signature", "unknown")

                # Calculate text length for dynamic prompt
                text_length = len(text_content)
                system_prompt = get_system_prompt(text_length)

                # Format for Gemini batch API - this is different from Vertex AI batch formatting
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
                        "tools": [],
                    },
                    "case_signature": case_signature,  # Include metadata for result processing
                }

                f.write(json.dumps(payload) + "\n")

        logger.info(f"Created batch file {batch_file} with {len(batch)} cases")

    # Submit batch jobs
    logger.info("Submitting batch jobs to Gemini API")

    jobs = submit_batch_jobs(
        batch_files, args.gcs_bucket, args.batch_size, args.model, args.region
    )

    # Print output locations for reference
    logger.info("Batch jobs submitted successfully")
    logger.info("Output will be available at the following locations:")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    gcs_output_prefix = f"batch_extract/output/{timestamp}"
    logger.info(f"gs://{args.gcs_bucket}/{gcs_output_prefix}/")

    # Track output URIs for downloading results
    output_uris = []

    for i, job in enumerate(jobs):
        logger.info(f"Job {i + 1}: {job['name']}")
        output_uri = job["output_uri"]
        logger.info(f"Output will be at: {output_uri}")
        output_uris.append(output_uri)

    # Wait for batch jobs to complete if requested
    if args.wait and jobs:
        logger.info("Waiting for batch jobs to complete...")

        # Set up client for checking job status
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")

        client = genai.Client(
            project=project_id,
            location=args.region,
            http_options=HttpOptions(api_version="v1"),
        )

        # States indicating job completion
        completed_states = {
            JobState.JOB_STATE_SUCCEEDED,
            JobState.JOB_STATE_FAILED,
            JobState.JOB_STATE_CANCELLED,
            JobState.JOB_STATE_PAUSED,
        }

        # Track jobs that are still running
        active_jobs = {job["name"]: job for job in jobs}

        while active_jobs:
            time.sleep(30)  # Check status every 30 seconds

            jobs_to_remove = []
            for job_name, job_info in active_jobs.items():
                updated_job = client.batches.get(name=job_name)
                logger.info(f"Job {job_name} state: {updated_job.state}")

                if updated_job.state in completed_states:
                    logger.info(
                        f"Job {job_name} completed with state: {updated_job.state}"
                    )
                    logger.info(f"Results available at: {job_info['output_uri']}")
                    jobs_to_remove.append(job_name)

            # Remove completed jobs from tracking
            for job_name in jobs_to_remove:
                del active_jobs[job_name]

        logger.info("All batch jobs have completed")

        # Download results if requested
        if args.download_results:
            # Get first job name for better naming (using just the job ID)
            first_job_name = jobs[0]["name"] if jobs else ""
            short_job_name = first_job_name.split("/")[-1] if first_job_name else ""

            # Pass the timestamp that was generated when creating the jobs
            download_results(
                output_dir, output_uris, args.gcs_bucket, short_job_name, timestamp
            )
            logger.info(f"Results have been downloaded to {output_dir}")
    else:
        logger.info(
            "All batch jobs submitted. Check GCS bucket for results when completed."
        )
        if args.download_results and not args.wait:
            logger.warning(
                "Cannot download results because --wait option is not enabled. "
                "Please use both --wait and --download-results together."
            )

    # Clean up temporary files - with error handling for non-empty directory
    try:
        for batch_file in batch_files:
            if batch_file.exists():
                batch_file.unlink()

        # Try to remove the directory, but don't fail if it's not empty
        try:
            temp_dir.rmdir()
        except OSError as e:
            logger.warning(
                f"Could not remove temp directory: {e}. This is normal if some files were created there."
            )
    except Exception as e:
        logger.warning(f"Error during cleanup: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
