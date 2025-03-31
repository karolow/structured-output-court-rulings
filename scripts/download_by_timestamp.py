#!/usr/bin/env python
"""
Script to download batch prediction results for all jobs with a specific timestamp prefix.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List

# Add parent directory to path to import utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
from google.cloud import storage

from utils import download_blobs_from_gcs, process_result_files

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def list_timestamp_dirs(bucket_name: str, timestamp_prefix: str) -> List[str]:
    """
    List all timestamp directories in the GCS bucket that match the given prefix.

    Args:
        bucket_name: GCS bucket name
        timestamp_prefix: Timestamp prefix to search for (e.g., '20250326_08')

    Returns:
        List of full timestamp directory paths found
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # If it looks like a full path, use it directly
    base_path = "batch_extract/output/"
    if timestamp_prefix.startswith(base_path):
        # User provided a full path like batch_extract/output/20250326_085508
        direct_path = timestamp_prefix
        # Remove trailing slash if present
        if direct_path.endswith("/"):
            direct_path = direct_path[:-1]

        # List objects under this path
        blobs = list(bucket.list_blobs(prefix=direct_path))

        if blobs:
            logger.info(f"Found {len(blobs)} objects under direct path {direct_path}")
            return [direct_path + "/"]

    # Extract the timestamp part if full path provided
    clean_prefix = timestamp_prefix
    if base_path in timestamp_prefix:
        clean_prefix = timestamp_prefix.split(base_path)[-1]

    # Remove any trailing slashes
    clean_prefix = clean_prefix.rstrip("/")

    # Find all output directories with this timestamp prefix
    all_output_dirs = set()

    # List blobs directly under output directory
    blobs = list(bucket.list_blobs(prefix=base_path, delimiter="/"))

    # Extract timestamp directories
    timestamp_dirs = set()
    for blob in blobs:
        if blob.name.endswith("/"):
            parts = blob.name.strip("/").split("/")
            if len(parts) >= 3:  # batch_extract/output/TIMESTAMP/
                timestamp = parts[2]  # Get the timestamp part
                if timestamp.startswith(clean_prefix):
                    timestamp_dirs.add(blob.name)
                    logger.info(f"Found timestamp directory: {blob.name}")

    # If we didn't find direct timestamp directories, check for nested structure
    if not timestamp_dirs:
        # Search for nested batch directories
        prefix_blob_iterator = bucket.list_blobs(prefix=base_path + clean_prefix)
        for blob in prefix_blob_iterator:
            blob_path = blob.name
            if "batch_" in blob_path and (
                "predictions.jsonl" in blob_path or "predictions-chunked" in blob_path
            ):
                # Extract the output directory path
                # Format example: batch_extract/output/20250326_085508/batch_0000/.../predictions.jsonl
                parts = blob_path.split("/")
                if len(parts) >= 3:
                    # Construct the output directory path including the timestamp
                    output_dir = "/".join(parts[:3]) + "/"
                    all_output_dirs.add(output_dir)

    # Combine direct timestamp dirs and nested output dirs
    combined_dirs = list(timestamp_dirs) + list(all_output_dirs)

    # Log what we found
    if combined_dirs:
        logger.info(
            f"Found {len(combined_dirs)} timestamp directories matching '{clean_prefix}':"
        )
        for directory in combined_dirs:
            logger.info(f"  - {directory}")
    else:
        logger.info(f"No timestamp directories found matching '{clean_prefix}'")

    return combined_dirs


def download_batch_results(
    bucket_name: str, job_dirs: List[str], output_dir: Path
) -> None:
    """
    Download batch prediction results from all specified job directories.

    Args:
        bucket_name: GCS bucket name
        job_dirs: List of job directory paths
        output_dir: Local directory to save the downloaded results
    """
    # Make sure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded_files = []
    processed_jobs = []

    for dir_path in job_dirs:
        # Extract job ID from the path
        job_id = "unknown"

        # Try different patterns to extract job ID
        if "genai_batch_job_" in dir_path:
            # For pattern: genai_batch_job_YYYYMMDDHHMMSS_NNNNN/
            parts = dir_path.rstrip("/").split("/")
            job_id = parts[-1]  # Get the last part
        else:
            # For other patterns, use timestamp as job_id
            parts = dir_path.rstrip("/").split("/")
            job_id = parts[-1] if len(parts) > 0 else "unknown"

        # Construct the full GCS URI
        output_uri = f"gs://{bucket_name}/{dir_path}"
        logger.info(f"Processing results from job: {job_id}")
        logger.info(f"Downloading from URI: {output_uri}")

        # Download blobs from this output URI
        job_files = download_blobs_from_gcs(output_uri, output_dir)
        if job_files:
            downloaded_files.extend(job_files)
            processed_jobs.append(job_id)
            logger.info(f"Downloaded {len(job_files)} files from job {job_id}")
        else:
            logger.warning(f"No files found for job {job_id}")

    # Process all downloaded files together with a combined descriptor
    if downloaded_files:
        # Use a job-based descriptor
        descriptor = "batch_jobs"
        if len(processed_jobs) == 1:
            descriptor = processed_jobs[0]
        elif len(processed_jobs) > 1:
            descriptor = f"multiple_jobs_{len(processed_jobs)}"

        # Process all files together
        process_result_files(
            downloaded_files,
            output_dir,
            job_name=descriptor,
            timestamp="",  # No timestamp needed since it's in the job name
        )
        logger.info(
            f"Processed {len(downloaded_files)} files from {len(processed_jobs)} jobs"
        )
    else:
        logger.warning("No result files found to download")


def list_all_objects(
    bucket_name: str, search_pattern: str = "", max_results: int = 100
) -> None:
    """
    List all objects in the bucket to help with debugging.

    Args:
        bucket_name: GCS bucket name
        search_pattern: Optional pattern to filter object names
        max_results: Maximum number of results to print
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # List all blobs in the bucket
    all_blobs = list(bucket.list_blobs())

    # Filter blobs if search pattern is provided
    if search_pattern:
        filtered_blobs = [
            blob for blob in all_blobs if search_pattern.lower() in blob.name.lower()
        ]
        logger.info(
            f"Found {len(filtered_blobs)} objects containing '{search_pattern}' in bucket {bucket_name}"
        )
        display_blobs = filtered_blobs[:max_results]
    else:
        logger.info(f"Found {len(all_blobs)} total objects in bucket {bucket_name}")
        display_blobs = all_blobs[:max_results]

    logger.info(f"Showing first {min(max_results, len(display_blobs))} objects:")

    # Print objects to understand the structure
    for i, blob in enumerate(display_blobs):
        logger.info(f"  {i + 1}. {blob.name}")


def main():
    # Load environment variables from .env file if present
    load_dotenv()

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Download batch prediction results for all jobs with a specific timestamp prefix"
    )
    parser.add_argument(
        "--timestamp-prefix",
        help="Timestamp prefix to search for (e.g., '20250326_08')",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to save downloaded results",
    )
    parser.add_argument(
        "--gcs-bucket",
        required=True,
        help="Google Cloud Storage bucket name",
    )
    parser.add_argument(
        "--list-objects",
        action="store_true",
        help="List objects in the bucket to help with debugging",
    )
    parser.add_argument(
        "--search",
        help="Filter object names by this pattern when listing",
    )
    args = parser.parse_args()

    # List all objects if requested
    if args.list_objects:
        list_all_objects(args.gcs_bucket, args.search or "")
        return 0

    # Check required arguments for download operation
    if not args.timestamp_prefix:
        logger.error("--timestamp-prefix is required when not using --list-objects")
        return 1

    if not args.output_dir:
        logger.error("--output-dir is required when not using --list-objects")
        return 1

    # Create Path object for the output directory
    output_dir = Path(args.output_dir)

    # List all timestamp directories matching the prefix
    timestamp_dirs = list_timestamp_dirs(args.gcs_bucket, args.timestamp_prefix)

    if not timestamp_dirs:
        logger.error(
            f"No timestamp directories found for prefix: {args.timestamp_prefix}"
        )
        return 1

    # Download and process results from all matching directories
    download_batch_results(args.gcs_bucket, timestamp_dirs, output_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
