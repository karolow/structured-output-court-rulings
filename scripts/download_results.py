#!/usr/bin/env python
"""
Script to download and process results from Vertex AI batch prediction jobs.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from google.cloud import aiplatform

from utils import download_results, find_output_uri

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def download_by_job_name(
    job_name: str,
    output_dir: Path,
    bucket_name: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> None:
    """
    Download batch prediction results from a specific job by its name.

    Args:
        job_name: Full Vertex AI job name
        (e.g., projects/my-project/locations/us-central1/batchPredictionJobs/12345)
        output_dir: Directory to save downloaded results
        bucket_name: Optional bucket name, if needed for fallback
        timestamp: Optional timestamp folder (e.g., 20250325_141757) to find results
    """
    # Make sure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Initialize Vertex AI SDK
        aiplatform.init()

        # Get the batch prediction job
        job = aiplatform.BatchPredictionJob(job_name)
        logger.info(f"Found batch prediction job: {job.display_name}")

        # Get output URI(s)
        output_uri = None

        # If timestamp is provided, construct the path directly
        if bucket_name and timestamp:
            output_uri = f"gs://{bucket_name}/batch_extract/output/{timestamp}"
            logger.info(f"Using timestamp-based output URI: {output_uri}")
        else:
            # Try to find URI from job metadata
            output_uri = find_output_uri(job, bucket_name)

        if not output_uri:
            logger.error(f"Could not find output URI for job {job_name}")
            return

        logger.info(f"Downloading results from {output_uri}")
        download_results(output_dir, [output_uri], bucket_name, job_name, timestamp)
        logger.info(f"Results downloaded to {output_dir}")

    except Exception as e:
        logger.error(f"Error processing job {job_name}: {e}", exc_info=True)


def check_jobs_status(
    job_names: List[str], wait: bool = True, timeout_minutes: int = 120
) -> bool:
    """
    Check the status of batch jobs.

    Args:
        job_names: List of job names to check
        wait: Whether to wait for jobs to complete
        timeout_minutes: Maximum time to wait in minutes

    Returns:
        True if all jobs are completed successfully, False otherwise
    """
    # Initialize Vertex AI
    aiplatform.init()

    if not job_names:
        logger.warning("No job names provided to check status")
        return False

    active_jobs = {job_name: job_name for job_name in job_names}
    start_time = time.time()
    timeout_seconds = timeout_minutes * 60

    logger.info(f"Checking status of {len(job_names)} jobs")

    while active_jobs and (time.time() - start_time < timeout_seconds):
        jobs_to_remove = []
        for job_name in list(active_jobs.keys()):
            try:
                job = aiplatform.BatchPredictionJob(job_name)
                state = job.state
                logger.info(f"Job {job_name}: {state}")

                if state in ["JOB_STATE_SUCCEEDED", "JOB_STATE_COMPLETED"]:
                    logger.info(f"Job {job_name} completed successfully")
                    jobs_to_remove.append(job_name)
                elif state in [
                    "JOB_STATE_FAILED",
                    "JOB_STATE_CANCELLED",
                    "JOB_STATE_EXPIRED",
                ]:
                    logger.error(f"Job {job_name} ended in state: {state}")
                    if not wait:
                        return False
                    jobs_to_remove.append(job_name)
            except Exception as e:
                logger.error(f"Error checking job {job_name}: {e}")
                if not wait:
                    return False

        # Remove completed/failed jobs from tracking
        for job_name in jobs_to_remove:
            del active_jobs[job_name]

        # If we're waiting and there are still active jobs, sleep
        if active_jobs and wait:
            logger.info(
                f"Waiting for {len(active_jobs)} jobs to complete... ({int((time.time() - start_time) / 60)} minutes elapsed)"
            )
            time.sleep(60)  # Check every minute
        elif not wait:
            break

    # Check if we exited due to timeout
    if active_jobs and wait:
        logger.warning(
            f"Timeout reached after {timeout_minutes} minutes. {len(active_jobs)} jobs still in progress."
        )
        return False

    return len(active_jobs) == 0


def main():
    # Load environment variables from .env file if present
    load_dotenv()

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Download and process Vertex AI batch prediction results"
    )
    parser.add_argument(
        "--job-name",
        help="Vertex AI batch prediction job name (full path)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save downloaded results",
    )
    parser.add_argument(
        "--gcs-bucket",
        help="Google Cloud Storage bucket name (optional, for fallback)",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for the job to complete before downloading",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Maximum time to wait in minutes",
    )
    parser.add_argument(
        "--timestamp",
        help="Timestamp folder to find results (e.g., 20250325_141757)",
    )
    args = parser.parse_args()

    # Create a Path object for the output directory
    output_dir = Path(args.output_dir)

    # Handle job status check and wait if requested
    job_complete = True
    if args.wait and args.job_name:
        logger.info(f"Waiting for job {args.job_name} to complete...")
        job_complete = check_jobs_status(
            [args.job_name], wait=True, timeout_minutes=args.timeout
        )

    # Download results if job is complete
    if job_complete and args.job_name:
        download_by_job_name(args.job_name, output_dir, args.gcs_bucket, args.timestamp)
    elif not job_complete:
        logger.warning("Job not complete, results not downloaded")
    else:
        logger.error("No job name provided")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
