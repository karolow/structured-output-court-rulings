# Legal Case Batch Processing with Vertex AI

This project enables batch processing of legal case rulings using Google Vertex AI and Gemini models. It extracts structured information from legal rulings in a scalable, efficient manner.

## Overview

The system processes legal rulings from the Polish Supreme Court, extracts structured information according to a defined schema, and stores the results in JSON format. The batch processing approach allows handling a large number of cases efficiently.

## Features

- Batch processing of legal rulings with Google Vertex AI
- Dynamic system prompts based on text length
- Scalable architecture for processing thousands of cases
- Structured output following a defined schema
- Progress monitoring and result aggregation

## Prerequisites

1. Google Cloud Platform account with Vertex AI enabled
2. Google Cloud Storage bucket
3. Appropriate GCP permissions:
   - Vertex AI User
   - Storage Admin
   - Service Account User

## Installation

1. Clone this repository
2. Install dependencies:

```bash
uv sync
```

3. Set up authentication:
   - Create a Google Cloud service account with necessary permissions
   - Download the service account key JSON file
   - Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your-service-account-key.json"
```
4. Create `.env` file with required variables.

## Environment Variables

The following environment variables are required:

```bash
GEMINI_API_KEY=your_api_key
GOOGLE_CLOUD_PROJECT=your-project-name
GOOGLE_CLOUD_LOCATION=europe-central2
GOOGLE_APPLICATION_CREDENTIALS=/path_to_creds.json
GOOGLE_GENAI_USE_VERTEXAI=True
```

You can set these in a `.env` file or directly in your environment.

## Usage

### Batch Processing

The main script for batch processing is `batch_extract.py`. Here's how to run it:

```bash
python batch_extract.py --input-file cases.json --output-dir output --gcs-bucket your-gcs-bucket-name
```

#### Command-Line Arguments for batch_extract.py

- `--input-file`: Path to input JSON file with cases (default: `first_1000_cases.json`)
- `--output-dir`: Local directory to store processed results (default: `output`)
- `--batch-size`: Number of cases per batch (default: 100)
- `--gcs-bucket`: GCS bucket name (required)
- `--model`: Vertex AI model name (default: `gemini-2.0-flash-001`)
- `--region`: GCP region (default: `europe-central2`)
- `--wait`: Wait for batch jobs to complete
- `--download-results`: Download and process results after jobs complete
- `--debug`: Output debug schema to file

#### Example

```bash
python batch_extract.py \
  --input-file sample.json \
  --output-dir output/extracted_data \
  --batch-size 200 \
  --gcs-bucket bucket-name \
  --model gemini-2.0-flash-001 \
  --region europe-central2 \
  --wait \
  --download-results
```

### Downloading Results

Two helper scripts are provided to download batch processing results:

#### 1. Download by Job Name (download_results.py)

Download results from a specific batch job by providing its Vertex AI job name:

```bash
python scripts/download_results.py \
  --job-name projects/your-project/locations/europe-central2/batchPredictionJobs/12345 \
  --output-dir output/job_results \
  --gcs-bucket your-gcs-bucket-name \
  --wait
```

Command-Line Arguments:
- `--job-name`: Vertex AI batch prediction job name (full path)
- `--output-dir`: Directory to save downloaded results (required)
- `--gcs-bucket`: Google Cloud Storage bucket name (optional, for fallback)
- `--wait`: Wait for the job to complete before downloading
- `--timeout`: Maximum time to wait in minutes (default: 120)
- `--timestamp`: Timestamp folder to find results (e.g., 20250325_141757)

#### 2. Download by Timestamp (download_by_timestamp.py)

Download results from all batch jobs with a specific timestamp prefix:

```bash
python scripts/download_by_timestamp.py \
  --timestamp-prefix 20250325_08 \
  --output-dir output/timestamp_results \
  --gcs-bucket your-gcs-bucket-name
```

Command-Line Arguments:
- `--timestamp-prefix`: Timestamp prefix to search for (e.g., '20250326_08')
- `--output-dir`: Directory to save downloaded results
- `--gcs-bucket`: Google Cloud Storage bucket name (required)
- `--list-objects`: List objects in the bucket to help with debugging
- `--search`: Filter object names by this pattern when listing

## Processing Workflow

1. **Prepare Data**: The script reads the input JSON file and splits it into multiple JSONL files with the appropriate format for batch prediction.
2. **Upload to GCS**: The JSONL files are uploaded to Google Cloud Storage.
3. **Submit Batch Jobs**: Batch prediction jobs are submitted to Vertex AI using the specified model.
4. **Monitor Progress**: The script monitors job progress and waits for all jobs to complete.
5. **Process Results**: Results are downloaded from GCS, processed, and saved to the output directory.

### Increasing Performance

- Adjust batch size based on your dataset characteristics
- Use regional buckets in the same region as Vertex AI processing
- Monitor memory usage when processing very large datasets
