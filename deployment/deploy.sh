#!/bin/bash
# Deployment script for Background Replacement System to Google Cloud Run

set -e

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-your-project-id}"
REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME="background-replacement"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "Deploying Background Replacement System to Google Cloud Run"
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Service: ${SERVICE_NAME}"

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "Error: gcloud CLI is not installed"
    exit 1
fi

# Set project
echo "Setting GCP project..."
gcloud config set project ${PROJECT_ID}

# Build container image
echo "Building container image..."
gcloud builds submit --tag ${IMAGE_NAME}

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME} \
    --platform managed \
    --region ${REGION} \
    --memory 2Gi \
    --cpu 2 \
    --timeout 540s \
    --max-instances 10 \
    --allow-unauthenticated

# Get service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --region ${REGION} --format 'value(status.url)')

echo "Deployment complete!"
echo "Service URL: ${SERVICE_URL}"

# Optional: Set up Cloud Scheduler
read -p "Do you want to set up Cloud Scheduler? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    SCHEDULER_JOB="${SERVICE_NAME}-job"
    
    echo "Creating Cloud Scheduler job..."
    gcloud scheduler jobs create http ${SCHEDULER_JOB} \
        --schedule="*/5 * * * *" \
        --uri="${SERVICE_URL}" \
        --http-method=GET \
        --time-zone="America/New_York" \
        --attempt-deadline=540s || echo "Job may already exist"
    
    echo "Cloud Scheduler job created: ${SCHEDULER_JOB}"
fi

echo "Deployment script completed successfully!"



