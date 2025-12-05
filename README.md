# Background Replacement System

Production-ready automated background replacement system that monitors Google Drive folders, uses configurable AI providers for inpainting, and logs operations to Google Sheets.

## Features

- **Configurable AI Providers**: Switch between Tencent (FREE), Alibaba ($0.003), or Stability AI ($0.02)
- **Automatic Fallback**: If primary provider fails, automatically try backup providers
- **Google Drive Integration**: Monitors folders, reads prompts, downloads/uploads images
- **Google Sheets Logging**: Comprehensive logging with provider tracking and cost analysis
- **Background Mask Generation**: Automatic mask generation with Rembg (FREE), Remove.bg fallback ($0.01), OpenCV fallback
- **Cloud Deployment**: Runs on Google Cloud Run with scheduled triggers

## Architecture

```
Google Drive → Cloud Scheduler → Cloud Run Function → AI Provider → Google Sheets
```

## Project Structure

```
background-replacement-system/
├── providers/              # Provider system
│   ├── base.py            # Base abstract class
│   ├── tencent_hunyuan.py # Tencent provider
│   ├── alibaba_tongyi.py  # Alibaba provider
│   ├── stability_ai.py    # Stability AI provider
│   ├── factory.py         # Factory + fallback wrapper
│   └── metrics.py         # Provider metrics tracking
├── src/
│   ├── config.py          # Configuration management
│   ├── pipeline.py        # Main processing pipeline
│   ├── mask_generator.py  # Background mask generation
│   ├── google_drive_client.py    # Google Drive integration
│   ├── google_sheets_logger.py   # Google Sheets logging
│   └── utils.py           # Utility functions
├── tests/                 # Unit and integration tests
├── deployment/            # Deployment configuration
│   ├── main.py           # Cloud Function entry point
│   ├── Dockerfile        # Container configuration
│   └── cloudfunctions.yml # GCP configuration
├── requirements.txt       # Python dependencies
└── env.example           # Example environment variables
```

## Setup Instructions

### 1. Prerequisites

- Python 3.11+
- Google Cloud Project with APIs enabled
- Service account with Google Drive/Sheets permissions
- Provider API credentials (Tencent/Alibaba/Stability AI)

### 2. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

Copy `env.example` to `.env` and fill in your credentials:

```bash
cp env.example .env
```

Required environment variables:

- `INPAINT_PROVIDER`: Primary provider (tencent, alibaba, stability)
- `TENCENT_SECRET_ID`, `TENCENT_SECRET_KEY`: Tencent credentials
- `GOOGLE_APPLICATION_CREDENTIALS`: Path to service account JSON
- `GOOGLE_DRIVE_ROOT_FOLDER_ID`: Root folder ID in Google Drive
- `REMOVEBG_API_KEY`: Remove.bg API key (optional fallback, $0.01/image)

See `env.example` for all available configuration options.

### 4. Google Drive Structure

Your Google Drive should be organized as follows:

```
GoogleDrive_Root/
├── Project_A/
│   ├── source_images_category1/
│   │   ├── image1.jpg
│   │   └── prompt.txt (content: "beach sunset with palm trees")
│   └── Output_Folder_category1/
│       └── (output images will be created here)
└── Project_B/
    └── (similar structure)
```

## Configuration Guide

### Provider Selection

**Cost-Optimized (Recommended)**:
```bash
export INPAINT_PROVIDER=tencent          # FREE
export ENABLE_FALLBACK=true
export FALLBACK_PROVIDERS=alibaba,stability
```

**Reliability-Focused**:
```bash
export INPAINT_PROVIDER=alibaba          # $0.003/img
export ENABLE_FALLBACK=true
export FALLBACK_PROVIDERS=tencent,stability
```

**Premium**:
```bash
export INPAINT_PROVIDER=stability        # $0.02/img
export ENABLE_FALLBACK=true
export FALLBACK_PROVIDERS=alibaba,tencent
```

## Usage

### Local Development

```python
from src.pipeline import BackgroundReplacementPipeline
from src.config import get_config

config = get_config()
pipeline = BackgroundReplacementPipeline(config)
pipeline.execute()
```

### Deployment to Google Cloud Run

1. Build and push container:
```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/background-replacement
```

2. Deploy to Cloud Run:
```bash
gcloud run deploy background-replacement \
  --image gcr.io/PROJECT_ID/background-replacement \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --timeout 540s
```

3. Set up Cloud Scheduler:
```bash
gcloud scheduler jobs create http background-replacement-job \
  --schedule="*/5 * * * *" \
  --uri="https://your-function-url" \
  --http-method=GET
```

## Testing

Run unit tests:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest --cov=src --cov=providers tests/
```

## Cost Analysis

### Monthly Cost (1000 images)

| Setup | Mask Generation | Inpainting | Total Monthly Cost |
|-------|-----------------|------------|-------------------|
| **Optimal** ✅ | Rembg (FREE) | Tencent (FREE) + Alibaba fallback | **$0.44-0.84** |
| **Reliable** | Rembg + Remove.bg fallback | Alibaba (primary) + Tencent fallback | **$3.44-3.84** |
| **Premium** | Rembg + Remove.bg fallback | Stability AI | **$20.44-20.84** |

**Breakdown (Optimal Setup):**
- Cloud Run: $0.38
- Mask Generation: $0.00-0.10 (Rembg FREE + Remove.bg fallback optional)
- Inpainting: $0.00-0.30 (Tencent FREE + Alibaba fallback)
- Network: $0.06
- **Total: $0.44-0.84/month**

**Recommendation**: Start with Rembg (FREE) + Tencent (FREE) as primary, use fallbacks as needed. This gives 99.95% cost savings compared to using expensive APIs like Clipdrop ($700+/month).

## Troubleshooting

### Common Issues

1. **Provider Authentication Fails**
   - Check credentials in environment variables
   - Verify provider account is active
   - Check API quotas/limits

2. **Google Drive Access Denied**
   - Verify service account has Drive permissions
   - Check folder structure matches expected pattern
   - Ensure service account has access to folders

3. **Mask Generation Fails**
   - Rembg should work automatically (no API key needed)
   - Verify Remove.bg API key if using fallback (optional)
   - Check image format is supported (JPG, PNG)
   - OpenCV fallback should activate automatically

4. **Processing Timeout**
   - Increase Cloud Run timeout setting
   - Check provider API response times
   - Review image sizes (large images take longer)

## Monitoring

- **Google Sheets**: Per-project logs with processing details
- **Google Drive**: Execution logs in `Background_Generation_Logs` folder
- **Cloud Logging**: Application logs in GCP Console

## License

[Specify your license]

## Support

[Add support contact information]

