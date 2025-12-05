# Quick Start Guide

## Running the Application Locally

### Step 1: Install Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure

1. Copy `env.example` to `.env`:
   ```bash
   copy env.example .env
   ```

2. Edit `.env` file with your credentials:
   ```env
   INPAINT_PROVIDER=tencent
   TENCENT_SECRET_ID=your_secret_id
   TENCENT_SECRET_KEY=your_secret_key
   TENCENT_REGION=ap-shanghai
   GOOGLE_APPLICATION_CREDENTIALS=C:\path\to\service-account.json
   GCP_PROJECT_ID=your-project-id
   GOOGLE_DRIVE_ROOT_FOLDER_ID=your_folder_id
   ENABLE_FALLBACK=false
   ```

### Step 3: Set Up Google Drive

1. Create folder structure in Google Drive:
   ```
   Root Folder/
   └── Test_Project/
       └── source_images_test/
           ├── image1.jpg
           └── prompt.txt
   ```

2. Share folder with service account email

3. Get folder ID from Drive URL

### Step 4: Run Locally

```bash
python run_local.py
```

That's it! The script will:
- ✅ Load configuration from `.env`
- ✅ Connect to Google Drive
- ✅ Process all images
- ✅ Show results and costs

## Troubleshooting

- **Missing .env file**: Copy from `env.example`
- **Credentials error**: Check file paths and keys
- **No images found**: Verify folder structure and sharing



