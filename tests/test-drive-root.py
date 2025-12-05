from google.oauth2 import service_account
from googleapiclient.discovery import build

FOLDER_ID = "12u7S77HBzehgilzegl7EbYRcnMuE9GKs"

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
creds = service_account.Credentials.from_service_account_file(
    "D:/Projects/Work/AdAssetGenerator/GoogleDriveKey.json",
    scopes=SCOPES,
)
service = build("drive", "v3", credentials=creds)

query = (
    f"'{FOLDER_ID}' in parents and "
    "mimeType='application/vnd.google-apps.folder' and trashed=false"
)
print("Query:", query)

resp = service.files().list(
    q=query,
    fields="files(id, name)",
    pageSize=100,
    includeItemsFromAllDrives=True,
    supportsAllDrives=True,
    corpora="allDrives",
).execute()

print("Result:", resp.get("files", []))