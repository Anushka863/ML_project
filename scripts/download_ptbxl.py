import os
import sys
import urllib.request
import urllib.parse
import http.cookiejar
import re
import zipfile
from pathlib import Path

def download_and_extract():
    file_id = "1proXkaBDxrPgIbopBDdM_uMgsfQaysSQ"
    target_dir = Path(r"c:\Users\asus\Desktop\ML_PROJECT-Anushka_branch\ML_project\data\ptbxl")
    target_dir.mkdir(parents=True, exist_ok=True)
    zip_path = target_dir / "ptb-xl-1.0.3.zip"

    # Step 1: Download from Google Drive
    if not zip_path.exists() or zip_path.stat().st_size < 1_800_000_000:
        print(f"Downloading PTB-XL dataset from Google Drive ({file_id}) to {zip_path}...")
        cookie_jar = http.cookiejar.CookieJar()
        opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))
        urllib.request.install_opener(opener)

        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

        with opener.open(req) as resp:
            html = resp.read().decode("utf-8", errors="ignore")

        confirm_val = re.search(r'name="confirm" value="([^"]+)"', html)
        uuid_val = re.search(r'name="uuid" value="([^"]+)"', html)
        action_match = re.search(r'action="([^"]+)"', html)

        params = {"id": file_id}
        if confirm_val:
            params["confirm"] = confirm_val.group(1)
        if uuid_val:
            params["uuid"] = uuid_val.group(1)

        base_url = action_match.group(1) if action_match else "https://drive.usercontent.google.com/download"
        download_url = f"{base_url}?{urllib.parse.urlencode(params)}"
        download_req = urllib.request.Request(download_url, headers={"User-Agent": "Mozilla/5.0"})

        with opener.open(download_req) as d_resp, open(zip_path, "wb") as out_f:
            total_len = d_resp.headers.get("Content-Length")
            total_len = int(total_len) if total_len else None
            downloaded = 0
            chunk_size = 1024 * 1024 * 4  # 4 MB chunks
            while True:
                chunk = d_resp.read(chunk_size)
                if not chunk:
                    break
                out_f.write(chunk)
                downloaded += len(chunk)
                if total_len:
                    pct = (downloaded / total_len) * 100
                    print(f"\rDownloaded: {downloaded / (1024*1024):.1f} MB / {total_len / (1024*1024):.1f} MB ({pct:.1f}%)", end="", flush=True)
                else:
                    print(f"\rDownloaded: {downloaded / (1024*1024):.1f} MB", end="", flush=True)
        print("\nDownload complete!")
    else:
        print(f"Zip file already exists at {zip_path} ({zip_path.stat().st_size / (1024*1024):.1f} MB). Skipping download.")

    # Step 2: Unzip if not already extracted
    db_csv = target_dir / "ptbxl_database.csv"
    nested_db = list(target_dir.glob("**/ptbxl_database.csv"))
    if not nested_db:
        print(f"Extracting {zip_path} into {target_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(target_dir)
        print("Extraction complete!")
    else:
        print(f"Extracted PTB-XL database already found at {nested_db[0]}")

if __name__ == "__main__":
    download_and_extract()
