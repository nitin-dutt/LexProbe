import os
import urllib.request
import zipfile

print("Downloading CUAD dataset from Zenodo (official source)...")
os.makedirs("data/raw", exist_ok=True)

url = "https://zenodo.org/records/4595826/files/CUAD_v1.zip"
zip_path = "data/raw/CUAD_v1.zip"
json_output_path = "data/raw/CUAD_v1.json"

try:
    print("Downloading zip file (this may take a minute or two)...")
    urllib.request.urlretrieve(url, zip_path)
    
    print("Extracting CUAD_v1.json...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Dynamically find the json file inside the zip so we don't guess the folder structure
        json_filename = next((name for name in zip_ref.namelist() if name.endswith('CUAD_v1.json')), None)
        if json_filename:
            with zip_ref.open(json_filename) as source, open(json_output_path, "wb") as target:
                target.write(source.read())
            print(f"\nSUCCESS! File extracted perfectly to: {json_output_path}")
        else:
            print("\nError: Could not find CUAD_v1.json inside the zip.")
            
    # Clean up the zip to save space
    os.remove(zip_path)
    
except Exception as e:
    print(f"\nError: {e}")
