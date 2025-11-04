#!/usr/bin/env python3
"""
Download and extract the iNaturalist dataset for OpenOOD.
"""
import os
import zipfile
import gdown

# Configuration
GOOGLE_DRIVE_ID = '1zfLfMvoUD0CUlKNnkk7LgxZZBnTBipdj'
DATA_DIR = '/home/junesang/OpenOOD/data/images_largescale/'
DATASET_NAME = 'inaturalist'
ZIP_PATH = os.path.join(DATA_DIR, f'{DATASET_NAME}.zip')
EXTRACT_PATH = os.path.join(DATA_DIR, DATASET_NAME)

def download_dataset():
    """Download iNaturalist dataset from Google Drive."""
    print(f"Downloading {DATASET_NAME} dataset from Google Drive...")
    print(f"Download ID: {GOOGLE_DRIVE_ID}")
    print(f"Target path: {ZIP_PATH}")

    # Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)

    # Download using gdown with resume support
    gdown.download(
        id=GOOGLE_DRIVE_ID,
        output=ZIP_PATH,
        quiet=False,
        resume=True
    )

    print(f"✓ Download complete: {ZIP_PATH}")
    return ZIP_PATH

def extract_dataset(zip_path):
    """Extract the downloaded ZIP file."""
    print(f"\nExtracting {zip_path}...")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get list of files in the archive
        file_list = zip_ref.namelist()
        print(f"Archive contains {len(file_list)} files")

        # Extract all files
        zip_ref.extractall(DATA_DIR)

    print(f"✓ Extraction complete to: {DATA_DIR}")

def cleanup_zip(zip_path):
    """Remove the ZIP file after extraction."""
    print(f"\nCleaning up ZIP file...")
    if os.path.exists(zip_path):
        os.remove(zip_path)
        print(f"✓ Removed: {zip_path}")
    else:
        print(f"⚠ ZIP file not found: {zip_path}")

def verify_dataset():
    """Verify that the dataset was extracted correctly."""
    print(f"\nVerifying dataset...")

    images_dir = os.path.join(DATA_DIR, DATASET_NAME, 'images')

    if not os.path.exists(images_dir):
        print(f"✗ Images directory not found: {images_dir}")
        return False

    # Count image files
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    num_images = len(image_files)

    print(f"✓ Found {num_images} image files in {images_dir}")

    if num_images == 10000:
        print("✓ Dataset verification successful! (Expected 10,000 images)")
        return True
    elif num_images > 0:
        print(f"⚠ Warning: Found {num_images} images, expected 10,000")
        return True
    else:
        print("✗ No images found!")
        return False

def main():
    """Main execution function."""
    print("=" * 70)
    print("iNaturalist Dataset Download and Setup")
    print("=" * 70)

    try:
        # Step 1: Download
        zip_path = download_dataset()

        # Step 2: Extract
        extract_dataset(zip_path)

        # Step 3: Cleanup
        cleanup_zip(zip_path)

        # Step 4: Verify
        success = verify_dataset()

        if success:
            print("\n" + "=" * 70)
            print("✓ iNaturalist dataset setup complete!")
            print("=" * 70)
        else:
            print("\n" + "=" * 70)
            print("✗ Dataset verification failed. Please check the logs.")
            print("=" * 70)
            return 1

        return 0

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit(main())
