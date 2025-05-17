import os
import zipfile
import re
from PIL import Image
import shutil
import glob

"""
Creates the required folders to run the structure-preserving CycleGAN:
    - trainA: synthetic frames
    - depthA: corresponding depth maps 
    - trainB: real frames
"""

def extract_zip(zip_path, extract_to):
    """Extract zip file to specified directory"""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted {zip_path} to {extract_to}")


def process_synthetic_data(synthetic_dirs, train_a_path, depth_a_path):
    """Process synthetic data to get RGB and depth images"""
    image_count = 0

    print(f"Number of synthetic directories to process: {len(synthetic_dirs)}")
    
    for dir_path in synthetic_dirs:

        # Get all RGB & depth frames
        # # SimCol
        # rgb_frames = glob.glob(os.path.join(dir_path, "**", "FrameBuffer_*.png"), recursive=True)
        # depth_frames = glob.glob(os.path.join(dir_path, "**", "Depth_*.png"), recursive=True)
        # C3VD
        rgb_frames = glob.glob(os.path.join(dir_path, "**", "*_color.png"), recursive=True)
        depth_frames = glob.glob(os.path.join(dir_path, "**", "*_depth.png"), recursive=True)

        print(f"  Found {len(rgb_frames)} RGB frames & {len(depth_frames)} depth frames")
        
        # Sort frames by number
        # SimCol
        # rgb_frames.sort(key=lambda x: int(re.search(r'FrameBuffer_(\d+)\.png', x).group(1)))
        # depth_frames.sort(key=lambda x: int(re.search(r'Depth_(\d+)\.png', x).group(1)))
        # C3VD
        rgb_frames.sort(key=lambda x: int(re.search(r'(\d+)_color\.png', x).group(1)))
        depth_frames.sort(key=lambda x: int(re.search(r'(\d+)_depth\.png', x).group(1)))
        
        # Ensure matching pairs
        min_frames = min(len(rgb_frames), len(depth_frames))
        
        for i in range(min_frames):

            # Copy RGB frames to trainA
            rgb_dest = os.path.join(train_a_path, f"{image_count:05}.png")
            shutil.copy(rgb_frames[i], rgb_dest)
            
            # Copy depth frames to depthA
            depth_dest = os.path.join(depth_a_path, f"{image_count:05d}.png")
            shutil.copy(depth_frames[i], depth_dest)
            
            image_count += 1
    
    return image_count


def process_real_data(real_dir, train_b_path):
    """Process real images for trainB"""
    image_count = 0
    
    # Get image files
    image_files = glob.glob(os.path.join(real_dir, "**", "frame*.jpg"), recursive=True)

    print(f"Found {len(image_files)} real images")
    
    # Copy images to trainB
    for img_path in image_files:
        img_dest = os.path.join(train_b_path, f"{image_count:05d}.jpg")
        try:
            # Ensure the image can be opened
            img = Image.open(img_path)
            img.close()
            shutil.copy(img_path, img_dest)
            image_count += 1
        except Exception as e:
            print(f"Skipping {img_path}: {e}")
    
    return image_count

def create_folders(dataset_path):
    for folder in ['trainA', 'trainB', 'depthA']:
    # for folder in ['trainA', 'depthA']:
        folder_path = os.path.join(dataset_path, folder)
        os.makedirs(folder_path, exist_ok=True)
    
    return (
        os.path.join(dataset_path, 'trainA'),
        os.path.join(dataset_path, 'trainB'),
        os.path.join(dataset_path, 'depthA')
    )


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_path',
        dest='dataset_path',
        required=True,
        help='Target dataset folder path'
    )
    parser.add_argument(
        '--synthetic1',
        dest='synthetic1',
        required=True,
        help='Path to first synthetic zip file'
    )
    parser.add_argument(
        '--synthetic2',
        dest='synthetic2',
        required=False,
        default=None, 
        help='Path to second synthetic zip file (Optional)'
    )
    parser.add_argument(
        '--real',
        dest='real',
        required=True,
        help='Path to real.zip'
    )
    args = parser.parse_args()

    # Use scratch space for temporary processing
    temp_dir = "/scratch0/temp_dataset"
    os.makedirs(temp_dir, exist_ok=True)

    # Create temp extraction directories
    synth1_dir = os.path.join(temp_dir, 'synth1')
    synth2_dir = os.path.join(temp_dir, 'synth2')
    real_dir = os.path.join(temp_dir, 'real')
    
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(synth1_dir, exist_ok=True)
    os.makedirs(synth2_dir, exist_ok=True)
    os.makedirs(real_dir, exist_ok=True)
    
    # Extract zip files
    extract_zip(args.synthetic1, synth1_dir)
    # extract_zip(args.synthetic2, synth2_dir)
    if args.synthetic2:  
        extract_zip(args.synthetic2, synth2_dir)
    extract_zip(args.real, real_dir)
    
    # Create required folders
    temp_train_a, temp_train_b, temp_depth_a = create_folders(temp_dir)
    # temp_train_a, temp_depth_a = create_folders(temp_dir)
    
    # Get all synthetic directories containing frames
    synth_dirs = []
    # for root_dir in [synth1_dir, synth2_dir]:
    #     print(f"Examining root directory: {root_dir}")
    #     for entry in os.listdir(root_dir):
    #         dir_path = os.path.join(root_dir, entry)
    #         if os.path.isdir(dir_path):
    #             synth_dirs.append(dir_path)
    #             print(f"Added directory: {dir_path}")
    for root_dir in [synth1_dir] + ([synth2_dir] if args.synthetic2 else []):
        print(f"Examining root directory: {root_dir}")
        for entry in os.listdir(root_dir):
            dir_path = os.path.join(root_dir, entry)
            if os.path.isdir(dir_path):
                synth_dirs.append(dir_path)
                print(f"Added directory: {dir_path}")
    print(f"Total synthetic directories found: {len(synth_dirs)}")
    
    # Process the data
    print("Processing synthetic data...")
    synthetic_count = process_synthetic_data(synth_dirs, temp_train_a, temp_depth_a)
    print(f"Added {synthetic_count} image pairs to trainA & depthA")
    
    print("Processing real data...")
    real_count = process_real_data(real_dir, temp_train_b)
    print(f"Added {real_count} images to trainB")

    # Create final destination directories
    train_a_path, train_b_path, depth_a_path = create_folders(args.dataset_path)
    # train_a_path, depth_a_path = create_folders(args.dataset_path)
    
    # Copy final processed data to target location
    print("Copying processed data to final destination...")
    for src, dst in [(temp_train_a, train_a_path), 
                    (temp_train_b, train_b_path), 
                    (temp_depth_a, depth_a_path)]:
        for file in os.listdir(src):
            shutil.copy(os.path.join(src, file), os.path.join(dst, file))

    
    # Clean up temp directory
    print("Cleaning up temporary files...")
    shutil.rmtree(temp_dir)
    
    print("Data preparation completed successfully")


if __name__ == '__main__':
    main()
