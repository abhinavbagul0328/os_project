import os
import glob

def main():
    # Target directory containing the images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, '..', 'data_pro'))
    
    if not os.path.exists(data_dir):
        print(f"Directory not found: {data_dir}")
        return

    # Find all PNG images
    image_files = glob.glob(os.path.join(data_dir, "*.png"))
    
    # Sort them alphabetically by file name
    image_files.sort(key=lambda x: os.path.basename(x))
    
    print(f"Found {len(image_files)} images to rename in {data_dir}.")
    
    if len(image_files) == 0:
        return
        
    # We will create a mapping file just in case you ever need the original names back,
    # or if you need to know which image was OK/NG based on the original name!
    mapping_file = os.path.join(data_dir, "name_mapping.csv")
    
    with open(mapping_file, 'w') as f:
        f.write("OriginalName,NewName\n")
        
        for i, filepath in enumerate(image_files, start=1):
            dirname = os.path.dirname(filepath)
            orig_basename = os.path.basename(filepath)
            ext = os.path.splitext(orig_basename)[1]
            
            # Create new name like 1.png, 2.png
            new_basename = f"{i}{ext}"
            new_filepath = os.path.join(dirname, new_basename)
            
            # Rename the file
            os.rename(filepath, new_filepath)
            
            # Record in the mapping file
            f.write(f"{orig_basename},{new_basename}\n")
            
            print(f"Renamed {orig_basename}  ->  {new_basename}")
            
    print(f"\nSuccessfully renamed {len(image_files)} images.")
    print(f"Saved the name mapping to: {mapping_file}")
    
    print("\n⚠️ WARNING ⚠️")
    print("The pipeline's utils.py currently relies on the '-OK' or '-NG' in the filename to determine labels.")
    print("Because the images are now just numbers, you will need to update utils.py to look up labels from 'name_mapping.csv' if you run the training again!")

if __name__ == "__main__":
    main()
