import json
from pathlib import Path


def filter_radar_data(input_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    filtered_frames = []
    original_count = len(data['data'])
    
    for frame in data['data']:
        try:
            if 'frameData' in frame and 'numDetectedPoints' in frame['frameData']:
                if frame['frameData']['numDetectedPoints'] >= 2:
                    filtered_frames.append(frame)
            elif 'frameData' in frame and 'pointCloud' in frame['frameData']:
                if len(frame['frameData']['pointCloud']) >= 2:
                    filtered_frames.append(frame)
        except:
            continue
    
    data['data'] = filtered_frames
    filtered_count = len(filtered_frames)
    removed_count = original_count - filtered_count
    
    with open(input_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    activity_path = str(input_file.parent.name)
    people_group = str(input_file.parent.parent.name)
    print(f"{people_group}/{activity_path}/{input_file.name}: {filtered_count}/{original_count} frames kept, {removed_count} removed")
    return filtered_count, original_count


def main():
    data_dir = Path("Social Exp Data")
    
    json_files = list(data_dir.rglob("*.json"))
    print(f"Found {len(json_files)} JSON files")
    
    total_original = 0
    total_filtered = 0
    
    for json_file in json_files:
        try:
            filtered, original = filter_radar_data(json_file)
            total_filtered += filtered
            total_original += original
        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
    
    print(f"\nTotal: {total_filtered}/{total_original} frames kept")
    print("Original files updated with filtered data")


if __name__ == "__main__":
    main()