import pandas as pd
import glob
import numpy as np
import json
from datetime import datetime
from dateutil.relativedelta import relativedelta
import os

def find_annot(activity):
    """Map activity names to numeric labels"""
    annots = {
        'approach': 0,
        'walking': 1,
        'splitting': 2,
        'standing': 3,
        'sitting': 4,
    }
    for act in annots.keys():
        if act in activity.lower():
            return annots[act]
    return None


def map_activity_type(a):
    """Map activities to their type classification"""
    # You can modify this based on your specific needs
    # For example, standing and sitting could be considered 'static' or 'micro'
    a_map = {
        0: 'dynamic',  # approach
        1: 'dynamic',  # walking
        2: 'dynamic',  # splitting
        3: 'static',   # standing
        4: 'static',   # sitting
    }
    return a_map.get(a, 'unknown')


def process_mmwave(f):
    """Process a single .txt file containing JSON data"""
    print(f"Processing: {os.path.basename(f)}")

    # Extract user from filename
    u = os.path.basename(f).split('_')[0]

    # Read JSON data (one JSON object per line)
    data = [json.loads(val) for val in open(f, "r")]

    if len(data) == 0:
        print(f"  WARNING: No data in {f}")
        return None

    # Get activity annotation
    annot = find_annot(data[0]['activity'])

    if annot is None:
        print(f"  WARNING: Unknown activity '{data[0]['activity']}' in {f}")
        return None

    # Handle date parsing
    if data[0]['datenow'].split('/')[1] == '0':
        new_date = '/'.join([data[0]['datenow'].split('/')[0], '1', data[0]['datenow'].split('/')[-1]])
        datetime_str = datetime.strftime(datetime.strptime(new_date, "%d/%m/%Y"), "%Y-%m-%d") + ' '
    else:
        datetime_str = datetime.strftime(
            datetime.strptime(data[0]['datenow'], "%d/%m/%Y") + relativedelta(months=1),
            "%Y-%m-%d"
        ) + ' '

    # Create DataFrame
    mmwave_df = pd.DataFrame.from_dict(data)

    # Add datetime column
    mmwave_df['datetime'] = mmwave_df['timenow'].apply(lambda e: datetime_str + ':'.join(e.split('_')))

    # Add user and activity
    mmwave_df['User'] = u
    mmwave_df['activity'] = annot

    # Check if doppz exists
    if 'doppz' in mmwave_df.columns:
        # Convert doppz to numpy arrays
        mmwave_df['doppz'] = list(np.array(mmwave_df['doppz'].values.tolist()))

        # Select relevant columns
        mmwave_df = mmwave_df[['datetime', 'rangeIdx', 'dopplerIdx', 'numDetectedObj',
                               'range', 'peakVal', 'x_coord', 'y_coord', 'doppz', 'activity']]
    else:
        print(f"  WARNING: No 'doppz' field in {f}")
        return None

    # Add activity type classification
    mmwave_df['activity_type'] = mmwave_df['activity'].map(lambda x: map_activity_type(x))

    print(f"  ✅ Processed {len(mmwave_df)} frames, Activity: {data[0]['activity']} (label={annot})")

    return mmwave_df


def read_mmwave(data_folder='raw_datasets'):
    """Read all .txt files from the data folder"""
    print(f"\n{'='*60}")
    print(f"Reading mmWave data from: {data_folder}")
    print(f"{'='*60}\n")

    mmwave_files = glob.glob(f'{data_folder}/*.txt')

    if len(mmwave_files) == 0:
        print(f"ERROR: No .txt files found in {data_folder}")
        return None

    print(f"Found {len(mmwave_files)} files to process\n")

    # Process all files
    dataframes = []
    for f in mmwave_files:
        df = process_mmwave(f)
        if df is not None:
            dataframes.append(df)

    if len(dataframes) == 0:
        print("\nERROR: No valid dataframes created")
        return None

    # Concatenate all dataframes
    print(f"\n{'='*60}")
    print("Combining all dataframes...")
    combined_df = pd.concat(dataframes, ignore_index=True)

    print(f"Total frames: {len(combined_df)}")
    print(f"Activities: {combined_df['activity'].unique()}")
    print(f"\nActivity distribution:")
    print(combined_df['activity'].value_counts())
    print(f"{'='*60}\n")

    return combined_df


def save_dataframes(mmwave_df, output_folder='processed_datasets'):
    """Save processed dataframes as .pkl files"""
    os.makedirs(output_folder, exist_ok=True)

    print(f"\n{'='*60}")
    print("Saving processed dataframes...")
    print(f"{'='*60}\n")

    # Save full dataset (all 5 activities)
    full_path = f'{output_folder}/my_data_full.pkl'
    mmwave_df.to_pickle(full_path)
    print(f"✅ Saved full dataset: {full_path} ({len(mmwave_df)} frames)")

    # Save by activity type
    dynamic_df = mmwave_df[mmwave_df.activity_type == 'dynamic']
    if len(dynamic_df) > 0:
        dynamic_path = f'{output_folder}/my_data_dynamic.pkl'
        dynamic_df.to_pickle(dynamic_path)
        print(f"✅ Saved dynamic activities: {dynamic_path} ({len(dynamic_df)} frames)")

    static_df = mmwave_df[mmwave_df.activity_type == 'static']
    if len(static_df) > 0:
        static_path = f'{output_folder}/my_data_static.pkl'
        static_df.to_pickle(static_path)
        print(f"✅ Saved static activities: {static_path} ({len(static_df)} frames)")

    print(f"\n{'='*60}")
    print("Dataset Summary")
    print(f"{'='*60}")
    print(f"Total frames: {len(mmwave_df)}")
    print(f"Dynamic activities (approach, walking, splitting): {len(dynamic_df)} frames")
    print(f"Static activities (standing, sitting): {len(static_df)} frames")
    print(f"\nActivity breakdown:")
    activity_names = {
        0: 'Approach',
        1: 'Walking',
        2: 'Splitting',
        3: 'Standing',
        4: 'Sitting'
    }
    for act_id, act_name in activity_names.items():
        count = len(mmwave_df[mmwave_df.activity == act_id])
        if count > 0:
            act_type = map_activity_type(act_id)
            print(f"  {act_name} ({act_type}): {count} frames")

    print(f"\nDoppz heatmap shape: {np.array(mmwave_df.iloc[0]['doppz']).shape}")
    print(f"{'='*60}\n")

    return dynamic_df, static_df


if __name__ == "__main__":
    # Process the data
    mmwave_df = read_mmwave('raw_datasets')

    if mmwave_df is not None:
        # Save as pickle files
        dynamic_df, static_df = save_dataframes(mmwave_df, 'processed_datasets')

        print("\n✅ Processing complete!")
        print("\nYour dataset files:")
        print("  - my_data_full.pkl (all 5 activities)")
        print("  - my_data_dynamic.pkl (approach, walking, splitting)")
        print("  - my_data_static.pkl (standing, sitting)")
        print("\nNext steps:")
        print("1. Create a classifier for 64x128 heatmaps")
        print("2. Train on your 5 activities")
        print("3. Evaluate the model")
    else:
        print("\n❌ Processing failed!")
