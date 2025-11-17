import os
import json
import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

# ========== Feature Extraction ==========
def safe_divide(a, b, default=0):
    if b == 0 or np.isnan(b) or np.isinf(b):
        return default
    return a / b

def extract_optimized_features(point_cloud):

    if not point_cloud or len(point_cloud) == 0:
        return np.zeros(10)
    
    points = np.array(point_cloud, dtype=np.float64)
    n_points = len(points)

    x_coords = points[:, 0]
    y_coords = points[:, 1]
    z_coords = points[:, 2]
    velocities = points[:, 3]

    features = [
        # Feature 1: Average depth distance from radar sensor to distinguish near vs far activities
        float(np.mean(y_coords)),  
        # Feature 2: Ratio of moving points (>0.1 m/s) to separate dynamic vs static activities  
        safe_divide(np.sum(np.abs(velocities) > 0.1), n_points),  
        # Feature 3: Average height to differentiate sitting vs standing group postures
        float(np.mean(z_coords)),  
        # Feature 4: Average lateral position to track group's sideways movement
        float(np.mean(x_coords)),  
        # Feature 5: Lateral spread range (max-min X) to measure group compactness vs dispersion
        float(np.ptp(x_coords)),   
    ]

    # Feature 6: Proportion in lower height bins (0-1.2m) to identify sitting activities
    z_hist, _ = np.histogram(z_coords, bins=5, range=(0, 3))
    z_hist = z_hist / np.sum(z_hist) if np.sum(z_hist) > 0 else z_hist
    features.append(float(np.sum(z_hist[:2])))

    features.extend([
        # Feature 7: Depth position variability to detect formation consistency vs scattered groups
        float(np.var(y_coords)),     
        # Feature 8: Total radar reflections count as group size indicator (more people = more points)
        float(n_points),             
        # Feature 9: Average Doppler velocity to capture overall movement direction and speed
        float(np.mean(velocities)),  
        # Feature 10: Depth spread range (max-min Y) to measure group's front-to-back formation
        float(np.ptp(y_coords))      
    ])

    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

# ========== Sequence Creation ==========
def create_sequences(features_list, sequence_length=10, overlap=0.5):
    sequences = []
    step_size = int(sequence_length * (1 - overlap))
    for i in range(0, len(features_list) - sequence_length + 1, step_size):
        seq = features_list[i:i + sequence_length]
        if len(seq) == sequence_length:
            sequences.append(seq)
    return np.array(sequences)

# ========== Process One Folder ==========
def process_activity_folder(folder_path, activity_label):
    all_sequences, all_labels = [], []
    json_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.json')])

    for jf in json_files:
        file_path = os.path.join(folder_path, jf)
        with open(file_path, 'r') as f:
            data = json.load(f)

        frame_features = []
        for frame in data["data"]:
            pc = frame["frameData"].get("pointCloud", [])
            features = extract_optimized_features(pc)
            frame_features.append(features)

        if frame_features:
            seqs = create_sequences(frame_features)
            all_sequences.extend(seqs)
            all_labels.extend([activity_label] * len(seqs))

    return np.array(all_sequences), np.array(all_labels)

# ========== Process Dataset (2 vs 3 People) ==========
def process_dataset(base_path, people_folder):
    people_path = [f for f in os.listdir(base_path) if people_folder in f][0]
    people_path = os.path.join(base_path, people_path)

    X_all, y_all = [], []
    for activity in sorted(os.listdir(people_path)):   # sorted = numeric order
        activity_path = os.path.join(people_path, activity)
        if os.path.isdir(activity_path):
            print(f"📂 Processing {activity} ({people_folder})")
            X, y = process_activity_folder(activity_path, activity)
            if X.size > 0:
                X_all.append(X)
                y_all.append(y)

    if X_all:
        X_all = np.vstack(X_all)
        y_all = np.hstack(y_all)

        # Label encode in numeric order
        le = LabelEncoder()
        y_all_encoded = le.fit_transform(y_all)
        mapping = {int(i): cls for i, cls in enumerate(le.classes_)}

        return X_all, y_all_encoded, mapping
    else:
        return np.array([]), np.array([]), {}

# ========== Main ==========
def main():
    base_path = r"C:\Users\14052\Desktop\Research\Research\3D_People_tracking\Github\TI_6843\Fall 25\Notebook\Social Exp\Dataset 1\Dataset 1 (PC)"
    script_dir = os.path.dirname(os.path.abspath(__file__))

    for people in ["2 People", "3 People"]:
        X, y, mapping = process_dataset(base_path, people)
        if X.size > 0:
            save_dir = os.path.join(script_dir, f"optimized_data_{people.replace(' ', '_')}")
            os.makedirs(save_dir, exist_ok=True)

            np.save(os.path.join(save_dir, "sequences.npy"), X)
            np.save(os.path.join(save_dir, "labels.npy"), y)

            with open(os.path.join(save_dir, "class_mapping.json"), "w") as f:
                json.dump(mapping, f, indent=2)

            print(f"\n Saved {people} dataset")
            print(f"   Shape: {X.shape}, Labels: {len(y)}")
            print(f"   Classes mapping: {mapping}\n")

if __name__ == "__main__":
    main()
