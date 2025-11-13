import os
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button
from tkinter import Tk, filedialog

# 🔹 Step 1: Select activity folder
Tk().withdraw()
activity_path = filedialog.askdirectory(title="Select Activity Folder")

if not activity_path:
    raise ValueError("No folder selected. Please run again and choose an activity folder.")

# 🔹 Step 2: Load ALL JSON files in the folder
json_files = sorted([f for f in os.listdir(activity_path) if f.endswith(".json")])
if not json_files:
    raise ValueError("No JSON files found in this folder.")

frames = []
for jf in json_files:
    with open(os.path.join(activity_path, jf), "r") as f:
        data = json.load(f)
        frames.extend(data["data"])   #  include ALL frames, even last partial JSONs

total_frames = len(frames)
total_points = sum(len(fr["frameData"].get("pointCloud", [])) for fr in frames)

# 🔹 Step 3: Labels
parent_folder = os.path.basename(os.path.dirname(activity_path))  # e.g. "Social Experinment (2 People)"
activity_name = os.path.basename(activity_path)                   # e.g. "Group Splitting"

# Extract only "2 People" / "3 People"
if "(" in parent_folder and ")" in parent_folder:
    people_folder = parent_folder.split("(")[-1].split(")")[0].strip()
else:
    people_folder = parent_folder

activity_label = f"{activity_name} ({people_folder})"

print(f"Selected Activity: {activity_label}")
print(f"Total JSON files: {len(json_files)}")
print(f"Total frames loaded: {total_frames}")
print(f"Total points detected: {total_points}")

# 🔹 Step 4: Extract frame data
def get_frame_data(frame_idx):
    frame_data = frames[frame_idx]["frameData"]
    pc = np.array(frame_data.get("pointCloud", []))
    if pc.size == 0:
        return np.empty(0), np.empty(0), np.empty(0)
    return pc[:, 0], pc[:, 1], pc[:, 2]

# 🔹 Step 5: Setup plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax_slider = plt.axes([0.2, 0.02, 0.65, 0.03], facecolor='lightgoldenrodyellow')
slider = Slider(ax_slider, 'Frame', 0, total_frames - 1, valinit=0, valstep=1)

ax_play = plt.axes([0.85, 0.02, 0.05, 0.04])
ax_pause = plt.axes([0.91, 0.02, 0.05, 0.04])
btn_play = Button(ax_play, 'Play')
btn_pause = Button(ax_pause, 'Pause')

playing = False

# 🔹 Step 6: Update plot
def update(frame_idx):
    ax.clear()
    frame_idx = int(frame_idx)
    frame_data = frames[frame_idx]["frameData"]

    actual_frame_num = frame_data.get("frameNum", frame_idx + 1)
    num_points = len(frame_data.get("pointCloud", []))

    x, y, z = get_frame_data(frame_idx)
    if x.size > 0:
        ax.scatter(x, y, z, c='blue', marker='o', s=20, alpha=0.7)

    ax.set_title(
        f"Current Activity: {activity_label}\n"
        f"Frame: {actual_frame_num} ({frame_idx+1}/{total_frames}) | "
        f"Points: {num_points}/{total_points}",
        fontsize=12
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=20, azim=30)

update(0)

# 🔹 Step 7: Play / Pause (ensures last frame shown)
def play(event):
    global playing
    playing = True
    while playing and slider.val < total_frames - 1:
        slider.set_val(slider.val + 1)
        plt.pause(0.1)
    if playing:
        update(total_frames - 1)

def pause(event):
    global playing
    playing = False

slider.on_changed(update)
btn_play.on_clicked(play)
btn_pause.on_clicked(pause)

plt.show()
