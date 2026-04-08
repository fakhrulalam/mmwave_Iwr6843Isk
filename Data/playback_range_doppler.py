import sys
import pandas as pd
import numpy as np
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QWidget, QPushButton, QLabel, QComboBox, QSlider, QSpinBox,
                             QFileDialog, QInputDialog)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage
import pyqtgraph as pg
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.ndimage import zoom


class RangeDopplerPlayback(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Range-Doppler Heatmap Playback")
        self.setGeometry(100, 100, 1200, 800)

        # Data variables
        self.current_activity = None
        self.activity_map = {}  # display name -> Path to sample folder
        self.heatmaps = []
        self.removed_frame_numbers = []
        self.current_frame = 0
        self.is_playing = False
        self.playback_speed = 100  # milliseconds between frames

        # Radar parameters (matching live configuration EXACTLY)
        self.maximum_range = 14.0  # meters (matching your live UI)
        self.unambiguous_velocity = 2.78  # m/s (10 km/h converted to m/s)
        self.min_value = 2048  # Default from live radar
        self.max_value = 4096  # Default from live radar
        self.grid_size = 250  # Grid resolution for interpolation
        self.use_mps = True  # Use m/s instead of km/h

        # Dataset path: default to the Research Data root (broadest coverage); allow override via CLI arg
        if len(sys.argv) > 1:
            self.dataset_root = Path(sys.argv[1]).expanduser().resolve()
        else:
            self.dataset_root = (Path(__file__).parent / "Research Data").resolve()


        # Setup UI
        self.setup_ui()

        # Setup timer for playback
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Load initial activity
        self.load_activities()

    def setup_ui(self):
        """Setup the user interface (matching live radar design)"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Top control panel - Activity and Playback controls
        control_layout = QHBoxLayout()

        # Dataset selector
        control_layout.addWidget(QLabel("Dataset:"))
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems([
            "Research Data",
            "Range-doppler Data/Single Activity Data",
            "Range-doppler Data/Social Exp Data"
        ])
        self.dataset_combo.setMaximumWidth(250)
        self.dataset_combo.currentTextChanged.connect(self.on_dataset_changed)
        control_layout.addWidget(self.dataset_combo)

        control_layout.addSpacing(20)

        # Activity selector
        control_layout.addWidget(QLabel("Activity:"))
        self.activity_combo = QComboBox()
        self.activity_combo.setMinimumWidth(120)
        self.activity_combo.currentTextChanged.connect(self.on_activity_changed)
        control_layout.addWidget(self.activity_combo)

        self.folder_button = QPushButton("Choose Folder")
        self.folder_button.clicked.connect(self.choose_dataset_folder)
        control_layout.addWidget(self.folder_button)

        self.file_button = QPushButton("Open CSV")
        self.file_button.clicked.connect(self.choose_csv_file)
        control_layout.addWidget(self.file_button)

        control_layout.addSpacing(20)

        # Play/Pause button
        self.play_button = QPushButton("▶ Play")
        self.play_button.setMinimumWidth(80)
        self.play_button.clicked.connect(self.toggle_playback)
        control_layout.addWidget(self.play_button)

        # Stop button
        self.stop_button = QPushButton("⏹ Stop")
        self.stop_button.setMinimumWidth(80)
        self.stop_button.clicked.connect(self.stop_playback)
        control_layout.addWidget(self.stop_button)

        # Export single whole-session plot
        self.export_button = QPushButton("Export Whole Plot")
        self.export_button.setMinimumWidth(130)
        self.export_button.clicked.connect(self.export_whole_plot)
        control_layout.addWidget(self.export_button)

        self.export_activity_button = QPushButton("Export Activity Plot")
        self.export_activity_button.setMinimumWidth(150)
        self.export_activity_button.clicked.connect(self.export_activity_plot)
        control_layout.addWidget(self.export_activity_button)

        self.video_button = QPushButton("Save Video")
        self.video_button.setMinimumWidth(100)
        self.video_button.clicked.connect(self.export_playback_video)
        control_layout.addWidget(self.video_button)

        control_layout.addSpacing(20)

        # Speed control
        control_layout.addWidget(QLabel("Speed (ms):"))
        self.speed_spinbox = QSpinBox()
        self.speed_spinbox.setRange(10, 1000)
        self.speed_spinbox.setValue(self.playback_speed)
        self.speed_spinbox.setSingleStep(10)
        self.speed_spinbox.setMinimumWidth(80)
        self.speed_spinbox.valueChanged.connect(self.on_speed_changed)
        control_layout.addWidget(self.speed_spinbox)

        control_layout.addStretch()
        main_layout.addLayout(control_layout)

        # Second row - Display parameters
        params_layout = QHBoxLayout()

        params_layout.addWidget(QLabel("Max Range (m):"))
        self.range_spinbox = QSpinBox()
        self.range_spinbox.setRange(1, 100)
        self.range_spinbox.setValue(int(self.maximum_range))
        self.range_spinbox.setMinimumWidth(60)
        self.range_spinbox.valueChanged.connect(self.on_range_changed)
        params_layout.addWidget(self.range_spinbox)

        params_layout.addSpacing(10)

        params_layout.addWidget(QLabel("Max Speed (m/s):"))
        self.velocity_spinbox = QSpinBox()
        self.velocity_spinbox.setRange(1, 50)
        self.velocity_spinbox.setValue(int(self.unambiguous_velocity))
        self.velocity_spinbox.setMinimumWidth(60)
        self.velocity_spinbox.valueChanged.connect(self.on_velocity_changed)
        params_layout.addWidget(self.velocity_spinbox)

        params_layout.addSpacing(10)

        params_layout.addWidget(QLabel("Min:"))
        self.min_spinbox = QSpinBox()
        self.min_spinbox.setRange(0, 8192)
        self.min_spinbox.setValue(self.min_value)
        self.min_spinbox.setMinimumWidth(70)
        self.min_spinbox.valueChanged.connect(self.on_min_changed)
        params_layout.addWidget(self.min_spinbox)

        params_layout.addSpacing(10)

        params_layout.addWidget(QLabel("Max:"))
        self.max_spinbox = QSpinBox()
        self.max_spinbox.setRange(0, 8192)
        self.max_spinbox.setValue(self.max_value)
        self.max_spinbox.setMinimumWidth(70)
        self.max_spinbox.valueChanged.connect(self.on_max_changed)
        params_layout.addWidget(self.max_spinbox)

        params_layout.addSpacing(20)

        # Preset buttons for common ranges
        preset1_btn = QPushButton("Preset 1\n(0-4096)")
        preset1_btn.clicked.connect(lambda: self.apply_preset(0, 4096))
        params_layout.addWidget(preset1_btn)

        preset2_btn = QPushButton("Preset 2\n(2048-4096)")
        preset2_btn.clicked.connect(lambda: self.apply_preset(2048, 4096))
        params_layout.addWidget(preset2_btn)

        preset3_btn = QPushButton("Preset 3\n(0-6000)")
        preset3_btn.clicked.connect(lambda: self.apply_preset(0, 6000))
        params_layout.addWidget(preset3_btn)

        params_layout.addStretch()
        main_layout.addLayout(params_layout)

        # Frame info and slider
        frame_control_layout = QHBoxLayout()

        self.frame_label = QLabel("Frame: 0 / 0")
        frame_control_layout.addWidget(self.frame_label)

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.valueChanged.connect(self.on_slider_changed)
        frame_control_layout.addWidget(self.frame_slider)

        main_layout.addLayout(frame_control_layout)

        # Range-Doppler heatmap plot (matching live radar exactly)
        self.heatmap_plot = pg.plot(title="Range-Doppler Heatmap")
        self.heatmap_plot.setLabel('bottom', 'Range [m]')
        self.heatmap_plot.setLabel('left', 'Speed [m/s]')
        self.heatmap_plot.showGrid(x=True, y=True)

        # Create heatmap image item
        self.heatmap_item = pg.ImageItem()
        self.heatmap_plot.addItem(self.heatmap_item)

        # Add colorbar (legend) - optional, comment out if it causes issues
        # Note: ColorBarItem may not be available in older pyqtgraph versions
        try:
            jet_cmap = plt.get_cmap('jet')
            jet_rgba = jet_cmap(np.linspace(0, 1, 256))
            jet_rgb = (jet_rgba[:, :3] * 255).astype(np.ubyte)
            jet_pos = np.linspace(0.0, 1.0, 256)
            jet_colormap = pg.ColorMap(jet_pos, jet_rgb)

            self.colorbar = pg.ColorBarItem(
                values=(0, 4096),
                colorMap=jet_colormap,
                label='Signal Strength'
            )
            self.colorbar.setImageItem(self.heatmap_item)
        except Exception as e:
            print(f"ColorBar not available: {e}")

        main_layout.addWidget(self.heatmap_plot)

        # Status bar
        self.status_label = QLabel("Ready")
        main_layout.addWidget(self.status_label)

    def load_activities(self):
        """Load all available samples by searching for range_doppler.csv recursively."""
        if not self.dataset_root.exists():
            self.status_label.setText(f"Error: Dataset path not found: {self.dataset_root}")
            return

        self.activity_map.clear()
        activities = []

        for rd_file in sorted(self.dataset_root.rglob("range_doppler.csv")):
            sample_dir = rd_file.parent
            display_name = str(sample_dir.relative_to(self.dataset_root)).replace("\\", "/")
            self.activity_map[display_name] = sample_dir
            activities.append(display_name)

        self.activity_combo.blockSignals(True)
        self.activity_combo.clear()
        if activities:
            self.activity_combo.addItems(sorted(activities))
            self.status_label.setText(f"Found {len(activities)} samples under {self.dataset_root.name}")
            # Auto-load first activity
            self.activity_combo.setCurrentIndex(0)
        else:
            self.status_label.setText(f"No range_doppler.csv found under {self.dataset_root}")
        self.activity_combo.blockSignals(False)
        
        # If we have activities, trigger the load for the first one
        if activities:
            self.on_activity_changed(sorted(activities)[0])

    def on_activity_changed(self, activity_name):
        """Handle activity selection change"""
        if not activity_name:
            return

        self.stop_playback()
        self.current_activity = activity_name
        self.load_range_doppler_data(activity_name)

    def on_dataset_changed(self, dataset_path):
        """Handle dataset selection change"""
        if not dataset_path:
            return
        
        self.stop_playback()
        self.dataset_root = (Path(__file__).parent / dataset_path).resolve()
        self.load_activities()

    def choose_dataset_folder(self):
        """Allow selecting a dataset root folder at runtime."""
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select Dataset Root Folder",
            str(self.dataset_root)
        )
        if not selected:
            return

        self.dataset_root = Path(selected).resolve()
        self.stop_playback()
        self.load_activities()

    def choose_csv_file(self):
        """Allow loading any single range_doppler CSV file directly."""
        csv_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select range_doppler CSV",
            str(self.dataset_root),
            "CSV Files (*.csv)"
        )
        if not csv_path:
            return

        selected_path = Path(csv_path).resolve()
        if selected_path.name.lower() != "range_doppler.csv":
            self.status_label.setText("Please select a file named range_doppler.csv")
            return

        if self.dataset_root in selected_path.parents:
            key = str(selected_path.parent.relative_to(self.dataset_root)).replace("\\", "/")
        else:
            key = selected_path.parent.name

        self.activity_map[key] = selected_path.parent

        existing = [self.activity_combo.itemText(i) for i in range(self.activity_combo.count())]
        if key not in existing:
            self.activity_combo.addItem(key)

        self.activity_combo.setCurrentText(key)

    def load_range_doppler_data(self, activity_name):
        """Load range-Doppler data from CSV file"""
        sample_dir = self.activity_map.get(activity_name)
        if sample_dir is None:
            self.status_label.setText(f"Error: Activity not found: {activity_name}")
            return

        csv_path = sample_dir / "range_doppler.csv"

        if not csv_path.exists():
            self.status_label.setText(f"Error: File not found: {csv_path}")
            return

        self.status_label.setText(f"Loading {activity_name}...")

        try:
            self.heatmaps, self.removed_frame_numbers = self.load_heatmaps_from_csv(csv_path)

            if self.heatmaps:
                self.current_frame = 0
                self.frame_slider.setMaximum(len(self.heatmaps) - 1)
                self.frame_slider.setValue(0)
                self.update_display()
                if self.removed_frame_numbers:
                    preview = self.removed_frame_numbers[:8]
                    suffix = "..." if len(self.removed_frame_numbers) > 8 else ""
                    self.status_label.setText(
                        f"Loaded {len(self.heatmaps)} frames from {activity_name} | "
                        f"removed {len(self.removed_frame_numbers)} anomalous frames {preview}{suffix}"
                    )
                else:
                    self.status_label.setText(f"Loaded {len(self.heatmaps)} frames from {activity_name}")
            else:
                self.status_label.setText("Error: No heatmaps loaded")

        except Exception as e:
            self.status_label.setText(f"Error loading data: {str(e)}")
            import traceback
            traceback.print_exc()

    def load_heatmaps_from_csv(self, csv_path):
        """Load one CSV and return filtered heatmaps plus removed frame numbers."""
        df = pd.read_csv(csv_path)

        if all(col in df.columns for col in ['frame_number', 'range_bin', 'doppler_bin', 'signal_strength']):
            heatmaps, frame_numbers = self.parse_longform_data(df)
        else:
            # Fallback for headerless files commonly used in Research Data.
            raw = pd.read_csv(csv_path, header=None)
            if raw.shape[1] >= 5:
                raw = raw.iloc[:, :5].copy()
                raw.columns = ['timestamp', 'frame_number', 'range_bin', 'doppler_bin', 'signal_strength']
            elif raw.shape[1] == 4:
                raw.columns = ['frame_number', 'range_bin', 'doppler_bin', 'signal_strength']
            else:
                raise ValueError("Unexpected CSV format")

            raw[['frame_number', 'range_bin', 'doppler_bin', 'signal_strength']] = raw[
                ['frame_number', 'range_bin', 'doppler_bin', 'signal_strength']
            ].apply(pd.to_numeric, errors='coerce')
            raw = raw.dropna(subset=['frame_number', 'range_bin', 'doppler_bin', 'signal_strength'])
            heatmaps, frame_numbers = self.parse_longform_data(raw)

        return self.filter_anomalous_frames(heatmaps, frame_numbers)

    def parse_longform_data(self, df):
        """Parse long-form DataFrame into list of 2D heatmaps and frame numbers."""
        heatmaps = []
        frame_numbers = []

        # Get dimensions
        max_range_bin = int(df['range_bin'].max())
        max_doppler_bin = int(df['doppler_bin'].max())
        num_range_bins = max_range_bin + 1
        num_doppler_bins = max_doppler_bin + 1

        # Group by frame number
        grouped = df.groupby('frame_number')

        for frame_num in sorted(grouped.groups.keys()):
            frame_data = grouped.get_group(frame_num)

            # Create zero matrix
            heatmap = np.zeros((num_range_bins, num_doppler_bins), dtype=np.float32)

            # Fill matrix with signal strength values
            range_bins = frame_data['range_bin'].to_numpy().astype(int)
            doppler_bins = frame_data['doppler_bin'].to_numpy().astype(int)
            signal_strengths = frame_data['signal_strength'].to_numpy().astype(np.float32)

            heatmap[range_bins, doppler_bins] = signal_strengths

            heatmaps.append(heatmap)
            frame_numbers.append(int(frame_num))

        return heatmaps, frame_numbers

    def filter_anomalous_frames(self, heatmaps, frame_numbers):
        """Remove clearly corrupted/saturated frames before playback."""
        if not heatmaps:
            return heatmaps, []

        if len(heatmaps) < 5:
            return heatmaps, []

        means = np.array([h.mean() for h in heatmaps], dtype=np.float32)
        stds = np.array([h.std() for h in heatmaps], dtype=np.float32)
        peaks = np.array([h.max() for h in heatmaps], dtype=np.float32)
        sat_counts = np.array([(h > 60000).sum() for h in heatmaps], dtype=np.float32)

        def robust_abs_z(values):
            med = np.median(values)
            mad = np.median(np.abs(values - med))
            if mad < 1e-9:
                return np.zeros_like(values)
            return np.abs((values - med) / (1.4826 * mad))

        z_mean = robust_abs_z(means)
        z_std = robust_abs_z(stds)
        z_peak = robust_abs_z(peaks)

        # Keep filtering conservative: remove only clearly corrupted frames.
        mask = (
            (sat_counts > 0)
            | (peaks > 10000)
            | (stds > 2000)
            | ((z_peak > 12.0) & (z_std > 12.0) & (z_mean > 12.0))
        )

        filtered_heatmaps = [h for i, h in enumerate(heatmaps) if not mask[i]]
        removed_frames = [int(frame_numbers[i]) for i in range(len(frame_numbers)) if mask[i]]

        return filtered_heatmaps, removed_frames

    def update_display(self):
        """Update the heatmap display with current frame (matching live radar exactly)"""
        if not self.heatmaps or self.current_frame >= len(self.heatmaps):
            return

        # Get current heatmap
        heatmap_data = self.heatmaps[self.current_frame]

        # Apply FFT shift to center zero velocity (exactly as in live system)
        heatmap_data = np.fft.fftshift(heatmap_data, axes=1)

        # Interpolate to higher resolution for smoother display (matching live radar)
        desired_rows = max(self.grid_size, heatmap_data.shape[0])
        desired_cols = max(self.grid_size, heatmap_data.shape[1])

        scale_y = desired_rows / heatmap_data.shape[0]
        scale_x = desired_cols / heatmap_data.shape[1]
        interpolated_heatmap = zoom(heatmap_data, (scale_y, scale_x), order=1)  # Bilinear interpolation

        # Normalize data (exactly as in live radar - NO clipping)
        heatmap_normalized = (interpolated_heatmap - self.min_value) / (self.max_value - self.min_value)

        # Debug: Print data range (comment out after debugging)
        if self.current_frame == 0:
            print(f"Data range: min={interpolated_heatmap.min():.1f}, max={interpolated_heatmap.max():.1f}, mean={interpolated_heatmap.mean():.1f}")
            print(f"Normalized range: min={heatmap_normalized.min():.3f}, max={heatmap_normalized.max():.3f}")

        # Create colormap lookup table (jet colormap as in live system)
        cmap = plt.get_cmap('jet')
        lookup_table = (cmap(np.linspace(0, 1, 256)) * 255).astype(np.uint8)

        # Apply lookup table (RGB only, no alpha)
        self.heatmap_item.setLookupTable(lookup_table[:, :3])

        # Set the image data
        self.heatmap_item.setImage(heatmap_normalized)

        # Set fixed color range [0, 1]
        self.heatmap_item.setLevels([0, 1])

        # Set axis ranges (exactly as in live radar)
        # Primary axis (bottom/x) = Range [m]
        # Secondary axis (left/y) = Speed [m/s]
        self.heatmap_item.setRect(
            0, -self.unambiguous_velocity,  # x_start, y_start
            self.maximum_range, 2 * self.unambiguous_velocity  # width, height
        )

        # Update frame label
        self.frame_label.setText(f"Frame: {self.current_frame + 1} / {len(self.heatmaps)}")

        # Update slider without triggering signal
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(self.current_frame)
        self.frame_slider.blockSignals(False)

    def update_frame(self):
        """Update to next frame during playback"""
        if not self.heatmaps:
            return

        self.current_frame += 1

        # Loop back to start
        if self.current_frame >= len(self.heatmaps):
            self.current_frame = 0

        self.update_display()

    def toggle_playback(self):
        """Toggle play/pause"""
        if self.is_playing:
            self.pause_playback()
        else:
            self.start_playback()

    def start_playback(self):
        """Start playback"""
        if not self.heatmaps:
            self.status_label.setText("Error: No data loaded")
            return

        self.is_playing = True
        self.play_button.setText("⏸ Pause")
        self.timer.start(self.playback_speed)
        self.status_label.setText("Playing...")

    def pause_playback(self):
        """Pause playback"""
        self.is_playing = False
        self.play_button.setText("▶ Play")
        self.timer.stop()
        self.status_label.setText("Paused")

    def stop_playback(self):
        """Stop playback and reset to first frame"""
        self.is_playing = False
        self.play_button.setText("▶ Play")
        self.timer.stop()
        self.current_frame = 0
        self.update_display()
        self.status_label.setText("Stopped")

    def on_slider_changed(self, value):
        """Handle slider value change"""
        self.current_frame = value
        self.update_display()

    def on_speed_changed(self, value):
        """Handle playback speed change"""
        self.playback_speed = value
        if self.is_playing:
            self.timer.setInterval(self.playback_speed)

    def on_range_changed(self, value):
        """Handle maximum range change"""
        self.maximum_range = float(value)
        self.update_display()

    def on_velocity_changed(self, value):
        """Handle maximum velocity change"""
        self.unambiguous_velocity = float(value)
        self.update_display()

    def on_min_changed(self, value):
        """Handle minimum value change"""
        self.min_value = value
        self.update_display()

    def on_max_changed(self, value):
        """Handle maximum value change"""
        self.max_value = value
        self.update_display()

    def apply_preset(self, min_val, max_val):
        """Apply a preset min/max range"""
        self.min_spinbox.setValue(min_val)
        self.max_spinbox.setValue(max_val)

    def _grab_plot_rgb_frame(self):
        """Grab the rendered plot widget as an RGB frame (includes axes and labels)."""
        pixmap = self.heatmap_plot.grab()
        qimg = pixmap.toImage().convertToFormat(QImage.Format_RGB888)
        w = qimg.width()
        h = qimg.height()
        bpl = qimg.bytesPerLine()
        ptr = qimg.bits()
        ptr.setsize(h * bpl)
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, bpl))
        arr = arr[:, : (w * 3)].reshape((h, w, 3)).copy()
        return arr

    def export_playback_video(self):
        """Export currently loaded playback frames to MP4 or GIF for presentation use."""
        if not self.heatmaps:
            self.status_label.setText("Error: No data loaded to export video")
            return

        safe_name = (self.current_activity or "session").replace("/", "_").replace("\\", "_")
        default_path = Path(__file__).parent / "Picture" / f"{safe_name}_playback.mp4"
        save_path_str, _ = QFileDialog.getSaveFileName(
            self,
            "Save Playback Video",
            str(default_path),
            "MP4 Video (*.mp4);;GIF Animation (*.gif);;All Files (*)",
        )
        if not save_path_str:
            self.status_label.setText("Video export canceled")
            return

        save_path = Path(save_path_str)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import imageio.v2 as imageio
        except Exception:
            self.status_label.setText("Video export needs imageio. Install: pip install imageio imageio-ffmpeg")
            return

        fps = max(1, int(round(1000.0 / float(self.playback_speed))))
        self.status_label.setText(f"Exporting video ({len(self.heatmaps)} frames @ {fps} fps)...")
        QApplication.processEvents()

        was_playing = self.is_playing
        if was_playing:
            self.pause_playback()

        original_frame = self.current_frame
        rgb_frames = []

        for i in range(len(self.heatmaps)):
            self.current_frame = i
            self.update_display()
            QApplication.processEvents()
            rgb_frames.append(self._grab_plot_rgb_frame())

        self.current_frame = original_frame
        self.update_display()
        if was_playing:
            self.start_playback()

        try:
            if save_path.suffix.lower() == ".gif":
                imageio.mimsave(save_path, rgb_frames, duration=1.0 / fps, loop=0)
            else:
                imageio.mimsave(save_path, rgb_frames, fps=fps, macro_block_size=None)
            self.status_label.setText(f"Saved playback video: {save_path}")
        except Exception as e:
            if save_path.suffix.lower() != ".gif":
                fallback = save_path.with_suffix(".gif")
                try:
                    imageio.mimsave(fallback, rgb_frames, duration=1.0 / fps, loop=0)
                    self.status_label.setText(f"MP4 failed ({e}); saved GIF instead: {fallback}")
                    return
                except Exception:
                    pass
            self.status_label.setText(f"Video export failed: {e}")

    def export_whole_plot(self):
        """Export one single playback-matched plot using sum aggregation over all loaded frames."""
        if not self.heatmaps:
            self.status_label.setText("Error: No data loaded to export")
            return

        whole = self.build_summed_plot_data(self.heatmaps)

        safe_name = (self.current_activity or "session").replace("/", "_").replace("\\", "_")
        default_path = Path(__file__).parent / "Picture" / f"{safe_name}_playback_whole_sum.png"

        default_title = self.current_activity or ""
        self.save_sum_plot(whole, default_path, "Save Whole-Session Plot", default_title)

    def export_activity_plot(self):
        """Export one summed plot for all sessions matching a selected activity name."""
        if not self.activity_map:
            self.status_label.setText("Error: No activities available")
            return

        activity_groups = {}
        for sample_dir in self.activity_map.values():
            group_label, _ = self.extract_activity_group_label(sample_dir)
            activity_groups.setdefault(group_label, []).append(sample_dir)

        activity_names = sorted(activity_groups.keys())
        if not activity_names:
            self.status_label.setText("Error: No activity folders found")
            return

        default_activity = None
        if self.current_activity in self.activity_map:
            default_activity, _ = self.extract_activity_group_label(self.activity_map[self.current_activity])
        default_index = activity_names.index(default_activity) if default_activity in activity_names else 0

        activity_name, ok = QInputDialog.getItem(
            self,
            "Select Activity",
            "Choose activity to aggregate across sessions:",
            activity_names,
            default_index,
            False,
        )
        if not ok or not activity_name:
            self.status_label.setText("Export canceled")
            return

        matched_dirs = activity_groups.get(activity_name, [])
        if not matched_dirs:
            self.status_label.setText(f"Error: No sessions found for activity '{activity_name}'")
            return

        all_heatmaps = []
        for sample_dir in matched_dirs:
            csv_path = sample_dir / "range_doppler.csv"
            if not csv_path.exists():
                continue
            try:
                heatmaps, _ = self.load_heatmaps_from_csv(csv_path)
                all_heatmaps.extend(heatmaps)
            except Exception as e:
                print(f"Skipping {csv_path}: {e}")

        if not all_heatmaps:
            self.status_label.setText(f"Error: Could not load valid frames for '{activity_name}'")
            return

        whole = self.build_summed_plot_data(all_heatmaps)

        safe_name = activity_name.replace("/", "_").replace("\\", "_")
        default_path = Path(__file__).parent / "Picture" / f"{safe_name}_all_sessions_whole_sum.png"
        default_title = f"{activity_name} (all sessions)"
        self.save_sum_plot(whole, default_path, "Save Whole-Activity Plot", default_title)
        self.status_label.setText(
            f"Saved whole activity plot for {activity_name} | "
            f"sessions: {len(matched_dirs)}, frames: {len(all_heatmaps)}"
        )

    def extract_activity_group_label(self, sample_dir):
        """Return scope-aware activity label, e.g., 'Single Activity Data | Sitting'."""
        try:
            rel_parts = sample_dir.relative_to(self.dataset_root).parts
        except Exception:
            rel_parts = sample_dir.parts

        def is_distance_like(name):
            txt = name.strip().lower().replace(" ", "")
            if txt.endswith("m"):
                txt = txt[:-1]
            if not txt:
                return False
            try:
                float(txt)
                return True
            except ValueError:
                return False

        def is_grouping_folder(name):
            txt = name.strip().lower()
            if not txt:
                return False
            # Social behavior often has middle folders like "2 People" or "3 people (update)"
            return "people" in txt

        scope = rel_parts[0].strip() if len(rel_parts) > 0 else self.dataset_root.name
        activity = None

        for part in reversed(rel_parts):
            p = part.strip()
            pl = p.lower()
            if not p:
                continue
            if pl.startswith("dataset"):
                continue
            if is_distance_like(p):
                continue
            if is_grouping_folder(p):
                continue
            if p == scope:
                continue
            activity = p
            break

        if not activity:
            activity = sample_dir.name

        return f"{scope} | {activity}", activity

    def build_summed_plot_data(self, heatmaps):
        """Create one summed playback-matched matrix from a list of frame heatmaps."""
        processed = []
        for frame in heatmaps:
            shifted = np.fft.fftshift(frame, axes=1)

            desired_rows = max(self.grid_size, shifted.shape[0])
            desired_cols = max(self.grid_size, shifted.shape[1])
            scale_y = desired_rows / shifted.shape[0]
            scale_x = desired_cols / shifted.shape[1]
            interp = zoom(shifted, (scale_y, scale_x), order=1)

            denom = max(self.max_value - self.min_value, 1e-9)
            norm = (interp - self.min_value) / denom
            processed.append(norm.astype(np.float32))

        return np.sum(np.stack(processed, axis=0), axis=0)

    def save_sum_plot(self, whole, default_path, dialog_title, default_title):
        """Prompt output path/title and save one summed heatmap figure."""
        save_path_str, _ = QFileDialog.getSaveFileName(
            self,
            dialog_title,
            str(default_path),
            "PNG Files (*.png);;All Files (*)",
        )
        if not save_path_str:
            self.status_label.setText("Export canceled")
            return False

        save_path = Path(save_path_str)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        title_text, ok = QInputDialog.getText(
            self,
            "Plot Title",
            "Enter plot title (leave blank for no title):",
            text=default_title,
        )
        if not ok:
            self.status_label.setText("Export canceled")
            return False

        fig, ax = plt.subplots(figsize=(10, 7))
        im = ax.imshow(
            whole.T,
            origin='lower',
            aspect='auto',
            cmap='jet',
            extent=[0.0, self.maximum_range, -self.unambiguous_velocity, self.unambiguous_velocity],
        )
        if title_text.strip():
            ax.set_title(title_text.strip())
        ax.set_xlabel('Range [m]')
        ax.set_ylabel('Doppler [m/s]')
        cb = fig.colorbar(im, ax=ax)
        cb.set_label('Summed Normalized Signal')
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)

        self.status_label.setText(f"Saved whole plot: {save_path}")
        return True


def main():
    app = QApplication(sys.argv)
    window = RangeDopplerPlayback()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()



