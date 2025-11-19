import sys
import pandas as pd
import numpy as np
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QWidget, QPushButton, QLabel, QComboBox, QSlider, QSpinBox)
from PyQt5.QtCore import QTimer, Qt
import pyqtgraph as pg
from matplotlib import cm
from scipy.ndimage import zoom


class RangeDopplerPlayback(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Range-Doppler Heatmap Playback - Dataset 2")
        self.setGeometry(100, 100, 1200, 800)

        # Data variables
        self.current_activity = None
        self.heatmaps = []
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

        # Dataset path
        self.dataset_root = Path(__file__).parent / "Social Exp Data\Dataset 2 (RD)\2 People" 


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

        # Activity selector
        control_layout.addWidget(QLabel("Activity:"))
        self.activity_combo = QComboBox()
        self.activity_combo.setMinimumWidth(120)
        self.activity_combo.currentTextChanged.connect(self.on_activity_changed)
        control_layout.addWidget(self.activity_combo)

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
            self.colorbar = pg.ColorBarItem(
                values=(0, 4096),
                colorMap='jet',
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
        """Load available activities from dataset"""
        if not self.dataset_root.exists():
            self.status_label.setText(f"Error: Dataset path not found: {self.dataset_root}")
            return

        activities = []
        for activity_dir in sorted(self.dataset_root.iterdir()):
            if activity_dir.is_dir():
                rd_file = activity_dir / "range_doppler.csv"
                if rd_file.exists():
                    activities.append(activity_dir.name)

        if activities:
            self.activity_combo.addItems(activities)
            self.status_label.setText(f"Found {len(activities)} activities")
        else:
            self.status_label.setText("No activities with range_doppler.csv found")

    def on_activity_changed(self, activity_name):
        """Handle activity selection change"""
        if not activity_name:
            return

        self.stop_playback()
        self.current_activity = activity_name
        self.load_range_doppler_data(activity_name)

    def load_range_doppler_data(self, activity_name):
        """Load range-Doppler data from CSV file"""
        csv_path = self.dataset_root / activity_name / "range_doppler.csv"

        if not csv_path.exists():
            self.status_label.setText(f"Error: File not found: {csv_path}")
            return

        self.status_label.setText(f"Loading {activity_name}...")

        try:
            # Read CSV file
            df = pd.read_csv(csv_path)

            # Check if it's long-form data (frame_number, range_bin, doppler_bin, signal_strength)
            if all(col in df.columns for col in ['frame_number', 'range_bin', 'doppler_bin', 'signal_strength']):
                self.heatmaps = self.parse_longform_data(df)
            else:
                self.status_label.setText("Error: Unexpected CSV format")
                return

            if self.heatmaps:
                self.current_frame = 0
                self.frame_slider.setMaximum(len(self.heatmaps) - 1)
                self.frame_slider.setValue(0)
                self.update_display()
                self.status_label.setText(f"Loaded {len(self.heatmaps)} frames from {activity_name}")
            else:
                self.status_label.setText("Error: No heatmaps loaded")

        except Exception as e:
            self.status_label.setText(f"Error loading data: {str(e)}")
            import traceback
            traceback.print_exc()

    def parse_longform_data(self, df):
        """Parse long-form DataFrame into list of 2D heatmaps"""
        heatmaps = []

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

        return heatmaps

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
        try:
            cmap = cm.get_cmap('jet')
        except AttributeError:
            # For newer matplotlib versions
            cmap = cm.colormaps.get_cmap('jet')
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


def main():
    app = QApplication(sys.argv)
    window = RangeDopplerPlayback()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
