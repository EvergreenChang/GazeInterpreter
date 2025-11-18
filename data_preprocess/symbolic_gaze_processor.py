import numpy as np
import os
import json
import pandas as pd
import re
import math
from pathlib import Path
from tqdm import tqdm
from scipy.signal import find_peaks
from typing import Tuple, List, Dict, Optional

from projectaria_tools.core import data_provider, mps
from projectaria_tools.core.mps.utils import (
    filter_points_from_confidence,
    get_gaze_vector_reprojection,
    get_nearest_eye_gaze,
    get_nearest_pose,
)
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core.sensor_data import TimeDomain
from nymeria.data_provider import NymeriaDataProvider


class SymbolicGazeAnalyzer:
    """
    Analyzer that converts raw gaze data streams into structured symbolic event sequences
    """
    
    def __init__(
        self,
        fps: int = 30,
        fixation_max_velocity: float = 30.0,
        saccade_min_velocity: float = 100.0,
        smoothing_window: int = 3,
        min_fixation_duration: float = 0.3,
        min_pursuit_duration: float = 0.2,
        min_pursuit_distance: float = 3.0,
        min_saccade_amplitude: float = 2.0,
    ):
        """
        Initialize the analyzer
        
        Parameters:
            fps: sampling frame rate
            fixation_max_velocity: fixation maximum velocity threshold (°/s)
            saccade_min_velocity: saccade minimum velocity threshold (°/s)
        """
        self.fps = fps
        self.fixation_max_velocity = fixation_max_velocity
        self.saccade_min_velocity = saccade_min_velocity
        self.smoothing_window = max(1, smoothing_window)
        self.min_fixation_duration = max(0.0, min_fixation_duration)
        self.min_pursuit_duration = max(0.0, min_pursuit_duration)
        self.min_pursuit_distance = max(0.0, min_pursuit_distance)
        self.min_saccade_amplitude = max(0.0, min_saccade_amplitude)
        self.dt = 1.0 / fps  
        self._gaze_data: np.ndarray = np.empty((0, 2))
        self._timestamps: np.ndarray = np.empty((0,))
    
    def compute_angular_velocity(self, gaze_data: np.ndarray) -> np.ndarray:
        """
        Compute instantaneous angular velocity
        
        Parameters:
            gaze_data: array of shape (n, 2), containing yaw and pitch data
            
        Returns:
            angular velocity array, length n-1
        """
        if len(gaze_data) < 2:
            return np.array([])
        
        # Compute the difference between adjacent points
        yaw_rad = np.deg2rad(gaze_data[:, 0])
        pitch_rad = np.deg2rad(gaze_data[:, 1])

        diff_yaw = np.diff(np.unwrap(yaw_rad)) * 180.0 / np.pi
        diff_pitch = np.diff(np.unwrap(pitch_rad)) * 180.0 / np.pi

        # Compute angular velocity (°/s)
        angular_velocity = np.sqrt(diff_yaw**2 + diff_pitch**2) / self.dt

        return self._smooth_series(angular_velocity)
    
    def classify_gaze_points(self, angular_velocity: np.ndarray) -> List[str]:
        """
        Classify data points based on dual velocity thresholds
        
        Parameters:
            angular_velocity: angular velocity array
            
        Returns:
            classification label list
        """
        classifications = []

        for velocity in angular_velocity:
            if velocity <= self.fixation_max_velocity:
                classifications.append('Fixation')
            elif velocity <= self.saccade_min_velocity:
                classifications.append('SmoothPursuit')
            else:
                classifications.append('Saccade')

        return self._enforce_min_constraints(classifications)
    
    def merge_consecutive_events(self, classifications: List[str], 
                                gaze_data: np.ndarray, 
                                timestamps: np.ndarray) -> List[Dict]:
        """
        Merge consecutive points of the same type into events
        
        Parameters:
            classifications: classification label list
            gaze_data: gaze data array
            timestamps: timestamp array
            
        Returns:
            event list
        """
        if len(classifications) == 0:
            return []

        events = []
        current_type = classifications[0]
        current_start_idx = 0
        
        for i in range(1, len(classifications)):
            if classifications[i] != current_type:
                # Current event ends, create event
                event = {
                    'event_type': current_type,
                    'start_idx': current_start_idx,
                    'end_idx': i,
                    'start_time': timestamps[current_start_idx],
                    'end_time': timestamps[i],
                    'gaze_points': gaze_data[current_start_idx:i+1]
                }
                events.append(event)
                
                # Start new event
                current_type = classifications[i]
                current_start_idx = i
        
        # Process the last event
        event = {
            'event_type': current_type,
            'start_idx': current_start_idx,
            'end_idx': len(classifications),
            'start_time': timestamps[current_start_idx],
            'end_time': timestamps[-1],
            'gaze_points': gaze_data[current_start_idx:]
        }
        events.append(event)

        return self._post_process_events(events)

    def _smooth_series(self, values: np.ndarray) -> np.ndarray:
        """Simple smoothing of the angular velocity series"""
        if len(values) == 0 or self.smoothing_window <= 1:
            return values

        window = min(self.smoothing_window, len(values))
        kernel = np.ones(window) / window
        padded = np.pad(values, (window // 2, window - 1 - window // 2), mode='edge')
        smoothed = np.convolve(padded, kernel, mode='valid')
        return smoothed

    def _enforce_min_constraints(
        self,
        classifications: List[str],
    ) -> List[str]:
        """Adjust classifications based on minimum duration and amplitude constraints"""
        if not classifications:
            return classifications

        labels = classifications[:]
        changed = True

        while changed:
            changed = False
            i = 0
            while i < len(labels):
                j = i + 1
                while j < len(labels) and labels[j] == labels[i]:
                    j += 1

                label = labels[i]
                duration = self._segment_duration(i, j)

                if label == 'Fixation' and duration < self.min_fixation_duration:
                    replacement = self._choose_replacement(labels, i, j)
                    if replacement and replacement != label:
                        for k in range(i, j):
                            labels[k] = replacement
                        changed = True
                elif label == 'SmoothPursuit':
                    distance = self._segment_distance(i, j)
                    if duration < self.min_pursuit_duration or distance < self.min_pursuit_distance:
                        replacement = self._choose_replacement(labels, i, j)
                        if replacement and replacement != label:
                            for k in range(i, j):
                                labels[k] = replacement
                            changed = True
                elif label == 'Saccade':
                    amplitude = self._segment_amplitude(i, j)
                    if amplitude < self.min_saccade_amplitude:
                        replacement = self._choose_replacement(labels, i, j, default='Fixation')
                        if replacement and replacement != label:
                            for k in range(i, j):
                                labels[k] = replacement
                            changed = True

                i = j

        return labels

    def _segment_duration(self, start_idx: int, end_idx: int) -> float:
        """Compute the duration of a classification segment"""
        if self._timestamps.size == 0:
            return 0.0

        start_frame = min(start_idx, len(self._timestamps) - 1)
        end_frame = min(end_idx, len(self._timestamps) - 1)
        start_time = self._timestamps[start_frame]
        end_time = self._timestamps[end_frame]
        return float(max(0.0, end_time - start_time))

    def _segment_distance(self, start_idx: int, end_idx: int) -> float:
        """Compute the cumulative displacement of a segment"""
        if self._gaze_data.size == 0:
            return 0.0

        start_frame = min(start_idx, len(self._gaze_data) - 1)
        end_frame = min(end_idx, len(self._gaze_data) - 1)
        segment = self._gaze_data[start_frame:end_frame + 1]
        if len(segment) < 2:
            return 0.0
        diffs = np.diff(segment, axis=0)
        distances = np.sqrt(np.sum(diffs**2, axis=1))
        return float(np.sum(distances))

    def _segment_amplitude(self, start_idx: int, end_idx: int) -> float:
        """Compute the start and end amplitude of a segment"""
        if self._gaze_data.size == 0:
            return 0.0

        start_frame = min(start_idx, len(self._gaze_data) - 1)
        end_frame = min(end_idx, len(self._gaze_data) - 1)
        segment = self._gaze_data[start_frame:end_frame + 1]
        if len(segment) < 2:
            return 0.0
        delta = segment[-1] - segment[0]
        return float(np.sqrt(np.sum(delta**2)))

    def _choose_replacement(
        self,
        labels: List[str],
        start_idx: int,
        end_idx: int,
        default: Optional[str] = None,
    ) -> Optional[str]:
        """Select replacement label, prioritize neighbors with longer duration"""
        left_label = labels[start_idx - 1] if start_idx > 0 else None
        right_label = labels[end_idx] if end_idx < len(labels) else None

        if left_label and left_label == right_label:
            return left_label

        left_duration = -1.0
        right_duration = -1.0

        if left_label:
            l = start_idx - 1
            while l >= 0 and labels[l] == left_label:
                l -= 1
            left_duration = self._segment_duration(l + 1, start_idx)

        if right_label:
            r = end_idx
            while r < len(labels) and labels[r] == right_label:
                r += 1
            right_duration = self._segment_duration(end_idx, r)

        if left_duration >= right_duration and left_duration >= 0:
            return left_label
        if right_duration > left_duration and right_duration >= 0:
            return right_label

        return default

    def extract_event_features(self, events: List[Dict], 
                             angular_velocity: np.ndarray,
                             motion_duration: float) -> List[Dict]:
        """
        Extract and structure event features
        
        Parameters:
            events: event list
            angular_velocity: angular velocity array
            motion_duration: duration of the entire motion
            
        Returns:
            structured event feature list
        """
        structured_events = []
        
        for event in events:
            event_type = event['event_type']
            start_time = event['start_time']
            end_time = event['end_time']
            duration = end_time - start_time
            gaze_points = event['gaze_points']
            
            # Basic features
            structured_event = {
                'event_type': event_type,
                'start_time': float(start_time),
                'end_time': float(end_time),
                'duration': float(duration),
                'duration_label': self._get_duration_label(duration, motion_duration)
            }
            
            # Add specific features based on event type
            if event_type == 'Fixation':
                structured_event.update(self._extract_fixation_features(gaze_points))
            elif event_type == 'Saccade':
                # Get the angular velocity for this event
                start_idx = event['start_idx']
                end_idx = event['end_idx']
                event_velocities = angular_velocity[start_idx:end_idx]
                structured_event.update(self._extract_saccade_features(gaze_points, event_velocities))
            elif event_type == 'SmoothPursuit':
                start_idx = event['start_idx']
                end_idx = event['end_idx']
                event_velocities = angular_velocity[start_idx:end_idx]
                structured_event.update(self._extract_pursuit_features(gaze_points, event_velocities))
            
            structured_events.append(structured_event)
        
        return structured_events
    
    def _extract_fixation_features(self, gaze_points: np.ndarray) -> Dict:
        """Extract fixation event features"""
        centroid_yaw = np.mean(gaze_points[:, 0])
        centroid_pitch = np.mean(gaze_points[:, 1])
        
        return {
            'point_count': len(gaze_points),
            'centroid_coordinates': {
                'yaw': float(centroid_yaw),
                'pitch': float(centroid_pitch)
            },
            'centroid_coordinates_label': self._get_direction_label(centroid_yaw, centroid_pitch)
        }
    
    def _extract_saccade_features(self, gaze_points: np.ndarray, 
                                 event_velocities: np.ndarray) -> Dict:
        """Extract saccade event features"""
        start_coords = gaze_points[0]
        end_coords = gaze_points[-1]
        
        # Compute amplitude
        delta_yaw = end_coords[0] - start_coords[0]
        delta_pitch = end_coords[1] - start_coords[1]
        amplitude = np.sqrt(delta_yaw**2 + delta_pitch**2)
        
        # Compute peak velocity
        finite_velocities = event_velocities[np.isfinite(event_velocities)] if event_velocities.size > 0 else np.array([])
        peak_velocity: Optional[float]
        if finite_velocities.size > 0:
            peak_velocity = float(np.max(finite_velocities))
        else:
            peak_velocity = None

        return {
            'start_coordinates': {
                'yaw': float(start_coords[0]),
                'pitch': float(start_coords[1])
            },
            'end_coordinates': {
                'yaw': float(end_coords[0]),
                'pitch': float(end_coords[1])
            },
            'amplitude': float(amplitude),
            'amplitude_label': self._get_amplitude_label(amplitude),
            'peak_velocity': peak_velocity,
            'peak_velocity_label': self._get_saccade_velocity_label(peak_velocity),
            'direction_label': self._get_direction_label_from_vector(delta_yaw, delta_pitch)
        }
    
    def _extract_pursuit_features(self, gaze_points: np.ndarray, 
                                 event_velocities: np.ndarray) -> Dict:
        """Extract smooth pursuit event features"""
        # Build trajectory
        trajectory = [
            {'yaw': float(point[0]), 'pitch': float(point[1])}
            for point in gaze_points
        ]
        
        # Compute total distance
        travelled_distance = 0
        for i in range(1, len(gaze_points)):
            dist = np.sqrt((gaze_points[i][0] - gaze_points[i-1][0])**2 + 
                          (gaze_points[i][1] - gaze_points[i-1][1])**2)
            travelled_distance += dist
        
        # Compute average velocity
        finite_velocities = event_velocities[np.isfinite(event_velocities)] if event_velocities.size > 0 else np.array([])
        average_velocity: Optional[float]
        if finite_velocities.size > 0:
            average_velocity = float(np.mean(finite_velocities))
        else:
            average_velocity = None
        
        # Main direction
        start_coords = gaze_points[0]
        end_coords = gaze_points[-1]
        delta_yaw = end_coords[0] - start_coords[0]
        delta_pitch = end_coords[1] - start_coords[1]
        
        return {
            'point_count': len(gaze_points),
            'trajectory': trajectory,
            'travelled_distance': float(travelled_distance),
            'travelled_distance_label': self._get_distance_label(travelled_distance),
            'average_velocity': average_velocity,
            'average_velocity_label': self._get_pursuit_velocity_label(average_velocity),
            'main_direction_label': self._get_direction_label_from_vector(delta_yaw, delta_pitch)
        }

    def _post_process_events(self, events: List[Dict]) -> List[Dict]:
        """Consolidate event list, remove empty events"""
        consolidated: List[Dict] = []
        for event in events:
            start_time = event['start_time']
            end_time = event['end_time']
            if end_time <= start_time:
                continue
            consolidated.append(event)
        return consolidated
    
    def _get_duration_label(self, duration: float, motion_duration: float) -> str:
        """Get label based on duration ratio"""
        ratio = duration / motion_duration if motion_duration > 0 else 0
        if ratio < 0.2:
            return 'Brief'
        elif ratio < 0.5:
            return 'Short'
        elif ratio < 0.75:
            return 'Moderate'
        else:
            return 'Long'
    
    def _get_amplitude_label(self, amplitude: float) -> str:
        """Get label based on amplitude"""
        if amplitude < 5:
            return 'Small'
        elif amplitude < 15:
            return 'Medium'
        else:
            return 'Large'
    
    def _get_distance_label(self, distance: float) -> str:
        """Get label based on distance"""
        if distance < 5:
            return 'Small'
        elif distance < 15:
            return 'Medium'
        else:
            return 'Large'
    
    def _get_saccade_velocity_label(self, velocity: Optional[float]) -> str:
        """Get label based on saccade velocity"""
        if velocity is None or not math.isfinite(velocity):
            return 'Unknown'
        if velocity < 150:
            return 'Slow'
        elif velocity < 350:
            return 'Fast'
        else:
            return 'Very Fast'

    def _get_pursuit_velocity_label(self, velocity: Optional[float]) -> str:
        """Get label based on smooth pursuit velocity"""
        if velocity is None or not math.isfinite(velocity):
            return 'Unknown'
        if velocity < 40:
            return 'Slow'
        elif velocity < 70:
            return 'Steady'
        else:
            return 'Fast'
    
    def _get_direction_label(self, yaw: float, pitch: float) -> str:
        """Get direction label based on coordinates"""
        if abs(yaw) < 5 and abs(pitch) < 5:
            return 'Center'
        elif abs(yaw) < 5:
            return 'Up' if pitch > 0 else 'Down'
        elif abs(pitch) < 5:
            return 'Right' if yaw > 0 else 'Left'
        else:
            h_dir = 'Right' if yaw > 0 else 'Left'
            v_dir = 'Up' if pitch > 0 else 'Down'
            return f'{h_dir}{v_dir}'
    
    def _get_direction_label_from_vector(self, delta_yaw: float, delta_pitch: float) -> str:
        """Get label based on direction vector"""
        if abs(delta_yaw) < 1e-6 and abs(delta_pitch) < 1e-6:
            return 'Stationary'
        
        angle = math.atan2(delta_pitch, delta_yaw)
        angle_degrees = math.degrees(angle)
        
        # Standardize angle to 0-360 degrees
        if angle_degrees < 0:
            angle_degrees += 360
        
        # 8-direction mapping
        if angle_degrees < 22.5 or angle_degrees >= 337.5:
            return 'Right'
        elif angle_degrees < 67.5:
            return 'Up-Right'
        elif angle_degrees < 112.5:
            return 'Up'
        elif angle_degrees < 157.5:
            return 'Up-Left'
        elif angle_degrees < 202.5:
            return 'Left'
        elif angle_degrees < 247.5:
            return 'Down-Left'
        elif angle_degrees < 292.5:
            return 'Down'
        else:
            return 'Down-Right'
    
    def analyze_motion_segment(self, gaze_data: np.ndarray, 
                             timestamps: np.ndarray,
                             motion_info: Dict) -> Dict:
        """
        Analyze gaze data for a single motion segment
        
        Parameters:
            gaze_data: gaze data array (n, 2) [yaw, pitch]
            timestamps: timestamp array
            motion_info: motion information dictionary
            
        Returns:
            analysis result dictionary
        """
        gaze_array = np.asarray(gaze_data, dtype=float)
        timestamp_array = np.asarray(timestamps, dtype=float)

        if gaze_array.ndim != 2 or timestamp_array.ndim != 1:
            return self._create_empty_result(motion_info)

        finite_mask = np.isfinite(timestamp_array)
        if gaze_array.size > 0:
            finite_mask &= np.all(np.isfinite(gaze_array), axis=1)

        gaze_array = gaze_array[finite_mask]
        timestamp_array = timestamp_array[finite_mask]

        if len(gaze_array) < 2:
            return self._create_empty_result(motion_info)

        self._gaze_data = gaze_array
        self._timestamps = timestamp_array

        # 1. Compute angular velocity
        angular_velocity = self.compute_angular_velocity(gaze_data)
        
        # 2. Classify data points
        classifications = self.classify_gaze_points(angular_velocity)
        
        # 3. Merge consecutive events
        events = self.merge_consecutive_events(classifications, gaze_data, timestamps)
        
        # 4. Extract event features
        motion_duration = motion_info['end_time'] - motion_info['start_time']
        structured_events = self.extract_event_features(events, angular_velocity, motion_duration)
        
        # 5. Build result
        result = {
            'metadata': {
                'motion_id': motion_info['motion_id'],
                'start_time': motion_info['start_time'],
                'end_time': motion_info['end_time'],
                'body_posture': motion_info['body_posture'],
                'arm_motion': motion_info['arm_motion'],
                'leg_motion': motion_info['leg_motion'],
                'focus_attention': motion_info['focus_attention'],
                'parameters': {
                    'fps': self.fps,
                    'fixation_max_velocity': self.fixation_max_velocity,
                    'saccade_min_velocity': self.saccade_min_velocity,
                    'smoothing_window': self.smoothing_window,
                    'min_fixation_duration': self.min_fixation_duration,
                    'min_pursuit_duration': self.min_pursuit_duration,
                    'min_pursuit_distance': self.min_pursuit_distance,
                    'min_saccade_amplitude': self.min_saccade_amplitude,
                }
            },
            'gaze_events': structured_events
        }
        
        return result
    
    def _create_empty_result(self, motion_info: Dict) -> Dict:
        """Create empty analysis result"""
        return {
            'metadata': {
                'motion_id': motion_info['motion_id'],
                'start_time': motion_info['start_time'],
                'end_time': motion_info['end_time'],
                'body_posture': motion_info['body_posture'],
                'arm_motion': motion_info['arm_motion'],
                'leg_motion': motion_info['leg_motion'],
                'focus_attention': motion_info['focus_attention'],
                'parameters': {
                    'fps': self.fps,
                    'fixation_max_velocity': self.fixation_max_velocity,
                    'saccade_min_velocity': self.saccade_min_velocity,
                    'smoothing_window': self.smoothing_window,
                    'min_fixation_duration': self.min_fixation_duration,
                    'min_pursuit_duration': self.min_pursuit_duration,
                    'min_pursuit_distance': self.min_pursuit_distance,
                    'min_saccade_amplitude': self.min_saccade_amplitude,
                }
            },
            'gaze_events': []
        }


class SymbolicGazeDataProcessor:
    """
    Combined data processor that integrates preprocessing and symbolic gaze analysis
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the processor
        
        Parameters:
            config: configuration dictionary
        """
        self.config = config
        self.fps = config.get('sample_fps', 20)
        self.smoothing_window = config.get('smoothing_window', 3)
        self.fixation_max_velocity = config.get('fixation_max_velocity', 30.0)
        self.saccade_min_velocity = config.get('saccade_min_velocity', 80.0)
        self.min_fixation_duration = config.get('min_fixation_duration', 0.3)
        self.min_pursuit_duration = config.get('min_pursuit_duration', 0.2)
        self.min_pursuit_distance = config.get('min_pursuit_distance', 3.0)
        self.min_saccade_amplitude = config.get('min_saccade_amplitude', 2.0)

        # Initialize symbolic gaze analyzer
        self.gaze_analyzer = SymbolicGazeAnalyzer(
            fps=self.fps,
            fixation_max_velocity=self.fixation_max_velocity,
            saccade_min_velocity=self.saccade_min_velocity,
            smoothing_window=self.smoothing_window,
            min_fixation_duration=self.min_fixation_duration,
            min_pursuit_duration=self.min_pursuit_duration,
            min_pursuit_distance=self.min_pursuit_distance,
            min_saccade_amplitude=self.min_saccade_amplitude,
        )
    
    def find_nearest_timestamp_idx(self, timestamps, target):
        """Find the index of the nearest timestamp to the target timestamp"""
        return min(range(len(timestamps)), key=lambda i: abs(timestamps[i] - target))
    
    def process_all_data(self):
        """Process all data folders"""
        data_dir = self.config['data_dir']
        output_dir = self.config['output_dir']
        
        # Ensure output directories exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'preprocessed'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'symbolic_gaze'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'personalized_gaze'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'general_gaze'), exist_ok=True)
        
        # Get list of folders to skip
        skip_folders = self.config.get('skip_folders', [])
        
        # Iterate over all subfolders in the data directory
        for subfolder in tqdm(os.listdir(data_dir)):
            subfolder_path = os.path.join(data_dir, subfolder)
            
            # Skip specified folders
            if subfolder in skip_folders:
                print(f"Skipping folder: {subfolder}")
                continue
            
            # Check if output files exist
            preprocessed_output = os.path.join(output_dir, 'preprocessed', f"{subfolder}.npy")
            symbolic_output = os.path.join(output_dir, 'symbolic_gaze', f"{subfolder}.json")
            
            if os.path.exists(preprocessed_output) and os.path.exists(symbolic_output):
                print(f"Folder {subfolder} already processed, skipping...")
                continue
            
            if os.path.isdir(subfolder_path):
                try:
                    self.process_single_folder(subfolder, subfolder_path, output_dir)
                except Exception as e:
                    print(f"Error processing folder {subfolder}: {str(e)}")
                    continue
    
    def process_single_folder(self, subfolder_name: str, subfolder_path: str, output_dir: str):
        """Process a single data folder"""
        print(f"Processing folder: {subfolder_path}")
        
        # Load NymeriaDataProvider
        nymeria_dp = NymeriaDataProvider(sequence_rootdir=Path(subfolder_path), load_wrist=False)
        
        if nymeria_dp.recording_head is None:
            print(f"Warning: {subfolder_name} has no recording_head data, skipping...")
            return
        
        # Check if gaze data is available
        has_g_gaze = nymeria_dp.recording_head.has_general_gaze
        has_p_gaze = nymeria_dp.recording_head.has_personalized_gaze
        
        if not has_g_gaze and not has_p_gaze:
            print(f"Warning: {subfolder_name} has no gaze data, skipping...")
            return
        
        # Get device calibration information
        device_calibration = nymeria_dp.recording_head.vrs_dp.get_device_calibration()
        T_device_CPF = device_calibration.get_transform_device_cpf()
        
        # Get time range
        t_ns_start, t_ns_end = nymeria_dp.timespan_ns
        dt: int = int(1e9 / self.fps)
        
        # Initialize data storage
        personalized_gaze = {'pitch_yaw': [], 'CPF': [], 'World': []}
        general_gaze = {'pitch_yaw': [], 'CPF': [], 'World': []}
        head_direction = []
        body_pose = []
        T_world_devices = []
        total_seconds = []
        
        # Extract temporal data
        print("Extracting temporal data...")
        for idx, t_ns in tqdm(enumerate(range(t_ns_start, t_ns_end, dt))):
            try:
                # Get pose information
                aria_pose, tdiff = nymeria_dp.recording_head.get_pose(t_ns, time_domain=TimeDomain.TIME_CODE)
                total_seconds.append(aria_pose.tracking_timestamp.total_seconds())
                
                T_W_D = aria_pose.transform_world_device
                T_world_CPF = T_W_D @ T_device_CPF
                
                # Extract personalized gaze data
                if has_p_gaze:
                    p_gaze = nymeria_dp.get_personalized_gaze(t_ns)["recording_head"]
                    personalized_gaze['pitch_yaw'].append([p_gaze.pitch, p_gaze.yaw])
                    p_CPF = nymeria_dp.recording_head.get_gaze_in_CPF(p_gaze)
                    personalized_gaze['CPF'].append(p_CPF)
                    p_world = T_world_CPF @ p_CPF
                    personalized_gaze['World'].append(p_world.reshape(3,))
                
                # Extract general gaze data
                if has_g_gaze:
                    g_gaze = nymeria_dp.get_general_gaze(t_ns)["recording_head"]
                    general_gaze['pitch_yaw'].append([g_gaze.pitch, g_gaze.yaw])
                    g_CPF = nymeria_dp.recording_head.get_gaze_in_CPF(g_gaze)
                    general_gaze['CPF'].append(g_CPF)
                    g_world = T_world_CPF @ g_CPF
                    general_gaze['World'].append(g_world.reshape(3,))
                
                # Extract head direction
                head_direction.append((T_world_CPF @ np.array([0, 0, 0])).reshape(3,))
                
                # Extract body pose
                skel = nymeria_dp.get_synced_poses(t_ns)['xsens']
                body_pose.append(skel)
                T_world_devices.append(T_W_D.to_matrix())
                
            except Exception as e:
                print(f"Error processing timestamp {t_ns}: {str(e)}")
                continue
        
        # Get narration data
        narration_provider = nymeria_dp.narrations
        
        # Process atomic narration
        atomic = narration_provider.get_start_end_narration(type_of_narration='atomic')
        atomic_narration = self._process_narration_data(atomic, total_seconds, narration_type='atomic')
        
        # Process activity narration
        activity = narration_provider.get_start_end_narration(type_of_narration='activity')
        activity_narration = self._process_narration_data(activity, total_seconds, narration_type='activity')
        
        # Process motion narration
        motion = narration_provider.get_start_end_narration(type_of_narration='motion')
        motion_narration = self._process_narration_data(motion, total_seconds, narration_type='motion')
        
        # Save preprocessed data
        preprocessed_data = {
            'personalized_gaze': personalized_gaze,
            'general_gaze': general_gaze,
            'head_origins': head_direction,
            'body_pose': body_pose,
            'T_device_CPF': T_device_CPF.to_matrix(),
            'T_world_devices': T_world_devices,
            'atomic_narration': atomic_narration,
            'activity_narration': activity_narration,
            'motion_narration': motion_narration,
            'total_seconds': total_seconds
        }
        
        preprocessed_output_path = os.path.join(output_dir, 'preprocessed', f"{subfolder_name}.npy")
        np.save(preprocessed_output_path, preprocessed_data)
        print(f"Preprocessed data saved to: {preprocessed_output_path}")
        
        # Generate symbolic gaze data
        if motion is not None and (has_g_gaze or has_p_gaze):
            symbolic_gaze_results, segment_exports = self._generate_symbolic_gaze_data(
                preprocessed_data, motion, total_seconds
            )
            
            if symbolic_gaze_results:
                # Save symbolic gaze data
                symbolic_output_path = os.path.join(output_dir, 'symbolic_gaze', f"{subfolder_name}.json")
                with open(symbolic_output_path, 'w', encoding='utf-8') as f:
                    json.dump(symbolic_gaze_results, f, indent=2, ensure_ascii=False)
                print(f"Symbolic gaze data saved to: {symbolic_output_path}")
                print(f"Analyzed {len(symbolic_gaze_results)} motion segments")

                self._export_motion_gaze_segments(
                    segment_exports,
                    os.path.join(output_dir, 'personalized_gaze', subfolder_name),
                    os.path.join(output_dir, 'general_gaze', subfolder_name),
                )
        else:
            print(f"Warning: {subfolder_name} has no motion data or gaze data, skipping symbolic analysis...")
    
    def _process_narration_data(self, narration_data, total_seconds: List, narration_type: str):
        """Process narration data, convert timestamps to frame indices"""
        if narration_data is None:
            return None
        
        narration_data_np = narration_data.to_numpy()
        processed_narration = []
        
        for i in range(len(narration_data_np)):
            start_time = narration_data_np[i][0]
            end_time = narration_data_np[i][1]
            
            start_idx = self.find_nearest_timestamp_idx(total_seconds, start_time)
            end_idx = self.find_nearest_timestamp_idx(total_seconds, end_time)
            
            if narration_type == 'motion':
                # motion narration has 6 columns: start_time, end_time, body_posture, arm_motion, leg_motion, focus_attention
                body_narration = narration_data_np[i][2]
                arm_narration = narration_data_np[i][3]
                leg_narration = narration_data_np[i][4]
                focus_narration = narration_data_np[i][5]
                processed_narration.append([start_idx, end_idx, body_narration, arm_narration, leg_narration, focus_narration])
            else:
                # atomic and activity narration has 3 columns: start_time, end_time, description
                description = narration_data_np[i][2]
                processed_narration.append([start_idx, end_idx, description])
        
        return processed_narration
    
    def _generate_symbolic_gaze_data(
        self,
        preprocessed_data: Dict,
        motion_data,
        total_seconds: List,
    ) -> Tuple[List[Dict], List[Dict]]:
        """Generate symbolic gaze data"""
        print("Generating symbolic gaze data...")

        personalized_raw = np.array(preprocessed_data['personalized_gaze']['pitch_yaw'])
        general_raw = np.array(preprocessed_data['general_gaze']['pitch_yaw'])

        # Select the gaze data to use (prefer personalized_gaze)
        if personalized_raw.size > 0:
            gaze_data = np.rad2deg(personalized_raw)
        elif general_raw.size > 0:
            gaze_data = np.rad2deg(general_raw)
        else:
            print("Warning: no available gaze data")
            return [], []

        # Note: the column order of gaze_data is [pitch, yaw], need to adjust to [yaw, pitch]
        gaze_data = gaze_data[:, [1, 0]]  # Swap column order

        total_seconds_np = np.asarray(total_seconds, dtype=float)
        motion_data_np = motion_data.to_numpy()

        personalized_deg = np.rad2deg(personalized_raw) if personalized_raw.size > 0 else None
        general_deg = np.rad2deg(general_raw) if general_raw.size > 0 else None

        # Filter out invalid values in timestamps or gaze data
        time_mask = np.isfinite(total_seconds_np)
        if not np.all(time_mask):
            total_seconds_np = total_seconds_np[time_mask]
            gaze_data = gaze_data[time_mask]
            if personalized_deg is not None:
                personalized_deg = personalized_deg[time_mask]
            if general_deg is not None:
                general_deg = general_deg[time_mask]

        gaze_mask = np.all(np.isfinite(gaze_data), axis=1)
        if not np.all(gaze_mask):
            total_seconds_np = total_seconds_np[gaze_mask]
            gaze_data = gaze_data[gaze_mask]
            if personalized_deg is not None:
                personalized_deg = personalized_deg[gaze_mask]
            if general_deg is not None:
                general_deg = general_deg[gaze_mask]

        if len(gaze_data) == 0:
            print("Warning: after invalid value filtering, no available gaze samples")
            return [], []

        symbolic_results = []
        segment_exports: List[Dict] = []
        
        # Apply character replacement
        original_char = self.config.get('original_character', 'C')
        replacement = self.config.get('replacement_character', 'The human')
        
        for idx in range(len(motion_data_np)):
            start_time = motion_data_np[idx][0]
            end_time = motion_data_np[idx][1]
            body_posture = motion_data_np[idx][2]
            arm_motion = motion_data_np[idx][3]
            leg_motion = motion_data_np[idx][4]
            focus_attention = motion_data_np[idx][5]
            
            # Filter gaze data within the time range
            mask = (total_seconds_np >= start_time) & (total_seconds_np <= end_time)
            segment_gaze_data = gaze_data[mask]
            segment_timestamps = total_seconds_np[mask]

            if len(segment_gaze_data) == 0:
                print(f"Warning: motion ID {idx} (time range: {start_time}-{end_time}) has no corresponding gaze data")
                continue

            segment_personalized = None
            segment_general = None

            if personalized_deg is not None:
                segment_personalized = personalized_deg[mask][:, [0, 1]]
            if general_deg is not None:
                segment_general = general_deg[mask][:, [0, 1]]

            segment_valid_mask = np.isfinite(segment_timestamps)
            if segment_gaze_data.size > 0:
                segment_valid_mask &= np.all(np.isfinite(segment_gaze_data), axis=1)

            if segment_personalized is not None and len(segment_personalized) > 0:
                segment_valid_mask &= np.all(np.isfinite(segment_personalized), axis=1)

            segment_gaze_data = segment_gaze_data[segment_valid_mask]
            segment_timestamps = segment_timestamps[segment_valid_mask]

            if segment_personalized is not None:
                segment_personalized = segment_personalized[segment_valid_mask]
            if segment_general is not None and len(segment_general) > 0:
                segment_general = segment_general[segment_valid_mask]

            if len(segment_gaze_data) == 0 or len(segment_timestamps) == 0:
                print(f"Warning: motion ID {idx} after cleaning has no valid gaze data")
                continue

            # Apply character replacement
            if body_posture and isinstance(body_posture, str):
                body_posture = self._replace_character(body_posture, original_char, replacement)
            if arm_motion and isinstance(arm_motion, str):
                arm_motion = self._replace_character(arm_motion, original_char, replacement)
            if leg_motion and isinstance(leg_motion, str):
                leg_motion = self._replace_character(leg_motion, original_char, replacement)
            if focus_attention and isinstance(focus_attention, str):
                focus_attention = self._replace_character(focus_attention, original_char, replacement)
            
            # Build motion information
            motion_info = {
                'motion_id': idx,
                'start_time': float(start_time),
                'end_time': float(end_time),
                'body_posture': body_posture,
                'arm_motion': arm_motion,
                'leg_motion': leg_motion,
                'focus_attention': focus_attention
            }
            
            # Perform symbolic analysis
            symbolic_result = self.gaze_analyzer.analyze_motion_segment(
                segment_gaze_data, segment_timestamps, motion_info
            )

            symbolic_results.append(symbolic_result)

            export_personalized = self._prepare_export_series(segment_timestamps, segment_personalized)
            export_general = self._prepare_export_series(segment_timestamps, segment_general)

            segment_exports.append(
                {
                    'motion_id': idx,
                    'personalized': export_personalized,
                    'general': export_general,
                }
            )

        return symbolic_results, segment_exports

    def _export_motion_gaze_segments(
        self,
        segment_exports: List[Dict],
        personalized_dir: str,
        general_dir: str,
    ) -> None:
        """Save gaze sequence data for each motion"""
        os.makedirs(personalized_dir, exist_ok=True)
        os.makedirs(general_dir, exist_ok=True)

        for segment in segment_exports:
            motion_id = segment['motion_id']

            export_personalized = segment.get('personalized')
            if export_personalized is not None:
                p_times = np.asarray(export_personalized['timestamps'], dtype=np.float32).reshape(-1, 1)
                p_values = np.asarray(export_personalized['values'], dtype=np.float32)

                if p_times.shape[0] > 0 and p_times.shape[0] == p_values.shape[0]:
                    personalized_array = np.hstack((p_times, p_values))
                    personalized_path = os.path.join(
                        personalized_dir,
                        f"motion_id_{motion_id:04d}.npy",
                    )
                    np.save(personalized_path, personalized_array)

            export_general = segment.get('general')
            if export_general is not None:
                g_times = np.asarray(export_general['timestamps'], dtype=np.float32).reshape(-1, 1)
                g_values = np.asarray(export_general['values'], dtype=np.float32)

                if g_times.shape[0] > 0 and g_times.shape[0] == g_values.shape[0]:
                    general_array = np.hstack((g_times, g_values))
                    general_path = os.path.join(
                        general_dir,
                        f"motion_id_{motion_id:04d}.npy",
                    )
                    np.save(general_path, general_array)

    def _prepare_export_series(
        self,
        timestamps: np.ndarray,
        values: Optional[np.ndarray],
    ) -> Optional[Dict[str, np.ndarray]]:
        """Clean the exported gaze sequence, remove invalid values"""
        if values is None or len(values) == 0:
            return None

        timestamps_array = np.asarray(timestamps, dtype=float).reshape(-1)
        values_array = np.asarray(values, dtype=float)

        if timestamps_array.size == 0 or values_array.size == 0:
            return None

        min_len = min(timestamps_array.shape[0], values_array.shape[0])
        timestamps_array = timestamps_array[:min_len]
        values_array = values_array[:min_len]

        if values_array.ndim == 1:
            valid_mask = np.isfinite(timestamps_array) & np.isfinite(values_array)
        else:
            valid_mask = np.isfinite(timestamps_array) & np.all(np.isfinite(values_array), axis=1)

        if not np.any(valid_mask):
            return None

        return {
            'timestamps': timestamps_array[valid_mask],
            'values': values_array[valid_mask],
        }

    def _replace_character(self, text: str, original_char: str, replacement: str) -> str:
        """Use regular expression to replace character identifiers in text"""
        if not text:
            return text
        
        # Escape special characters (to prevent regular expression errors)
        escaped_char = re.escape(original_char)
        
        # Use regular expression to replace
        pattern = rf'\b{escaped_char}\b(?=\s|\'s|[,.!?;:]|$)'
        result = re.sub(pattern, replacement, text)
        
        return result


def create_default_config():
    """Create default configuration"""
    config = {
        # Input output paths
        'data_dir': "/Users/.../data/",
        'output_dir': "/Users/.../output_symbolic/",
        
        # Processing parameters
        'sample_fps': 20,  # Sampling frame rate

        # Symbolic analysis parameters
        'fixation_max_velocity': 30.0,  # Fixation maximum velocity threshold (°/s)
        'saccade_min_velocity': 80.0,  # Saccade minimum velocity threshold (°/s)
        'smoothing_window': 3,  # Angular velocity smoothing window (frames)
        'min_fixation_duration': 0.3,  # Fixation event minimum duration (seconds)
        'min_pursuit_duration': 0.2,  # Smooth pursuit event minimum duration (seconds)
        'min_pursuit_distance': 3.0,  # Smooth pursuit cumulative displacement threshold (°)
        'min_saccade_amplitude': 2.0,  # Saccade minimum amplitude (°)
        
        # Character replacement
        'original_character': 'C',
        'replacement_character': 'The human',
        
        # Folders to skip
        'skip_folders': [
            # Can add folders to skip here
        ]
    }
    return config


def main():
    """Main function"""
    # Create default configuration
    config = create_default_config()
 
    print("Configuration information:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Create processor instance
    processor = SymbolicGazeDataProcessor(config)
    
    # Start processing all data
    print("Processing data...")
    processor.process_all_data()
    print("All data processing completed!")


if __name__ == "__main__":
    main() 
