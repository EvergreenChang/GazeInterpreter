from symbolic_gaze_processor import SymbolicGazeDataProcessor, create_default_config

def main():
    """Example: How to use the symbolic gaze data processor"""
    
    # 1. Create configuration
    config = create_default_config()
    
    # 2. Custom configuration (optional)
    config.update({
        'data_dir': "./data/",
        'output_dir': "./nymeria_symbolic_output/",
        'sample_fps': 20,
        'fixation_max_velocity': 30.0,  # fixation maximum velocity threshold (°/s)
        'saccade_min_velocity': 80.0,  # saccade minimum velocity threshold (°/s)
        'smoothing_window': 3,  # smoothing window for angular velocity averaging (frames)
        'min_fixation_duration': 0.3,  # minimum duration for a fixation event (seconds)
        'min_pursuit_duration': 0.2,  # minimum duration for smooth pursuit (seconds)
        'min_pursuit_distance': 3.0,  # cumulative displacement threshold for smooth pursuit (°)
        'min_saccade_amplitude': 2.0,  # minimum amplitude for saccade (°)
        'original_character': 'C',  # original character
        'replacement_character': 'The human',  # replacement character
        'skip_folders': []  # list of scene directories to skip
    })
    
    print("Configuration information:")
    print(f"  Data directory: {config['data_dir']}")
    print(f"  Output directory: {config['output_dir']}")
    print(f"  Sampling frequency: {config['sample_fps']} fps")
    print(f"  Fixation maximum velocity: {config['fixation_max_velocity']} °/s")
    print(f"  Saccade minimum velocity: {config['saccade_min_velocity']} °/s")
    print(f"  Smoothing window: {config['smoothing_window']} frames")
    print(f"  Minimum fixation duration: {config['min_fixation_duration']} seconds")
    print(f"  Minimum pursuit duration: {config['min_pursuit_duration']} seconds")
    print(f"  Minimum pursuit distance: {config['min_pursuit_distance']} °")
    print(f"  Minimum saccade amplitude: {config['min_saccade_amplitude']} °")
    print()
    
    # 3. Create processor
    processor = SymbolicGazeDataProcessor(config)
    
    # 4. Process data
    print("Processing data...")
    processor.process_all_data()
    print("Data processing completed!")
    
    print("\nOutput file description:")
    print(f"  Preprocessed data (npy): {config['output_dir']}/preprocessed/")
    print(f"  Symbolic gaze data (json): {config['output_dir']}/symbolic_gaze/")
    print(f"  Personalized gaze sequence (npy): {config['output_dir']}/personalized_gaze/")
    print(f"  General gaze sequence (npy): {config['output_dir']}/general_gaze/")

if __name__ == "__main__":
    main() 


## Configuration parameters
"""
Parameter Description:

data_dir: Root directory of Nymeria raw data.
    Default value: /Users/.../data/

output_dir: Output directory for results.
    Default value: /Users/.../output_symbolic/

sample_fps: Gaze sampling frequency (Hz).
    Default value: 20

fixation_max_velocity: Maximum angular velocity threshold for fixation (°/s).
    Default value: 30.0

saccade_min_velocity: Minimum angular velocity threshold for saccade (°/s).
    Default value: 80.0

smoothing_window: Smoothing window for angular velocity averaging (frames).
    Default value: 3

min_fixation_duration: Minimum duration for a fixation event (seconds).
    Default value: 0.3

min_pursuit_duration: Minimum duration for smooth pursuit (seconds).
    Default value: 0.2

min_pursuit_distance: Cumulative displacement threshold for smooth pursuit (°).
    Default value: 3.0

min_saccade_amplitude: Minimum amplitude for saccade (°).
    Default value: 2.0

original_character / replacement_character: Character replacement in narrative text.
    Default value: C / The human

skip_folders: List of scene directories to skip.
    Default value: []
"""
