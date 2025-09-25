
# Ocean Wave

This project generates a pixel-art style tidal animation video based on Hong Kong Observatory tide data, synchronized with music rhythm.

## Main Files
- `OceanWaveFinal.py`: Main script to generate the pixel-art tidal animation video
- `led_counter-7.ttf`: Custom pixel font for overlay text
- `tide_extreme_2025.csv`, `tide_hourly_2025.csv`: Tide data files (from Hong Kong Observatory)
- `setup_py311_env.sh`: Shell script to set up a Python 3.11 virtual environment and install dependencies
- `requirements.txt`: Python dependencies list (must be present)
- `OceanWave-DecaJoins.mp3`: Audio file for background music

## Quick Start

1. Install Python 3.11 (on macOS: `brew install python@3.11`)
2. Set up the virtual environment:
   ```sh
   zsh setup_py311_env.sh
   source .venv311/bin/activate
   ```
3. Install dependencies (if not done by the script):
   ```sh
   pip install -r requirements.txt
   ```
4. Run the main script:
   ```sh
   python OceanWaveFinal.py
   ```
5. The output video will be saved as `tide_pixel_art.mp4` in the project directory.

## Dependencies
- pandas
- numpy
- pillow
- librosa
- soundfile
- ffmpeg (must be installed on your system)

## Notes
- The audio file `OceanWave-DecaJoins.mp3` is included in the project.
- Generated frame images will be saved in the `frames_output/` folder.
- The final video is named `tide_pixel_art.mp4`.
- The text "Ocean Save" will appear in the top-right corner at 27 seconds.

## Data Description
- `tide_extreme_2025.csv` and `tide_hourly_2025.csv` are official tide data files from the Hong Kong Observatory, used for animation generation.

## Reproducibility
To reproduce the project after cloning from GitHub:
1. Ensure all files listed above are present in the project directory.
2. Follow the Quick Start steps.

## License & Credits
- Tide data: © Hong Kong Observatory
- Font: `led_counter-7.ttf` (see font license if applicable)
- Music: Audio file `OceanWave-DecaJoins.mp3` included in project

---

For further assistance, please open an issue or contact the maintainer.
cd /Users/a1/Desktop/polyu/5913/AS2