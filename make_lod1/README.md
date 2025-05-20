# Global LoD-1 Building Model Generation

## Overview

This project generates Level of Detail 1 (LoD-1) 3D building models by combining fused building footprint data with inferred height maps. The output includes building instances annotated with estimated heights and associated uncertainties.

## Usage

### 1. Preprocessing

Ensure you have:
- All fused building footprint GeoJSON files ready.
- Inferred height maps stored under a directory, referred to as `height_result_dir`.

### 2. Prepare Input List

Create a plain text file that lists the full paths to all the input GeoJSON files you wish to process. Add one file path per line.

### 3. Run the Script

Execute the main processing script with the following command:

```bash
python main.py --input_file /path/to/input_list.txt --processes 0
```
- `--input_file`: Path to the text file containing the list of GeoJSON inputs.
- `--processes`: Number of parallel processes to use. Set to 0 to utilize all available GPU cores.

## Processing Details
Each fused building footprint tile is spatially cropped to match the extent of corresponding 0.2° × 0.2° predicted height map tiles. 

For each building instance:
The maximum height value from the height map within the footprint is extracted and assigned, together with the estimated variance as an uncertainty measure.
