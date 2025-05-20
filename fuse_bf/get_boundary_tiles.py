from glob import glob
import os
import json
from tqdm import tqdm

def quick_get_extent(multi_polygon_coords):
    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')
    # Iterate through the coordinates to find the bounding box
    for polygon_coords in multi_polygon_coords:
        for ring in polygon_coords:  # Iterate through exterior and any interior rings
            for coord in ring:
                x, y = coord
                # Update min and max values
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
    return min_x, min_y, max_x, max_y
        

full_file = "city_boundaries.geojson"

with open(full_file, "r") as f:
    lines = f.readlines()

created_files = glob("boundary_tiles/*.geojson")
created_files = [os.path.basename(cf) for cf in created_files]

for line in tqdm(lines, desc="Tiling ..."):
    if "coordinates" not in line:
        continue
    if "\"COUNTRY\": \"Antarctica\"" in line:
        continue
    xmin, ymin, xmax, ymax = quick_get_extent(json.loads(line.rstrip().rstrip(','))["geometry"]["coordinates"])

    long_min = int(xmin // 5 * 5)
    long_max = int(xmax // 5 * 5 + 5)
    lat_min = int(ymin // 5 * 5)
    lat_max = int(ymax // 5 * 5 + 5)

    # long_min = int(min_x // 5 * 5)
    # long_max = max(int(max_x // 5 * 5), int(long_min + 5))
    # lat_min = int(min_y // 5 * 5)
    # lat_max = max(int(max_y // 5 * 5), int(lat_min + 5))
    for x1 in range(long_min, long_max, 5):
        x2 = x1 + 5
        x1_flag = "w" if x1 < 0 else "e"
        x2_flag = "w" if x2 < 0 else "e"
        x1 = abs(x1)
        x2 = abs(x2)

        for y1 in range(lat_min, lat_max, 5):
            y2 = y1 + 5
            y1_flag = "s" if y1 < 0 else "n"
            y2_flag = "s" if y2 < 0 else "n"
            y1 = abs(y1)
            y2 = abs(y2)

            filename = "boundary_tiles/" + f"GUF04_DLR_v02_{x1_flag}{x1:03d}_{y2_flag}{y2:02d}_{x2_flag}{x2:03d}_{y1_flag}{y1:02d}_OGR04.geojson"
    
            if filename in created_files:
                with open(filename, "a") as f:
                    f.writelines(line)
            else:
                # Define the GeoJSON structure with the dynamic name
                geojson_data = {
                    "type": "FeatureCollection",
                    "name": filename.split(".")[0],
                    "crs": {
                        "type": "name",
                        "properties": {
                            "name": "urn:ogc:def:crs:OGC:1.3:CRS84"
                        }
                    },
                    "features": []
                }
                geojson_string = json.dumps(geojson_data, indent=0)
                geojson_lines = geojson_string.splitlines()
                with open(filename, "w") as f:
                    for geojson_line in geojson_lines[:-2]:
                        f.writelines(geojson_line+"\n")
                    f.writelines("\"features\":[\n" )
                    f.writelines(line)
                created_files.append(filename)

for created_file in created_files:
    with open(created_file, "r") as f:
        contents = f.read().rstrip()

    if contents.endswith(","):
        contents = contents.rstrip(",")
    
    with open(created_file, "w") as f:
        f.write(contents)
        f.write("\n")
        f.writelines("]\n")
        f.writelines("}\n")