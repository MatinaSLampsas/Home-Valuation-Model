import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

## import house data: 
csv_file_path = 'data/all_repeated_sales.csv'
data = pd.read_csv(csv_file_path)
print(data.head())

# Define the function to calculate distance based on coordinates - 
# ONLY USED FOR CP SLO AND DOWNTOWN DIST
def haversine(lat1, lon1, lat2, lon2):
    # Radius of the Earth in kilometers
    R = 6371.0
    # Convert coordinates from degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance

"""
DIST CAL POLY AND DOWNTOWN
"""

# Coordinates for CP SLO AND DOWNTOWN 
downtown_slo_coords = (35.282752, -120.659616)
calpoly_coords = (35.300399, -120.662362)

data['dist_downtown'] = data.apply(lambda row: haversine(row['Latitude'], row['Longitude'], downtown_slo_coords[0], downtown_slo_coords[1]), axis=1)
data['dist_calpoly'] = data.apply(lambda row: haversine(row['Latitude'], row['Longitude'], calpoly_coords[0], calpoly_coords[1]), axis=1)

"""
DIST FOR COAST
"""

# Create a GeoDataFrame from the CSV data
geometry = [Point(xy) for xy in zip(data['Longitude'], data['Latitude'])]
homes_gdf = gpd.GeoDataFrame(data, geometry=geometry)

# Load the shapefile - for the coast
shapefile_path = 'CCal/Cencal_1998_2002.shp'  # Update with the correct path
coastline = gpd.read_file(shapefile_path)

 #Make sure both GeoDataFrames use the same coordinate reference system (CRS)
homes_gdf = homes_gdf.set_crs(epsg=4326)  # Assuming your CSV data uses WGS 84
coastline = coastline.to_crs(epsg=4326)
# Change CRS with meters (e.g., EPSG:3857 or UTM)
homes_gdf = homes_gdf.to_crs(epsg=3857)  # Web Mercator, distances in meters
coastline = coastline.to_crs(epsg=3857)

# Calculate the distance from each house to the coastline - divide by 1000 to convert to kilometers
homes_gdf['dist_to_coast'] = homes_gdf.geometry.apply(lambda house: coastline.distance(house).min())/1000

"""
DIST TO HIGH, MIDDLE, ELEMENTARY SCHOOL
"""
school_shapefile_path = 'Schools/SchoolSites2324.shp'
schools_gdf = gpd.read_file(school_shapefile_path)
#print(schools_gdf.head())

"""
HIGHSCHOOL
"""

## Isolate only the highschools from the data
highschools_gdf = schools_gdf[schools_gdf['SchoolType'] == 'High']

# Reproject both GeoDataFrames to a CRS with meters for distance calc
homes_gdf = homes_gdf.to_crs(epsg=3857)
highschools_gdf = highschools_gdf.to_crs(epsg=3857)

# Define a function to calculate the minimum distance to the nearest high school
def distance_to_nearest_highschool(house_geometry):
    distances = highschools_gdf.geometry.distance(house_geometry)
    return distances.min() if not distances.empty else float('inf')

# Apply the function to each house in homes_gdf
homes_gdf['dist_closest_highschool'] = homes_gdf.geometry.apply(distance_to_nearest_highschool)/1000

print(homes_gdf.head())

"""
MIDDLESCHOOL

basically doing the same thing as the highschool but with all the middleschool variables
"""

middleschools_gdf = schools_gdf[schools_gdf['SchoolType'] == 'Middle']
# Set CRS for both GeoDataFrames if not already set
if homes_gdf.crs is None:
    homes_gdf = homes_gdf.set_crs(epsg=4326)
if middleschools_gdf.crs is None:
    middleschools_gdf = middleschools_gdf.set_crs(epsg=4326)

# Reproject both GeoDataFrames to a CRS with meters (e.g., EPSG:3857) for distance calculation
homes_gdf = homes_gdf.to_crs(epsg=3857)
middleschools_gdf = middleschools_gdf.to_crs(epsg=3857)

# Create spatial index for high schools
middleschools_gdf.sindex

middleschools_gdf = middleschools_gdf[middleschools_gdf.geometry.notnull()]
# Define a function to calculate the minimum distance to the nearest high school
def distance_to_nearest_highschool(house_geometry):
    distances = middleschools_gdf.geometry.distance(house_geometry)
    return distances.min() if not distances.empty else float('inf')

# Apply the function to each house in homes_gdf
homes_gdf['dist_closest_middleschool'] = homes_gdf.geometry.apply(distance_to_nearest_highschool)/1000

print(homes_gdf.head())


"""
ELEMENTARY
"""

elementary_gdf = schools_gdf[schools_gdf['SchoolType'] == 'Elementary']
# Set CRS for both GeoDataFrames if not already set
if homes_gdf.crs is None:
    homes_gdf = homes_gdf.set_crs(epsg=4326)
if elementary_gdf.crs is None:
    elementary_gdf = elementary_gdf.set_crs(epsg=4326)

# Reproject both GeoDataFrames to a CRS with meters (e.g., EPSG:3857) for distance calculation
homes_gdf = homes_gdf.to_crs(epsg=3857)
elementary_gdf = elementary_gdf.to_crs(epsg=3857)

# Create spatial index for high schools
elementary_gdf.sindex

middleschools_gdf = middleschools_gdf[middleschools_gdf.geometry.notnull()]
# Define a function to calculate the minimum distance to the nearest high school
def distance_to_nearest_highschool(house_geometry):
    distances = elementary_gdf.geometry.distance(house_geometry)
    return distances.min() if not distances.empty else float('inf')

# Apply the function to each house in homes_gdf
homes_gdf['dist_closest_elementary'] = homes_gdf.geometry.apply(distance_to_nearest_highschool)/1000
