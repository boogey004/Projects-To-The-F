"""
Extract extra public climate / environment features for every Train+Test record
from Google Earth Engine (all sources are public, as the competition allows).

UNTESTED -- written without internet access. If a dataset id or band name errors,
check it in the GEE catalogue (https://developers.google.com/earth-engine/datasets).

SETUP (once):
  pip install earthengine-api pandas
  python -c "import ee; ee.Authenticate()"        # sign in with a Google account
  Set PROJECT below to your Google Cloud project id with Earth Engine enabled.

RUN:   python gee_extract_features.py   (from the folder with Train.csv / Test.csv)
OUT:   files 'climate_extra_batch_N.csv' appear in your Google Drive folder
       'zindi_climate'. Download them, concatenate, and upload the result here.
"""
try:
    import ee
except ImportError:
    ee = None

import pandas as pd

if ee is None:
    raise ImportError("Install the Earth Engine client library before running this script: pip install earthengine-api pandas")

PROJECT = "YOUR_GOOGLE_CLOUD_PROJECT_ID"
BATCH = 400                       # records per export task (reduce if tasks time out)
WINDOWS = [3, 7, 14, 30, 60, 90, 180]   # days before the death date

ee.Initialize(project=PROJECT)

tr = pd.read_csv("Train.csv")
te = pd.read_csv("Test.csv")
pts = pd.concat([tr[["ID", "latitude", "longitude", "deathdate"]],
                 te[["ID", "latitude", "longitude", "deathdate"]]], ignore_index=True)

# ---------------- datasets ----------------
chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").select("precipitation")
era = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
t_mean = era.select("temperature_2m")
t_max = era.select("temperature_2m_max")
t_min = era.select("temperature_2m_min")
dew = era.select("dewpoint_temperature_2m")
soil = era.select("volumetric_soil_water_layer_1")
ndvi = ee.ImageCollection("MODIS/061/MOD13Q1").select(["NDVI", "EVI"])
lst = ee.ImageCollection("MODIS/061/MOD11A1").select("LST_Day_1km")
pop = ee.ImageCollection("WorldPop/GP/100m/pop").mosaic().rename("pop_density")
water = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").unmask(0)
wc = ee.ImageCollection("ESA/WorldCover/v200").first()
landcover = ee.Image.cat([
    wc.eq(10).rename("lc_trees"), wc.eq(40).rename("lc_crop"), wc.eq(50).rename("lc_built"),
    wc.eq(80).rename("lc_water"), wc.eq(90).rename("lc_wetland")])
access = ee.Image("Oxford/MAP/accessibility_to_cities_2015_v1_0").select("accessibility") \
    .rename("travel_min_to_city")
nightlight = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG") \
    .filterDate("2014-01-01", "2023-12-31").select("avg_rad").mean().rename("night_light")
srtm = ee.Image("USGS/SRTMGL1_003").select("elevation")
gaul = ee.FeatureCollection("FAO/GAUL/2015/level2")          # district boundaries


def add_features(f):
    d = ee.Date(f.get("date"))
    end = d.advance(1, "day")                      # include the death day
    g = f.geometry()
    bands = []
    for w in WINDOWS:
        s = d.advance(-w + 1, "day")
        r = chirps.filterDate(s, end)
        bands += [
            r.sum().rename(f"x_rain_sum_{w}d"),
            r.max().rename(f"x_rain_max_{w}d"),
            r.map(lambda i: i.gt(1)).sum().rename(f"x_rain_days_{w}d"),
            t_mean.filterDate(s, end).mean().subtract(273.15).rename(f"x_tmean_{w}d"),
            t_max.filterDate(s, end).max().subtract(273.15).rename(f"x_tmax_{w}d"),
            t_min.filterDate(s, end).min().subtract(273.15).rename(f"x_tmin_{w}d"),
            dew.filterDate(s, end).mean().subtract(273.15).rename(f"x_dewpt_{w}d"),
            soil.filterDate(s, end).mean().rename(f"x_soilw_{w}d"),
        ]
    for w in [30, 90, 180]:
        s = d.advance(-w + 1, "day")
        bands += [
            ndvi.filterDate(s, end).select("NDVI").mean().multiply(0.0001).rename(f"x_ndvi_{w}d"),
            ndvi.filterDate(s, end).select("EVI").mean().multiply(0.0001).rename(f"x_evi_{w}d"),
        ]
    s = d.advance(-29, "day")
    bands.append(lst.filterDate(s, end).mean().multiply(0.02).subtract(273.15).rename("x_lst_day_30d"))
    img = ee.Image.cat(bands)
    vals = img.reduceRegion(ee.Reducer.first(), g, 1000)
    near = ee.Image.cat([pop, access, nightlight, landcover]).reduceRegion(
        ee.Reducer.mean(), g.buffer(1000), 100)                      # ~1 km surroundings
    w5 = water.rename("water_occ_5km").reduceRegion(ee.Reducer.mean(), g.buffer(5000), 200)
    w20 = water.rename("water_occ_20km").reduceRegion(ee.Reducer.mean(), g.buffer(20000), 500)
    relief = srtm.rename("elev").reduceRegion(
        ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True), g.buffer(5000), 90)
    dist = gaul.filterBounds(g)
    admin = ee.Dictionary({"district": dist.aggregate_first("ADM2_NAME"),
                           "region": dist.aggregate_first("ADM1_NAME")})
    return f.set(vals).set(near).set(w5).set(w20).set(relief).set(admin)


records = [
    ee.Feature(ee.Geometry.Point([float(r.longitude), float(r.latitude)]),
               {"ID": r.ID, "date": r.deathdate})
    for r in pts.itertuples()
]

n_batches = (len(records) + BATCH - 1) // BATCH
for b in range(n_batches):
    fc = ee.FeatureCollection(records[b * BATCH:(b + 1) * BATCH]).map(add_features)
    task = ee.batch.Export.table.toDrive(
        collection=fc,
        description=f"climate_extra_batch_{b}",
        folder="zindi_climate",
        fileNamePrefix=f"climate_extra_batch_{b}",
        fileFormat="CSV",
    )
    task.start()
    print(f"started export {b + 1}/{n_batches}")
print("Monitor progress at https://code.earthengine.google.com/tasks")
