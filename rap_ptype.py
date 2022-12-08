import numpy as np
import pygrib as pg
import matplotlib
import sys
from osgeo import gdal,osr
import json
import tensorflow as tf
import joblib
import sys
sys.path.insert(1,'./')
from tiles import *

# Inteprolation function
def interp(var,hgts):
    varinterp = np.interp(height_levels,hgts[::-1],var[::-1])
    return varinterp


# Tile generation function
colordict = {0:(0,0,0),1:(255,164,249),2:(234,47,47),3:(46,143,64),4:(0,178,255)}
def tilemaker(data,modeltype,colordict):
        rgbarr = np.zeros((3,shape0,shape1))
        for (key,val) in colordict.items():
                inds = np.where(data==key)
                ind0 = inds[0]
                ind1 = inds[1]
                rgbarr[0,ind0,ind1] = val[0]
                rgbarr[1,ind0,ind1] = val[1]
                rgbarr[2,ind0,ind1] = val[2]
        bands = rgbarr.shape[0]
        rows = rgbarr.shape[1]
        cols = rgbarr.shape[2]
        originX = leftmostlon
        originY = bottomlat
        gridspace = delta
        driver = gdal.GetDriverByName("GTiff")
        options = ['PHOTOMETRIC=RGB', 'PROFILE=GeoTIFF']
        outRaster = driver.Create('./app/static/outimages/rap.tif',cols,rows,bands,gdal.GDT_Byte,options=options)
        outRaster.SetGeoTransform([originX,gridspace,0,originY,0,gridspace])
        srs = osr.SpatialReference()
        srs.SetWellKnownGeogCS("WGS84")
        outRaster.SetProjection(srs.ExportToWkt())

        for band in range(bands):
                outRaster.GetRasterBand(band+1).WriteArray(rgbarr[band,:,:])
        del outRaster
        main(['./app/static/outimages/rap.tif' , './app/static/outimages/%s_tiles/' % (modeltype),'--zoom=4-7','--srcnodata=0,0,0','-x','--resampling=near','-v'])

# Load standard scaler and CNN model 
scaler = joblib.load("./app/static/height_scaler.save")
mymodel = tf.keras.models.load_model("./app/static/height_model.h5")

# Read interpolated RAP or HRRR file (need to be on regular grid)
filename = "./app/static/rap_regrid.grib2"
grbs = pg.open(filename)
shape0 = grbs[1].values.shape[0]
shape1 = grbs[1].values.shape[1]

lats,lons = grbs[1].latlons()
lons = lons - 360.
leftmostlon = np.around(lons[0,0],decimals=3)
bottomlat = np.around(lats[0,0],decimals=3) 
delta = np.around(lats[1,0] - lats[0,0],3)
print(bottomlat,leftmostlon,delta)
tempgrbs = grbs.select(name="Temperature",typeOfLevel="isobaricInhPa")[2:-1]
dewgrbs = grbs.select(name="Dew point temperature",typeOfLevel="isobaricInhPa")[2:-1]
uwndgrbs = grbs.select(name="U component of wind",typeOfLevel="isobaricInhPa")[2:-1]
vwndgrbs = grbs.select(name="V component of wind",typeOfLevel="isobaricInhPa")[2:-1]
geogrbs = grbs.select(name="Geopotential height",typeOfLevel="isobaricInhPa")[2:-1]
sfchgt = np.array(grbs[608].values)

# Combine categorical p-types into one array
ptypes = np.zeros((shape0,shape1))
crain = np.array(grbs[641].values)
csnow = np.array(grbs[638].values)
cicep = np.array(grbs[639].values)
cfrzr = np.array(grbs[640].values)
print(cfrzr.shape)
ptypes[cfrzr==1] = 1
ptypes[cicep==1] = 2
ptypes[crain==1] = 3
ptypes[csnow==1] = 4

temps = np.zeros((shape0,shape1,37))
dews = np.zeros((shape0,shape1,37))
uwnds = np.zeros((shape0,shape1,37))
vwnds = np.zeros((shape0,shape1,37))
agls = np.zeros((shape0,shape1,37))

pres_levels = np.arange(100,1025,25)
height_levels = np.arange(0, 16750, 250)
print("here")
print(len(tempgrbs))
print(shape0,shape1)
for i in range(len(tempgrbs)):  
    print(i)
    temps[:,:,i] = np.array(tempgrbs[i].values) - 273.15
    print(np.array(tempgrbs[i].values) - 273.15)
    dews[:,:,i] = np.array(dewgrbs[i].values) - 273.15
    uwnds[:,:,i] = np.array(uwndgrbs[i].values)
    vwnds[:,:,i] = np.array(vwndgrbs[i].values)
    agls[:,:,i] = np.array (geogrbs[i].values) - sfchgt    
print(shape0,shape1)
tempsinterp = np.zeros((shape0,shape1,67))
dewsinterp = np.zeros((shape0,shape1,67))
uwndsinterp = np.zeros((shape0,shape1,67))
vwndsinterp = np.zeros((shape0,shape1,67))
pressinterp = np.zeros((shape0,shape1,67))

print("there")
# Interpolate to height AGL coordinates
for j in range(temps.shape[0]):
    for k in range(temps.shape[1]):
        tempinterp = interp(temps[j,k,:],agls[j,k,:])
        dewinterp = interp(dews[j,k,:],agls[j,k,:])
        uwndinterp = interp(uwnds[j,k,:],agls[j,k,:])
        vwndinterp = interp(vwnds[j,k,:],agls[j,k,:])
        presinterp = interp(pres_levels,agls[j,k,:])
        tempsinterp[j,k,:] = tempinterp
        dewsinterp[j,k,:] = dewinterp
        uwndsinterp[j,k,:] = uwndinterp
        vwndsinterp[j,k,:] = vwndinterp
        pressinterp[j,k,:] = presinterp

		# Write output to jsons for D3 soundings
        out = {"temp":[],"dew":[],"pres":[],"uwnd":[],"vwnd":[]}
        out["temp"] = np.around(tempinterp[::-1],decimals=2).tolist()
        out["dew"] = np.around(dewinterp[::-1],decimals=2).tolist()
        out["uwnd"] = np.around(uwndinterp[::-1],decimals=2).tolist()
        out["vwnd"] = np.around(vwndinterp[::-1],decimals=2).tolist()
        out["pres"] = np.around(presinterp[::-1],decimals=2).tolist()
        mylat = lats[j,k]
        mylon = lons[j,k]
        strlat = "{0:.2f}".format(mylat)
        strlon = "{0:.2f}".format(mylon)
        coordinates = strlat + "_" + strlon
        with open("./app/static/jsons/%s.json" % (coordinates),"w") as outfile:
            json.dump(out,outfile)

# Combine features, scale with standard scalerm and make predictions
features = np.concatenate((tempsinterp,dewsinterp,uwndsinterp,vwndsinterp),axis=2)
features = features.reshape(shape0*shape1,268)
featurestransformed = scaler.transform(features)
featurestransformed = featurestransformed.reshape(shape0*shape1,67,4,order='F')
predictions = mymodel.predict(featurestransformed)
predictions = predictions.reshape(shape0,shape1,4)

# Get highest probability p-type
maxprediction = np.argmax(predictions,axis=2) + 1
maxprediction[np.where((crain==0) & (csnow==0) & (cicep==0) & (cfrzr==0))] = 0

# Mask where data is undefined
maxprediction[temps[:,:,0]>9000.] = 0

# Generate image tiles for model predictions and categorical p-types
tilemaker(maxprediction,"model",colordict)
tilemaker(ptypes,"rap",colordict)
