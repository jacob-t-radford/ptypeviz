# ptypeviz
 
This is an interactive web viewer for visualizing precipitation type output. A Leaflet map on the left displays gridded precipitation type from either the RAP categorical p-type fields or the CNN. Clicking on a location on the map brings up a skew-t for that location on the right as well as the probabilities of each p-type category in a bar graph. The temperature and dewpoint profiles of the skew-t can be dragged to new values, which then uses Flask to calculate new precipitation type probabilities with the CNN.

Steps to run:

1) Install dependencies with ptypeviz.yml file
2) Regrid RAP or HRRR data to a uniform grid (example file: /app/static/rap_regrid.grib2) - I suggest with wgrib2
3) Run rap_ptype.py pointing to regridded file to produce jsons representing skew-t data and image tiles for the Leaflet map
4) In top level directory enter "flask run" 
5) Navigate to http://127.0.0.1:5000/ in web browser
