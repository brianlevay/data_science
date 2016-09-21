## 3. Basemap ##

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
%matplotlib inline

m = Basemap(projection='merc', llcrnrlat=-80, urcrnrlat=80, llcrnrlon=-180, urcrnrlon=180)

## 4. Converting from spherical to cartesian coordinates ##

m = Basemap(projection='merc', llcrnrlat=-80, urcrnrlat=80, llcrnrlon=-180, urcrnrlon=180)

# Convert from Series objects to List objects.
longitudes = airports["longitude"].tolist()
latitudes = airports["latitude"].tolist()

# Convert latitude and longitude to x and y coordinates.
x, y = m(longitudes, latitudes)

# Display original longitude values
print(longitudes[0:5])
# Display original latitude values
print(latitudes[0:5])
# Display x-axis coordinates
print(x[0:5])
# Display y-axis coordinates
print(y[0:5])

## 5. Generating a scatter plot ##

m = Basemap(projection='merc', llcrnrlat=-80, urcrnrlat=80, llcrnrlon=-180, urcrnrlon=180)
x, y = m(longitudes, latitudes)
m.scatter(x,y,s=1)
plt.show()

## 6. Customizing the plot using Basemap ##

m.scatter(x,y,s=1)
m.drawcoastlines()
plt.show()

## 7. Customizing the plot using Matplotlib ##

# Add code here, before creating the Basemap instance.
fig = plt.figure(figsize=(15,20))
ax = fig.add_subplot(1,1,1)
ax.set_title('Scaled Up Earth With Coastlines')

m = Basemap(projection='merc', llcrnrlat=40.496044, urcrnrlat=80, llcrnrlon=-180, urcrnrlon=180)
longitudes = airports["longitude"].tolist()
latitudes = airports["latitude"].tolist()
m.scatter(x, y, s=1)
x, y = m(longitudes, latitudes)
m.drawcoastlines()
plt.show()

## 8. Great circles ##

geo_routes = pd.read_csv("geo_routes.csv")
print(geo_routes.columns)
print(geo_routes.head(5))

## 9. Displaying great circles ##

fig = plt.figure(figsize=(15,20))
m = Basemap(projection='merc', llcrnrlat=-80, urcrnrlat=80, llcrnrlon=-180, urcrnrlon=180)
m.drawcoastlines()

def create_great_circles(df):
    for index, row in df.iterrows():
        start_lon = row['start_lon']
        start_lat = row['start_lat']
        end_lon = row['end_lon']
        end_lat = row['end_lat']

        if abs(end_lat - start_lat) < 180 and abs(end_lon - start_lon) < 180:
            m.drawgreatcircle(start_lon, start_lat, end_lon, end_lat, linewidth=1)
            
dfw = geo_routes[geo_routes['source'] == 'DFW']
create_great_circles(dfw)