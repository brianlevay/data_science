## 2. Finding correlations ##

correlations = combined.corr()
correlations = correlations["sat_score"]
correlations

## 3. Plotting enrollment ##

%matplotlib inline
import matplotlib.pyplot as plt

combined.plot.scatter(x="total_enrollment", y="sat_score")
plt.show()

## 4. Exploring schools with low SAT scores and enrollment ##

selection = (combined['total_enrollment'] < 1000) & (combined['sat_score'] < 1000)
low_enrollment = combined.loc[selection]

print(low_enrollment['School Name'])

# International high schools

## 5. Plotting language learning percentage ##

combined.plot.scatter(x="ell_percent", y="sat_score")
plt.show()

## 6. Mapping the schools ##

from mpl_toolkits.basemap import Basemap
m = Basemap(
    projection='merc', 
    llcrnrlat=40.496044, 
    urcrnrlat=40.915256, 
    llcrnrlon=-74.255735, 
    urcrnrlon=-73.700272,
    resolution='i'
)

m.drawmapboundary(fill_color='#85A6D9')
m.drawcoastlines(color='#6D5F47', linewidth=.4)
m.drawrivers(color='#6D5F47', linewidth=.4)

longitudes = list(combined["lon"])
latitudes = list(combined["lat"])

m.scatter(longitudes, latitudes, s=20, zorder=2, latlon=True)
plt.show()

## 7. Plotting out statistics ##

m = Basemap(
    projection='merc', 
    llcrnrlat=40.496044, 
    urcrnrlat=40.915256, 
    llcrnrlon=-74.255735, 
    urcrnrlon=-73.700272,
    resolution='i'
)

m.drawmapboundary(fill_color='#85A6D9')
m.drawcoastlines(color='#6D5F47', linewidth=.4)
m.drawrivers(color='#6D5F47', linewidth=.4)

longitudes = list(combined["lon"])
latitudes = list(combined["lat"])

m.scatter(longitudes, latitudes, c=combined["ell_percent"], cmap="summer", s=20, zorder=2, latlon=True)
plt.show()

## 8. Calculating district level statistics ##

import numpy

districts = combined.groupby("school_dist")
districts = districts.agg(numpy.mean)
districts.reset_index(inplace=True)

print(districts.head(5))

## 9. Plotting ell_percent by district ##

m = Basemap(
    projection='merc', 
    llcrnrlat=40.496044, 
    urcrnrlat=40.915256, 
    llcrnrlon=-74.255735, 
    urcrnrlon=-73.700272,
    resolution='i'
)

m.drawmapboundary(fill_color='#85A6D9')
m.drawcoastlines(color='#6D5F47', linewidth=.4)
m.drawrivers(color='#6D5F47', linewidth=.4)

longitudes = list(districts["lon"])
latitudes = list(districts["lat"])

m.scatter(longitudes, latitudes, c=districts["ell_percent"], cmap="summer", s=20, zorder=2, latlon=True)
plt.show()