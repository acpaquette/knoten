"""
A set of classes that represent the target surface. Each class implements the
get_height and get_radius functions for computing the height and radius respectively
at a given ground location (geocentric latitude and longitude).
"""
from typing import overload

import numpy as np

from knoten.utils import reproject

from plio.io.io_gdal import GeoDataset

class EllipsoidDem:
    """
    A biaxial ellipsoid surface model.
    """

    def __init__(self, semi_major, semi_minor = None):
        """
        Create an ellipsoid DEM from a set of radii

        Parameters
        ----------
        semi_major : float
                     The equatorial semi-major radius of the ellipsoid.
        semi_minor : float
                     The polar semi-minor radius of the ellipsoid.
        """
        self.a = semi_major
        self.b = semi_major
        self.c = semi_major

        if semi_minor is not None:
            self.c = semi_minor

    def get_lla_height(self, lat, lon):
        """
        Get the height above the ellipsoid at a ground location

        Parameters
        ----------
        lat : float
              The geocentric latitude in degrees
        lon : float
              The longitude in degrees
        """
        return 0

    def get_ecef_height(self, x: float, y: float, z: float) -> float:
        """
        Get the height above the ellipsoid at a ground location

        Parameters
        ----------
        lat : float
              The geocentric latitude in degrees
        lon : float
              The longitude in degrees
        """
        lat, lon, _ = reproject([x, y, z], 
                                 self.a, 
                                 self.c, 
                                 'geocent', 
                                 'latlon')
        return self.get_lla_height(lat, lon)

    def get_lla_radius(self, lat, lon):
        """
        Get the radius at a ground location

        Parameters
        ----------
        lat : float
              The geocentric latitude in degrees
        lon : float
              The longitude in degrees
        """
        cos_lon = np.cos(np.deg2rad(lon))
        sin_lon = np.sin(np.deg2rad(lon))
        cos_lat = np.cos(np.deg2rad(lat))
        sin_lat = np.sin(np.deg2rad(lat))

        denom = self.b * self.b * cos_lon * cos_lon
        denom += self.a * self.a * sin_lon * sin_lon
        denom *= self.c * self.c * cos_lat * cos_lat
        denom += self.a * self.a * self.b * self.b * sin_lat * sin_lat
        radius = (self.a * self.b * self.c) / np.sqrt(denom)
        return radius

    def get_ecef_radius(self, x: float, y: float, z: float) -> float:
        """
        Get the height above the ellipsoid at a ground location

        Parameters
        ----------
        lat : float
              The geocentric latitude in degrees
        lon : float
              The longitude in degrees
        """
        lat, lon, _ = reproject([x, y, z], 
                                 self.a, 
                                 self.c, 
                                 'geocent', 
                                 'latlon')
        return self.get_lla_radius(lat, lon)

class GdalDem(EllipsoidDem):
    """
    A raster DEM surface model.
    """

    def __init__(self, dem, semi_major, semi_minor = None, dem_type=None):
        """
        Create a GDAL dem from a dem file

        Parameters
        ----------
        dem : str
              The DEM file
        semi_major : float
                     The equatorial semi-major radius of the reference ellipsoid.
        semi_minor : float
                     The polar semi-minor radius of the reference ellipsoid.
        dem_type : str
                   The type of DEM, either height above reference ellipsoid or radius.
        """
        super().__init__(semi_major, semi_minor)
        dem_types = ('height', 'radius')
        if dem_type is None:
            dem_type = dem_types[0]
        if dem_type not in dem_types:
            raise ValueError(f'DEM type {dem_type} is not a valid option.')
        self.dem = GeoDataset(dem)
        self.dem_type = dem_type

    def get_lla_raster_value(self, lat: float, lon: float) -> float:
        """
        Get the value of the dem raster at a ground location

        Parameters
        ----------
        lat : float
              The geocentric latitude in degrees
        lon : float
              The longitude in degrees
        """
        px, py = self.dem.latlon_to_pixel(lat, lon)
        value = self.dem.read_array(1, [px, py, 1, 1])[0][0]
        if value == self.dem.no_data_value:
            return None
        return value

    def get_ecef_raster_value(self, x: float, y: float, z: float) -> float:
        """
        Get the value of the dem raster at a ground location

        Parameters
        ----------
        lat : float
              The geocentric latitude in degrees
        lon : float
              The longitude in degrees
        """
        lat, lon, _ = reproject([x, y, z], 
                                 self.semi_major, 
                                 self.semi_minor, 
                                 'geocent', 
                                 'latlon')
        return self.get_lla_raster_value(lat, lon)

    def get_lla_height(self, lat: float, lon: float) -> float:
        """
        Get the height above the ellipsoid at a ground location

        Parameters
        ----------
        lat : float
              The geocentric latitude in degrees
        lon : float
              The longitude in degrees
        """
        height = self.get_lla_raster_value(lat, lon)
        if self.dem_type == 'radius' and height is not None:
            height -= super().get_lla_radius(lat, lon)
        return height

    def get_lla_radius(self, lat: float, lon: float) -> float:
        """
        Get the radius at a ground location

        Parameters
        ----------
        lat : float
              The geocentric latitude in degrees
        lon : float
              The longitude in degrees
        """
        radius = self.get_lla_raster_value(lat, lon)
        if self.dem_type == 'height':
            radius += super().get_lla_radius(lat, lon)
        return radius
