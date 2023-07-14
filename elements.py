class OpticalElement():
    def propagate_ray(self):
        # raise error

class SphericalRefraction(OpticalElement):
    def __init__(self, z_0, aperture, curvature, n_1, n_2):
    def intercept(self, ray):
        # return either None or new position vector
    def propagate_ray(self, ray):
        # return either None or nothing if propagated successfully

class OutputPlane(OpticalElement):
    def __init__(self, z_0):
    def intercept(self, ray):
        # return either None or new position vector
    def propagate_ray(self, ray):
        # return either None or nothing if propagated successfully

class PlanarRefraction(OpticalElement):
    def __init__(self, z_0, aperture, n_1, n_2):
    def intercept(self, ray):
        # return either None or new position vector
    def propagate_ray(self, ray):
        # return either None or nothing if propagated successfully
