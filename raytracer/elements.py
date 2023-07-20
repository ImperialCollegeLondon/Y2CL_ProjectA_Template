class PlanarRefraction(OpticalElement):
    def __init__(self, z_0, aperture, n_1, n_2):
        pass

    def intercept(self, ray):
        # return either None or new position vector
        pass

    def propagate_ray(self, ray):
        # return either None or nothing if propagated successfully
        pass
