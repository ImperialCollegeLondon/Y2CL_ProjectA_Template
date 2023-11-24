from inspect import getmembers, isclass, getsource
from pprint import pformat
import re

DATA_ATTRIBUTE_REGEX = re.compile(r"self\.([_a-zA-Z0-9]*)", re.MULTILINE)


class TestAdvancedDesign:
    def test_planarrefraction_exists(self, elements):
        assert hasattr(elements, "PlanarRefraction")
        assert issubclass(elements.PlanarRefraction, elements.OpticalElement)

    def test_convexplano_exists(self, elements, lenses):
        assert hasattr(lenses, "ConvexPlano")
        assert issubclass(lenses.ConvexPlano, elements.OpticalElement)

    def test_sr_tree(self, elements):
        test_tree = set(elements.SphericalRefraction.mro())
        assert len(test_tree) > 3
        assert test_tree.difference({elements.SphericalRefraction,
                                     elements.OpticalElement,
                                     object})

    def test_intercept_not_in_sr(self, elements):
        assert "intercept" not in vars(elements.SphericalRefraction)

    def test_op_tree(self, elements):
        test_tree = set(elements.OutputPlane.mro())
        assert len(test_tree) > 3
        assert test_tree.difference({elements.OutputPlane,
                                     elements.OpticalElement,
                                     object})
        
    def test_intercept_not_in_op(self, elements):
        assert "intercept" not in vars(elements.OutputPlane)

    # def test_hidden_variables(self, rays, elements, lenses):
    #     r = rays.Ray(pos=[1, 2, 3], direc=[4, 5, 6])
    #     rb = rays.RayBundle(rmax=5., nrings=5)
    #     sr = elements.SphericalRefraction(z_0=100, aperture=35., curvature=0.02, n_1=1., n_2=1.5)
    #     op = elements.OutputPlane(z_0=250.)
    #     objects = [r, rb, sr, op]

    #     non_hidden_variables = set()
    #     for o in objects:
    #         for name in o.__dict__:
    #             if not name.startswith('_'):
    #                 non_hidden_variables.add(f"{o.__class__.__name__}.{name}")

    #     assert not non_hidden_variables

    def test_hidden_variables(self, rays, elements, lenses):
        non_hidden_vars = set()
        for module in (rays, elements, lenses):
            for name, cls in getmembers(module, isclass):
                if init_func := vars(cls).get("__init__", False):
                    non_hidden_vars.update(f"{name}.{var}" for var in DATA_ATTRIBUTE_REGEX.findall(getsource(init_func))
                                           if not var.startswith('_'))

        assert not non_hidden_vars, f"Non hidden data attributes:\n {pformat(non_hidden_vars)}"
