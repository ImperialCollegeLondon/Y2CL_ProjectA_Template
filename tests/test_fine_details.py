from types import MethodType, FunctionType
from inspect import signature, getmembers, isclass, getsource
from unittest.mock import MagicMock, PropertyMock
import numpy as np
from pprint import pformat
import re
import pytest
from matplotlib.figure import Figure


class TestRayInternals:

    def test_partial_construction(self, rays):
        rays.Ray([1.,2.,3.])
        rays.Ray(direc=[4.,5.,6.])

    def test_pos_list_or_array_check(self, rays):
        with pytest.raises(Exception) as excinfo:
            rays.Ray(pos=12)

    def test_pos_too_long_check(self, rays):
        with pytest.raises(Exception) as excinfo:
            rays.Ray(pos=[1., 2., 3., 4.])

    def test_pos_too_short_check(self, rays):
        with pytest.raises(Exception) as excinfo:
            rays.Ray(pos=[1., 2.])

    def test_pos_type(self, default_ray_pos, test_ray_pos):
        assert isinstance(default_ray_pos, list)
        assert isinstance(test_ray_pos, list)

    def test_pos_single_element(self, default_ray_pos, test_ray_pos):
        assert len(default_ray_pos) == 1
        assert len(test_ray_pos) == 1

    def test_pos_ndarray(self, default_ray_pos, test_ray_pos):
        assert isinstance(default_ray_pos[0], np.ndarray)
        assert isinstance(test_ray_pos[0], np.ndarray)

    def test_pos_dtype(self, default_ray_pos, test_ray_pos):
        init_pos_d = default_ray_pos[0]
        init_pos_t = test_ray_pos[0]
        assert init_pos_d.dtype == float
        assert init_pos_t.dtype == float

    def test_pos_correct(self, test_ray_pos):
        assert np.allclose(test_ray_pos[0], [1., 2., 3.])

    def test_dir_list_or_array_check(self, rays):
        with pytest.raises(Exception) as excinfo:
            rays.Ray(direc=12)

    def test_dir_too_long_check(self, rays):
        with pytest.raises(Exception) as excinfo:
            rays.Ray(direc=[1., 2., 3., 4.])

    def test_dir_too_short_check(self, rays):
        with pytest.raises(Exception) as excinfo:
            rays.Ray(direc=[1., 2.])

    def test_dir_ndarray(self, default_ray_direc, test_ray_direc):
        assert isinstance(default_ray_direc, np.ndarray)
        assert isinstance(test_ray_direc, np.ndarray)

    def test_dir_dtype(self, default_ray_direc, test_ray_direc):
        assert default_ray_direc.dtype == float
        assert test_ray_direc.dtype == float

    def test_dir_normalised(self, default_ray_direc, test_ray_direc):
        assert np.isclose(np.linalg.norm(default_ray_direc), 1.)
        assert np.isclose(np.linalg.norm(test_ray_direc), 1.)

    def test_dir_sensible_default(self, default_ray_direc):
        assert np.allclose(default_ray_direc, [0., 0., 1.])

    def test_dir_correct(self, test_ray_direc):
        test_k = np.array([4., 5., 6.])
        assert np.allclose(test_ray_direc, test_k / np.linalg.norm(test_k))

    def test_constructor_default_types(self, rays):
        params = signature(rays.Ray.__init__).parameters
        pos_default = params["pos"].default
        direc_default = params["direc"].default
        assert isinstance(pos_default, (type(None), list, np.ndarray))
        assert isinstance(direc_default, (type(None), list, np.ndarray))

    def test_append_pos_list_or_array_check(self, default_ray):
        with pytest.raises(Exception):
            default_ray.append(pos=12, direc=[1.,2.,3.])

    def test_append_pos_too_long_check(self, default_ray):
        with pytest.raises(Exception):
            default_ray.append(pos=[1., 2., 3., 4.], direc=[1.,2.,3.])

    def test_append_pos_too_short_check(self, default_ray):
        with pytest.raises(Exception):
            default_ray.append(pos=[1., 2.], direc=[1.,2.,3.])

    def test_append_direc_list_or_array_check(self, default_ray):
        with pytest.raises(Exception):
            default_ray.append(pos=[1.,2.,3.], direc=12)

    def test_append_direc_too_long_check(self, default_ray):
        with pytest.raises(Exception):
            default_ray.append(pos=[1.,2.,3.], direc=[1., 2., 3., 4.])

    def test_append_direc_too_short_check(self, default_ray):
        with pytest.raises(Exception):
            default_ray.append(pos=[1.,2.,3.], direc=[1., 2.])

#####################################################
    def test_append_increases_length(self, default_ray, var_name_map):
        pos = getattr(default_ray, var_name_map['pos'])
        assert len(pos) == 1
        default_ray.append([1., 2., 3.], [4., 5., 6.])
        assert len(pos) == 2

    def test_append_pos_type(self, default_ray, var_name_map):
        assert isinstance(getattr(default_ray, var_name_map["pos"]), list)
        default_ray.append(pos=[4., 5., 6.], direc=[7., 8., 9.])
        assert isinstance(getattr(default_ray, var_name_map["pos"]), list)

    def test_append_pos_array(self, default_ray, var_name_map):
        default_ray.append(pos=[4., 5., 6.], direc=[7., 8., 9.])
        pos = getattr(default_ray, var_name_map["pos"])
        assert isinstance(pos[-1], np.ndarray)

    def test_append_pos_correct(self, default_ray, var_name_map):
        default_ray.append(pos=[1., 2., 3.], direc=[4., 5., 6.])
        pos = getattr(default_ray, var_name_map['pos'])
        assert np.allclose(pos[-1], [1., 2., 3.])

        pos = default_ray.pos
        if isinstance(pos, MethodType):
            pos = pos()
        assert np.allclose(pos, [1., 2., 3.])

    def test_append_direc_array(self, default_ray, var_name_map):
        default_ray.append(pos=[4., 5., 6.], direc=[7., 8., 9.])
        direc = getattr(default_ray, var_name_map['direc'])
        assert isinstance(direc, np.ndarray)

    def test_append_direc_correct(self, default_ray, var_name_map):
        default_ray.append([1., 2., 3.], [4., 5., 6.])
        direc = getattr(default_ray, var_name_map["direc"])
        test_k = np.array([4., 5., 6.])
        assert np.allclose(direc, test_k / np.linalg.norm(test_k))

        direc = default_ray.direc
        if isinstance(direc, MethodType):
            direc = direc()
        assert np.allclose(direc, test_k / np.linalg.norm(test_k))


DATA_ATTRIBUTE_REGEX = re.compile(r"^\s*self\.([_a-zA-Z0-9]+)[^=]*=(?!=)", re.MULTILINE)


class TestAdvancedDesign:
    # def test_planarrefraction_exists(self, elements):
    #     assert hasattr(elements, "PlanarRefraction")
    #     assert issubclass(elements.PlanarRefraction, elements.OpticalElement)

    def test_biconvex_pc_share_base(self, lenses):
        assert lenses.BiConvex.__bases__ == lenses.PlanoConvex.__bases__

    def test_srefl_tree(self, elements):
        test_tree = set(elements.SphericalReflection.mro())
        assert len(test_tree) > 3
        assert test_tree.difference({elements.SphericalReflection,
                                     elements.OpticalElement,
                                     object})

    def test_intercept_not_in_srefl(self, elements):
        assert "intercept" not in vars(elements.SphericalReflection)

    def test_convexplano_exists(self, elements, lenses):
        assert "ConvexPlano" in vars(lenses)
        assert issubclass(lenses.ConvexPlano, elements.OpticalElement)

    def test_sr_tree(self, elements):
        test_tree = set(elements.SphericalRefraction.mro())
        assert len(test_tree) > 3
        assert test_tree.difference({elements.SphericalRefraction,
                                     elements.OpticalElement,
                                     object})

    def test_intercept_not_in_sr(self, rays, elements):
        assert "intercept" not in vars(elements.SphericalRefraction)
        with pytest.raises((NotImplementedError, TypeError)):
            elements.OpticalElement().intercept(rays.Ray([0., 0., 0.], [0., 0., 1.]))

    def test_op_tree(self, elements):
        test_tree = set(elements.OutputPlane.mro())
        assert len(test_tree) > 3
        assert test_tree.difference({elements.OutputPlane,
                                     elements.OpticalElement,
                                     object})

    def test_intercept_not_in_op(self, rays, elements):
        assert "intercept" not in vars(elements.OutputPlane)
        with pytest.raises((NotImplementedError, TypeError)):
            elements.OpticalElement().intercept(rays.Ray([0., 0., 0.], [0., 0., 1.]))

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
            for cls_name, cls in getmembers(module, isclass):
                if module == rays and cls == Figure:
                    continue
                if init_func := vars(cls).get("__init__", False):
                    non_hidden_vars.update(f"{cls_name}.{var}" for var in DATA_ATTRIBUTE_REGEX.findall(getsource(init_func))
                                           if not var.startswith('_'))

        assert not non_hidden_vars, f"Non hidden data attributes:\n {pformat(non_hidden_vars)}"

    def test_intercept_calls_ray_pos_once(self, rays, elements, monkeypatch):
        with monkeypatch.context() as m:
            if isinstance(rays.Ray.pos, FunctionType):
                pos_mock = MagicMock(return_value=np.array([0., 0., 0.]))
            else:
                pos_mock = PropertyMock(return_value=np.array([0., 0., 0.]))
            m.setattr(rays.Ray, "pos", pos_mock)
            if hasattr(elements, "Ray"):
                m.setattr(elements.Ray, "pos", pos_mock)
            elements.SphericalRefraction(z_0=10, curvature=-0.02, n_1=1., n_2=1.5, aperture=50.).intercept(rays.Ray())
        pos_mock.assert_called_once()

    def test_intercept_calls_ray_direc_once(self, rays, elements, monkeypatch):
        with monkeypatch.context() as m:
            if isinstance(rays.Ray.pos, FunctionType):
                direc_mock = MagicMock(return_value=np.array([0., 0., 1.]))
            else:
                direc_mock = PropertyMock(return_value=np.array([0., 0., 1.]))
            m.setattr(rays.Ray, "direc", direc_mock)
            if hasattr(elements, "Ray"):
                m.setattr(elements.Ray, "direc", direc_mock)
            elements.SphericalRefraction(z_0=10, curvature=-0.02, n_1=1., n_2=1.5, aperture=50.).intercept(rays.Ray())
        direc_mock.assert_called_once()

    def test_propagate_ray_calls_intercept_once(self, default_ray, elements, monkeypatch):
        sr = elements.SphericalRefraction(z_0=10, curvature=-0.02, n_1=1., n_2=1.5, aperture=50.)
        intercept_mock = MagicMock(wraps=sr.intercept)
        with monkeypatch.context() as m:
            m.setattr(sr, "intercept", intercept_mock)
            sr.propagate_ray(default_ray)
        intercept_mock.assert_called_once()
