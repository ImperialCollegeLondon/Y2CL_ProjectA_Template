from inspect import signature, getmembers, isclass, getsource
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

    def test_default_pos_list(self, default_ray_pos):
        assert isinstance(default_ray_pos, list)
 
    def test_pos_list(self, test_ray_pos):
        assert isinstance(test_ray_pos, list)

    def test_default_pos_single_element(self, default_ray_pos):
        assert len(default_ray_pos) == 1

    def test_pos_single_element(self, test_ray_pos):
        assert len(test_ray_pos) == 1

    def test_default_pos_ndarray(self, default_ray_pos):
        assert isinstance(default_ray_pos[0], np.ndarray)

    def test_pos_ndarray(self, test_ray_pos):
        assert isinstance(test_ray_pos[0], np.ndarray)

    def test_default_pos_dtype(self, default_ray_pos):
        init_pos = default_ray_pos[0]
        assert init_pos.dtype == np.dtype("float64") or \
               init_pos.dtype == np.dtype("float32")

    def test_pos_dtype(self, test_ray_pos):
        init_pos = test_ray_pos[0]
        assert init_pos.dtype == np.dtype("float64") or \
               init_pos.dtype == np.dtype("float32")

    def test_pos_set(self, test_ray_pos):
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

    def test_default_dir_ndarray(self, default_ray_direc):
        assert isinstance(default_ray_direc, np.ndarray)

    def test_dir_ndarray(self, test_ray_direc):
        assert isinstance(test_ray_direc, np.ndarray)
        
    def test_default_dir_dtype(self, default_ray_direc):
        assert default_ray_direc.dtype == np.dtype("float64") or \
               default_ray_direc.dtype == np.dtype("float32")
        
    def test_dir_dtype(self, test_ray_direc):
        assert test_ray_direc.dtype == np.dtype("float64") or \
               test_ray_direc.dtype == np.dtype("float32")

    def test_default_dir_normalised(self, default_ray_direc):
        assert np.isclose(np.linalg.norm(default_ray_direc), 1.)

    def test_dir_normalised(self, test_ray_direc):
        assert np.isclose(np.linalg.norm(test_ray_direc), 1.)

    def test_dir_sensible_default(self, default_ray_direc):
        assert np.allclose(default_ray_direc, [0., 0., 1.])

    def test_k_set(self, test_ray_direc):
        test_k = np.array([4., 5. ,6.])
        test_k /= np.linalg.norm(test_k)
        assert np.allclose(test_ray_direc, test_k)

    def test_hidden_variables(self, rays):
        for var in vars(rays.Ray()):
            assert var.startswith('_')

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
        default_ray.append([1.,2.,3.], [4.,5.,6.])
        assert len(pos) == 2

    def test_append_pos_type(self, default_ray, var_name_map):
        assert isinstance(getattr(default_ray,var_name_map["pos"]), list)
        default_ray.append(pos=[4.,5.,6.], direc=[7.,8.,9.])
        assert isinstance(getattr(default_ray,var_name_map["pos"]), list)

    def test_append_pos_array(self, default_ray, var_name_map):
        default_ray.append(pos=[4.,5.,6.], direc=[7.,8.,9.])
        pos = getattr(default_ray,var_name_map["pos"])
        assert isinstance(pos[-1], np.ndarray)

    def test_append_pos(self, default_ray, var_name_map):
        default_ray.append(pos=[1., 2., 3.], direc=[4., 5., 6.])
        pos = getattr(default_ray, var_name_map['pos'])
        assert np.allclose(pos[1], [1., 2., 3.])
        assert np.allclose(default_ray.pos(), [1., 2., 3.])

    def test_append_direc_array(self, default_ray, var_name_map):
        default_ray.append(pos=[4.,5.,6.], direc=[7.,8.,9.])
        direc = getattr(default_ray, var_name_map['direc'])
        assert isinstance(direc, np.ndarray)

    def test_append_direc(self, default_ray, var_name_map):
        default_ray.append([1., 2., 3.], [4., 5., 6.])
        direc = getattr(default_ray, var_name_map["direc"])
        test_k = np.array([4.,5.,6.])
        test_k /= np.linalg.norm(test_k)
        assert np.allclose(direc, test_k)
        assert np.allclose(default_ray.direc(), test_k)
#################################################


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
            for cls_name, cls in getmembers(module, isclass):
                if module == rays and cls == Figure:
                    continue
                if init_func := vars(cls).get("__init__", False):
                    non_hidden_vars.update(f"{cls_name}.{var}" for var in DATA_ATTRIBUTE_REGEX.findall(getsource(init_func))
                                           if not var.startswith('_'))

        assert not non_hidden_vars, f"Non hidden data attributes:\n {pformat(non_hidden_vars)}"



## OpticalElement.propagate_ray calls intercept
        
    ## testing intercept only calls pos and direc once to avoid repetition
    def test_ray_pos_called_once(self, rays, elements, monkeypatch):
        pos_mock = MagicMock(return_value=np.array([0., 0., 10.]))
        monkeypatch.setattr(rays.Ray, "pos", pos_mock)
        r = rays.Ray()
        sr = elements.SphericalRefraction(z_0=10, curvature=-0.02, n_1=1., n_2=1.5, aperture=50.)
        sr.intercept(r)
        pos_mock.assert_called_once()

    def test_ray_direc_called_once(self, rays, elements, monkeypatch):
        direc_mock = MagicMock(return_value=np.array([0., 0., 10.]))
        monkeypatch.setattr(rays.Ray, "direc", direc_mock)
        r = rays.Ray()
        sr = elements.SphericalRefraction(z_0=10, curvature=-0.02, n_1=1., n_2=1.5, aperture=50.)
        sr.intercept(r)
        direc_mock.assert_called_once()