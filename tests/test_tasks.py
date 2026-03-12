from types import FunctionType, MethodType
from operator import itemgetter
from itertools import product
from inspect import signature
from string import ascii_lowercase
from unittest.mock import MagicMock, PropertyMock, patch, call
from inspect import getmembers, isclass, signature, getsource
from importlib import import_module, reload
from base64 import b64encode, b64decode
import numpy as np
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import pytest
# from utils import check_figures_equal


class TestTask2:

    def test_rays_exists(self, rays):
        pass

    def test_docstring_present(self, rays):
        assert rays.__doc__ is not None

    def test_docstring_not_blank(self, rays):
        assert rays.__doc__ != ""

    def test_ray_exists(self, rays):
        assert "Ray" in vars(rays)

    def test_ray_construction_args(self, rays):
        assert {"pos", "direc"}.issubset(signature(rays.Ray).parameters.keys())

    def test_ray_construction(self, default_ray, test_ray):
        pass

    def test_pos_exists(self, rays):  # Check method not overridden by variable
        assert isinstance(rays.Ray.pos, (FunctionType, property))

    def test_pos_type(self, default_ray, test_ray):
        default_pos = default_ray.pos
        if isinstance(default_pos, MethodType):
            default_pos = default_pos()
        assert isinstance(default_pos, np.ndarray)

        pos = test_ray.pos
        if isinstance(pos, MethodType):
            pos = pos()
        assert isinstance(pos, np.ndarray)

    def test_pos_correct(self, test_ray):
        pos = test_ray.pos
        if isinstance(pos, MethodType):
            pos = pos()
        assert np.allclose(pos, [1., 2., 3.])

    def test_direc_exists(self, rays):  # Check method not overridden by variable
        assert isinstance(rays.Ray.direc, (FunctionType, property))

    def test_direc_type(self, default_ray, test_ray):
        default_direc = default_ray.direc
        if isinstance(default_direc, MethodType):
            default_direc = default_direc()
        assert isinstance(default_direc, np.ndarray)

        direc = test_ray.direc
        if isinstance(direc, MethodType):
            direc = direc()
        assert isinstance(direc, np.ndarray)

    def test_direc_correct(self, test_ray):
        test_k = np.array([4., 5., 6.])
        direc = test_ray.direc
        if isinstance(direc, MethodType):
            direc = direc()
        assert np.allclose(direc, test_k) or np.allclose(direc, test_k / np.linalg.norm(test_k))

    def test_append_exists(self, rays):
        assert isinstance(rays.Ray.append, FunctionType)

    def test_append_correct(self, test_ray):
        test_ray.append([10., 12., 14.], [11., 13., 15.])
        pos = test_ray.pos
        if isinstance(pos, MethodType):
            pos = pos()
        direc = test_ray.direc
        if isinstance(direc, MethodType):
            direc = direc()
        vertices = test_ray.vertices
        if isinstance(vertices, MethodType):
            vertices = vertices()

        assert isinstance(pos, np.ndarray)
        assert np.allclose(pos, [10., 12., 14.])
        assert isinstance(direc, np.ndarray)
        test_k = np.array([11., 13., 15.])
        assert np.allclose(direc, test_k) or np.allclose(direc, test_k / np.linalg.norm(test_k))
        assert len(vertices) == 2
        assert np.allclose(vertices, [[1., 2., 3.],
                                      [10., 12., 14.]])
        for vertex in vertices:
            assert isinstance(vertex, np.ndarray)

    def test_vertices_exists(self, rays):
        assert isinstance(rays.Ray.vertices, (FunctionType, property))

    def test_vertices_type(self, default_ray, test_ray):
        default_vertices = default_ray.vertices
        if isinstance(default_vertices, MethodType):
            default_vertices = default_vertices()
        assert isinstance(default_vertices, list)

        for default_vertex in default_vertices:
            assert isinstance(default_vertex, np.ndarray)

        vertices = test_ray.vertices
        if isinstance(vertices, MethodType):
            vertices = vertices()
        assert isinstance(vertices, list)

        for vertex in vertices:
            assert isinstance(vertex, np.ndarray)

    def test_vertices_correct(self, default_ray, test_ray):
        default_vertices = default_ray.vertices
        if isinstance(default_vertices, MethodType):
            default_vertices = default_vertices()

        assert len(default_vertices) == 1
        default_ray.append([4., 5., 6.], [7., 8., 9.])
        default_vertices = default_ray.vertices
        if isinstance(default_vertices, MethodType):
            default_vertices = default_vertices()
        assert len(default_vertices) == 2

        for vertex in default_vertices:
            assert isinstance(vertex, np.ndarray)

        test_vertices = test_ray.vertices
        if isinstance(test_vertices, MethodType):
            test_vertices = test_vertices()

        assert len(test_vertices) == 1
        test_ray.append([4., 5., 6.], [7., 8., 9.])
        test_vertices = test_ray.vertices
        if isinstance(test_vertices, MethodType):
            test_vertices = test_vertices()
        assert len(test_vertices) == 2

        for vertex in test_vertices:
            assert isinstance(vertex, np.ndarray)

        assert np.allclose(test_vertices, [[1., 2., 3.],
                                           [4., 5., 6.]])


class TestTask3:

    def test_elements_exists(self, elements):
        pass

    def test_oe_exists(self, elements):
        assert "OpticalElement" in vars(elements)
        # assert "OpticalElement" in (name for name, _ in getmembers(elements, isclass))

    def test_intercept_raises(self, default_ray, elements):
        oe = elements.OpticalElement()
        with pytest.raises(NotImplementedError):
            oe.intercept(default_ray)

    def test_pr_raises(self, default_ray, elements):
        oe = elements.OpticalElement()
        with pytest.raises(NotImplementedError):
            oe.propagate_ray(default_ray)


class TestTask4:

    def test_sr_exists(self, elements):
        assert "SphericalRefraction" in vars(elements)

    def test_inheritance(self, elements):
        # assert elements.OpticalElement in elements.SphericalRefraction.mro()
        assert issubclass(elements.SphericalRefraction, elements.OpticalElement)

    def test_construction_args(self, elements):
        assert {"z_0", "aperture", "curvature", "n_1", "n_2"}.issubset(signature(elements.SphericalRefraction).parameters.keys())

    def test_construction(self, elements):
        elements.SphericalRefraction(z_0=10., aperture=5., curvature=0.2, n_1=1., n_2=1.5)

    def test_z0(self, elements):
        assert isinstance(elements.SphericalRefraction.z_0, (FunctionType, property))
        sr = elements.SphericalRefraction(z_0=10., aperture=5., curvature=0.2, n_1=1., n_2=1.5)
        z0 = sr.z_0
        if isinstance(z0, MethodType):
            z0 = z0()
        assert z0 == 10.

    def test_aperture(self, elements):
        assert isinstance(elements.SphericalRefraction.aperture, (FunctionType, property))
        sr = elements.SphericalRefraction(z_0=10., aperture=5., curvature=0.2, n_1=1., n_2=1.5)
        aperture = sr.aperture
        if isinstance(aperture, MethodType):
            aperture = aperture()
        assert aperture == 5.

    def test_curvature(self, elements):
        assert isinstance(elements.SphericalRefraction.curvature, (FunctionType, property))
        sr = elements.SphericalRefraction(z_0=10., aperture=5., curvature=0.2, n_1=1., n_2=1.5)
        curvature = sr.curvature
        if isinstance(curvature, MethodType):
            curvature = curvature()
        assert curvature == 0.2

    def test_n1(self, elements):
        assert isinstance(elements.SphericalRefraction.n_1, (FunctionType, property))
        sr = elements.SphericalRefraction(z_0=10., aperture=5., curvature=0.2, n_1=1., n_2=1.5)
        n1 = sr.n_1
        if isinstance(n1, MethodType):
            n1 = n1()
        assert n1 == 1.

    def test_n2(self, elements):
        assert isinstance(elements.SphericalRefraction.n_2, (FunctionType, property))
        sr = elements.SphericalRefraction(z_0=10., aperture=5., curvature=0.2, n_1=1., n_2=1.5)
        n2 = sr.n_2
        if isinstance(n2, MethodType):
            n2 = n2()
        assert n2 == 1.5


class TestTask5:

    def test_intercept_exists(self, elements):
        assert isinstance(elements.SphericalRefraction.intercept, FunctionType)
    
    def test_intercept_not_crash(self, default_ray, elements):
        sr = elements.SphericalRefraction(z_0=10, curvature=0.02, n_1=1., n_2=1.5, aperture=50.)
        sr.intercept(default_ray)

    def test_no_intercept(self, rays, elements):
        ray = rays.Ray(pos=[10., 0., 0.])
        sr = elements.SphericalRefraction(z_0=10, curvature=0.02, n_1=1., n_2=1.5, aperture=5.)
        assert sr.intercept(ray) is None

    def test_intercept_type(self, default_ray, elements):
        sr = elements.SphericalRefraction(z_0=10, curvature=0.02, n_1=1., n_2=1.5, aperture=50.)
        assert isinstance(sr.intercept(default_ray), np.ndarray)

    def test_onaxis_intercept(self, rays, elements):
        ray = rays.Ray(pos=[0., 0., 0.])
        sr = elements.SphericalRefraction(z_0=10, curvature=0.02, n_1=1., n_2=1.5, aperture=50.)
        assert np.allclose(sr.intercept(ray), [0., 0., 10.])

        sr = elements.SphericalRefraction(z_0=10, curvature=-0.02, n_1=1., n_2=1.5, aperture=50.)
        assert np.allclose(sr.intercept(ray), [0., 0., 10.])

    def test_offaxis_intercept(self, rays, elements):
        ray1 = rays.Ray(pos=[1., 0., 0.])
        ray2 = rays.Ray(pos=[-1., 0., 0.])
        sr = elements.SphericalRefraction(z_0=10, curvature=0.02, n_1=1., n_2=1.5, aperture=50.)
        assert np.allclose(sr.intercept(ray1), [1., 0., 10.010001])
        assert np.allclose(sr.intercept(ray2), [-1., 0., 10.010001])

        sr = elements.SphericalRefraction(z_0=10, curvature=-0.02, n_1=1., n_2=1.5, aperture=50.)
        assert np.allclose(sr.intercept(ray1), [1., 0., 9.989999])
        assert np.allclose(sr.intercept(ray2), [-1., 0., 9.989999])

    def test_intercept_aperture(self, rays, elements):
        ray1 = rays.Ray(pos=[7., 0., 0.])
        ray2 = rays.Ray(pos=[-7., 0., 0.])
        ray3 = rays.Ray(pos=[10., 0., 0.])
        ray4 = rays.Ray(pos=[-10., 0., 0.])

        sr = elements.SphericalRefraction(z_0=10, curvature=0.02, n_1=1., n_2=1.5, aperture=50.)
        assert np.allclose(sr.intercept(ray1), [7., 0., 10.49242482])
        assert np.allclose(sr.intercept(ray2), [-7., 0., 10.49242482])
        assert np.allclose(sr.intercept(ray3), [10., 0., 11.01020514])
        assert np.allclose(sr.intercept(ray4), [-10., 0., 11.01020514])

        sr = elements.SphericalRefraction(z_0=10, curvature=-0.02, n_1=1., n_2=1.5, aperture=50.)
        assert np.allclose(sr.intercept(ray1), [7., 0., 9.50757518])
        assert np.allclose(sr.intercept(ray2), [-7., 0., 9.50757518])
        assert np.allclose(sr.intercept(ray3), [10., 0., 8.98979486])
        assert np.allclose(sr.intercept(ray4), [-10., 0., 8.98979486])

        sr = elements.SphericalRefraction(z_0=10, curvature=0.02, n_1=1., n_2=1.5, aperture=9.)
        assert np.allclose(sr.intercept(ray1), [7., 0., 10.49242482])
        assert np.allclose(sr.intercept(ray2), [-7., 0., 10.49242482])
        assert sr.intercept(ray3) is None
        assert sr.intercept(ray4) is None

        sr = elements.SphericalRefraction(z_0=10, curvature=-0.02, n_1=1., n_2=1.5, aperture=9.)
        assert np.allclose(sr.intercept(ray1), [7., 0., 9.50757518])
        assert np.allclose(sr.intercept(ray2), [-7., 0., 9.50757518])
        assert sr.intercept(ray3) is None
        assert sr.intercept(ray4) is None


class TestTask6:

    def test_physics_exists(self, ph):
        pass

    def test_refract_exists(self, ph):
        assert isinstance(ph.refract, FunctionType)

    def test_refract_args(self, ph):
        assert {"direc", "normal", "n_1", "n_2"}.issubset(signature(ph.refract).parameters.keys())

    def test_refract_type(self, ph):
        assert isinstance(ph.refract(direc=[1, 2, 3], normal=[4, 5, 6], n_1=1, n_2=1.5), np.ndarray)

    def test_returns_unitvector(self, ph):
        assert np.isclose(np.linalg.norm(ph.refract(direc=np.array([0., 0.05, 1.]),
                                                    normal=np.array([0., -1.9, -1.]),
                                                    n_1=1.0, n_2=1.5)), 1.)

    def test_onaxis_refract(self, ph):
        assert np.allclose(ph.refract(direc=np.array([0., 0., 1.]),
                                      normal=np.array([0., 0., -1.]),
                                      n_1=1.0, n_2=1.5), np.array([0., 0., 1.]))

    def test_offaxis_refract(self, ph):
        direc = np.array([0., 0., 1.])
        norm_lower = np.array([0., -1., -1.])
        norm_lower /= np.linalg.norm(norm_lower)
        assert np.allclose(ph.refract(direc=direc,
                                      normal=norm_lower,
                                      n_1=1.0, n_2=1.5), np.array([0., 0.29027623, 0.9569429]))

        norm_upper = np.array([0., 1., -1.])
        norm_upper /= np.linalg.norm(norm_upper)
        assert np.allclose(ph.refract(direc=direc,
                                      normal=norm_upper,
                                      n_1=1.0, n_2=1.5), np.array([0., -0.29027623, 0.9569429]))

    def test_equal_ref_indices_onaxis(self, ph):
        norm = np.array([0., 0., -1.])
        norm /= np.linalg.norm(norm)
        assert np.allclose(ph.refract(direc=np.array([0., 0., 1.]),
                                      normal=norm,
                                      n_1=1.0, n_2=1.0), np.array([0., 0., 1.]))

    def test_equal_ref_indices_offaxis(self, ph):
        direc = np.array([0., 0., 1.])
        norm_upper = np.array([0., 1., -1.])
        norm_upper /= np.linalg.norm(norm_upper)
        print(ph.refract(direc=direc,normal=norm_upper,n_1=11.0, n_2=1.0), np.array([0., 0., 1.]))
        assert np.allclose(ph.refract(direc=direc,
                                      normal=norm_upper,
                                      n_1=1.0, n_2=1.0), np.array([0., 0., 1.]))

        norm_lower = np.array([0., -1., -1.])
        norm_lower /= np.linalg.norm(norm_lower)
        assert np.allclose(ph.refract(direc=direc,
                                      normal=norm_lower,
                                      n_1=1.0, n_2=1.0), np.array([0., 0., 1.]))

    def test_TIR(self, ph):
        direc = np.array([0., 0., 1.])
        norm = np.array([0., 1., -1.])
        norm /= np.linalg.norm(norm)
        assert ph.refract(direc=direc, normal=norm, n_1=1.5, n_2=1.0) is None


class TestTask7:

    def test_pr_exists(self, elements):
        assert isinstance(elements.SphericalRefraction.propagate_ray, FunctionType)

    def test_pr_calls_intercept_once(self, rays, elements, monkeypatch):
        ray = rays.Ray([0., 0., 0.], [0., 0., 1.])
        intercept_mock = MagicMock(return_value=np.array([0., 0., 10.]))
        with monkeypatch.context() as m:
            m.setattr(elements.SphericalRefraction, "intercept", intercept_mock)
            sr = elements.SphericalRefraction(z_0=10, curvature=0.02, n_1=1., n_2=1.5, aperture=50.)
            sr.propagate_ray(ray)
        intercept_mock.assert_called_once_with(ray)

    def test_pr_calls_refract_once(self, rays, elements, ph, monkeypatch):
        ray = rays.Ray([0., 0., 0.], [0., 0., 1.])
        refract_mock = MagicMock(return_value=np.array([0., 0., 1.]))
        with monkeypatch.context() as m:
            m.setattr(ph, "refract", refract_mock)
            if hasattr(elements, "refract"):
                m.setattr(elements, "refract", refract_mock)
            sr = elements.SphericalRefraction(z_0=10, curvature=0.02, n_1=1., n_2=1.5, aperture=50.)
            sr.propagate_ray(ray)
        refract_mock.assert_called_once()

        call_dict = {}
        for name, val in zip(signature(ph.refract).parameters.keys(), refract_mock.call_args.args):
            call_dict[name] = val
        call_dict.update(refract_mock.call_args.kwargs)

        assert np.allclose(call_dict["direc"], [0., 0., 1.])
        assert np.allclose(call_dict["normal"] / np.linalg.norm(call_dict["normal"]), [0., 0., -1.])
        assert call_dict["n_1"] == 1.0
        assert call_dict["n_2"] == 1.5

    def test_pr_calls_append_once(self, rays, elements, monkeypatch):
        append_mock = MagicMock(return_value=None)
        with monkeypatch.context() as m:
            m.setattr(rays.Ray, "append", append_mock)
            if hasattr(elements, "Ray"):
                m.setattr(elements.Ray, "append", append_mock)

            sr = elements.SphericalRefraction(z_0=10, curvature=0.02, n_1=1., n_2=1.5, aperture=50.)
            sr.propagate_ray(rays.Ray([1., 2., 0.], [0., 0., 1.]))
        append_mock.assert_called_once()

        call_dict = {}
        for name, val in zip(signature(rays.Ray().append).parameters.keys(), append_mock.call_args.args):
            call_dict[name] = val
        call_dict.update(append_mock.call_args.kwargs)

        assert np.allclose(call_dict["pos"], [1., 2., 10.05002503])
        assert np.allclose(call_dict["direc"], [-0.00667112, -0.01334223, 0.99988873])


class TestTask8:

    TASK8_DEFAULT = b'ZGVmIHRhc2s4KCk6CiAgICAiIiIKICAgIFRhc2sgOC4KCiAgICBJbiB0aGlzIGZ1bmN0aW9uIHlvdSBzaG91bGQgY2hlY2sgeW91ciBwcm9wYWdhdGVfcmF5IGZ1bmN0aW9uIHByb3Blcmx5CiAgICBmaW5kcyB0aGUgY29ycmVjdCBpbnRlcmNlcHQgYW5kIGNvcnJlY3RseSByZWZyYWN0cyBhIHJheS4gRG9uJ3QgZm9yZ2V0CiAgICB0byBjaGVjayB0aGF0IHRoZSBjb3JyZWN0IHZhbHVlcyBhcmUgYXBwZW5kZWQgdG8geW91ciBSYXkgb2JqZWN0LgogICAgIiIiCg=='

    def test_doesnt_crash(self, task8_output, an):
        attempted = getsource(an.task8).encode('utf-8') != b64decode(TestTask8.TASK8_DEFAULT)
        assert attempted, "Task8 not attempted."

    def test_ray_created(self, ray_mock, task8_output):
        ray_mock.assert_called()
        assert ray_mock.call_count > 1, "Only a single ray created."

    def test_sr_created(self, sr_mock, task8_output):
        sr_mock.assert_called()

    def test_propagate_ray_called(self, pr_mock, task8_output):
        pr_mock.assert_called()
        assert pr_mock.call_count > 1, "Propagate only called once"


class TestTask9:

    def test_op_exists(self, elements):
        assert "OutputPlane" in vars(elements)

    def test_inheritance(self, elements):
        # assert elements.OpticalElement in elements.OutputPlane.mro()
        assert issubclass(elements.OutputPlane, elements.OpticalElement)

    def test_construction_args(self, elements):
        assert {"z_0"}.issubset(signature(elements.OutputPlane).parameters.keys())

    def test_construction(self, elements):
        elements.OutputPlane(z_0=250.)

    def test_intercept_exists(self, elements):
        assert isinstance(elements.OutputPlane.intercept, FunctionType)

    def test_pr_exists(self, elements):
        assert isinstance(elements.OutputPlane.propagate_ray, FunctionType)

    def test_pr_calls_intercept_once(self, default_ray, elements, monkeypatch):
        intercept_patch = MagicMock(return_value=np.array([1., 2., 3.]))
        with monkeypatch.context() as m:
            m.setattr(elements.OutputPlane, "intercept", intercept_patch)
            op = elements.OutputPlane(z_0=10)
            op.propagate_ray(default_ray)
        intercept_patch.assert_called_once_with(default_ray)

    def test_pr_doesnt_call_refract(self, default_ray, ph, elements, monkeypatch):
        refract_mock = MagicMock(return_value=np.array([0., 0., 1.]))
        # mock both places incase they have imported the functing into elements
        with monkeypatch.context() as m:
            m.setattr(ph, "refract", refract_mock)
            if hasattr(elements, "refract"):
                m.setattr(elements, "refract", refract_mock)
            op = elements.OutputPlane(z_0=10)
            op.propagate_ray(default_ray)
        refract_mock.assert_not_called()

    def test_pr_calls_append_once(self, rays, elements, monkeypatch):
        append_mock = MagicMock(return_value=None)
        with monkeypatch.context() as m:
            m.setattr(rays.Ray, "append", append_mock)
            if hasattr(elements, "Ray"):
                m.setattr(elements.Ray, "append", append_mock)
            op = elements.OutputPlane(z_0=10)
            op.propagate_ray(rays.Ray())
        append_mock.assert_called_once()

        call_dict = {}
        for name, val in zip(signature(rays.Ray().append).parameters.keys(), append_mock.call_args.args):
            call_dict[name] = val
        call_dict.update(append_mock.call_args.kwargs)

        assert np.allclose(call_dict["pos"], [0., 0., 10.])
        assert np.allclose(call_dict["direc"], [0., 0., 1.])

    def test_parallel_intercept(self, rays, elements):
        ray = rays.Ray(pos=[0., 10., 0.])
        op = elements.OutputPlane(z_0=10)
        intercept = op.intercept(ray)
        assert np.allclose(intercept, [0., 10., 10.])

    def test_nonparallel_intercept(self, rays, elements):
        ray = rays.Ray(pos=[1., 10., 2.], direc=[-0.15, -0.5, 1.])
        op = elements.OutputPlane(z_0=10)
        intercept = op.intercept(ray)
        assert np.allclose(intercept, [-0.2, 6., 10.])


class TestTask10:

    TASK10_DEFAULT = b'QFNhdmVPdXRwdXQoInRhc2sxMCIpCmRlZiB0YXNrMTAoKToKICAgICIiIgogICAgVGFzayAxMC4KCiAgICBJbiB0aGlzIGZ1bmN0aW9uIHlvdSBzaG91bGQgY3JlYXRlIFJheSBvYmplY3RzIHdpdGggdGhlIGdpdmVuIGluaXRpYWwgcG9zaXRpb25zLgogICAgVGhlc2UgcmF5cyBzaG91bGQgYmUgcHJvcGFnYXRlZCB0aHJvdWdoIHRoZSBzdXJmYWNlLCB1cCB0byB0aGUgb3V0cHV0IHBsYW5lLgogICAgWW91IHNob3VsZCB0aGVuIHBsb3QgdGhlIHRyYWNrcyBvZiB0aGVzZSByYXlzLgogICAgVGhpcyBmdW5jdGlvbiBzaG91bGQgcmV0dXJuIHRoZSBtYXRwbG90bGliIGZpZ3VyZSBvZiB0aGUgcmF5IHBhdGhzLgoKICAgIFJldHVybnM6CiAgICAgICAgRmlndXJlOiB0aGUgcmF5IHBhdGggcGxvdC4KICAgICIiIgogICAgcmV0dXJuCg=='

    def test_doesnt_crash(self, task10_output, an):
        attempted = getsource(an.task10).encode('utf-8') != b64decode(TestTask10.TASK10_DEFAULT)
        assert attempted, "Task10 not attempted."

    def test_sr_created_once(self, sr_mock, task10_output):
        sr_mock.assert_called_once()

    def test_sr_setup_correctly(self, elements, an, monkeypatch):
        params = signature(elements.SphericalRefraction).parameters.keys()
        sr = MagicMock(wraps=elements.SphericalRefraction)
        with monkeypatch.context() as m:
            m.setattr(elements, "SphericalRefraction", sr)
            m.setattr(an, "SphericalRefraction", sr)
            an.task10()

        call_dict = {}
        for name, val in zip(params, sr.call_args.args):
            call_dict[name] = val
        call_dict.update(sr.call_args.kwargs)

        assert call_dict["z_0"] == 100
        assert call_dict['curvature'] == 0.03
        assert call_dict["n_1"] == 1.0
        assert call_dict["n_2"] == 1.5

    def test_op_created_once(self, op_mock, task10_output):
        op_mock.assert_called_once()

    def test_op_setup_correctly(self, op_mock, elements, task10_output):
        params = signature(elements.OutputPlane.__init__).parameters.keys()
        call_dict = {name: val for name, val in zip(params, op_mock.call_args.args)}
        call_dict.update(op_mock.call_args.kwargs)

        assert call_dict["z_0"] == 250.

    def test_rays_created(self, ray_mock, task10_output):
        ray_mock.assert_called()
        assert ray_mock.call_count > 1, "Only a single ray created"

    def test_ray_vertices_called(self, vert_mock, task10_output):
        vert_mock.assert_called()
        assert vert_mock.call_count > 1, "only called vertices on one ray"

    def test_output_fig_produced(self, task10_output):
        assert isinstance(task10_output, Figure)

    def test_plot_ready_for_marking(self, plots_for_marking_dir):
        for ext in ['png', 'pkl']:
            plot = plots_for_marking_dir / f"task10.{ext}"
            assert plot.exists(), f"plot file {plot.name!r} does not exist in {plot.parent}"
            assert plot.stat().st_size > 0, f"plot file {plot.name!r} is empty"


class TestTask11:

    TASK11_DEFAULT = b'QFNhdmVPdXRwdXQoInRhc2sxMSIsIHBsb3Rfb3V0cHV0X2luZGljZXM9aXRlbWdldHRlcigwKSkKZGVmIHRhc2sxMSgpOgogICAgIiIiCiAgICBUYXNrIDExLgoKICAgIEluIHRoaXMgZnVuY3Rpb24geW91IHNob3VsZCBwcm9wYWdhdGUgdGhlIHRocmVlIGdpdmVuIHBhcmF4aWFsIHJheXMgdGhyb3VnaCB0aGUgc3lzdGVtCiAgICB0byB0aGUgb3V0cHV0IHBsYW5lIGFuZCB0aGUgdHJhY2tzIG9mIHRoZXNlIHJheXMgc2hvdWxkIHRoZW4gYmUgcGxvdHRlZC4KICAgIFRoaXMgZnVuY3Rpb24gc2hvdWxkIHJldHVybiB0aGUgZm9sbG93aW5nIGl0ZW1zIGFzIGEgdHVwbGUgaW4gdGhlIGZvbGxvd2luZyBvcmRlcjoKICAgIDEuIHRoZSBtYXRwbG90bGliIGZpZ3VyZSBvYmplY3QgZm9yIHJheSBwYXRocwogICAgMi4gdGhlIGNhbGN1bGF0ZWQgZm9jYWwgcG9pbnQuCgogICAgUmV0dXJuczoKICAgICAgICB0dXBsZVtGaWd1cmUsIGZsb2F0XTogdGhlIHJheSBwYXRoIHBsb3QgYW5kIHRoZSBmb2NhbCBwb2ludAogICAgIiIiCiAgICByZXR1cm4K'

    def test_focal_point_exists(self, elements):
        assert isinstance(elements.SphericalRefraction.focal_point, (FunctionType, property))

    def test_focal_point(self, elements):
        sr = elements.SphericalRefraction(z_0=100,
                                          curvature=0.03,
                                          n_1=1.0,
                                          n_2=1.5,
                                          aperture=34)
        focal_point = sr.focal_point
        if isinstance(focal_point, MethodType):
            focal_point = focal_point()
        assert np.isclose(focal_point, 200.)

    def test_doesnt_crash(self, task11_output, an):
        attempted = getsource(an.task11).encode('utf-8') != b64decode(TestTask11.TASK11_DEFAULT)
        assert attempted, "Task11 not attempted."

    def test_output(self, task11_output):
        assert isinstance(task11_output[0], Figure)
        assert np.isclose(task11_output[1], 200.)

    def test_analysis_uses_focal_point(self, elements, an, monkeypatch):
        focal_point_mock = PropertyMock(return_value=200.)
        if isinstance(elements.SphericalRefraction.focal_point, FunctionType):
            focal_point_mock = MagicMock(return_value=200.)
        with monkeypatch.context() as m:
            m.setattr(elements.SphericalRefraction, "focal_point", focal_point_mock)
            if hasattr(an, "SphericalRefraction"):
                m.setattr(an.SphericalRefraction, "focal_point", focal_point_mock)
            an.task11()
        focal_point_mock.assert_called()

    def test_plot_ready_for_marking(self, plots_for_marking_dir):
        for ext in ['png', 'pkl']:
            plot = plots_for_marking_dir / f"task11.{ext}"
            assert plot.exists(), f"plot file {plot.name!r} does not exist in {plot.parent}"
            assert plot.stat().st_size > 0, f"plot file {plot.name!r} is empty"


class TestTask12:

    TASK12_DEFAULT = b'QFNhdmVPdXRwdXQoInRhc2sxMiIpCmRlZiB0YXNrMTIoKToKICAgICIiIgogICAgVGFzayAxMi4KCiAgICBJbiB0aGlzIGZ1bmN0aW9uIHlvdSBzaG91bGQgY3JlYXRlIGEgUmF5QnVuYmxlIGFuZCBwcm9wYWdhdGUgaXQgdG8gdGhlIG91dHB1dCBwbGFuZQogICAgYmVmb3JlIHBsb3R0aW5nIHRoZSB0cmFja3Mgb2YgdGhlIHJheXMuCiAgICBUaGlzIGZ1bmN0aW9uIHNob3VsZCByZXR1cm4gdGhlIG1hdHBsb3RsaWIgZmlndXJlIG9mIHRoZSB0cmFjayBwbG90LgoKICAgIFJldHVybnM6CiAgICAgICAgRmlndXJlOiB0aGUgdHJhY2sgcGxvdC4KICAgICIiIgogICAgcmV0dXJuCg=='

    def test_bundle_exists(self, rays):
        assert "RayBundle" in vars(rays)

    def test_bundle_class(self, rays):
        assert isinstance(rays.RayBundle, type)

    def test_genpolar_exists(self, genpolar):
        assert "rtrings" in vars(genpolar)

    def test_using_rtrings(self, rtrings_mock, rays, monkeypatch):
        with monkeypatch.context() as m:
            m.setattr(rays, "rtrings", rtrings_mock)
            rays.RayBundle()
        rtrings_mock.assert_called_once_with(rmax=5, nrings=5, multi=6)

    def test_bundles_rays(self, rays, bundle_var_name_map):
        rb = rays.RayBundle()
        rb_rays = getattr(rb, bundle_var_name_map['rays'])
        assert len(rb_rays) == 91

        positions = sorted((r.pos if not isinstance(r.pos, MethodType) else r.pos() for r in rb_rays), key=itemgetter(0, 1))
        expected_positions = sorted([[-5.000000e+00, 6.123234e-16, 0.000000e+00],
                                     [-4.890738, -1.03955845, 0.],
                                     [-4.890738, 1.03955845, 0.],
                                     [-4.56772729, -2.03368322, 0.],
                                     [-4.56772729, 2.03368322, 0.],
                                     [-4.04508497, -2.93892626, 0.],
                                     [-4.04508497, 2.93892626, 0.],
                                     [-4.0000000e+00, 4.8985872e-16, 0.0000000e+00],
                                     [-3.86370331, -1.03527618, 0.],
                                     [-3.86370331, 1.03527618, 0.],
                                     [-3.46410162, -2., 0.],
                                     [-3.46410162, 2., 0.],
                                     [-3.34565303, -3.71572413, 0.],
                                     [-3.34565303, 3.71572413, 0.],
                                     [-3.0000000e+00, 3.6739404e-16, 0.0000000e+00],
                                     [-2.82842712, -2.82842712, 0.],
                                     [-2.82842712, 2.82842712, 0.],
                                     [-2.81907786, -1.02606043, 0.],
                                     [-2.81907786, 1.02606043, 0.],
                                     [-2.5, -4.33012702, 0.],
                                     [-2.5, 4.33012702, 0.],
                                     [-2.29813333, -1.92836283, 0.],
                                     [-2.29813333, 1.92836283, 0.],
                                     [-2., -3.46410162, 0.],
                                     [-2.0000000e+00, 2.4492936e-16, 0.0000000e+00],
                                     [-2., 3.46410162, 0.],
                                     [-1.73205081, -1., 0.],
                                     [-1.73205081, 1., 0.],
                                     [-1.54508497, -4.75528258, 0.],
                                     [-1.54508497, 4.75528258, 0.],
                                     [-1.5, -2.59807621, 0.],
                                     [-1.5, 2.59807621, 0.],
                                     [-1.03527618, -3.86370331, 0.],
                                     [-1.03527618, 3.86370331, 0.],
                                     [-1., -1.73205081, 0.],
                                     [-1.0000000e+00, 1.2246468e-16, 0.0000000e+00],
                                     [-1., 1.73205081, 0.],
                                     [-0.52264232, -4.97260948, 0.],
                                     [-0.52264232, 4.97260948, 0.],
                                     [-0.52094453, -2.95442326, 0.],
                                     [-0.52094453, 2.95442326, 0.],
                                     [-0.5, -0.8660254, 0.],
                                     [-0.5, 0.8660254, 0.],
                                     [-7.34788079e-16, -4.00000000e+00, 0.00000000e+00],
                                     [-3.6739404e-16, -2.0000000e+00, 0.0000000e+00],
                                     [0., 0., 0.],
                                     [1.2246468e-16, 2.0000000e+00, 0.0000000e+00],
                                     [2.4492936e-16, 4.0000000e+00, 0.0000000e+00],
                                     [0.5, -0.8660254, 0.],
                                     [0.5, 0.8660254, 0.],
                                     [0.52094453, -2.95442326, 0.],
                                     [0.52094453, 2.95442326, 0.],
                                     [0.52264232, -4.97260948, 0.],
                                     [0.52264232, 4.97260948, 0.],
                                     [1., -1.73205081, 0.],
                                     [1., 0., 0.],
                                     [1., 1.73205081, 0.],
                                     [1.03527618, -3.86370331, 0.],
                                     [1.03527618, 3.86370331, 0.],
                                     [1.5, -2.59807621, 0.],
                                     [1.5, 2.59807621, 0.],
                                     [1.54508497, -4.75528258, 0.],
                                     [1.54508497, 4.75528258, 0.],
                                     [1.73205081, -1., 0.],
                                     [1.73205081, 1., 0.],
                                     [2., -3.46410162, 0.],
                                     [2., 0., 0.],
                                     [2., 3.46410162, 0.],
                                     [2.29813333, -1.92836283, 0.],
                                     [2.29813333, 1.92836283, 0.],
                                     [2.5, -4.33012702, 0.],
                                     [2.5, 4.33012702, 0.],
                                     [2.81907786, -1.02606043, 0.],
                                     [2.81907786, 1.02606043, 0.],
                                     [2.82842712, -2.82842712, 0.],
                                     [2.82842712, 2.82842712, 0.],
                                     [3., 0., 0.],
                                     [3.34565303, -3.71572413, 0.],
                                     [3.34565303, 3.71572413, 0.],
                                     [3.46410162, -2., 0.],
                                     [3.46410162, 2., 0.],
                                     [3.86370331, -1.03527618, 0.],
                                     [3.86370331, 1.03527618, 0.],
                                     [4., 0., 0.],
                                     [4.04508497, -2.93892626, 0.],
                                     [4.04508497, 2.93892626, 0.],
                                     [4.56772729, -2.03368322, 0.],
                                     [4.56772729, 2.03368322, 0.],
                                     [4.890738, -1.03955845, 0.],
                                     [4.890738, 1.03955845, 0.],
                                     [5., 0., 0.]], key=itemgetter(0, 1))
        assert np.allclose(positions, expected_positions)

    def test_bundle_not_inheritance(self, rays):
        # assert rays.Ray not in rays.RayBundle.mro()
        assert not issubclass(rays.RayBundle, rays.Ray)

    def test_bundle_args(self, rays):
        assert {"rmax", "nrings", "multi"}.issubset(signature(rays.RayBundle).parameters.keys())

    def test_bundle_construction(self, rays):
        rays.RayBundle(rmax=5., nrings=5)

    def test_prop_bundle_exists(self, rays):
        assert isinstance(rays.RayBundle.propagate_bundle, FunctionType)

    def test_prop_bundle_calles_prop_ray(self, rays, pr_mock, elements):
        sr = elements.SphericalRefraction(z_0=100, aperture=35., curvature=0.2, n_1=1., n_2=1.5)
        rb = rays.RayBundle(rmax=5., nrings=5)
        rb.propagate_bundle([sr])
        pr_mock.assert_called()

    def test_track_plot_exists(self, rays):
        assert isinstance(rays.RayBundle.track_plot, FunctionType)

    def test_track_plot_type(self, rays):
        rb = rays.RayBundle(rmax=5., nrings=5)
        ret = rb.track_plot()
        assert isinstance(ret, (Figure, tuple))
        if isinstance(ret, tuple):
            assert isinstance(ret[0], Figure)
            assert isinstance(ret[1], Axes)

    def test_track_plot_calles_vertices(self, vert_mock, rays):
        rb = rays.RayBundle(rmax=5., nrings=5)
        rb.track_plot()
        vert_mock.assert_called()

    def test_doesnt_crash(self, task12_output, an):
        attempted = getsource(an.task12).encode('utf-8') != b64decode(TestTask12.TASK12_DEFAULT)
        assert attempted, "Task12 not attempted."

    def test_ouput_fig_produced(self, task12_output):
        assert isinstance(task12_output, Figure)

    def test_analysis_uses_track_plot(self, track_plot_mock, task12_output):
        track_plot_mock.assert_called()

    def test_plot_ready_for_marking(self, plots_for_marking_dir):
        for ext in ['png', 'pkl']:
            plot = plots_for_marking_dir / f"task12.{ext}"
            assert plot.exists(), f"plot file {plot.name!r} does not exist in {plot.parent}"
            assert plot.stat().st_size > 0, f"plot file {plot.name!r} is empty"


class TestTask13:

    TASK13_DEFAULT = b'QFNhdmVPdXRwdXQoInRhc2sxMyIsIHBsb3Rfb3V0cHV0X2luZGljZXM9aXRlbWdldHRlcigwKSkKZGVmIHRhc2sxMygpOgogICAgIiIiCiAgICBUYXNrIDEzLgoKICAgIEluIHRoaXMgZnVuY3Rpb24geW91IHNob3VsZCBhZ2FpbiBjcmVhdGUgYW5kIHByb3BhZ2F0ZSBhIFJheUJ1bmRsZSB0byB0aGUgb3V0cHV0IHBsYW5lCiAgICBiZWZvcmUgcGxvdHRpbmcgdGhlIHNwb3QgcGxvdC4KICAgIFRoaXMgZnVuY3Rpb24gc2hvdWxkIHJldHVybiB0aGUgZm9sbG93aW5nIGl0ZW1zIGFzIGEgdHVwbGUgaW4gdGhlIGZvbGxvd2luZyBvcmRlcjoKICAgIDEuIHRoZSBtYXRwbG90bGliIGZpZ3VyZSBvYmplY3QgZm9yIHRoZSBzcG90IHBsb3QKICAgIDIuIHRoZSBzaW11bGF0aW9uIFJNUwoKICAgIFJldHVybnM6CiAgICAgICAgdHVwbGVbRmlndXJlLCBmbG9hdF06IHRoZSBzcG90IHBsb3QgYW5kIHJtcwogICAgIiIiCiAgICByZXR1cm4K'

    def test_spot_plot_exists(self, rays):
        assert isinstance(rays.RayBundle.spot_plot, FunctionType)

    def test_rms_exists(self, rays):
        assert isinstance(rays.RayBundle.rms, (FunctionType, property))

    def test_doesnt_crash(self, task13_output, an):
        attempted = getsource(an.task13).encode('utf-8') != b64decode(TestTask13.TASK13_DEFAULT)
        assert attempted, "Task13 not attempted."

    def test_output(self, task13_output):
        assert isinstance(task13_output[0], Figure)
        assert np.isclose(task13_output[1], 0.016176669411515444)

    def test_analysis_uses_spot_plot_rms(self, spot_plot_mock, rms_mock, task13_output):
        spot_plot_mock.assert_called()
        rms_mock.assert_called()

    def test_plot_ready_for_marking(self, plots_for_marking_dir):
        for ext in ['png', 'pkl']:
            plot = plots_for_marking_dir / f"task13.{ext}"
            assert plot.exists(), f"plot file {plot.name!r} does not exist in {plot.parent}"
            assert plot.stat().st_size > 0, f"plot file {plot.name!r} is empty"


class TestTask14:

    TASK14_DEFAULT = b'QFNhdmVPdXRwdXQoInRhc2sxNCIsIHBsb3Rfb3V0cHV0X2luZGljZXM9aXRlbWdldHRlcigwKSkKZGVmIHRhc2sxNCgpOgogICAgIiIiCiAgICBUYXNrIDE0LgoKICAgIEluIHRoaXMgZnVuY3Rpb24geW91IHdpbGwgdHJhY2UgYSBudW1iZXIgb2YgUmF5QnVuZGxlcyB0aHJvdWdoIHRoZSBvcHRpY2FsIHN5c3RlbSBhbmQKICAgIHBsb3QgdGhlIFJNUyBhbmQgZGlmZnJhY3Rpb24gc2NhbGUgZGVwZW5kZW5jZSBvbiBpbnB1dCBiZWFtIHJhZGlpLgogICAgVGhpcyBmdW5jdGlvbiBzaG91bGQgcmV0dXJuIHRoZSBmb2xsb3dpbmcgaXRlbXMgYXMgYSB0dXBsZSBpbiB0aGUgZm9sbG93aW5nIG9yZGVyOgogICAgMS4gdGhlIG1hdHBsb3RsaWIgZmlndXJlIG9iamVjdCBmb3IgdGhlIGRpZmZyYWN0aW9uIHNjYWxlIHBsb3QKICAgIDIuIHRoZSBzaW11bGF0aW9uIFJNUyBmb3IgaW5wdXQgYmVhbSByYWRpdXMgMi41CiAgICAzLiB0aGUgZGlmZnJhY3Rpb24gc2NhbGUgZm9yIGlucHV0IGJlYW0gcmFkaXVzIDIuNQoKICAgIFJldHVybnM6CiAgICAgICAgdHVwbGVbRmlndXJlLCBmbG9hdCwgZmxvYXRdOiB0aGUgcGxvdCwgdGhlIHNpbXVsYXRpb24gUk1TIHZhbHVlLCB0aGUgZGlmZnJhY3Rpb24gc2NhbGUuCiAgICAiIiIKICAgIHJldHVybgo='

    def test_doesnt_crash(self, task14_output, an):
        attempted = getsource(an.task14).encode('utf-8') != b64decode(TestTask14.TASK14_DEFAULT)
        assert attempted, "Task14 not attempted."

    def test_output(self, task14_output):
        assert isinstance(task14_output[0], Figure)
        assert np.isclose(task14_output[1], 0.0020035841295443506)
        assert np.isclose(task14_output[2], 0.01176)

    def test_plot_ready_for_marking(self, plots_for_marking_dir):
        for ext in ['png', 'pkl']:
            plot = plots_for_marking_dir / f"task14.{ext}"
            assert plot.exists(), f"plot file {plot.name!r} does not exist in {plot.parent}"
            assert plot.stat().st_size > 0, f"plot file {plot.name!r} is empty"


class TestTask15:

    TASK15_DEFAULT = b'QFNhdmVPdXRwdXQoWyJ0YXNrMTVhIiwgInRhc2sxNWIiXSwgcGxvdF9vdXRwdXRfaW5kaWNlcz1pdGVtZ2V0dGVyKDAsIDIpKQpkZWYgdGFzazE1KCk6CiAgICAiIiIKICAgIFRhc2sgMTUuCgogICAgSW4gdGhpcyBmdW5jdGlvbiB5b3Ugd2lsbCBjcmVhdGUgcGxhbm8tY29udmV4IGxlbnNlcyBpbiBlYWNoIG9yaWVudGF0aW9uIGFuZCBwcm9wYWdhdGUgYSBSYXlCdW5kbGUKICAgIHRocm91Z2ggZWFjaCB0byB0aGVpciByZXNwZWN0aXZlIGZvY2FsIHBvaW50LiBZb3Ugc2hvdWxkIHRoZW4gcGxvdCB0aGUgc3BvdCBwbG90IGZvciBlYWNoIG9yaWVudGF0aW9uLgogICAgVGhpcyBmdW5jdGlvbiBzaG91bGQgcmV0dXJuIHRoZSBmb2xsb3dpbmcgaXRlbXMgYXMgYSB0dXBsZSBpbiB0aGUgZm9sbG93aW5nIG9yZGVyOgogICAgMS4gdGhlIG1hdHBsb3RsaWIgZmlndXJlIG9iamVjdCBmb3IgdGhlIHNwb3QgcGxvdCBmb3IgdGhlIHBsYW5vLWNvbnZleCBzeXN0ZW0KICAgIDIuIHRoZSBmb2NhbCBwb2ludCBmb3IgdGhlIHBsYW5vLWNvbnZleCBsZW5zCiAgICAzLiB0aGUgbWF0cGxvdGxpYiBmaWd1cmUgb2JqZWN0IGZvciB0aGUgc3BvdCBwbG90IGZvciB0aGUgY29udmV4LXBsYW5vIHN5c3RlbQogICAgNCAgdGhlIGZvY2FsIHBvaW50IGZvciB0aGUgY29udmV4LXBsYW5vIGxlbnMKCgogICAgUmV0dXJuczoKICAgICAgICB0dXBsZVtGaWd1cmUsIGZsb2F0LCBGaWd1cmUsIGZsb2F0XTogdGhlIHNwb3QgcGxvdHMgYW5kIHJtcyBmb3IgcGxhbm8tY29udmV4IGFuZCBjb252ZXgtcGxhbm8uCiAgICAiIiIKICAgIHJldHVybgo='

    def test_lenses_exists(self, lenses):
        pass

    def test_planoconvex_exists(self, lenses):
        assert "PlanoConvex" in vars(lenses)

    def test_construction_args(self, lenses):
        params = signature(lenses.PlanoConvex).parameters.keys()
        basic = {"z_0", "curvature1", "curvature2", "n_inside", "n_outside", "thickness", "aperture"}
        advanced = {"z_0", "curvature", "n_inside", "n_outside", "thickness", "aperture"}
        assert basic.issubset(params) or advanced.issubset(params)

    def test_doesnt_crash(self, task15_output, an):
        attempted = getsource(an.task15).encode('utf-8') != b64decode(TestTask15.TASK15_DEFAULT)
        assert attempted, "Task15 not attempted."

    def test_output(self, task15_output):
        pc_fig, pc_focal_point, cp_fig, cp_focal_point = task15_output
        assert isinstance(pc_fig, Figure)
        assert isinstance(cp_fig, Figure)
        assert np.isclose(pc_focal_point, 201.74922600619198)
        assert np.isclose(cp_focal_point, 198.45281250408226)

    def test_2SR_objects_created(self, sr_mock, task15_output):
        assert sr_mock.call_count >= 2

    def test_OP_object_created(self, op_mock, task15_output):
        assert op_mock.call_count == 2

    def test_plots_ready_for_marking(self, plots_for_marking_dir):
        for index, ext in product(ascii_lowercase[:2], ['png', 'pkl']):
            plot = plots_for_marking_dir / f"task15{index}.{ext}"
            assert plot.exists(), f"plot file {plot.name!r} does not exist in {plot.parent}"
            assert plot.stat().st_size > 0, f"plot file {plot.name!r} is empty"


class TestTask16:

    TASK16_DEFAULT = b'QFNhdmVPdXRwdXQoInRhc2sxNiIsIHBsb3Rfb3V0cHV0X2luZGljZXM9aXRlbWdldHRlcigwKSkKZGVmIHRhc2sxNigpOgogICAgIiIiCiAgICBUYXNrIDE2LgoKICAgIEluIHRoaXMgZnVuY3Rpb24geW91IHdpbGwgYmUgYWdhaW4gcGxvdHRpbmcgdGhlIHJhZGlhbCBkZXBlbmRlbmNlIG9mIHRoZSBSTVMgYW5kIGRpZmZyYWN0aW9uIHZhbHVlcwogICAgZm9yIGVhY2ggb3JpZW50YXRpb24gb2YgeW91ciBsZW5zLgogICAgVGhpcyBmdW5jdGlvbiBzaG91bGQgcmV0dXJuIHRoZSBmb2xsb3dpbmcgaXRlbXMgYXMgYSB0dXBsZSBpbiB0aGUgZm9sbG93aW5nIG9yZGVyOgogICAgMS4gdGhlIG1hdHBsb3RsaWIgZmlndXJlIG9iamVjdCBmb3IgdGhlIGRpZmZyYWN0aW9uIHNjYWxlIHBsb3QKICAgIDIuIHRoZSBSTVMgZm9yIGlucHV0IGJlYW0gcmFkaXVzIDMuNSBmb3IgdGhlIHBsYW5vLWNvbnZleCBzeXN0ZW0KICAgIDMuIHRoZSBSTVMgZm9yIGlucHV0IGJlYW0gcmFkaXVzIDMuNSBmb3IgdGhlIGNvbnZleC1wbGFubyBzeXN0ZW0KICAgIDQgIHRoZSBkaWZmcmFjdGlvbiBzY2FsZSBmb3IgaW5wdXQgYmVhbSByYWRpdXMgMy41CgogICAgUmV0dXJuczoKICAgICAgICB0dXBsZVtGaWd1cmUsIGZsb2F0LCBmbG9hdCwgZmxvYXRdOiB0aGUgcGxvdCwgUk1TIGZvciBwbGFuby1jb252ZXgsIFJNUyBmb3IgY29udmV4LXBsYW5vLCBkaWZmcmFjdGlvbiBzY2FsZS4KICAgICIiIgogICAgcmV0dXJuCg=='

    def test_doesnt_crash(self, task16_output, an):
        attempted = getsource(an.task16).encode('utf-8') != b64decode(TestTask16.TASK16_DEFAULT)
        assert attempted, "Task16 not attempted."

    def test_output(self, task16_output):
        fig, pc_rms, cp_rms, diff = task16_output
        assert isinstance(fig, Figure)
        assert np.isclose(pc_rms, 0.012687332076619933)
        assert np.isclose(cp_rms, 0.0031927627499460415)
        assert np.isclose(diff, 0.008126934984520126)

    def test_plot_ready_for_marking(self, plots_for_marking_dir):
        for ext in ['png', 'pkl']:
            plot = plots_for_marking_dir / f"task16.{ext}"
            assert plot.exists(), f"plot file {plot.name!r} does not exist in {plot.parent}"
            assert plot.stat().st_size > 0, f"plot file {plot.name!r} is empty"

class TestTask17:

    TASK17_DEFAULT = b'QFNhdmVPdXRwdXQoInRhc2sxNyIsIHBsb3Rfb3V0cHV0X2luZGljZXM9aXRlbWdldHRlcigwKSkKZGVmIHRhc2sxNygpOgogICAgIiIiCiAgICBUYXNrIDE3LgoKICAgIEluIHRoaXMgZnVuY3Rpb24geW91IHdpbGwgYmUgZmlyc3QgcGxvdHRpbmcgdGhlIHNwb3QgcGxvdCBmb3IgeW91ciBQbGFub0NvbnZleCBsZW5zIHdpdGggdGhlIGN1cnZlZAogICAgc2lkZSBmaXJzdCAoYXQgdGhlIGZvY2FsIHBvaW50KS4gWW91IHdpbGwgdGhlbiBiZSBvcHRpbWlzaW5nIHRoZSBjdXJ2YXR1cmVzIG9mIGEgQmlDb252ZXggbGVucwogICAgaW4gb3JkZXIgdG8gbWluaW1pc2UgdGhlIFJNUyBzcG90IHNpemUgYXQgdGhlIHNhbWUgZm9jYWwgcG9pbnQuIFRoaXMgZnVuY3Rpb24gc2hvdWxkIHJldHVybgogICAgdGhlIGZvbGxvd2luZyBpdGVtcyBhcyBhIHR1cGxlIGluIHRoZSBmb2xsb3dpbmcgb3JkZXI6CiAgICAxLiBUaGUgY29tcGFyaXNvbiBzcG90IHBsb3QgZm9yIGJvdGggUGxhbm9Db252ZXggKGN1cnZlZCBzaWRlIGZpcnN0KSBhbmQgQmlDb252ZXggbGVuc2VzIGF0IFBsYW5vQ29udmV4IGZvY2FsIHBvaW50LgogICAgMi4gVGhlIFJNUyBzcG90IHNpemUgZm9yIHRoZSBQbGFub0NvbnZleCBsZW5zIGF0IGZvY2FsIHBvaW50CiAgICAzLiB0aGUgUk1TIHNwb3Qgc2l6ZSBmb3IgdGhlIEJpQ29udmV4IGxlbnMgYXQgUGxhbm9Db252ZXggZm9jYWwgcG9pbnQKCiAgICBSZXR1cm5zOgogICAgICAgIHR1cGxlW0ZpZ3VyZSwgZmxvYXQsIGZsb2F0XTogVGhlIGNvbWJpbmVkIHNwb3QgcGxvdCwgUk1TIGZvciB0aGUgUEMgbGVucywgUk1TIGZvciB0aGUgQmlDb252ZXggbGVucwogICAgIiIiCiAgICByZXR1cm4K'

    def test_doesnt_crash(self, task17_output, an):
        attempted = getsource(an.task17).encode('utf-8') != b64decode(TestTask17.TASK17_DEFAULT)
        assert attempted, "Task17 not attempted."

    def test_biconvex_exists(self, lenses):
        assert "BiConvex" in vars(lenses)

    def test_not_pc_inheritance(self, lenses):
        assert not issubclass(lenses.BiConvex, lenses.PlanoConvex)

    def test_output(self, task17_output):
        fig, cp_rms, bc_rms = task17_output
        assert isinstance(fig, Figure)
        assert np.isclose(cp_rms, 0.009341789683984076)
        assert bc_rms < cp_rms

    def test_plot_ready_for_marking(self, plots_for_marking_dir):
        for ext in ['png', 'pkl']:
            plot = plots_for_marking_dir / f"task17.{ext}"
            assert plot.exists(), f"plot file {plot.name!r} does not exist in {plot.parent}"
            assert plot.stat().st_size > 0, f"plot file {plot.name!r} is empty"

class TestTask18:

    TASK18_DEFAULT = b'QFNhdmVPdXRwdXQoInRhc2sxOCIsIHBsb3Rfb3V0cHV0X2luZGljZXM9aXRlbWdldHRlcigwKSkKZGVmIHRhc2sxOCgpOgogICAgIiIiCiAgICBUYXNrIDE4LgoKICAgIEluIHRoaXMgZnVuY3Rpb24geW91IHdpbGwgYmUgdGVzdGluZyB5b3VyIHJlZmxlY3Rpb24gbW9kZWxsaW5nLiBDcmVhdGUgYSBuZXcgU3BoZXJpY2FsUmVmbGVjdGluZyBzdXJmYWNlCiAgICBhbmQgdHJhY2UgYSBSYXlCdW5kbGUgdGhyb3VnaCBpdCB0byB0aGUgT3V0cHV0UGxhbmUuVGhpcyBmdW5jdGlvbiBzaG91bGQgcmV0dXJuCiAgICB0aGUgZm9sbG93aW5nIGl0ZW1zIGFzIGEgdHVwbGUgaW4gdGhlIGZvbGxvd2luZyBvcmRlcjoKICAgIDEuIFRoZSB0cmFjayBwbG90IHNob3dpbmcgcmVmbGVjdGluZyByYXkgYnVuZGxlIG9mZiBTcGhlcmljYWxSZWZsZWN0aW9uIHN1cmZhY2UuCiAgICAyLiBUaGUgZm9jYWwgcG9pbnQgb2YgdGhlIFNwaGVyaWNhbFJlZmxlY3Rpb24gc3VyZmFjZS4KCiAgICBSZXR1cm5zOgogICAgICAgIHR1cGxlW0ZpZ3VyZSwgZmxvYXRdOiBUaGUgdHJhY2sgcGxvdCwgdGhlIGZvY2FsIHBvaW50CiAgICAiIiIKICAgIHJldHVybgo='

    def test_doesnt_crash(self, task18_output, an):
        attempted = getsource(an.task18).encode('utf-8') != b64decode(TestTask18.TASK18_DEFAULT)
        assert attempted, "Task18 not attempted."

    def test_srefl_exists(self, elements):
        assert "SphericalReflection" in vars(elements)

    def test_output(self, task18_output):
        fig, refl_fp = task18_output
        assert isinstance(fig, Figure)
        assert np.isclose(refl_fp, 75.0)

    def test_plot_ready_for_marking(self, plots_for_marking_dir):
        for ext in ['png', 'pkl']:
            plot = plots_for_marking_dir / f"task18.{ext}"
            assert plot.exists(), f"plot file {plot.name!r} does not exist in {plot.parent}"
            assert plot.stat().st_size > 0, f"plot file {plot.name!r} is empty"


class TestTask19:

    TASK19_DEFAULT = b'QFNhdmVPdXRwdXQoWyJ0YXNrMTlhIiwgInRhc2sxOWIiXSkKZGVmIHRhc2sxOSgpOgogICAgIiIiCiAgICBUYXNrIDE5LgoKICAgIEluIHRoaXMgZnVuY3Rpb24geW91IHdpbGwgYmUgcXVhbnRpZmluZyB0aGUgYW1vdW50IG9mIHNwaGVyaWNhbCBhYmVycmF0aW9uLgogICAgQ3JlYXRlIGEgbmV3IENvbnZleFBsYW5vIGxlbnMgYW5kIHRyYWNlIGEgUmF5QnVuZGxlIHRocm91Z2ggaXQgdG8gdGhlIE91dHB1dFBsYW5lCiAgICBsb2NhdGVkIGF0IHRoZSBmb2NhbCBwb2ludC4KICAgIFRoaXMgZnVuY3Rpb24gc2hvdWxkIHJldHVybiB0d28gbWF0cGxvdGxpYiBmaWd1cmVzIGFzIGEgdHVwbGUgaW4gdGhlIGZvbGxvd2luZyBvcmRlcjoKICAgIDEuIFRoZSBwbG90IHNob3dpbmcgdGhlIHRyYW5zdmVyc2Ugc3BoZXJpY2FsIGFiZXJyYXRpb24gaS5lLiBybXMgYXMgYSBmdW5jdGlvbiBvZiBiZWFtIHJhZGl1cyBhdCBmb2NhbCBwb2ludC4KICAgIDIuIFRoZSBwbG90IHNob3dpbmcgdGhlIGxvbmdpdHVkaW5hbCBzcGhlcmljYWwgYWJlcnJhdGlvbiBpLmUuIHogaW50ZXJjZXB0IG9mIGEgcmF5IGFzIGEgZnVuY3Rpb24KICAgICAgIG9mIHRyYW5zdmVyc2UgZGlzdGFuY2UgZnJvbSBvcHRpY2FsIGF4aXMuCgogICAgUmV0dXJuczoKICAgICAgICB0dXBsZVtGaWd1cmUsIEZpZ3VyZV06IFRyYW5zdmVyc2Ugc3BoZXJpY2FsIGFiZXJyYXRpb24gcGxvdCwgbG9uZ2l0dWRpbmFsIHNwaGVyaWNhbCBhYmVycmF0aW9uIHBsb3QKICAgICIiIgogICAgcmV0dXJuCg=='

    def test_doesnt_crash(self, task19_output, an):
        attempted = getsource(an.task19).encode('utf-8') != b64decode(TestTask19.TASK19_DEFAULT)
        assert attempted, "Task19 not attempted."

    def test_z_int(self, rays):
        assert "z_int" in vars(rays.Ray)
        r1 = rays.Ray(pos=[4., 4., 0.], direc=[-1., -1., 1.])
        z1int = r1.z_int
        if isinstance(z1int, MethodType):
            z1int = z1int()()
        assert z1int == 4.
        r2 = rays.Ray(pos=[3., 4., 0.], direc=[-3./7, -4./7, 1.])
        z2int = r2.z_int
        if isinstance(z2int, MethodType):
            z2int = z2int()()
        assert z2int == 7.

    def test_plots_ready_for_marking(self, plots_for_marking_dir):
        for index, ext in product(ascii_lowercase[:2], ['png', 'pkl']):
            plot = plots_for_marking_dir / f"task19{index}.{ext}"
            assert plot.exists(), f"plot file {plot.name!r} does not exist in {plot.parent}"
            assert plot.stat().st_size > 0, f"plot file {plot.name!r} is empty"


class TestTask20:

    TASK20_DEFAULT = b'QFNhdmVPdXRwdXQoInRhc2syMCIpCmRlZiB0YXNrMjAoKToKICAgICIiIgogICAgVGFzayAyMC4KCiAgICBJbiB0aGlzIGZ1bmN0aW9uIHlvdSB3aWxsIGludmVzdGlnYXRlIGRpc3BlcnNpb24uIENyZWF0ZSBzZXZlcmFsIFJheXMgd2l0aCBkaWZmZXJlbnQgd2F2ZWxlbmd0aHMuCiAgICBQbG90IHRoZSBwYXRocyBvZiB0aGVzZSByYXlzIHRocm91Z2ggYSBCaUNvbnZleCBsZW5zIG1hZGUgb2YgQks3IGdsYXNzIHRvIGFuIE91dHB1dFBsYW5lIGF0IHo9MjAwLgogICAgVGhpcyBmdW5jdGlvbiBzaG91bGQgcmV0dXJuIHRoZSB0cmFjayBwbG90IG9mIHRoZSBwYXRoIG9mIHlvdXIgUmF5cyB0aHJvdWdoIHRoZSBnbGFzcyBsZW5zLgoKICAgIFJldHVybnM6CiAgICAgICAgRmlndXJlOiBUaGUgcmF5IHRyYWNrIHBsb3QuCiAgICAiIiIKICAgIHJldHVybgo='

    def test_doesnt_crash(self, task20_output, an):
        attempted = getsource(an.task20).encode('utf-8') != b64decode(TestTask20.TASK20_DEFAULT)
        assert attempted, "Task20 not attempted."

    def test_ray_wavelength(self, rays):
        assert "wavelength" in signature(rays.Ray).parameters
        assert "wavelength" in vars(rays.Ray)

    def test_dispersive_material(self, ph, elements):
        assert "DispersiveMaterial" in vars(ph)
        assert "ref_index" in vars(ph.DispersiveMaterial)
        assert "wavelength" in signature(ph.DispersiveMaterial.ref_index).parameters

    def test_SR_n1_n2(self, ph, elements):
        assert "wavelength" in signature(elements.SphericalRefraction.n_1).parameters
        assert "wavelength" in signature(elements.SphericalRefraction.n_2).parameters

        disp_mat1 = MagicMock(spec=ph.DispersiveMaterial)
        disp_mat1.ref_index = MagicMock()
        sr1 = elements.SphericalRefraction(z_0=100, curvature=0.03, n_1=disp_mat1, n_2=1.0, aperture=34)
        assert sr1.n_2() == 1.
        sr1.n_1(wavelength=700e-9)
        assert disp_mat1.ref_index.called

        disp_mat2 = MagicMock(spec=ph.DispersiveMaterial)
        disp_mat2.ref_index = MagicMock()
        sr1 = elements.SphericalRefraction(z_0=100, curvature=0.03, n_1=1.0, n_2=disp_mat2, aperture=34)
        assert sr1.n_1() == 1.
        sr1.n_2(wavelength=700e-9)
        assert disp_mat2.ref_index.called

    def test_SR_propagate(self, rays, elements, ph, monkeypatch):
        ray = rays.Ray(pos=[0., 1., 0.,], direc=[0., 0., 1.], wavelength=700e-9)
        disp_mat = MagicMock(spec=ph.DispersiveMaterial)
        disp_mat.ref_index = MagicMock(return_value=1.5)
        sr = elements.SphericalRefraction(z_0=100, curvature=0.03, n_1=1.0, n_2=disp_mat, aperture=34)
        n_2 = MagicMock(wraps=sr.n_2)
        with monkeypatch.context() as ctx:
            ctx.setattr(sr, "n_2", n_2)
            sr.propagate_ray(ray)
        assert n_2.called
        assert disp_mat.ref_index.called

    def test_plot_ready_for_marking(self, plots_for_marking_dir):
        for ext in ['png', 'pkl']:
            plot = plots_for_marking_dir / f"task20.{ext}"
            assert plot.exists(), f"plot file {plot.name!r} does not exist in {plot.parent}"
            assert plot.stat().st_size > 0, f"plot file {plot.name!r} is empty"
