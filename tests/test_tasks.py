from types import FunctionType, MethodType, NoneType
from unittest.mock import MagicMock, PropertyMock
from inspect import getmembers, isclass, signature
import numpy as np
import matplotlib as mpl
from matplotlib.figure import Figure
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
        assert np.linalg.norm(ph.refract(direc=np.array([0., 0.05, 1.]),
                                         normal=np.array([0., -1.9, -1.]),
                                         n_1=1.0, n_2=1.5)) == 1.

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

    def test_doesnt_crash(self, task8_output):
        pass

    def test_ray_created(self, an, rays, monkeypatch):
        mock_ray_init = MagicMock(wraps=rays.Ray)
        with monkeypatch.context() as m:
            m.setattr(rays, "Ray", mock_ray_init)
            if hasattr(an, "Ray"):
                m.setattr(an, "Ray", mock_ray_init)
            an.task8()
        mock_ray_init.assert_called()
        assert mock_ray_init.call_count > 1, "Only a single ray created."

    def test_sr_created(self, an, elements, monkeypatch):
        mock_sr_class = MagicMock(wraps=elements.SphericalRefraction)
        with monkeypatch.context() as m:
            m.setattr(elements, "SphericalRefraction", mock_sr_class)
            if hasattr(an, "SphericalRefraction"):
                m.setattr(an, "SphericalRefraction", mock_sr_class)
            an.task8()
        mock_sr_class.assert_called()

    def test_propagate_ray_called(self, an, elements, monkeypatch):
        mock_pr = MagicMock()
        with monkeypatch.context() as m:
            m.setattr(elements.SphericalRefraction, "propagate_ray", mock_pr)
            if hasattr(an, "SphericalRefraction"):
                m.setattr(an.SphericalRefraction, "propagate_ray", mock_pr)
            an.task8()
        mock_pr.assert_called()
        assert mock_pr.call_count > 1, "Propagate only called once"


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

    def test_doesnt_crash(self, task10_output):
        pass

    def test_sr_created_once(self, sr_mock, an):
        an.task10()
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

    def test_op_created_once(self, op_mock, an):
        an.task10()
        op_mock.assert_called_once()

    def test_op_setup_correctly(self, elements, an, monkeypatch):
        params = signature(elements.OutputPlane).parameters.keys()
        op = MagicMock(wraps=elements.OutputPlane)
        with monkeypatch.context() as m:
            m.setattr(elements, "OutputPlane", op)
            m.setattr(an, "OutputPlane", op)
            an.task10()
        call_dict = {}
        for name, val in zip(params, op.call_args.args):
            call_dict[name] = val
        call_dict.update(op.call_args.kwargs)

        assert call_dict["z_0"] == 250.

    def test_rays_created(self, ray_mock, an):
        an.task10()
        ray_mock.assert_called()
        assert ray_mock.call_count > 1, "Only a single ray created"

    def test_ray_vertices_called(self, vert_mock, an):
        an.task10()
        vert_mock.assert_called()
        assert vert_mock.call_count > 1, "only called vertices on one ray"

    def test_output_fig_produced(self, task10_output):
        assert isinstance(task10_output, Figure)

    # @check_figures_equal(ref_path="task10", tol=32)
    # def test_plot10(self, task10_output):
    #     return task10_output


class TestTask11:

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

    def test_doesnt_crash(self, task11_output):
        pass

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

    # @check_figures_equal(ref_path="task11", tol=32)
    # def test_plot10(self, task11_output):
    #     return task11_output[0]


class TestTask12:

    def test_bundle_exists(self, rays):
        assert "RayBundle" in vars(rays)

    def test_bundle_class(self, rays):
        assert isinstance(rays.RayBundle, type)

    def test_bundle_not_inheritance(self, rays):
        # assert rays.Ray not in rays.RayBundle.mro()
        assert not issubclass(rays.RayBundle, rays.Ray)

    def test_bundle_args(self, rays):
        assert {"rmax", "nrings", "multi"}.issubset(signature(rays.RayBundle).parameters.keys())

    def test_bundle_construction(self, rays):
        rays.RayBundle(rmax=5., nrings=5)

    def test_prop_bundle_exists(self, rays):
        assert isinstance(rays.RayBundle.propagate_bundle, FunctionType)

    def test_prop_bundle_calles_prop_ray(self, rays, elements, pr_mock):
        sr = elements.SphericalRefraction(z_0=100, aperture=35., curvature=0.2, n_1=1., n_2=1.5)
        rb = rays.RayBundle(rmax=5., nrings=5)
        rb.propagate_bundle([sr])
        pr_mock.assert_called()

    def test_track_plot_exists(self, rays):
        assert isinstance(rays.RayBundle.track_plot, FunctionType)

    def test_track_plot_type(self, rays):
        rb = rays.RayBundle(rmax=5., nrings=5)
        assert isinstance(rb.track_plot(), Figure)

    def test_track_plot_calles_vertices(self, rays, vert_mock):
        rb = rays.RayBundle(rmax=5., nrings=5)
        rb.track_plot()
        vert_mock.assert_called()

    def test_doesnt_crash(self, task12_output):
        pass

    def test_ouput_fig_produced(self, task12_output):
        assert isinstance(task12_output, Figure)

    def test_analysis_uses_track_plot(self, rays, an, monkeypatch):
        track_plot_mock = MagicMock()
        with monkeypatch.context() as m:
            m.setattr(rays.RayBundle, "track_plot", track_plot_mock)
            if hasattr(an, "RayBundle"):
                m.setattr(an.RayBundle, "track_plot", track_plot_mock)
            an.task12()
        track_plot_mock.assert_called()

    # @check_figures_equal(ref_path="task12", tol=32)
    # def test_plot12(self, task12_output):
    #     return task12_output


class TestTask13:

    def test_spot_plot_exists(self, rays):
        assert isinstance(rays.RayBundle.spot_plot, FunctionType)

    def test_rms_exists(self, rays):
        assert isinstance(rays.RayBundle.rms, (FunctionType, property))

    def test_doesnt_crash(self, task13_output):
        pass

    def test_output(self, task13_output):
        assert isinstance(task13_output[0], Figure)
        assert np.isclose(task13_output[1], 0.016176669411515444)

    def test_analysis_uses_spot_plot_rms(self, rays, an, monkeypatch):
        spot_plot_mock = MagicMock()
        pm = PropertyMock
        if isinstance(rays.RayBundle.rms, FunctionType):
            pm = MagicMock
        rms_mock = pm()
        with monkeypatch.context() as m:
            m.setattr(rays.RayBundle, "spot_plot", spot_plot_mock)
            m.setattr(rays.RayBundle, "rms", rms_mock)
            if hasattr(an, "RayBundle"):
                m.setattr(an.RayBundle, "spot_plot", spot_plot_mock)
                m.setattr(an.RayBundle, "rms", rms_mock)
            an.task13()
        spot_plot_mock.assert_called()
        rms_mock.assert_called()

    # @check_figures_equal(ref_path="task13", tol=33)
    # def test_plot13(self, task13_output):
    #     return task13_output[0]


class TestTask14:

    def test_doesnt_crash(self, task14_output):
        pass

    def test_output(self, task14_output):
        assert isinstance(task14_output[0], Figure)
        assert np.isclose(task14_output[1], 0.0020035841295443506)
        assert np.isclose(task14_output[2], 0.01176)

    # @check_figures_equal(ref_path="task14", tol=33)
    # def test_plot14(self, task14_output):
    #     return task14_output


class TestTask15:

    def test_lenses_exists(self, lenses):
        pass

    def test_planoconvex_exists(self, lenses):
        assert "PlanoConvex" in vars(lenses)

    def test_construction_args(self, lenses):
        params = signature(lenses.PlanoConvex).parameters.keys()
        basic = {"z_0", "curvature1", "curvature2", "n_inside", "n_outside", "thickness", "aperture"}
        advanced = {"z_0", "curvature", "n_inside", "n_outside", "thickness", "aperture"}
        assert basic.issubset(params) or advanced.issubset(params)

    def test_doesnt_crash(self, task15_output):
        pass

    def test_output(self, task15_output):
        pc_fig, pc_focal_point, cp_fig, cp_focal_point = task15_output
        assert isinstance(pc_fig, Figure)
        assert isinstance(cp_fig, Figure)
        assert np.isclose(pc_focal_point, 201.74922600619198)
        assert np.isclose(cp_focal_point, 198.45281250408226)

    def test_2SR_objects_created(self, sr_mock_with_lenses, an):
        an.task15()
        assert sr_mock_with_lenses.call_count >= 2

    def test_OP_object_created(self, op_mock, an):
        an.task15()
        assert op_mock.call_count == 2

    # @check_figures_equal(ref_path="task15pc", tol=33)
    # def test_plot15pc(self, task15_output):
    #     return task15_output[0]

    # @check_figures_equal(ref_path="task15cp", tol=33)
    # def test_plot15cp(self, task15_output):
    #     return task15_output[2]


class TestTask16:
    def test_doesnt_crash(self, task16_output):
        pass

    def test_output(self, task16_output):
        fig, pc_rms, cp_rms, diff = task16_output
        assert isinstance(fig, Figure)
        assert np.isclose(pc_rms, 0.012687332076619933)
        assert np.isclose(cp_rms, 0.0031927627499460415)
        assert np.isclose(diff, 0.008126934984520126)

    # @check_figures_equal(ref_path="task16", tol=33)
    # def test_plot16(self, task16_output):
    #     return task16_output
