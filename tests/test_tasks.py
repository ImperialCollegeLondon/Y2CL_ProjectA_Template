from types import FunctionType, MethodType, NoneType
from unittest.mock import MagicMock
from inspect import getmembers, isclass, signature
import numpy as np
import matplotlib as mpl
import pytest
from utils import check_figures_equal


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
        assert elements.OpticalElement in elements.SphericalRefraction.mro()

    def test_construction_args(self, elements):
        assert {"z_0", "aperture", "curvature", "n_1", "n_2"}.issubset(signature(elements.SphericalRefraction).parameters.keys())

    def test_construction(self, elements):
        elements.SphericalRefraction(z_0=10., aperture=5., curvature=0.2, n_1=1., n_2=1.5)


class TestTask5:

    def test_intercept_exists(self, elements):
        assert isinstance(elements.SphericalRefraction.intercept, FunctionType)

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

    ## TODO: pick up here.

    def test_sr_created(self, an, elements, monkeypatch):
        mock_sr_class = MagicMock()
        monkeypatch.setattr(elements, "SphericalRefraction", mock_sr_class)
        if hasattr(an, "SphericalRefraction"):
            monkeypatch.setattr(an, "SphericalRefraction", mock_sr_class)
        try:
            an.task7()
        except: pass
        mock_sr_class.assert_called()

    def test_sr_calls_propagate(self, an, elements, monkeypatch):
        mock_pr = MagicMock()
        monkeypatch.setattr(elements.SphericalRefraction, "propagate_ray", mock_pr)
        if hasattr(an, "SphericalRefraction"):
            monkeypatch.setattr(an.SphericalRefraction, "propagate_ray", mock_pr)
        try:
            an.task7()
        except: pass
        mock_pr.assert_called()


    def test_multiple_propagates(self, an, elements, monkeypatch):
        mock_pr = MagicMock()
        monkeypatch.setattr(elements.SphericalRefraction, "propagate_ray", mock_pr)
        if hasattr(an, "SphericalRefraction"):
            monkeypatch.setattr(an.SphericalRefraction, "propagate_ray", mock_pr)
        try:
            an.task7()
        except: pass
        assert mock_pr.call_count > 1


class TestTask8ish:
    def test_op_exists(self, elements):
        assert hasattr(elements, "OutputPlane")

    def test_inheritance(self, elements):
        assert elements.OpticalElement in elements.OutputPlane.mro()

    def test_construction(self, elements):
        elements.OutputPlane(z_0=250.)

    def test_pr_calls_intercept_once(self, default_ray, elements, monkeypatch):
        intercept_patch = MagicMock(return_value=np.array([1., 2., 3.]))
        monkeypatch.setattr(elements.OutputPlane, "intercept", intercept_patch)
        op = elements.OutputPlane(z_0=10)
        op.propagate_ray(default_ray)
        intercept_patch.assert_called_once_with(default_ray)

    def test_pr_doesnt_call_refract(self, default_ray, ph, elements, monkeypatch):
        refract_mock = MagicMock(return_value=np.array([0., 0., 1.]))
        # mock both places incase they have imported the functing into elements
        monkeypatch.setattr(ph, "refract", refract_mock)
        if hasattr(elements, "refract"):
            monkeypatch.setattr(elements, "refract", refract_mock)
        op = elements.OutputPlane(z_0=10)
        op.propagate_ray(default_ray)
        refract_mock.assert_not_called()

    def test_pr_calls_append_once(self, default_ray, rays, elements, monkeypatch):
        append_mock = MagicMock(return_value=None)
        monkeypatch.setattr(rays.Ray, "append", append_mock)
        if hasattr(elements, "Ray"):
            monkeypatch.setattr(elements.Ray, "append", append_mock)
        op = elements.OutputPlane(z_0=10)
        intercept = op.intercept(default_ray)
        op.propagate_ray(default_ray)
        append_mock.assert_called_once()
        new_pos, _ = append_mock.call_args.args
        assert np.allclose(new_pos, intercept)

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


class TestTask9:

    def test_doesnt_crash(self, task9_output):
        pass

    def test_sr_created_once(self, sr_mock, task9_output):
        sr_mock.assert_called_once()

    def test_sr_setup_correctly(self, sr_mock, task9_output):
        sr_mock.assert_called_once()
        args = set(sr_mock.call_args.args)
        args.update(sr_mock.call_args.kwargs.values())
        assert len(args) == 5
        assert not {100, 0.03, 1., 1.5}.difference(args)

    def test_op_created_once(self, op_mock, task9_output):
        op_mock.assert_called_once()

    def test_op_setup_correctly(self, op_mock, task9_output):
        op_mock.assert_called_once()
        args = op_mock.call_args.args
        args += tuple(op_mock.call_args.kwargs.values())
        assert len(args) == 1
        assert args[0] == 250

    def test_ray_created(self, ray_mock, task9_output):
        ray_mock.assert_called()

    def test_multiple_rays_created(self, ray_mock, task9_output):
        assert ray_mock.call_count > 1

    def test_ray_vertices_called(self, vert_mock, task9_output):
        vert_mock.assert_called()

    def test_ray_vertices_called_multiple(self, vert_mock, task9_output):
        assert vert_mock.call_count > 1

    def test_output_fig_produced(self, task9_output):
        assert isinstance(task9_output, mpl.figure.Figure)

    @check_figures_equal(ref_path="task9", tol=32)
    def test_plot9(self, task9_output):
        return task9_output


class TestTask10:

    def test_focal_point_exists(self, elements):
        assert hasattr(elements.SphericalRefraction, "focal_point")

    def test_focal_point(self, elements):
        sr = elements.SphericalRefraction(z_0=100,
                                          curvature=0.03,
                                          n_1=1.0,
                                          n_2=1.5,
                                          aperture=34)
        assert np.isclose(sr.focal_point(), 200.)

    def test_doesnt_crash(self, task10_output):
        pass

    def test_output_fp(self, task10_output):
        assert np.isclose(task10_output[1], 200.)

    def test_ouput_fig_produced(self, task10_output):
        assert isinstance(task10_output[0], mpl.figure.Figure)

    @check_figures_equal(ref_path="task10", tol=32)
    def test_plot10(self, task10_output):
        return task10_output[0]


class TestTask11:

    def test_ray_bundle_exists(self, ray_bundle):
        assert ray_bundle is not None

    def test_ray_bundle_class(self, ray_bundle):
        assert isinstance(ray_bundle, type)

    def test_bundle_not_inheritance(self, rays, ray_bundle):
        assert rays.Ray not in ray_bundle.mro()

    def test_bundle_construction(self, ray_bundle):
        ray_bundle(rmax=5., nrings=5)

    def test_propbundle_exists(self, ray_bundle):
        assert hasattr(ray_bundle, "propagate_bundle")

    def test_propbundle_calles_propray(self, rays, elements, pr_mock):
        sr = elements.SphericalRefraction(z_0=100, aperture=35., curvature=0.2, n_1=1., n_2=1.5)
        rb = rays.RayBundle(rmax=5., nrings=5)
        rb.propagate_bundle([sr])
        pr_mock.assert_called()

    def test_track_plot_exists(self, ray_bundle):
        assert hasattr(ray_bundle, "track_plot")

    def test_track_plot(self, rays, monkeypatch):
        vert_mock = MagicMock(return_value=[np.array([0., 0., 0.]), np.array([0., 0., 1.])])
        monkeypatch.setattr(rays.Ray, "vertices", vert_mock)
        rb = rays.RayBundle(rmax=5., nrings=5)
        rb.track_plot()
        vert_mock.assert_called()

    def test_doesnt_crash(self, task11_output):
        pass

    def test_ouput_fig_produced(self, task11_output):
        assert isinstance(task11_output, mpl.figure.Figure)

    @check_figures_equal(ref_path="task11", tol=32)
    def test_plot11(self, task11_output):
        return task11_output


class TestTask12:

    def test_spot_plot_exists(self, rays):
        assert hasattr(rays.RayBundle, "spot_plot")

    def test_rms_exists(self, rays):
        assert hasattr(rays.RayBundle, "rms")

    def test_doesnt_crash(self, task12_output):
        pass

    def test_rms(self, task12_output):
        assert np.isclose(task12_output[1], 0.016176669411515444)

    def test_output_fig_produced(self, task12_output):
        assert isinstance(task12_output[0], mpl.figure.Figure)

    @check_figures_equal(ref_path="task12", tol=33)
    def test_plot12(self, task12_output):
        return task12_output[0]


class TestTask13:

    def test_output_fig_produced(self, task13_output):
        assert isinstance(task13_output, mpl.figure.Figure)

    @check_figures_equal(ref_path="task13", tol=33)
    def test_plot13(self, task13_output):
        return task13_output


class TestTask14:

    def test_lenses_exists(self, lenses):
        pass

    def test_planoconvex_exists(self, lenses):
        assert hasattr(lenses, "PlanoConvex")

    def test_planoconvex_inheritance(self, elements, lenses):
        assert issubclass(lenses.PlanoConvex, elements.OpticalElement)

    def test_doesnt_crash(self, task14_output):
        pass

    def test_output_pcfig(self, task14_output):
        assert isinstance(task14_output[0], mpl.figure.Figure)

    def test_output_cpfig(self, task14_output):
        assert isinstance(task14_output[2], mpl.figure.Figure)

    def test_pc_focalpoint(self, task14_output):
        assert np.isclose(task14_output[1], 201.74922600619198)

    def test_cp_focalpoint(self, task14_output):
        assert np.isclose(task14_output[3], 198.45281250408226)

    def test_2SR_objects_created(self, sr_mock_with_lenses, an):
        an.task14()
        assert sr_mock_with_lenses.call_count >= 2

    def test_OP_object_created(self, op_mock, an):
        an.task14()
        assert op_mock.called

    @check_figures_equal(ref_path="task14pc", tol=33)
    def test_plot14pc(self, task14_output):
        return task14_output[0]

    @check_figures_equal(ref_path="task14cp", tol=33)
    def test_plot14cp(self, task14_output):
        return task14_output[2]


class TestTask15:
    def test_doesnt_crash(self, task15_output):
        pass

    def test_output_fig(self, task15_output):
        assert isinstance(task15_output, mpl.figure.Figure)    
    
    @check_figures_equal(ref_path="task15", tol=33)
    def test_plot15(self, task15_output):
        return task15_output
