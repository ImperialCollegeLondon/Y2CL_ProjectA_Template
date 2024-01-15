from types import FunctionType
from unittest.mock import MagicMock
import numpy as np
import matplotlib as mpl
import pytest
from utils import check_figures_equal

class TestTask1:
    def test_docstring_present(self, ph):
        assert ph.__doc__ != None
    def test_docstring_not_blank(self, ph):
        assert ph.__doc__ != ""

class TestTask2:
    def test_pos_method(self, rays):  # Check method not overridden by variable
        assert isinstance(rays.Ray.pos, FunctionType)

    def test_default_p_array(self, rays):
        r = rays.Ray()
        assert isinstance(r.pos(), np.ndarray)

    def test_p_array(self, test_ray):
        assert isinstance(test_ray.pos(), np.ndarray)

    def test_p_set(self, test_ray):
        assert np.allclose(test_ray.pos(), [1., 2., 3.])

    def test_direc_method(self, rays):  # Check method not overridden by variable
        assert isinstance(rays.Ray.direc, FunctionType)

    def test_k_array(self, test_ray):
        assert isinstance(test_ray.direc(), np.ndarray)

    def test_k_set(self, test_ray):
        test_k = np.array([4.,5.,6.])
        test_k /= np.linalg.norm(test_k)
        assert np.allclose(test_ray.direc(), test_k)

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

    def test_vertices_method(self, rays):  # Check method not overridden by variable
        assert isinstance(rays.Ray.vertices, FunctionType)

    def test_vertices_list(self, default_ray):
        assert isinstance(default_ray.vertices(), list)

    def test_vertices_array(self, default_ray):
        default_ray.append([4.,5.,6.], [7.,8.,9.])
        for vertex in default_ray.vertices():
            assert isinstance(vertex, np.ndarray)

    def test_vertices(self, test_ray):
        assert np.allclose(test_ray.vertices(), [[1.,2.,3.]])

    def test_multiple_vertices(self, test_ray):
        test_ray.append([4.,5.,6.], [7.,8.,9.])
        assert np.allclose(test_ray.vertices(),
                           [[1.,2.,3.],
                            [4.,5.,6.]])

class TestTask2ish:
    def test_oe_exists(self, elements):
        assert hasattr(elements, "OpticalElement")

    def test_intercept_raises(self, elements):
        oe = elements.OpticalElement()
        with pytest.raises(NotImplementedError):
            oe.intercept(1)

    def test_pr_raises(self, elements):
        oe = elements.OpticalElement()
        with pytest.raises(NotImplementedError):
            oe.propagate_ray(1)

class TestTask3:
    def test_sr_class_exists(self, elements):
        assert hasattr(elements, "SphericalRefraction")

    def test_inheritance(self, elements):
        assert elements.OpticalElement in elements.SphericalRefraction.mro()

    def test_construction(self, elements):
        elements.SphericalRefraction(z_0=10, aperture=5, curvature=0.2, n_1=1., n_2=1.5)


class TestTask4:

    def test_intercept_exists(self, elements):
        assert hasattr(elements.SphericalRefraction, "intercept")

    def test_no_intercept(self, rays, elements):
        ray = rays.Ray(pos=[10., 0., 0.])
        sr = elements.SphericalRefraction(z_0=10, curvature=0.02, n_1=1., n_2=1.5, aperture=5.)
        assert sr.intercept(ray) is None

    def test_intercept_array(self, default_ray, elements):
        sr = elements.SphericalRefraction(z_0=10, curvature=0.02, n_1=1., n_2=1.5, aperture=50.)
        assert isinstance(sr.intercept(default_ray), np.ndarray)

    def test_onaxis_ray_convex_intercept(self, rays, elements):
        ray = rays.Ray(pos=[0.,0.,0.])
        sr = elements.SphericalRefraction(z_0=10, curvature=0.02, n_1=1., n_2=1.5, aperture=50.)
        assert np.allclose(sr.intercept(ray), [0., 0., 10.])

    def test_onaxis_ray_concave_intercept(self, rays, elements):
        ray = rays.Ray(pos=[0.,0.,0.])
        sr = elements.SphericalRefraction(z_0=10, curvature=-0.02, n_1=1., n_2=1.5, aperture=50.)
        assert np.allclose(sr.intercept(ray), [0., 0., 10.])

    def test_offaxis_convex_intercept(self, rays, elements):
        ray1 = rays.Ray(pos=[1., 0., 0.])
        ray2 = rays.Ray(pos=[-1., 0., 0.])
        sr = elements.SphericalRefraction(z_0=10, curvature=0.02, n_1=1., n_2=1.5, aperture=50.)
        assert np.allclose(sr.intercept(ray1), [1., 0., 10.010001])
        assert np.allclose(sr.intercept(ray2), [-1., 0., 10.010001])

    def test_offaxis_concave_intercept(self, rays, elements):
        ray1 = rays.Ray(pos=[1., 0., 0.])
        ray2 = rays.Ray(pos=[-1., 0., 0.])
        sr = elements.SphericalRefraction(z_0=10, curvature=-0.02, n_1=1., n_2=1.5, aperture=50.)
        assert np.allclose(sr.intercept(ray1), [1., 0., 9.989999])
        assert np.allclose(sr.intercept(ray2), [-1., 0., 9.989999])

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


class TestTask5:
    def test_refract_exists(self, ph):
        assert hasattr(ph, "refract")

    def test_calling_args(self, ph):
        ph.refract(direc=[1, 2, 3], normal=[4, 5, 6], n_1=1, n_2=1.5)

    def test_returns_unitvector(self, ph):
        dir = np.array([0., 0.05, 1.])
        norm = np.array([0., -1.9, -1.])
        assert np.linalg.norm(ph.refract(dir, norm, 1.0, 1.5)) == 1.

    def test_onaxis_refract(self, ph):
        dir = np.array([0., 0., 1.])
        norm = np.array([0., 0., -1.])
        assert np.allclose(ph.refract(dir, norm, 1.0, 1.5), np.array([0., 0., 1.]))

    def test_offaxis_refract_lower(self, ph):
        dir = np.array([0., 0., 1.])
        norm = np.array([0., -1., -1.])
        norm /= np.linalg.norm(norm)
        assert np.allclose(ph.refract(dir, norm, 1.0, 1.5), np.array([0., 0.29027623, 0.9569429]))

    def test_offaxis_refract_upper(self, ph):
        dir = np.array([0., 0., 1.])
        norm = np.array([0., 1., -1.])
        norm /= np.linalg.norm(norm)
        assert np.allclose(ph.refract(dir, norm, 1.0, 1.5), np.array([0., -0.29027623, 0.9569429]))

    def test_equal_ref_indices_onaxis(self, ph):
        dir = np.array([0., 0., 1.])
        norm = np.array([0., 0., -1.])
        norm /= np.linalg.norm(norm)
        assert np.allclose(ph.refract(dir, norm, 1.0, 1.0), np.array([0., 0., 1.]))
    
    def test_equal_ref_indices_upper(self, ph):
        dir = np.array([0., 0., 1.])
        norm = np.array([0., 1., -1.])
        norm /= np.linalg.norm(norm)
        assert np.allclose(ph.refract(dir, norm, 1.0, 1.0), np.array([0., 0., 1.]))

    def test_equal_ref_indices_lower(self, ph):
        dir = np.array([0., 0., 1.])
        norm = np.array([0., -1., -1.])
        norm /= np.linalg.norm(norm)
        assert np.allclose(ph.refract(dir, norm, 1.0, 1.0), np.array([0., 0., 1.]))

    def test_TIR(self, ph):
        dir = np.array([0., 0., 1.])
        norm = np.array([0., 1., -1.])
        norm /= np.linalg.norm(norm)
        assert ph.refract(dir, norm, 1.5, 1.0) is None

class TestTask6:

    def test_pr_exists(self, elements):
        assert hasattr(elements.SphericalRefraction, "propagate_ray")

    def test_pr_calls_intercept_once(self, rays, elements, monkeypatch):
        intercept_mock = MagicMock(return_value=np.array([0., 0., 10.]))
        monkeypatch.setattr(elements.SphericalRefraction, "intercept", intercept_mock)
        sr = elements.SphericalRefraction(z_0=10, curvature=0.02, n_1=1., n_2=1.5, aperture=50.)
        ray = rays.Ray([0., 0., 0.], [0., 0., 1.])
        sr.propagate_ray(ray)
        intercept_mock.assert_called_once_with(ray)

    def test_pr_calls_refract_once(self, rays, elements, ph, monkeypatch):
        refract_mock = MagicMock(return_value=np.array([0., 0., 1.]))
        monkeypatch.setattr(ph, "refract", refract_mock)
        if hasattr(elements, "refract"):
            monkeypatch.setattr(elements, "refract", refract_mock)
        sr = elements.SphericalRefraction(z_0=10, curvature=0.02, n_1=1., n_2=1.5, aperture=50.)
        ray = rays.Ray([0.,0.,0.], [0.,0.,1.])
        sr.propagate_ray(ray)
        refract_mock.assert_called_once()
        k, norm, n1, n2 = refract_mock.call_args.args
        norm /= np.linalg.norm(norm)
        assert np.allclose(k, ray.direc())
        assert np.allclose(norm, np.array([0., 0., -1.]))
        assert n1 == 1.0
        assert n2 == 1.5

    def test_pr_calls_append_once(self, rays, elements, monkeypatch):
        append_mock = MagicMock(return_value=None)
        monkeypatch.setattr(rays.Ray, "append", append_mock)
        sr = elements.SphericalRefraction(z_0=10, curvature=0.02, n_1=1., n_2=1.5, aperture=50.)
        sr.propagate_ray(rays.Ray([1., 2., 0.], [0., 0., 1.]))
        append_mock.assert_called_once()
        new_pos, new_direc = append_mock.call_args.args
        assert np.allclose(new_pos, [1., 2., 10.05002503])
        assert np.allclose(new_direc, [-0.00667112, -0.01334223, 0.99988873])

class TestTask7:
    def test_doesnt_crash(self, an):
        an.task7()

    def test_ray_created(self, an, rays, monkeypatch):
        mock_ray_init = MagicMock(wraps=rays.Ray)
        monkeypatch.setattr(rays, "Ray", mock_ray_init)
        if hasattr(an, "Ray"):
            monkeypatch.setattr(an, "Ray", mock_ray_init)
        try:
            an.task7()
        except: pass
        mock_ray_init.assert_called()

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

    def test_multiple_rays_created(self, an, rays, monkeypatch):
        mock_ray_init = MagicMock(wraps=rays.Ray)
        monkeypatch.setattr(rays, "Ray", mock_ray_init)
        if hasattr(an, "Ray"):
            monkeypatch.setattr(an, "Ray", mock_ray_init)
        try:
            an.task7()
        except: pass
        assert mock_ray_init.call_count > 1

    def test_multiple_propagates(self, an, elements, monkeypatch):
        mock_pr = MagicMock()
        monkeypatch.setattr(elements.SphericalRefraction, "propagate_ray", mock_pr)
        if hasattr(an, "SphericalRefraction"):
            monkeypatch.setattr(an.SphericalRefraction, "propagate_ray", mock_pr)
        try:
            an.task7()
        except: pass
        assert mock_pr.call_count > 1


class TestTask8:
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
