"""Pytest fixtures."""
from importlib import import_module
from pathlib import Path
from types import FunctionType
from unittest.mock import MagicMock, patch, PropertyMock
import pytest
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("agg")  # Non-interactive backend more stable for testing than interactive Tk


# pylint: disable=redefined-outer-name


class CombinedNamespace:
    def __init__(self, *modules):
        self._modules = modules

    def __getattr__(self, name):
        for module in self._modules:
            if (var := getattr(module, name, None)) is not None:
                return var

        raise AttributeError("No such attribute: " + name)

    def __iter__(self):
        for module in self._modules:
            yield from vars(module).keys()


@pytest.fixture
def rt():
    rays_module = import_module("raytracer.rays")
    elements_module = import_module("raytracer.elements")
    lenses_module = import_module("raytracer.lenses")
    return CombinedNamespace(rays_module, elements_module, lenses_module)


@pytest.fixture(scope="function")
def ph():
    return import_module("raytracer.physics")


@pytest.fixture(scope="function")
def rays():
    return import_module("raytracer.rays")


@pytest.fixture(scope="function")
def elements():
    return import_module("raytracer.elements")


@pytest.fixture(scope="function")
def physics():
    return import_module("raytracer.physics")

@pytest.fixture(scope="function")
def genpolar():
    return import_module("raytracer.genpolar")


@pytest.fixture(scope="function")
def lenses():
    return import_module("raytracer.lenses")


@pytest.fixture(scope="function")
def an():
    yield import_module("raytracer.analysis")
    plt.close('all')


@pytest.fixture
def var_name_map(rays):
    pos = [1., 2., 3.]
    direc = [4., 5., 6.]
    udirec = np.array(direc) / np.linalg.norm(direc)
    r = rays.Ray(pos=pos, direc=direc)

    ret = {}
    for var_name, val in vars(r).items():
        if np.allclose(val, pos):
            ret['pos'] = var_name
        elif np.allclose(val, direc) or np.allclose(val, udirec):
            ret['direc'] = var_name
    return ret

@pytest.fixture
def bundle_var_name_map(rays):
    r = rays.RayBundle()

    ret = {}
    for var_name, val in vars(r).items():
        if isinstance(val, (list, np.ndarray)):
            ret['rays'] = var_name
    return ret

@pytest.fixture
def default_ray(rays):
    return rays.Ray()


@pytest.fixture
def default_ray_pos(rays, var_name_map):
    return getattr(rays.Ray(), var_name_map["pos"])


@pytest.fixture
def default_ray_direc(rays, var_name_map):
    return getattr(rays.Ray(), var_name_map["direc"])


@pytest.fixture
def test_ray(rays):
    return rays.Ray(pos=[1., 2., 3.], direc=[4., 5., 6.])


@pytest.fixture
def test_ray_pos(rays, var_name_map):
    return getattr(rays.Ray(pos=[1., 2., 3.], direc=[4., 5., 6.]), var_name_map["pos"])


@pytest.fixture
def test_ray_direc(rays, var_name_map):
    return getattr(rays.Ray(pos=[1., 2., 3.], direc=[4., 5., 6.]), var_name_map["direc"])


@pytest.fixture
def ray_bundle(rays):
    return getattr(rays, "RayBundle", getattr(rays, "Bundle", None))


@pytest.fixture(scope="module")
def source_files():
    excluded_files = ()
    src_files = [str(file_) for file_ in Path(__file__).parent.parent.glob("raytracer/[a-zA-Z]*.py")
                 if file_.name not in excluded_files]
    assert src_files, "No source files found to check!"
    return src_files


@pytest.fixture(scope="module")
def source_files_str(source_files):
    return ' '.join(source_files)


@pytest.fixture(scope="function")
def rtrings_mock(genpolar):
    with patch.object(genpolar, "rtrings", spec=genpolar.rtrings, wraps=genpolar.rtrings) as rtrm:
        yield rtrm

@pytest.fixture(scope="function")
def sr_mock(elements):
    with patch.object(elements.SphericalRefraction,
                      "__init__",
                      autospec=True,
                      side_effect=elements.SphericalRefraction.__init__) as srm:
        yield srm


@pytest.fixture(scope="function")
def op_mock(elements):
    with patch.object(elements.OutputPlane,
                      "__init__",
                      autospec=True,
                      side_effect=elements.OutputPlane.__init__) as opm:
        yield opm


@pytest.fixture(scope="function")
def ray_mock(rays):
    with patch.object(rays.Ray, "__init__", autospec=True, side_effect=rays.Ray.__init__) as rm:
        yield rm


@pytest.fixture(scope="function")
def pr_mock(elements):
    with patch.object(elements.SphericalRefraction,
                      "propagate_ray",
                      autospec=True,
                      side_effect=elements.SphericalRefraction.propagate_ray) as prm:
        yield prm


@pytest.fixture(scope="function")
def planoconvex_mock(lenses, an):
    pc = MagicMock(wraps=lenses.PlanoConvex)
    with (patch.object(lenses, "PlanoConvex", pc),
          patch.object(an, "PlanoConvex", pc)):
        yield pc


@pytest.fixture(scope="function")
def vert_mock(rays):
    mock_type = PropertyMock(return_value=[np.array([0., 0., 0.]), np.array([0., 0., 1.])])
    if isinstance(rays.Ray.vertices, FunctionType):
        mock_type = MagicMock(return_value=[np.array([0., 0., 0.]), np.array([0., 0., 1.])])
    with patch.object(rays.Ray, "vertices", mock_type) as v_mock:
        yield v_mock


@pytest.fixture(scope="function")
def track_plot_mock(rays):
    with patch.object(rays.RayBundle, "track_plot", autospec=True, side_effect=rays.RayBundle.track_plot) as tp_mock:
        yield tp_mock


@pytest.fixture(scope="function")
def spot_plot_mock(rays):
    with patch.object(rays.RayBundle, "spot_plot", autospec=True, side_effect=rays.RayBundle.spot_plot) as sp_mock:
        yield sp_mock


@pytest.fixture(scope="function")
def rms_mock(rays):
    rmsm = PropertyMock(return_value=123)
    if isinstance(rays.RayBundle.rms, FunctionType):
        rmsm = MagicMock(return_value=123)
    with patch.object(rays.RayBundle, "rms", rmsm):
        yield rmsm


@pytest.fixture(scope="function")
def task8_output(an):
    yield an.task8()


@pytest.fixture(scope="function")
def task10_output(an):
    yield an.task10()


@pytest.fixture(scope="function")
def task11_output(an):
    yield an.task11()


@pytest.fixture(scope="function")
def task12_output(an):
    yield an.task12()


@pytest.fixture(scope="function")
def task13_output(an):
    yield an.task13()


@pytest.fixture(scope="function")
def task14_output(an):
    yield an.task14()


@pytest.fixture(scope="function")
def task15_output(an):
    yield an.task15()


@pytest.fixture(scope="function")
def task16_output(an):
    yield an.task16()


@pytest.fixture(scope="function")
def task17_output(an):
    yield an.task17()


@pytest.fixture(scope="function")
def task18_output(an):
    yield an.task18()
