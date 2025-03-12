"""Pytest fixtures."""
from importlib import import_module
from pathlib import Path
from types import FunctionType
from unittest.mock import MagicMock, patch, PropertyMock
import pytest
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


class CombinedNamespace:
    def __init__(self, *modules):
        self._modules = modules

    def __getattr__(self, name):
        for module in self._modules:
            if (var:=getattr(module, name, None)) is not None:
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

@pytest.fixture(scope="session")
def ph():
    return import_module("raytracer.physics")

@pytest.fixture(scope="session")
def rays():
    return import_module("raytracer.rays")

@pytest.fixture(scope="session")
def elements():
    return import_module("raytracer.elements")

@pytest.fixture(scope="session")
def physics():
    return import_module("raytracer.physics")

@pytest.fixture(scope="session")
def lenses():
    return import_module("raytracer.lenses")

@pytest.fixture(scope="session")
def an():
    matplotlib.use("agg")  # Non-interactive backend more stable for testing that interactive Tk
    yield import_module("raytracer.analysis")
    plt.close()

@pytest.fixture
def var_name_map(rays):
    pos = [1.,2.,3.]
    direc = [4., 5., 6.]
    udirec = np.array(direc) / np.linalg.norm(direc)
    r = rays.Ray(pos=pos, direc=direc)

    ret={}
    for var_name, val in vars(r).items():
        if np.allclose(val, pos):
            ret['pos'] = var_name
        elif np.allclose(val, direc) or np.allclose(val, udirec):
            ret['direc'] = var_name
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
    return rays.Ray(pos=[1.,2.,3.], direc=[4.,5.,6.])

@pytest.fixture
def test_ray_pos(rays, var_name_map):
    return getattr(rays.Ray(pos=[1.,2.,3.], direc=[4.,5.,6.]), var_name_map["pos"])

@pytest.fixture
def test_ray_direc(rays, var_name_map):
    return getattr(rays.Ray(pos=[1.,2.,3.], direc=[4.,5.,6.]), var_name_map["direc"])

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
def sr_mock(elements, an):
    sr = MagicMock(wraps=elements.SphericalRefraction)
    with patch.object(elements, "SphericalRefraction", sr), \
         patch.object(an, "SphericalRefraction", sr):
        yield sr

@pytest.fixture(scope="function")
def sr_mock_with_lenses(elements, lenses, an):
    sr = MagicMock(wraps=elements.SphericalRefraction)
    with (patch.object(elements, "SphericalRefraction", sr),
          patch.object(lenses, "SphericalRefraction", sr),
          patch.object(an, "SphericalRefraction", sr)):
        yield sr


@pytest.fixture(scope="function")
def op_mock(elements, an):
    op = MagicMock(wraps=elements.OutputPlane)
    with patch.object(elements, "OutputPlane", op), \
         patch.object(an, "OutputPlane", op):
        yield op

@pytest.fixture(scope="function")
def ray_mock(rays, an):
    ray = MagicMock(wraps=rays.Ray)
    with patch.object(rays, "Ray", ray), \
         patch.object(an, "Ray", ray):
        yield ray

@pytest.fixture()
def pr_mock(elements, an):
    pr = MagicMock()
    with patch.object(elements.SphericalRefraction, "propagate_ray", pr):
        if hasattr(an, "SphericalRefraction"):
            with patch.object(an.SphericalRefraction, "propagate_ray", pr):
                yield pr
        else:
            yield pr

@pytest.fixture(scope="function")
def planoconvex_mock(lenses, an):
    pc = MagicMock(wraps=lenses.PlanoConvex)
    with (patch.object(lenses, "PlanoConvex", pc),
          patch.object(an, "PlanoConvex", pc)):
        yield pc

@pytest.fixture(scope="function")
def vert_mock(rays, an):
    v_mock = PropertyMock(return_value=[np.array([0., 0., 0.]), np.array([0., 0., 1.])])
    if isinstance(rays.Ray.vertices, FunctionType):
        v_mock = MagicMock(return_value=[np.array([0., 0., 0.]), np.array([0., 0., 1.])])

    with (patch.object(rays.Ray, "vertices", v_mock),
          patch.object(an.Ray, "vertices", v_mock)):
        yield v_mock

@pytest.fixture(scope="class")
def task8_output(an):
    yield an.task8()
    plt.close("all")    


@pytest.fixture(scope="class")
def task10_output(an):
    yield an.task10()
    plt.close("all")

@pytest.fixture(scope="class")
def task11_output(an):
    yield an.task11()
    plt.close("all")

@pytest.fixture(scope="class")
def task12_output(an):
    yield an.task12()
    plt.close("all")

@pytest.fixture(scope="class")
def task13_output(an):
    yield an.task13()
    plt.close("all")

@pytest.fixture(scope="class")
def task14_output(an):
    yield an.task14()
    plt.close("all")

@pytest.fixture(scope="class")
def task15_output(an):
    yield an.task15()
    plt.close("all")

@pytest.fixture(scope="class")
def task16_output(an):
    yield an.task16()
    plt.close("all")

@pytest.fixture(scope="class")
def task17_output(an):
    yield an.task17()
    plt.close("all")

@pytest.fixture(scope="class")
def task18_output(an):
    yield an.task18()
    plt.close("all")
