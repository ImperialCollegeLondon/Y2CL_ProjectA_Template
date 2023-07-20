import pytest

class TestOpticalElement:
    def test_oe_exists(self, elements):
        assert hasattr(elements, "OpticalElement")

    def test_pr_raises(self, elements):
        with pytest.raises(NotImplementedError):
            elements.OpticalElement.propagate_ray(1, 2)


class TestRayBundle:
    def test_ray_bundle_exists(self, ray_bundle):
        assert ray_bundle is not None

    def test_ray_bundle_class(self, ray_bundle):
        assert isinstance(ray_bundle, type)

    def test_bundle_not_inheritance(self, rays, ray_bundle):
        assert rays.Ray not in ray_bundle.mro()


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
