import pytest






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
