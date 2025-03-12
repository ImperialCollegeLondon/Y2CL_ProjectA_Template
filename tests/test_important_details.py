"""
Module for testing important details.

This module tests that students haven't used relative imports for their modules.
Doing so can cause errors in the testing framework.
"""
import re
from textwrap import dedent
from inspect import getsource


MODULES_TO_TEST_FOR = '|'.join(['rays', 'elements', 'genpolar', 'physics', 'lenses'])
RELATIVE_IMPORTS_REGEX = re.compile(rf"^\s*((?:import|from)\s+[.]?(?:{MODULES_TO_TEST_FOR}).*?)$", re.MULTILINE)
MSG = """
    {n_rel_imports} Relative imports found in {module}.py:
    --------------------------------
    {rel_imports}
    --------------------------------
    These should use absolute imports i.e. raytracer.<module>

    """


class TestNoRelativeImports:
    """Test for relative imports."""

    def _test_module(self, module_name, module):
        """
        Search for relative imports from local modules.

        Common code for scanning a module for any relative references to local modules.

        Args:
            module_name (str): The name of the module to scan
            module (module): The actual module to scan
        """
        relative_imports = RELATIVE_IMPORTS_REGEX.findall(getsource(module))
        relative_imports_str = f"\n{' '*4}".join(f"{i+1}: {import_line.strip()}"
                                                 for i, import_line in enumerate(relative_imports))
        assert not relative_imports, dedent(MSG.format(n_rel_imports=len(relative_imports),
                                                       module=module_name,
                                                       rel_imports=relative_imports_str))

    def test_rays_module(self, rays):
        """
        Test the rays.py module.

        Args:
            rays (fixture): The rays fixture
        """
        self._test_module('rays', rays)

    def test_elements_module(self, elements):
        """
        Test the elements.py module.

        Args:
            elements (fixture): The elements fixture
        """
        self._test_module('elements', elements)

    def test_physics_module(self, physics):
        """
        Test the physics.py module.

        Args:
            physics (fixture): The physics fixture
        """
        self._test_module('physics', physics)

    def test_lenses_module(self, lenses):
        """
        Test the lenses.py module.

        Args:
            lenses (fixture): The lenses fixture
        """
        self._test_module('lenses', lenses)
