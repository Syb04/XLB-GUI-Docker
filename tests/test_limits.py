import json
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
from workbench.limits import positive_env
from workbench.schema import default_project, validate_project
from workbench.server import sampled_cells


class LimitsTests(unittest.TestCase):
    def test_three_million_cells_accepted_by_default(self):
        project = default_project()
        project['mesh']['cells'] = [300, 100, 100]
        self.assertEqual(validate_project(project)['mesh']['cells'], [300, 100, 100])

    def test_environment_reaches_schema_and_geometry(self):
        env = dict(os.environ, XLB_MAX_TOTAL_CELLS='2000000', XLB_MAX_AXIS_CELLS='768',
                   XLB_MAX_BOUNDARY_LINK_MIB='256')
        code = """
import json
from workbench import geometry, schema
from workbench.limits import configured_limits
p = schema.default_project()
p['mesh']['cells'] = [300,100,100]
try:
    schema.validate_project(p)
except ValueError as e:
    assert 'XLB_MAX_TOTAL_CELLS' in str(e)
else:
    raise AssertionError('configured limit ignored')
assert geometry.MAX_TOTAL_CELLS == 2000000
assert geometry.MAX_AXIS_CELLS == 768
assert geometry.MAX_BOUNDARY_LINK_BYTES == 256 * 1024**2
print(json.dumps(configured_limits()))
"""
        result = subprocess.run([sys.executable, '-c', code], env=env, check=True,
                                capture_output=True, text=True)
        self.assertEqual(json.loads(result.stdout)['max_total_cells'], 2_000_000)

    def test_bad_environment_is_rejected(self):
        for raw in ('0', '-1', 'NaN', '1.5', '9999999999999999'):
            with patch.dict(os.environ, {'XLB_TEST_LIMIT': raw}):
                with self.assertRaisesRegex(ValueError, 'XLB_TEST_LIMIT'):
                    positive_env('XLB_TEST_LIMIT', 10)

    def test_preview_sampling_preserves_global_stride_across_chunks(self):
        mask = np.random.default_rng(4).random((93, 83, 61)) > .4
        count, sampled = sampled_cells(mask)
        expected = np.argwhere(mask)
        stride = max(1, (len(expected) + 5999) // 6000)
        self.assertEqual(count, len(expected))
        self.assertLessEqual(len(sampled), 6000)
        np.testing.assert_array_equal(sampled, expected[::stride])
        count, sampled = sampled_cells(np.zeros((5, 6, 7), dtype=bool))
        self.assertEqual(count, 0)
        self.assertEqual(sampled.shape, (0, 3))
