import unittest
from pathlib import Path

from scripts.validate_colab_notebook import validate_notebook


class ColabNotebookTests(unittest.TestCase):
    def test_notebook_contract(self):
        root = Path(__file__).resolve().parents[1]
        report = validate_notebook(root / "notebooks/Landslide_Training_Colab.ipynb")
        self.assertGreaterEqual(report["cells"], 20)
        self.assertGreaterEqual(report["code_cells"], 15)
        self.assertEqual(report["parameter_cells"], 1)


if __name__ == "__main__":
    unittest.main()
