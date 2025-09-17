"""
This module contains example tests for a Kedro project.
Tests should be placed in ``src/tests``, in modules that mirror your
project's structure, and in files named test_*.py.
"""
import pytest
from pathlib import Path
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

# The tests below are here for the demonstration purpose
# and should be replaced with the ones testing the project
# functionality

class TestKedroRun:
    def test_kedro_run_success(self):
        """Test que el pipeline de Kedro se ejecuta correctamente."""
        bootstrap_project(Path.cwd())

        # El pipeline debe ejecutarse sin errores
        with KedroSession.create(project_path=Path.cwd()) as session:
            result = session.run()
            
        # Verificar que la ejecución fue exitosa
        assert result is not None, "El pipeline debe ejecutarse exitosamente"
