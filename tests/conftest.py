import sys
from pathlib import Path

import pytest
from dbos import DBOS, DBOSConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture()
def dbos_env():
    DBOS.destroy()
    config: DBOSConfig = {
        "name": "test-app",
        "database_url": "sqlite:///openai.sqlite",
    }
    DBOS(config=config)
    DBOS.reset_system_database()
    DBOS.launch()
    yield
    DBOS.destroy()
