from typing import Iterator

import pytest
from dbos import DBOS, DBOSConfig


@pytest.fixture()
def dbos_env(tmp_path: str) -> Iterator[None]:
    DBOS.destroy()
    config: DBOSConfig = {
        "name": "test-app",
        "database_url": f"sqlite:///{tmp_path}/test.sqlite",
    }
    DBOS(config=config)
    DBOS.reset_system_database()
    DBOS.launch()
    yield
    DBOS.destroy()
