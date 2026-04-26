import memgraph
import memgraph.kg
import memgraph.retrieval
import memgraph.api
import memgraph.ui


def test_version():
    assert memgraph.__version__ == "0.1.0"
