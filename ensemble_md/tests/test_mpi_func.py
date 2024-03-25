import pytest
from mpi4py import MPI

@pytest.mark.mpi
def test_size():
    comm = MPI.COMM_WORLD
    assert comm.size == 4
