from model.jagged import concatenate
try:
    import awkward0 as awkward
except ModuleNotFoundError:
    import awkward
import numpy as np


def test_concat():
    ja1 = awkward.JaggedArray.fromiter([[1, 2, 3], [4, 5]])
    ja2 = awkward.JaggedArray.fromiter([[7, 8], [9, 10, 11], [12]])

    ja12 = concatenate([ja1, ja2])
    ja_known = awkward.JaggedArray.fromiter(
        [[1, 2, 3], [4, 5], [7, 8], [9, 10, 11], [12]]
    )

    assert np.all(ja12.starts == ja_known.starts)
    assert np.all(ja12.stops == ja_known.stops)
    assert np.all(ja12.content == ja_known.content)
