import subprocess
try:
    import uproot3 as uproot
except ModuleNotFoundError:
    import uproot
import sys
import numpy as np
import pathlib
import pytest
import platform
import shutil
from pathlib import Path
from pytest import approx

MAIN_DIR = Path(__file__).resolve().parents[1]
ANA_DIR = MAIN_DIR / "ana"
DAT_DIR = MAIN_DIR / "dat"


def makehist_core(*, split: bool, dat: Path, build: Path):
    result = "result_10pvs.root"
    pv = "pv_10pvs.root"
    kwargs = dict(text=True, check=True, capture_output=True)

    shutil.copy(DAT_DIR / result, dat / result)
    shutil.copy(DAT_DIR / pv, dat / pv)

    if split:
        output = subprocess.run([build / "make_tracks", "10pvs", "data", dat], **kwargs).stdout
        output += subprocess.run(
            [build / "make_histogram_from_tracks", "10pvs", "trks", dat], **kwargs
        ).stdout

    else:
        output = subprocess.run(
            [build / "make_histogram", "10pvs", "data", dat], **kwargs
        ).stdout

    f1 = uproot.open(dat / result)["kernel"]
    f2 = uproot.open(dat / "kernel_10pvs.root")["kernel"]
    return f1, f2, output


@pytest.fixture(scope="session")
def compile_code(tmpdir_factory):
    build = tmpdir_factory.mktemp("build")
    kwargs = dict(stdout=sys.stdout, stderr=sys.stderr)
    subprocess.run(["cmake", "-S", ANA_DIR, "-B", build, "-G", "Ninja"], **kwargs)
    subprocess.run(["cmake", "--build", build], **kwargs)
    return build


@pytest.fixture(scope="session")
def makehist_single(compile_code, tmpdir_factory):
    data_single = tmpdir_factory.mktemp("single")
    return makehist_core(split=False, dat=data_single, build=compile_code)


@pytest.fixture(scope="session")
def makehist_split(compile_code, tmpdir_factory):
    data_split = tmpdir_factory.mktemp("split")
    return makehist_core(split=True, dat=data_split, build=compile_code)


def test_simple_run(makehist_single):
    _, _, output = makehist_single

    expected = """
Number of entries to read in: 10
Entry 0/10 (183 good, 2 bad) AnyTracks: 185 PVs: 8 SVs: 8
Entry 1/10 (149 good, 15 bad) AnyTracks: 164 PVs: 5 SVs: 7
Entry 2/10 (231 good, 33 bad) AnyTracks: 264 PVs: 12 SVs: 6
Entry 3/10 (186 good, 22 bad) AnyTracks: 208 PVs: 13 SVs: 4
Entry 4/10 (395 good, 98 bad) AnyTracks: 493 PVs: 18 SVs: 16
Entry 5/10 (82 good, 1 bad) AnyTracks: 83 PVs: 5 SVs: 0
Entry 6/10 (169 good, 14 bad) AnyTracks: 183 PVs: 10 SVs: 0
Entry 7/10 (65 good, 2 bad) AnyTracks: 67 PVs: 2 SVs: 2
Entry 8/10 (106 good, 0 bad) AnyTracks: 106 PVs: 6 SVs: 7
Entry 9/10 (29 good, 0 bad) AnyTracks: 29 PVs: 4 SVs: 0
"""
    assert expected.strip() == output.strip()


BRANCHES = [
    "zdata",
    "xmax",
    "ymax",
    "pv_cat",
    "pv_loc",
    "pv_loc_x",
    "pv_loc_y",
    "pv_ntrks",
    "sv_cat",
    "sv_loc",
    "sv_loc_x",
    "sv_loc_y",
    "sv_ntrks",
]


@pytest.mark.parametrize("branchname", BRANCHES)
def test_with_uproot_single(makehist_single, branchname):
    f1, f2, output = makehist_single
    arr1 = f1.array(branchname).flatten()
    arr2 = f2.array(branchname).flatten()
    diff = np.isclose(arr1, arr2, rtol=1e-04, atol=1e-07)
    arange = np.arange(len(arr1))
    arrdiff = np.stack([arange // 4000, arange % 4000, arr1, arr2]).T[~diff]
    for row in arrdiff:
        print("{0:1g} {1:4g} : {2:11g} <-> {3:11g}".format(*row))

    assert len(arrdiff) < 2

    # assert arr1 == approx(arr2, rel=1e-04, abs=1e-07)


@pytest.mark.parametrize("branchname", BRANCHES)
def test_with_uproot_split(makehist_split, branchname):
    f1, f2, _ = makehist_split
    arr1 = f1.array(branchname).flatten()
    arr2 = f2.array(branchname).flatten()
    diff = np.isclose(arr1, arr2, rtol=1e-04, atol=1e-07)
    arange = np.arange(len(arr1))
    arrdiff = np.stack([arange // 4000, arange % 4000, arr1, arr2]).T[~diff]
    for row in arrdiff:
        print("{0:1g} {1:4g} : {2:11g} <-> {3:11g}".format(*row))

    assert len(arrdiff) < 2

    # assert arr1 == approx(arr2, rel=1e-04, abs=1e-07)
