import subprocess
import uproot
import sys
import numpy as np
import pathlib
import pytest
import platform

from pathlib import Path
MAIN_DIR = Path(__file__).resolve().parents[1]
ANA_DIR = MAIN_DIR / 'ana'
DAT_DIR = MAIN_DIR / 'dat'


copt = '++' if platform.system() == 'Linux' else ''

@pytest.fixture(scope='module')
def makehist_arrs():
    output = subprocess.run(['root', '-b', '-q', f'makehist.C{copt}("10pvs", "../dat")'],
                            cwd = ANA_DIR)

    f1 = uproot.open(DAT_DIR / 'result_10pvs.root')['kernel']
    f2 = uproot.open(DAT_DIR / 'kernel_10pvs.root')['kernel']
    return f1, f2



def test_simple_run(request, monkeypatch):
    print(dir(request.fspath))
    monkeypatch.chdir(request.fspath.dirpath().dirpath() / 'ana')

    output = subprocess.check_output(['root', '-b', '-q', f'makehist.C{copt}("10pvs", "../dat")'],
                                    encoding=sys.stdout.encoding)
    expected = '''\
Number of entries to read in: 10
Entry 0/10 Total tracks: 185 good tracks: 183 bad tracks: 2 PVs: 8 SVs: 8
Entry 1/10 Total tracks: 164 good tracks: 149 bad tracks: 15 PVs: 5 SVs: 7
Entry 2/10 Total tracks: 264 good tracks: 231 bad tracks: 33 PVs: 12 SVs: 6
Entry 3/10 Total tracks: 208 good tracks: 186 bad tracks: 22 PVs: 13 SVs: 4
Entry 4/10 Total tracks: 493 good tracks: 395 bad tracks: 98 PVs: 18 SVs: 16
Entry 5/10 Total tracks: 83 good tracks: 82 bad tracks: 1 PVs: 5 SVs: 0
Entry 6/10 Total tracks: 183 good tracks: 169 bad tracks: 14 PVs: 10 SVs: 0
Entry 7/10 Total tracks: 67 good tracks: 65 bad tracks: 2 PVs: 2 SVs: 2
Entry 8/10 Total tracks: 106 good tracks: 106 bad tracks: 0 PVs: 6 SVs: 7
Entry 9/10 Total tracks: 29 good tracks: 29 bad tracks: 0 PVs: 4 SVs: 0'''

    assert expected in output

@pytest.mark.parametrize('branchname', [
    'zdata', 'xmax', 'ymax',
    'pv_n', 'sv_n',
    'pv_cat', 'pv_loc', 'pv_loc_x', 'pv_loc_y', 'pv_ntrks',
    'sv_cat', 'sv_loc', 'sv_loc_x', 'sv_loc_y', 'sv_ntrks'])
def test_with_uproot(makehist_arrs, branchname):
    f1, f2 = makehist_arrs
    arr1 = f1.array(branchname).flatten()
    arr2 = f2.array(branchname).flatten()
    diff = np.isclose(arr1, arr2, rtol=1e-04, atol=1e-07)
    arange = np.arange(len(arr1))
    arrdiff = np.stack([arange//4000, arange%4000, arr1, arr2]).T[~diff]
    for row in arrdiff:
        print('{0:1g} {1:4g} : {2:11g} <-> {3:11g}'.format(*row))
    np.testing.assert_allclose(arr1, arr2, rtol=1e-04, atol=1e-07)
