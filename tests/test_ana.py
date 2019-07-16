import subprocess
import uproot
import sys
import numpy as np

copt = ''

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

def test_with_uproot(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirpath().dirpath() / 'ana')

    output = subprocess.run(['root', '-b', '-q', f'makehist.C{copt}("10pvs", "../dat")'])
    assert output.returncode == 0
    f1 = uproot.open('../dat/result_10pvs.root')['kernel']
    f2 = uproot.open('../dat/kernel_10pvs.root')['kernel']
    for branchname in f1.keys():
        arr1 = f1.array(branchname)
        arr2 = f2.array(branchname)
        print(branchname)
        np.testing.assert_allclose(arr1.flatten(), arr2.flatten())
