import subprocess


def test_cmake(request, tmpdir, monkeypatch):
    ana_path = request.fspath.dirpath().dirpath() / 'ana'
    build_path = tmpdir.mkdir('build_ana')

    monkeypatch.chdir(build_path)

    subprocess.check_call(['cmake', ana_path])
    subprocess.check_call(['cmake', '--build', '.'])
