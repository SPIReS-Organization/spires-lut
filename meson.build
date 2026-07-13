project('pyDISORT', 'c', 'fortran',
  meson_version: '>=1.5.0',
)

add_languages('fortran')

py_mod = import('python')
py = py_mod.find_installation(pure: false)
py_dep = py.dependency()

incdir_numpy = run_command(py,
  ['-c', 'import numpy; print(numpy.get_include())'],
  check : true
).stdout().strip()

incdir_f2py = run_command(py,
    ['-c', 'import numpy.f2py; print(numpy.f2py.get_include())'],
    check : true
).stdout().strip()

inc_np = include_directories(incdir_numpy, incdir_f2py)

sources = [
  'spires_lut/BDREF.f', 'spires_lut/DISOBRDF.f', 'spires_lut/DISORT.f', 
  'spires_lut/ERRPACK.f', 'spires_lut/LAPACK.f', 'spires_lut/LINPAK.f', 'spires_lut/RDI1MACH.f'
]

disort_source = custom_target('disortmodule.c',
  input : sources,
  output : ['disortmodule.c', 'disort-f2pywrappers.f'],
  command : [py, '-m', 'numpy.f2py', '@INPUT@', '-m', 'disort', '--lower']
)

py.extension_module('disort',
  [sources, disort_source],
  incdir_f2py / 'fortranobject.c',
  include_directories: inc_np,
  dependencies : py_dep,
  install : true,
)