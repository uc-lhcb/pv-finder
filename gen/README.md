Generator code.

### Files 
* `scatter.h/c`: Class Scatter, which only has one method, `smear`.
* `velo.h/c`: Material, Sensor, Integral, Distance, Segment, Intersect, VeloMaterial
* `velo.py`: Loads the C++ classes into Python (3 materials)
* `gen.py`: Loads scatter, velo; generates and writes out events

To run generator:

```bash
./gen.py 2018xxxx
```

See options with -h.

Multiprocessing does not speed up much; use

```bash
for i in 1 2 3 4; do ./gen.py --events 10 tmp_$i & done
```

For about .4 events per second rather than .25 or so.
