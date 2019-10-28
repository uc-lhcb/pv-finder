Generator code.

### Files
* `scatter.h/c`: Class Scatter, which only has one method, `smear`.
    - `scatter.py`: Loads C++ classes into Python
* `velo.h/c`: Material, Sensor, Integral, Distance, Segment, Intersect, VeloMaterial
    * `velo.py`: Loads the C++ classes into Python (VeloMaterial, FoilMaterial, ModuleMaterial)
* `gen.py`: Loads scatter, velo; generates and writes out events

To run the generator:

```bash
./gen.py --threads=24 2018xxxx
```

See options with `-h`.

### Requirements

* Python
* ROOT
* Pythia8


### Output ROOT file design


| ROOT leaves                       | # | Description                                |
|-----------------------------------|---|--------------------------------------------|
| `pvr_x`, `pvr_y`, `pvr_z`         | P | PV locations, randomly generated from LHCb expected Gaussian distribution |
| `svr_x`, `svr_y`, `svr_z`         | T | Heavy flavor SVs, from Pythia decay points |
| `svr_pvr`                         | T | The ID of the owning PV for the SV         |
| `hit_x`, `hit_y`, `hit_z`         | H | Recorded location of the hits, with some xy smearing applied |
| `hit_prt`                         | H | The ID of the owning PV for the hit        |
| `prt_pid`                         | T | The particle ID of the particle            |
| `prt_px`, `prt_py`, `prt_pz`      | T | The direction of the particle              |
| `prt_e`                           | T | The energy of the particle                 |
| `prt_x`, `prt_y`, `prt_z`         | T | The location of the particle               |
| `prt_hits`                        | T | The number of hits this particle had       |
| `prt_pvr`                         | T | The ID of the owning PV                    |
| `ntrks_prompt`                    | P | The number of prompt tracks in this event  |

The lengths are:

* T: The total number of particles
* S: The number of particles decaying into SVs
* H: The total number of hits
* P: One per PV in the event

All three of the first lengths have an ID number back to the original PVs (`*_pvr`). These are jagged arrays, one array per event.
