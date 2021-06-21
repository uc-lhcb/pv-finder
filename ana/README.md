Code to analyze hits or tracks and produce tracks or kernels.

This is now fully compilable. Like any CMake project, you run (from this `/ana` folder):

```bash
cmake -S . -B build
cmake --build build
```

There are three executables:

```bash
./make_histogram             # hits -> kernel
./make_tracks                # hits -> tracks
./make_histogram_from_tracks # tracks -> kernel
```

All executables take the same three optional arguments; `base`, `tree`, and `folder`.
The hits files will always start with `pv_`, the track files with `trks_`,
and the kernel files with `kernel_`.

TODO: Rename `pv_` to `hits_`. Rename `histogram` to `kernel`.
Move the headers to a folder to keep the listing from being so long.

There is a file in `/dat` with 10 events with 10 (visible) collisions each to
play with. The "correct" result is stored in the repository as `result_10pvs.root`,
which is used for comparisons (`/dat/compare_runs.py`) and tests.
This is the default if you run the executables in place.


## Developing

You should be in the `pvfinder` or `pvfindergpu` Conda environment on macOS or Linux. If you want to use Docker, you can:

```bash
docker run --rm -v $PWD:/pv -it gitlab-registry.cern.ch/lhcb-reco-dev/pv-finder:latest
cmake -S /pv/ana/ -B /build-ana -GNinja
cmake --build build-ana/
./build-ana/make_histogram 10pvs pv/dat
```

Note that there are minor variations in the Minuit minimizations
between systems, so the `pytest` based testing may report variations on macOS.
The stored `results_10pvs.root` is designed to work correctly with
the Docker version since that is what is used in CI.

In `make_histogram` and family, the following procedure is followed. First, the input
and output ROOT files are opened. The trees are then used in the constructors of three families of readers:

* `Core*In`: Input
    * `CoreHitsIn`: Only for hits. `hit_x`, `hit_y`, `hit_z`, `hit_prt` (ID of hit)
    * `CoreNHitsIn`: Stored in tracks too. `prt_hits` (number of hits)
    * `CorePVsIn`: `pvr_x`, `pvr_y`, `pvr_z`, `prt_pvr` (ID), `svr_x`, `svr_y`, `svr_z`, `srt_pvr`
    * `CoreTruthTracksIn`: `prt_px`, `prt_py`, `prt_pz`, `prt_x`, `prt_y`, `prt_z`, `ntrks_prompt`. Really only used to compute category information.
    * `CoreReconTracksIn`: Tracks specific `recon_x`, `recon_y`, `recon_z`, `recon_tx`, `recon_ty`, `recon_chi2`
* `Core*Out`: Output (same as above, only for output)
* `Data*Out`: Final Output
    * `DataPVsOut`: `pv_cat`, `pv_loc`, `pv_loc_x`, `pv_loc_y`, `pv_ntrks`, `sv_cat`, `sv_loc`, `sv_loc_x`, `sv_loc_y`, `sv_ntrks`  (all float)
    * `DataKernelOut`: `zdata`, `xmax`, `ymax` (all 4,000 long, all float)

Note that `sv_n` and `pv_n` were in `DataPVsOut`, but have been removed.
All of these are stored in a single tree. `CoreHits`, `CoreNHits`, `CorePVs`, and `CoreTruthTracks` are
required for Hits reconstruction. If you replace `CoreHits` with `CoreReconTracks`, you can use Track-only reconstruction.

You should call `GetEntry` to load up the branches with one event, then run `make_tracks()`
to convert hits to `AnyTracks`, or directly create an `AnyTracks` with a `CoreReconTracksIn`
instance. Use `copy_in_pvs` to add the Core readers to the PVs out instance if you are creating
a kernel. `makez` computes the kernel from `AnyTracks`. Fill the tree then repeat.


### Classes

* `Point`: A location in `x`, `y`, `z`
* `Trajectory`: Holds a `Point` and a direction
* `Hit`: The location of a hit - a `Point` and some ids
* `Hits`: Holds `Hit`'s in bins of phi, sorted
* `TripletBase`: Holds a `Point`, `Trajectory`, and `chi2` and can do closest approach and PDF calculations
    * `TripletToy`: Holds three hits
* `AnyTracks`: Holds a collection of `TripletBase`
    * `Tracks`: Holds a collection of `TripletToy` - is convertible to `AnyTracks`



### Planned changes:

* Track, from `VeloTracks` will hold:
    - `ClosestToBeam`: The state(x,y,z,tx,tz,q/p) at which the (extrapolated) track came closest to the beam. Number\[6\]
    - `errCTBState`: Covariance matrix of closest to beam state. Only non-zero elements (x,y,tx,ty, Cov(x,tx)). Cov(x,tx)= Cov(x,ty)= Cov(y,tx)= Cov(y,ty) Number\[5\]
    - Based on [this description.](https://gitlab.cern.ch/BCForward/RAPID-data/blob/master/Event_format.md)


## Notebook

To run on Goofy:

```bash
ssh goofy -L 8888:localhost:8888
module purge
conda activate pvfindergpu
cd <the ana directory>
jupyter lab --no-browser
```

Copy and paste the `http://localhost:8888/...` link into your browser on your local computer.

