import pandas as pd
import numpy as np
import awkward as ak
import os
from model.utilities import Timer

from model.cmsdata import (
    process_vertex_info,
    process_track_info,
    process_truth_vertex_info,
    process_truth_track_info,
    create_root_file
)

vfile = "vertex_info_v3.txt"
tfile = "tracks_info_v3.txt"
tvfile = "truch_vertex_info_v3.txt"
ttfile = "truch_track_info_v3.txt"
output_name = "/home/ekauffma/trks_cmstest4.root"

def main(output_name, vfile, tfile, tvfile, ttfile):
    
    # read in vertex info
    with Timer(start = "Processing Vertex Info"):
        vertex_info = process_vertex_info(vfile)

    # read in track info
    with Timer(start = "Processing Track Info"):
        track_info = process_track_info(tfile)

    # read in truth vertex info
    with Timer(start = "Processing Truth Vertex Info"):
        truth_vertex_info = process_vertex_info(tvfile)

    # read in truth track info
    with Timer(start = "Processing Truth Track Info"):
        truth_track_info = process_truth_track_info(ttfile)

    # Create the output TFile and TTree.
    with Timer(start = "Creating ROOT File"):
        create_root_file(output_name, "trks", truth_vertex_info, track_info, truth_track_info)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This processes multiple .txt files into a ROOT file containing tracks. "
        "An example of the usage is as follows: "
        "python ProcessDataCMS.py -o /output_path/output_file.root -v vertex_info.txt -t track_info.txt"
        "--tv truth_vertex_info.txt --tt truth_track_info.txt"
    )

    parser.add_argument(
        "-o", "--output", type=Path, required=True, help="Set the output file (.root)"
    )
    parser.add_argument(
        "-v", "--vfile", type=Path, required=True, help="Set the vertex file (.txt)"
    )
    parser.add_argument(
        "-t", "--tfile", type=Path, required=True, help="Set the track file (.txt)"
    )
    parser.add_argument(
        "--tvfile", type=Path, required=True, help="Set the truth vertex file (.txt)"
    )
    parser.add_argument(
        "--ttfile", type=Path, required=True, help="Set the truth track file (.txt)"
    )

    args = parser.parse_args()

    main(args.output, args.vfile, args.tfile, args.tvfile, args.ttfile)