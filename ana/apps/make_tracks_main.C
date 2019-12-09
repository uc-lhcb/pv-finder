#include <TString.h>
#include "CLI11.hpp"

void make_tracks(TString input, TString tree, TString folder);


int main(int argc, char** argv) {

    CLI::App app("make_tracks", "Make tracks from a hits file");
    app.option_defaults()->always_capture_default();

    TString base = "10pvs";
    app.add_option("base,--base", base);

    TString tree = "data";
    app.add_option("tree,--tree", tree);

    TString folder = "../../dat";
    app.add_option("folder,--folder", folder);

    CLI11_PARSE(app, argc, argv);

    make_tracks(base, tree, folder);
}
