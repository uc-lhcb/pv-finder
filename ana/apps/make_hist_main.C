#include <TString.h>
#include "CLI11.hpp"

void makehist(TString input, TString tree, TString folder, int nevents);

int main(int argc, char** argv) {

    CLI::App app("make_histogram", "Make kernel from a hits file");
    app.option_defaults()->always_capture_default();

    TString base = "10pvs";
    app.add_option("base,--base", base);

    TString tree = "data";
    app.add_option("tree,--tree", tree);

    TString folder = "../../dat";
    app.add_option("folder,--folder", folder);

    int nevents = 0;
    app.add_option("nevents,--nevents", nevents);

    CLI11_PARSE(app, argc, argv);

    makehist(base, tree, folder, nevents);
}
