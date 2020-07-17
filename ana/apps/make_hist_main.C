#include <TString.h>
#include "CLI11.hpp"

void makehist(TString input, TString tree, TString folder);

int main(int argc, char** argv) {

    CLI::App app("make_histogram", "Make kernel from a hits file");
    app.option_defaults()->always_capture_default();

    TString base = "10pvs";
    app.add_option("base,--base", base);

    TString tree = "data";
    app.add_option("tree,--tree", tree);

    TString folder = "../../dat";
    app.add_option("folder,--folder", folder);

    bool verbose_track_info = false;
    app.add_option("verbose_track_info,--verbose_track_info", verbose_track_info);
    
    bool write_track_info = false;
    app.add_option("write_track_info,--write_track_info", write_track_info);

    CLI11_PARSE(app, argc, argv);

    makehist(base, tree, folder, write_track_info, verbose_track_info);
}
