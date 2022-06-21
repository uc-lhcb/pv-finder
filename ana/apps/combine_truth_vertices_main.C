#include <TString.h>
#include "CLI11.hpp"

void combinetruthvertices(TString input, TString output, TString tree, TString folder, int nevents);


int main(int argc, char** argv) {

    CLI::App app("combine_truth_vertices", "CMS data: combine vertices to form primary vertices");
    app.option_defaults()->always_capture_default();

    TString base = "10pvs";
    app.add_option("base,--base", base);
    
    TString base2 = "10pvs";
    app.add_option("base2,--base2", base2);

    TString tree = "trks";
    app.add_option("tree,--tree", tree);

    TString folder = "../../dat";
    app.add_option("folder,--folder", folder);

    int nevents = 0;
    app.add_option("nevents,--nevents", nevents);

    CLI11_PARSE(app, argc, argv);

    combinetruthvertices(base, base2, tree, folder, nevents);
}
