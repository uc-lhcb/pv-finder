#include "TFile.h"
#include "TH1F.h"
#include "TMinuit.h"
#include "TTree.h"
#include "hits.h"
#include "tracklets.h"
#include <iostream>

void resetMarked(int marked[SENSORS][PHI_BINS][MAX_HITS_PER_BIN]) {
    for(int s = 0; s < SENSORS; s++) {
        for(int i = 0; i < PHI_BINS; i++) {
            for(int j = 0; j < MAX_HITS_PER_BIN; j++)
                marked[s][i][j] = 0;
        }
    }
}

// counts the number of truth-matched real and fake tracks
void checkTracks(int n, Tracklet tracks[], int &ngood, int &nbad) {
    ngood = 0;
    nbad = 0;
    for(int i = 0; i < n; i++) {
        if(tracks[i].good())
            ngood++;
        else
            nbad++;
    }
}

// number of truth-level reconstructible particles in the event
int nreco(Data &data) {
    int n = data.nhits->size();
    int sum = 0;
    for(int i = 0; i < n; i++) {
        if(data.nhits->at(i) >= 3)
            sum++;
    }
    return sum;
}

int main() {

    TFile f("test_nu76.root");
    TTree *t = (TTree *)f.Get("data");
    Data data;
    data.init(t);

    HitArray *hit_array = HitArray::instance();
    int nevents = t->GetEntries();
    // nevents = 1;

    Tracklet tracks[1500];
    int marked[SENSORS][PHI_BINS][MAX_HITS_PER_BIN];

    for(int event = 0; event < nevents; event++) {
        t->GetEntry(event);
        hit_array->newEvent(data);

        // loop here just repeats tracking the event 100 times to make timing more consistent
        for(int i = 0; i < 100; i++) {
            resetMarked(marked);
            int ntracks = 0;
            makeTracklets(marked, ntracks, tracks);
        }

        /*
        cout << ntracks << endl;
        int ngood,nbad;
        checkTracks(ntracks,tracks,ngood,nbad);
        cout << ngood << " " << nbad << endl;
        cout << nreco(data) << endl;
        */
    }

    return 0;
}
