#ifndef DATA_H
#define DATA_H

using namespace std;

class Data {
  public:
    vector<double> *pvx;
    vector<double> *pvy;
    vector<double> *pvz;
    vector<double> *svx;
    vector<double> *svy;
    vector<double> *svz;
    vector<double> *sv_ipv;
    vector<double> *px;
    vector<double> *py;
    vector<double> *pz;
    vector<double> *x;
    vector<double> *y;
    vector<double> *z;
    vector<double> *hx;
    vector<double> *hy;
    vector<double> *hz;
    vector<double> *hid;
    vector<double> *hs;
    vector<double> *ntrks;
    vector<double> *ipv;
    vector<double> *nhits;

    Data() {
        pvx = new vector<double>();
        pvy = new vector<double>();
        pvz = new vector<double>();
        svx = new vector<double>();
        svy = new vector<double>();
        svz = new vector<double>();
        sv_ipv = new vector<double>();
        px = new vector<double>();
        py = new vector<double>();
        pz = new vector<double>();
        x = new vector<double>();
        y = new vector<double>();
        z = new vector<double>();
        hx = new vector<double>();
        hy = new vector<double>();
        hz = new vector<double>();
        hid = new vector<double>();
        hs = new vector<double>();
        ntrks = new vector<double>();
        ipv = new vector<double>();
        nhits = new vector<double>();
    }

    ~Data() {
        delete pvx;
        delete pvy;
        delete pvz;
        delete svx;
        delete svy;
        delete svz;
        delete sv_ipv;
        delete px;
        delete py;
        delete pz;
        delete x;
        delete y;
        delete z;
        delete hx;
        delete hy;
        delete hz;
        delete hid;
        delete hs;
        delete ntrks;
        delete ipv;
        delete nhits;
    }

    void init(TTree *t) {
        t->SetBranchAddress("pvr_x", &pvx);
        t->SetBranchAddress("pvr_y", &pvy);
        t->SetBranchAddress("pvr_z", &pvz);
        t->SetBranchAddress("svr_x", &svx);
        t->SetBranchAddress("svr_y", &svy);
        t->SetBranchAddress("svr_z", &svz);
        t->SetBranchAddress("svr_pvr", &sv_ipv);
        t->SetBranchAddress("prt_px", &px);
        t->SetBranchAddress("prt_py", &py);
        t->SetBranchAddress("prt_pz", &pz);
        t->SetBranchAddress("prt_x", &x);
        t->SetBranchAddress("prt_y", &y);
        t->SetBranchAddress("prt_z", &z);
        t->SetBranchAddress("prt_pvr", &ipv);
        t->SetBranchAddress("prt_hits", &nhits);
        t->SetBranchAddress("hit_x", &hx);
        t->SetBranchAddress("hit_y", &hy);
        t->SetBranchAddress("hit_z", &hz);
        t->SetBranchAddress("hit_prt", &hid);
        t->SetBranchAddress("hit_sensor", &hs);
        t->SetBranchAddress("ntrks_prompt", &ntrks);
    }
};

#endif
