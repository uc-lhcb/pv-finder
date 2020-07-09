//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Mon Jan 22 11:51:26 2018 by ROOT version 6.12/04
// from TTree data/
// found on file: test_100pvs.root
//////////////////////////////////////////////////////////

#ifndef mds_ana_h
#define mds_ana_h

#include <TChain.h>
#include <TFile.h>
#include <TROOT.h>

// Header file for the classes stored in the TTree if any.
#include "vector"

class mds_ana {
  public:
    TTree *fChain;  //! pointer to the analyzed TTree or TChain
    Int_t fCurrent; //! current Tree number in a TChain

    // Fixed size dimensions of array or collections stored in the TTree if any.

    // Declaration of leaf types
    vector<double> *pvr_x;
    vector<double> *pvr_y;
    vector<double> *pvr_z;
    vector<double> *svr_x;
    vector<double> *svr_y;
    vector<double> *svr_z;
    vector<double> *svr_pvr;
    vector<double> *hit_x;
    vector<double> *hit_y;
    vector<double> *hit_z;
    vector<double> *hit_prt;
    vector<double> *prt_pid;
    vector<double> *prt_px;
    vector<double> *prt_py;
    vector<double> *prt_pz;
    vector<double> *prt_e;
    vector<double> *prt_x;
    vector<double> *prt_y;
    vector<double> *prt_z;
    vector<double> *prt_hits;
    vector<double> *prt_pvr;
    vector<double> *ntrks_prompt;

    // List of branches
    TBranch *b_pvr_x;        //!
    TBranch *b_pvr_y;        //!
    TBranch *b_pvr_z;        //!
    TBranch *b_svr_x;        //!
    TBranch *b_svr_y;        //!
    TBranch *b_svr_z;        //!
    TBranch *b_svr_pvr;      //!
    TBranch *b_hit_x;        //!
    TBranch *b_hit_y;        //!
    TBranch *b_hit_z;        //!
    TBranch *b_hit_prt;      //!
    TBranch *b_prt_pid;      //!
    TBranch *b_prt_px;       //!
    TBranch *b_prt_py;       //!
    TBranch *b_prt_pz;       //!
    TBranch *b_prt_e;        //!
    TBranch *b_prt_x;        //!
    TBranch *b_prt_y;        //!
    TBranch *b_prt_z;        //!
    TBranch *b_prt_hits;     //!
    TBranch *b_prt_pvr;      //!
    TBranch *b_ntrks_prompt; //!

    mds_ana(TTree *tree = 0);
    virtual ~mds_ana();
    virtual Int_t Cut(Long64_t entry);
    virtual Int_t GetEntry(Long64_t entry);
    virtual Long64_t LoadTree(Long64_t entry);
    virtual void Init(TTree *tree);
    virtual void Loop();
    virtual Bool_t Notify();
    virtual void Show(Long64_t entry = -1);
};

#endif

#ifdef mds_ana_cxx
mds_ana::mds_ana(TTree *tree) : fChain(0) {
    // if parameter tree is not specified (or zero), connect the file
    // used to generate this class and read the Tree.
    if(tree == 0) {
        TFile *f = (TFile *)gROOT->GetListOfFiles()->FindObject("../dat/test_100pvs.root");
        if(!f || !f->IsOpen()) {
            f = new TFile("../dat/test_100pvs.root");
        }
        f->GetObject("data", tree);
    }
    Init(tree);
}

mds_ana::~mds_ana() {
    if(!fChain)
        return;
    delete fChain->GetCurrentFile();
}

Int_t mds_ana::GetEntry(Long64_t entry) {
    // Read contents of entry.
    if(!fChain)
        return 0;
    return fChain->GetEntry(entry);
}
Long64_t mds_ana::LoadTree(Long64_t entry) {
    // Set the environment to read one entry
    if(!fChain)
        return -5;
    Long64_t centry = fChain->LoadTree(entry);
    if(centry < 0)
        return centry;
    if(fChain->GetTreeNumber() != fCurrent) {
        fCurrent = fChain->GetTreeNumber();
        Notify();
    }
    return centry;
}

void mds_ana::Init(TTree *tree) {
    // The Init() function is called when the selector needs to initialize
    // a new tree or chain. Typically here the branch addresses and branch
    // pointers of the tree will be set.
    // It is normally not necessary to make changes to the generated
    // code, but the routine can be extended by the user if needed.
    // Init() will be called many times when running on PROOF
    // (once per file to be processed).

    // Set object pointer
    pvr_x = 0;
    pvr_y = 0;
    pvr_z = 0;
    svr_x = 0;
    svr_y = 0;
    svr_z = 0;
    svr_pvr = 0;
    hit_x = 0;
    hit_y = 0;
    hit_z = 0;
    hit_prt = 0;
    prt_pid = 0;
    prt_px = 0;
    prt_py = 0;
    prt_pz = 0;
    prt_e = 0;
    prt_x = 0;
    prt_y = 0;
    prt_z = 0;
    prt_hits = 0;
    prt_pvr = 0;
    ntrks_prompt = 0;
    // Set branch addresses and branch pointers
    if(!tree)
        return;
    fChain = tree;
    fCurrent = -1;
    fChain->SetMakeClass(1);

    fChain->SetBranchAddress("pvr_x", &pvr_x, &b_pvr_x);
    fChain->SetBranchAddress("pvr_y", &pvr_y, &b_pvr_y);
    fChain->SetBranchAddress("pvr_z", &pvr_z, &b_pvr_z);
    fChain->SetBranchAddress("svr_x", &svr_x, &b_svr_x);
    fChain->SetBranchAddress("svr_y", &svr_y, &b_svr_y);
    fChain->SetBranchAddress("svr_z", &svr_z, &b_svr_z);
    fChain->SetBranchAddress("svr_pvr", &svr_pvr, &b_svr_pvr);
    fChain->SetBranchAddress("hit_x", &hit_x, &b_hit_x);
    fChain->SetBranchAddress("hit_y", &hit_y, &b_hit_y);
    fChain->SetBranchAddress("hit_z", &hit_z, &b_hit_z);
    fChain->SetBranchAddress("hit_prt", &hit_prt, &b_hit_prt);
    fChain->SetBranchAddress("prt_pid", &prt_pid, &b_prt_pid);
    fChain->SetBranchAddress("prt_px", &prt_px, &b_prt_px);
    fChain->SetBranchAddress("prt_py", &prt_py, &b_prt_py);
    fChain->SetBranchAddress("prt_pz", &prt_pz, &b_prt_pz);
    fChain->SetBranchAddress("prt_e", &prt_e, &b_prt_e);
    fChain->SetBranchAddress("prt_x", &prt_x, &b_prt_x);
    fChain->SetBranchAddress("prt_y", &prt_y, &b_prt_y);
    fChain->SetBranchAddress("prt_z", &prt_z, &b_prt_z);
    fChain->SetBranchAddress("prt_hits", &prt_hits, &b_prt_hits);
    fChain->SetBranchAddress("prt_pvr", &prt_pvr, &b_prt_pvr);
    fChain->SetBranchAddress("ntrks_prompt", &ntrks_prompt, &b_ntrks_prompt);
    Notify();
}

Bool_t mds_ana::Notify() {
    // The Notify() function is called when a new file is opened. This
    // can be either for a new TTree in a TChain or when when a new TTree
    // is started when using PROOF. It is normally not necessary to make changes
    // to the generated code, but the routine can be extended by the
    // user if needed. The return value is currently not used.

    return kTRUE;
}

void mds_ana::Show(Long64_t entry) {
    // Print contents of entry.
    // If entry is not specified, print current entry
    if(!fChain)
        return;
    fChain->Show(entry);
}
Int_t mds_ana::Cut(Long64_t entry) {
    // This function may be called from Loop.
    // returns  1 if entry is accepted.
    // returns -1 otherwise.
    return 1;
}
#endif // #ifdef mds_ana_cxx
