#pragma once

#include <TTree.h>
#include <vector>

struct In {};
struct Out {};

template <class T, class F> struct VecIO {};

template <class F> struct VecIO<In, F> {
    std::vector<F> *data = nullptr;
    TString name;

    template <class P> VecIO(P *parent, TString name) : name(name) {
        parent->vecs.push_back(this);
        parent->t->SetBranchAddress(name, &data);
    }

    const std::vector<F> *operator->() const { return data; }
};

template <class F> struct VecIO<Out, F> {
    std::vector<F> data;
    TString name;

    template <class P> VecIO(P *parent, TString name) : name(name) {
        parent->vecs.push_back(this);
        parent->t->Branch(name, &data);
    }

    void clear() { data.clear(); }

    std::vector<F> *operator->() { return &data; }
    const std::vector<F> *operator->() const { return &data; }
};

template <class T, class F> struct CoreIO {};

template <class F> struct CoreIO<In, F> {
    using Vec = VecIO<In, F>;
    TTree *t;
    std::vector<Vec *> vecs;

    CoreIO<In, F>(TTree *t) : t(t) {}
};

template <class F> struct CoreIO<Out, F> {
    using Vec = VecIO<Out, F>;
    TTree *t;
    std::vector<Vec *> vecs;

    CoreIO<Out, F>(TTree *t) : t(t) {}

    void clear() {
        for(Vec *item : vecs)
            item->clear();
    }

    void copy_in(CoreIO<In, F> &input) {
        clear();
        for(int i = 0; i < vecs.size(); ++i) {
            (*vecs[i]).data = *(*input.vecs[i]).data;
        }
    }
};

// Usage:
// NHits<In>;
// NHits<Out>;
