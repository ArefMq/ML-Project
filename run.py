#!/usr/bin/env python2

import sys
from classification.classification import run_classification
from clustering.clustering import run_k_means_clustering, run_hierarchical_clustering
from nn.neural_networks import run_nn
from regression.bench_mark import run_bench_mark
from regression.feature_selection import run_feature_selection
from regression.linear_regression import run_linear_regression
from svm.tree_based_regression import run_tree_based_regression
from svm.classification import run_tree_based_classification

USAGE_MESSAGE = '''Usage:

    classification or cl:           Runs Classification
    bench-mark or bm:               Runs Bench-Mark
    feature-selection or fs:        Runs Feature Selection on regression data
    linear-regression or lr:        Runs Linear Regression
    tree-based-regression or tbr:   Runs Tree-Based Regression
    svm:                            Runs Random-Forest, Decision-Tree, and SVM (SVC) Classifications
    reg-all:                        Runs all Regression based modules
    kmeans or km:                   Run k-means clustering
    hierarchical-clustering or hc:  Run Hierarchical Clustering (Agglomerative-Clustering)
    neural-networks or nn:          Run Neural Networks Classifications
    
You can also run these commands for running projects based on the sessions:
    
    p1:     Runs #1 session project (Regression)
    p2:     Runs #2 session project (Classification)
    p3:     Runs #3 session project (SVM)
    p4:     Runs #4 session project (Clustering)
    p5:     Runs #5 session project (Neural Networks)
'''

MODULES = {
    # TODO: add valid-form and name for each functionality
    'classification': run_classification,
    'cl': run_classification,
    'bench-mark': run_bench_mark,
    'bm': run_bench_mark,
    'feature-selection': run_feature_selection,
    'fs': run_feature_selection,
    'linear-regression': run_linear_regression,
    'lr': run_linear_regression,
    'tree-based-regression': run_tree_based_regression,
    'tbr': run_tree_based_regression,
    'svm': run_tree_based_classification,
    'reg-all': [run_linear_regression, run_feature_selection, run_bench_mark, run_tree_based_regression],
    'kmeans': run_k_means_clustering,
    'km': run_k_means_clustering,
    'hierarchical-clustering': run_hierarchical_clustering,
    'hc': run_hierarchical_clustering,
    'neural-networks': run_nn,
    'nn': run_nn,

    'p1': [run_linear_regression, run_feature_selection, run_bench_mark],
    'p2': run_classification,
    'p3': [run_tree_based_regression, run_tree_based_classification],
    'p4': [run_k_means_clustering, run_hierarchical_clustering],
    'p5': [run_nn],
}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print USAGE_MESSAGE
        exit(0)
    modules = sys.argv[1:]

    for m in modules:
        if m not in MODULES:
            print "module '%s' not found.\n" % m
            print USAGE_MESSAGE
            continue

        if isinstance(MODULES[m], list):
            for f in MODULES[m]:
                print "Running '%s':\n" % f.__name__
                f()
        else:
            print "Running '%s':\n" % m
            MODULES[m]()
