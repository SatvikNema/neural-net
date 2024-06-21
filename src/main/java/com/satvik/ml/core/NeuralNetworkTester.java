package com.satvik.ml.core;

import com.satvik.ml.pojo.Pair;
import com.satvik.ml.util.Matrix;

import java.util.List;

interface NeuralNetworkTester {
    double validate(List<Pair<Matrix, Matrix>> trainingData);
}
