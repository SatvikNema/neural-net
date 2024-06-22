package com.satvik.ml.core;

import com.satvik.ml.util.Matrix;
import java.util.ArrayList;
import java.util.List;

public class NNBuilder {
  public static NeuralNetwork create(
      int inputRows, int outputRows, List<Integer> hiddenLayersNeuronsCount) {
    List<Matrix> weights = new ArrayList<>();
    List<Matrix> biases = new ArrayList<>();

    int nHiddenLayers = hiddenLayersNeuronsCount.size();
    for (Integer integer : hiddenLayersNeuronsCount) {
      biases.add(Matrix.random(integer, 1, -1, 1));
    }

    // last layer's biases
    biases.add(Matrix.random(outputRows, 1, -1, 1));

    int previousLayerNeuronsCount = inputRows;
    for (int i = 0; i < nHiddenLayers; i++) {
      weights.add(Matrix.random(hiddenLayersNeuronsCount.get(i), previousLayerNeuronsCount, -1, 1));
      previousLayerNeuronsCount = hiddenLayersNeuronsCount.get(i);
    }
    weights.add(Matrix.random(outputRows, previousLayerNeuronsCount, -1, 1));

    return NeuralNetwork.builder()
        .weights(weights)
        .biases(biases)
        .layers(hiddenLayersNeuronsCount.size() + 1)
        .build();
  }
}
