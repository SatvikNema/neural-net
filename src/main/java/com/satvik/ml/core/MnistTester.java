package com.satvik.ml.core;

import com.satvik.ml.pojo.Pair;
import com.satvik.ml.util.Matrix;
import java.util.List;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Builder
@AllArgsConstructor
@NoArgsConstructor
@Data
public class MnistTester implements NeuralNetworkTester {
  private NeuralNetwork neuralNetwork;

  public double validate(List<Pair<Matrix, Matrix>> trainingData) {
    double error = 0;
    int countMissed = 0;
    for (Pair<Matrix, Matrix> trainingDatum : trainingData) {
      neuralNetwork.feedforward(trainingDatum.getA());
      Matrix output = neuralNetwork.getLayerOutputs().getLast();
      int predicted = output.max().getB()[0];
      int actual = trainingDatum.getB().max().getB()[0];
      if (predicted != actual) {
        countMissed++;
      }

      Matrix errorMatrix = output.subtract(trainingDatum.getB());
      error += errorMatrix.apply(x -> x * x).sum() / trainingData.size();
    }
    System.out.printf("Total: %s, wrong: %s%n", trainingData.size(), countMissed);
    return error;
  }
}
