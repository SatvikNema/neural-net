package com.satvik.ml.core;

import com.satvik.ml.pojo.Pair;
import com.satvik.ml.util.MathUtils;
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
public class MnistTrainer implements NeuralNetworkTrainer {
    private NeuralNetwork neuralNetwork;
    private int iterations;
    private double learningRate;

    public void train(List<Pair<Matrix, Matrix>> trainingData) {
        int mod = iterations / 100 == 0 ? 1 : iterations / 100;
        double error = 0;
        for (int t = 0; t < iterations; t++) {
            for (Pair<Matrix, Matrix> trainingDatum : trainingData) {
                neuralNetwork.trainForOneInput(trainingDatum, learningRate);
                double errorAdditionTerm =
                        neuralNetwork.getOutputErrorDiff().apply(x -> x * x).sum()
                                / trainingData.size();
                error += errorAdditionTerm;
            }

            neuralNetwork.setAverageError(error);

            if ((t == 0) || ((t + 1) % mod == 0)) {
                System.out.println("after " + (t + 1) + " epochs, average error: " + error);
            }
            error = 0;
            trainingData = MathUtils.shuffle(trainingData);
        }
    }
}
