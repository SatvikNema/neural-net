package com.satvik.ml.core;

import com.satvik.ml.pojo.Pair;
import com.satvik.ml.util.Functions;
import com.satvik.ml.util.MathUtils;
import com.satvik.ml.util.Matrix;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

@Builder
@AllArgsConstructor
@NoArgsConstructor
@Data
public class MnistTrainer implements NeuralNetworkTrainer{
    private NeuralNetwork neuralNetwork;
    private int iterations;
    private double learningRate;

    public void train(List<Pair<Matrix, Matrix>> trainingData){
        int mod = iterations / 100 == 0 ? 1 : iterations / 100;
        double error = 0;
        for(int t = 0; t< iterations; t++) {
            for (Pair<Matrix, Matrix> trainingDatum : trainingData) {
                neuralNetwork = trainForOneInput(trainingDatum, neuralNetwork);
                double errorAdditionTerm = neuralNetwork.getOutputErrorDiff().apply(x -> x*x).sum() / trainingData.size();
                error += errorAdditionTerm;
            }

            neuralNetwork.setAverageError(error);

            if((t == 0) || ((t+1)%mod == 0)) {
                System.out.println("after " + (t + 1) + " epochs, average error: " + error);
            }
            error = 0;
            trainingData = MathUtils.shuffle(trainingData);
        }
    }

    private NeuralNetwork trainForOneInput(Pair<Matrix, Matrix> trainingData, NeuralNetwork nnConfig) {

        nnConfig.feedforward(trainingData.getA());

        // back prop
        // last layer's calculation is different from hidden layers
        int layerInProcessing = nnConfig.getLayers() - 1;
        Matrix outputLayerErrorTerm = nnConfig.getLayerOutputs().get(layerInProcessing).subtract(trainingData.getB());
        Matrix deltaWeightLast = outputLayerErrorTerm.multiply(nnConfig.getLayerOutputs().get(layerInProcessing-1).apply(Functions::sigmoid));

        Matrix newWeights = nnConfig.getWeights().get(layerInProcessing).subtract(deltaWeightLast.apply(x -> learningRate*x));
        nnConfig.getWeights().set(layerInProcessing, newWeights);

        Matrix newBiases = nnConfig.getBiases().get(layerInProcessing).subtract(outputLayerErrorTerm.apply(x -> learningRate*x));
        nnConfig.getBiases().set(layerInProcessing, newBiases);

        Matrix nextLayerErrorTerm = outputLayerErrorTerm;

        int i;
        for(i=layerInProcessing-1;i>0;i--){
            Matrix thisLayerErrorTerm = nnConfig.getLayerOutputs().get(i).apply(Functions::differentialSigmoid).dot(nnConfig.getWeights().get(i+1).transpose().cross(nextLayerErrorTerm));
            Matrix deltaWeightI = thisLayerErrorTerm.multiply(nnConfig.getLayerOutputs().get(i-1).apply(Functions::sigmoid));
            newWeights = nnConfig.getWeights().get(i).subtract(deltaWeightI.apply(x -> learningRate*x));
            nnConfig.getWeights().set(i, newWeights);

            newBiases = nnConfig.getBiases().get(i).subtract(thisLayerErrorTerm.apply(x -> learningRate*x));
            nnConfig.getBiases().set(i, newBiases);

            nextLayerErrorTerm = thisLayerErrorTerm;
        }

        // for the first hidden layer, previous layer is the input. handle that accordingly
        Matrix thisLayerErrorTerm = nnConfig.getLayerOutputs().get(i).apply(Functions::differentialSigmoid).dot(nnConfig.getWeights().get(i+1).transpose().cross(nextLayerErrorTerm));
        Matrix deltaWeightI = thisLayerErrorTerm.multiply(trainingData.getA());
        newWeights = nnConfig.getWeights().get(i).subtract(deltaWeightI.apply(x -> learningRate*x));
        nnConfig.getWeights().set(i, newWeights);

        newBiases = nnConfig.getBiases().get(i).subtract(thisLayerErrorTerm.apply(x -> learningRate*x));
        nnConfig.getBiases().set(i, newBiases);

        nnConfig.setOutputErrorDiff(outputLayerErrorTerm);

        return nnConfig;
    }
}
