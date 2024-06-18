package com.satvik.ml;

import com.satvik.ml.util.Functions;
import com.satvik.ml.util.MathUtils;
import com.satvik.ml.util.Matrix;

import java.util.ArrayList;
import java.util.List;

public class Main {

    double ALPHA = 0.1;
    public static void main(String[] args) {

        // neural network which determines if the binary input is divisible by 3
        List<Pair<Matrix, Matrix>> trainingData = List.of(
                Pair.of(new Matrix(new double[][]{{0, 1, 1, 1, 0}}).transpose(), new Matrix(new double[][]{{0, 1}}).transpose()), //14
                Pair.of(new Matrix(new double[][]{{0, 1, 0, 0, 1}}).transpose(), new Matrix(new double[][]{{1, 0}}).transpose()), //9
                Pair.of(new Matrix(new double[][]{{1, 0, 1, 1, 0}}).transpose(), new Matrix(new double[][]{{0, 1}}).transpose()), //22
                Pair.of(new Matrix(new double[][]{{1, 1, 0, 0, 0}}).transpose(), new Matrix(new double[][]{{1, 0}}).transpose()), //24
                Pair.of(new Matrix(new double[][]{{1, 0, 0, 0, 0}}).transpose(), new Matrix(new double[][]{{0, 1}}).transpose()), //16
                Pair.of(new Matrix(new double[][]{{1, 1, 1, 1, 1}}).transpose(), new Matrix(new double[][]{{0, 1}}).transpose()), //31
                Pair.of(new Matrix(new double[][]{{0, 1, 1, 1, 1}}).transpose(), new Matrix(new double[][]{{1, 0}}).transpose()), //15
                Pair.of(new Matrix(new double[][]{{0, 0, 0, 1, 1}}).transpose(), new Matrix(new double[][]{{1, 0}}).transpose()), //3
                Pair.of(new Matrix(new double[][]{{0, 0, 1, 0, 0}}).transpose(), new Matrix(new double[][]{{0, 1}}).transpose()) //4
        );

        Main m = new Main();
        List<Integer> hiddenLayersNeuronsCount = List.of(3, 4, 3);
        m.train(trainingData, hiddenLayersNeuronsCount);


    }

    public void train(List<Pair<Matrix, Matrix>> trainingData, List<Integer> hiddenLayersNeuronsCount){


        int inputRows = trainingData.getFirst().getA().getRows();
        int outputRows = trainingData.getFirst().getB().getRows();

        List<Matrix> weights = new ArrayList<>();
        List<Matrix> biases = new ArrayList<>();

        int nHiddenLayers = hiddenLayersNeuronsCount.size();
        for(int i=0;i<nHiddenLayers;i++){
            biases.add(Matrix.random(hiddenLayersNeuronsCount.get(i), 1, -1, 1));
        }

        // last layer's biases
        biases.add(Matrix.random(outputRows, 1, -1, 1));

        int previousLayerNeuronsCount = inputRows;
        for(int i=0;i<nHiddenLayers;i++){
            weights.add(Matrix.random(hiddenLayersNeuronsCount.get(i), previousLayerNeuronsCount, -1, 1));
            previousLayerNeuronsCount = hiddenLayersNeuronsCount.get(i);
        }
        weights.add(Matrix.random(outputRows, previousLayerNeuronsCount, -1, 1));

        int iterations = 100_000;
        NeuralNetwork neuralNetwork = NeuralNetwork
                .builder()
                .weights(weights)
                .biases(biases)
                .layers(hiddenLayersNeuronsCount.size()+1)
                .build();

        for(int t = 0;t<iterations;t++) {
            for (Pair<Matrix, Matrix> trainingDatum : trainingData) {
                neuralNetwork = trainForEachSamplePeepeee(trainingDatum, neuralNetwork);
            }
            if((t+1)%10 == 0) {
                System.out.println("after " + (t + 1) + " epochs, error: \n" + neuralNetwork.getOutputErrorDiff());
            }
            trainingData = MathUtils.shuffle(trainingData);
        }

    }

    private NeuralNetwork trainForEachSamplePeepeee(Pair<Matrix, Matrix> trainingData, NeuralNetwork nnConfig) {
        List<Matrix> weights = nnConfig.getWeights();
        List<Matrix> biases = nnConfig.getBiases();
        List<Matrix> layerOutputs = new ArrayList<>();

        // first input is without any activation function
        Matrix input = trainingData.getA();
        Matrix bias = biases.getFirst();
        Matrix weight = weights.getFirst();
        Matrix outputLayer1 = bias.add(weight.cross(input));
        layerOutputs.add(outputLayer1);
        Matrix prevLayerOutput = outputLayer1;

        for(int i=1;i<nnConfig.getLayers();i++){
            input = prevLayerOutput.apply(Functions::sigmoid);
            bias = biases.get(i);
            weight = weights.get(i);
            Matrix outputLayerI = bias.add(weight.cross(input));
            layerOutputs.add(outputLayerI);

            prevLayerOutput = outputLayerI;
        }
        nnConfig.setLayerOutputs(layerOutputs);

        // back prop
        // last layer's calculation is different from hidden layers
        int layerInProcessing = nnConfig.getLayers() - 1;
        Matrix outputLayerErrorTerm = nnConfig.getLayerOutputs().get(layerInProcessing).subtract(trainingData.getB());
        Matrix deltaWeightLast = outputLayerErrorTerm.multiply(nnConfig.getLayerOutputs().get(layerInProcessing-1).apply(Functions::sigmoid));

        Matrix newWeights = nnConfig.getWeights().get(layerInProcessing).subtract(deltaWeightLast.apply(x -> ALPHA*x));
        nnConfig.getWeights().set(layerInProcessing, newWeights);

        Matrix newBiases = nnConfig.getBiases().get(layerInProcessing).subtract(outputLayerErrorTerm.apply(x -> ALPHA*x));
        nnConfig.getBiases().set(layerInProcessing, newBiases);

        Matrix nextLayerErrorTerm = outputLayerErrorTerm;

        int i;
        for(i=layerInProcessing-1;i>0;i--){
            Matrix thisLayerErrorTerm = nnConfig.getLayerOutputs().get(i).apply(Functions::differentialSigmoid).dot(nnConfig.getWeights().get(i+1).transpose().cross(nextLayerErrorTerm));
            Matrix deltaWeightI = thisLayerErrorTerm.multiply(nnConfig.getLayerOutputs().get(i-1).apply(Functions::sigmoid));
            newWeights = nnConfig.getWeights().get(i).subtract(deltaWeightI.apply(x -> ALPHA*x));
            nnConfig.getWeights().set(i, newWeights);

            newBiases = nnConfig.getBiases().get(i).subtract(thisLayerErrorTerm.apply(x -> ALPHA*x));
            nnConfig.getBiases().set(i, newBiases);

            nextLayerErrorTerm = thisLayerErrorTerm;
        }

        // for the first hidden layer, previous layer is the input. handle that accordingly
        Matrix thisLayerErrorTerm = nnConfig.getLayerOutputs().get(i).apply(Functions::differentialSigmoid).dot(nnConfig.getWeights().get(i+1).transpose().cross(nextLayerErrorTerm));
        Matrix deltaWeightI = thisLayerErrorTerm.multiply(trainingData.getA().apply(Functions::sigmoid));
        newWeights = nnConfig.getWeights().get(i).subtract(deltaWeightI.apply(x -> ALPHA*x));
        nnConfig.getWeights().set(i, newWeights);

        newBiases = nnConfig.getBiases().get(i).subtract(thisLayerErrorTerm.apply(x -> ALPHA*x));
        nnConfig.getBiases().set(i, newBiases);

        nnConfig.setOutputErrorDiff(outputLayerErrorTerm);

        return nnConfig;
    }

}
