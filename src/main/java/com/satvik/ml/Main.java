package com.satvik.ml;

import com.satvik.ml.reader.MnistReader;
import com.satvik.ml.util.Functions;
import com.satvik.ml.util.MathUtils;
import com.satvik.ml.util.Matrix;
import com.satvik.ml.util.NNBuilder;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static java.lang.Double.NaN;

public class Main {

    public static final int ITERATIONS = 10;
    double ALPHA = 0.01;
    private static final String rootPath = "/Users/satvik.nema/Documents/mnist_dataset/";
    private double error = 0;
    public static void main(String[] args) {

        // neural network which determines if the binary input is divisible by 3

//        List<Pair<Matrix, Matrix>> trainingData = List.of(
//                Pair.of(new Matrix(new double[][]{{0, 1, 1, 1, 0}}).transpose(), new Matrix(new double[][]{{0, 1}}).transpose()), //14
//                Pair.of(new Matrix(new double[][]{{0, 1, 0, 0, 1}}).transpose(), new Matrix(new double[][]{{1, 0}}).transpose()), //9
//                Pair.of(new Matrix(new double[][]{{1, 0, 1, 1, 0}}).transpose(), new Matrix(new double[][]{{0, 1}}).transpose()), //22
//                Pair.of(new Matrix(new double[][]{{1, 1, 0, 0, 0}}).transpose(), new Matrix(new double[][]{{1, 0}}).transpose()), //24
//                Pair.of(new Matrix(new double[][]{{1, 0, 0, 0, 0}}).transpose(), new Matrix(new double[][]{{0, 1}}).transpose()), //16
//                Pair.of(new Matrix(new double[][]{{1, 1, 1, 1, 1}}).transpose(), new Matrix(new double[][]{{0, 1}}).transpose()), //31
//                Pair.of(new Matrix(new double[][]{{0, 1, 1, 1, 1}}).transpose(), new Matrix(new double[][]{{1, 0}}).transpose()), //15
//                Pair.of(new Matrix(new double[][]{{0, 0, 0, 1, 1}}).transpose(), new Matrix(new double[][]{{1, 0}}).transpose()), //3
//                Pair.of(new Matrix(new double[][]{{0, 0, 1, 0, 0}}).transpose(), new Matrix(new double[][]{{0, 1}}).transpose()) //4
//        );
        NeuralNetwork neuralNetwork;


//        Main m = new Main();
//        List<Integer> hiddenLayersNeuronsCount = List.of(3, 3);
//        NeuralNetwork neuralNetwork = m.train(trainingData, hiddenLayersNeuronsCount);
//
//        List<Matrix> outputs = m.feedforward(new Matrix(new double[][]{{0, 0, 0, 1, 1}}).transpose(), neuralNetwork);
//        System.out.println(outputs.getLast());


        String trainImagesPath =  rootPath + "train-images.idx3-ubyte";
        String trainLabelsPath =  rootPath + "train-labels.idx1-ubyte";
        List<Pair<Matrix, Matrix>> mnistTrainingData = MnistReader.getDataForNN(trainImagesPath, trainLabelsPath);
        Main m = new Main();
        List<Integer> hiddenLayersNeuronsCount = List.of(16, 16);
        neuralNetwork = m.train(mnistTrainingData, hiddenLayersNeuronsCount);

        String modelName = String.format("%s-%s.txt", mnistTrainingData.size(), ITERATIONS);
        try {
            neuralNetwork.serialise("/Users/satvik.nema/practise/nerual-net/src/main/resources/"+modelName);
        } catch (IOException e) {
            System.out.println("failed while trying to save model");
            throw new RuntimeException(e);
        }

        // read the pre-trained model
        try {
            neuralNetwork = NeuralNetwork.deserialise("/Users/satvik.nema/practise/nerual-net/src/main/resources/"+modelName);
            int x = 1;
        } catch (IOException e) {
            System.out.println("failed to load the model from disk");
            throw new RuntimeException(e);
        }

        double error = m.validate(neuralNetwork, mnistTrainingData);
        System.out.println(error);
    }

    private double validate(NeuralNetwork neuralNetwork, List<Pair<Matrix, Matrix>> trainingData) {
        double error = 0;
        for (Pair<Matrix, Matrix> trainingDatum : trainingData) {
            neuralNetwork.feedforward(trainingDatum.getA());
            Matrix output = neuralNetwork.getLayerOutputs().getLast();

            Matrix errorMatrix = output.subtract(trainingDatum.getB());
            error += errorMatrix.apply(x -> x*x).sum() / trainingData.size();
        }
        return error;
    }

    public NeuralNetwork train(List<Pair<Matrix, Matrix>> trainingData, List<Integer> hiddenLayersNeuronsCount){

        int inputRows = trainingData.getFirst().getA().getRows();
        int outputRows = trainingData.getFirst().getB().getRows();

        NeuralNetwork neuralNetwork = NNBuilder.create(inputRows, outputRows, hiddenLayersNeuronsCount);

        int mod = ITERATIONS / 100 == 0 ? 1 : ITERATIONS / 100;
        for(int t = 0; t< ITERATIONS; t++) {
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

        return neuralNetwork;
    }

    private NeuralNetwork trainForOneInput(Pair<Matrix, Matrix> trainingData, NeuralNetwork nnConfig) {

        nnConfig.feedforward(trainingData.getA());

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
        Matrix deltaWeightI = thisLayerErrorTerm.multiply(trainingData.getA());
        newWeights = nnConfig.getWeights().get(i).subtract(deltaWeightI.apply(x -> ALPHA*x));
        nnConfig.getWeights().set(i, newWeights);

        newBiases = nnConfig.getBiases().get(i).subtract(thisLayerErrorTerm.apply(x -> ALPHA*x));
        nnConfig.getBiases().set(i, newBiases);

        nnConfig.setOutputErrorDiff(outputLayerErrorTerm);

        return nnConfig;
    }

}
