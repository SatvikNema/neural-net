package com.satvik.ml;

import com.satvik.ml.util.Functions;
import com.satvik.ml.util.MathUtils;
import com.satvik.ml.util.Matrix;

import java.security.KeyPair;
import java.util.Collections;
import java.util.List;
import java.util.Random;

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
        m.train(trainingData);


    }

    public void train(List<Pair<Matrix, Matrix>> trainingData){


        int r = trainingData.getFirst().getA().getRows();
        int outputRows = trainingData.getFirst().getB().getRows();

        // 5 -> 3 -> 3 -> 2 neural net
        Matrix biases1 = Matrix.random(3, 1, -1, 1);
        Matrix biases2 = Matrix.random(3, 1, -1, 1);
        Matrix biases3 = Matrix.random(2, 1, -1, 1);


        Matrix weights1 = Matrix.random(3, r, -1, 1);
        Matrix weights2 = Matrix.random(3, 3, -1, 1);
        Matrix weights3 = Matrix.random(outputRows, 3, -1, 1);

        NeuralNetworkConfig nnConfig = NeuralNetworkConfig
                .builder()
                .biases1(biases1)
                .biases2(biases2)
                .biases3(biases3)
                .weights1(weights1)
                .weights2(weights2)
                .weights3(weights3)
                .build();

        int size = trainingData.size();
        int iterations = 100_000;

        for(int t = 0;t<iterations;t++) {
            for (Pair<Matrix, Matrix> trainingDatum : trainingData) {
                nnConfig = trainForEachSample(trainingDatum, nnConfig);
            }
            if((t+1)%1000 == 0) {
                System.out.println("after " + (t + 1) + " epochs, error: \n" + nnConfig.getOutputErrorDiff());
            }
            trainingData = MathUtils.shuffle(trainingData);
        }

    }

    private NeuralNetworkConfig trainForEachSample(Pair<Matrix, Matrix> trainingData, NeuralNetworkConfig nnConfig) {
        Matrix weights1 = nnConfig.getWeights1();
        Matrix weights2 = nnConfig.getWeights2();
        Matrix weights3 = nnConfig.getWeights3();
        Matrix biases1 = nnConfig.getBiases1();
        Matrix biases2 = nnConfig.getBiases2();
        Matrix biases3 = nnConfig.getBiases3();

        Matrix input = trainingData.getA();
        Matrix bias = biases1;
        Matrix weights = weights1;

        Matrix outputLayer1 = bias.add(weights.cross(input));

        // for layer 2
        input = outputLayer1.apply(Functions::sigmoid);
        bias = biases2;
        weights = weights2;
        Matrix outputLayer2 = bias.add(weights.cross(input));

        // for layer 3
        input = outputLayer2.apply(Functions::sigmoid);
        bias = biases3;
        weights = weights3;
        Matrix outputLayer3 = bias.add(weights.cross(input));

        // error factor for layer 3


        // backward prop (assuming no activation function for the output layer)
        Matrix outputLayerErrorFactorForEachWeight = outputLayer3
                .subtract(trainingData.getB());
        Matrix deltaWeight3 = outputLayerErrorFactorForEachWeight.multiply(outputLayer2.apply(Functions::sigmoid));
        weights3 = weights3.subtract(deltaWeight3.apply(x -> ALPHA*x));
        biases3 = biases3.subtract(outputLayerErrorFactorForEachWeight.apply(x -> ALPHA*x));

        Matrix secondLayerErrorFactorForEachWeight = outputLayer2.apply(Functions::differentialSigmoid)
                .dot(weights3.transpose().cross(outputLayerErrorFactorForEachWeight));
        Matrix deltaWeight2 = secondLayerErrorFactorForEachWeight.multiply(outputLayer1.apply(Functions::sigmoid));
        weights2 = weights2.subtract(deltaWeight2.apply(x -> ALPHA*x));
        biases2 = biases2.subtract(secondLayerErrorFactorForEachWeight.apply(x -> ALPHA*x));

        Matrix firstLayerErrorFactorForEachWeight = outputLayer1.apply(Functions::differentialSigmoid)
                .dot(weights2.transpose().cross(secondLayerErrorFactorForEachWeight));
        Matrix deltaWeight1 = firstLayerErrorFactorForEachWeight.multiply(trainingData.getA().apply(Functions::sigmoid));
        weights1 = weights1.subtract(deltaWeight1.apply(x -> ALPHA*x));
        biases1 = biases1.subtract(firstLayerErrorFactorForEachWeight.apply(x -> ALPHA*x));

        return NeuralNetworkConfig
                .builder()
                .biases1(biases1)
                .biases2(biases2)
                .biases3(biases3)
                .weights1(weights1)
                .weights2(weights2)
                .weights3(weights3)
                .outputErrorDiff(outputLayerErrorFactorForEachWeight)
                .build();
    }

}
