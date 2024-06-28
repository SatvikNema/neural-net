package com.satvik.ml.core;

import com.satvik.ml.pojo.Pair;
import com.satvik.ml.util.Functions;
import com.satvik.ml.util.Matrix;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class NeuralNetwork {

    private List<Matrix> weights;
    private List<Matrix> biases;
    private List<Matrix> layerOutputs;
    private int layers;
    private Matrix outputErrorDiff;
    private double averageError;

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        //        getLayerOutputs().forEach(e -> sb.append(e).append("\n"));
        getWeights().forEach(e -> sb.append(e).append("\n"));
        getBiases().forEach(e -> sb.append(e).append("\n"));
        return sb.toString();
    }

    private String getSerialisedContent() {
        StringBuilder content = new StringBuilder(layers + "\n");
        for (Matrix w : weights) {
            content.append(w.getContentToSerialise());
        }
        for (Matrix b : biases) {
            content.append(b.getContentToSerialise());
        }
        content.append(averageError).append("\n");
        return content.toString();
    }

    public void serialise(String filePath) throws IOException {
        try (PrintWriter printWriter = new PrintWriter(filePath)) {
            printWriter.print(getSerialisedContent());
        }
    }

    public static NeuralNetwork deserialise(String filePath) throws IOException {

        List<Matrix> weights = new ArrayList<>();
        List<Matrix> biases = new ArrayList<>();
        int layers;
        NeuralNetwork neuralNetwork = NeuralNetwork.builder().build();
        try (Stream<String> linesStream =
                Files.lines(Path.of(filePath), Charset.defaultCharset())) {
            List<String> lines = linesStream.toList();
            int index = 0;
            layers = Integer.parseInt(lines.get(index++));
            index = extractComponent(weights, layers, lines, index);
            index = extractComponent(biases, layers, lines, index);
            double avgError = Double.parseDouble(lines.get(index));
            neuralNetwork =
                    NeuralNetwork.builder()
                            .weights(weights)
                            .biases(biases)
                            .averageError(avgError)
                            .layers(layers)
                            .build();
        }
        return neuralNetwork;
    }

    private static int extractComponent(
            List<Matrix> component, int layers, List<String> lines, int index) {
        for (int i = 0; i < layers; i++) {
            String[] dimensions = lines.get(index++).split("\\s+");
            int rows = Integer.parseInt(dimensions[0]);
            int columns = Integer.parseInt(dimensions[1]);

            double[][] bias = new double[rows][columns];
            for (int row = 0; row < rows; row++) {
                List<Double> doubleList =
                        Stream.of(lines.get(index++).split("\\s+"))
                                .map(Double::parseDouble)
                                .toList();
                double[] thisRow = new double[columns];
                for (int j = 0; j < columns; j++) {
                    thisRow[j] = doubleList.get(j);
                }
                bias[row] = thisRow;
            }
            component.add(new Matrix(bias));
        }
        return index;
    }

    public void trainForOneInput(Pair<Matrix, Matrix> trainingData, double learningRate) {
        feedforward(trainingData.getA());
        backpropagation(trainingData, learningRate);
    }

    public void feedforward(Matrix input) {
        List<Matrix> layerOutputs = new ArrayList<>();

        // first input is without any activation function
        Matrix bias = biases.getFirst();
        Matrix weight = weights.getFirst();
        Matrix outputLayer1 = bias.add(weight.cross(input));
        layerOutputs.add(outputLayer1);
        Matrix prevLayerOutput = outputLayer1;

        for (int i = 1; i < getLayers(); i++) {
            input = prevLayerOutput.apply(Functions::sigmoid);
            bias = biases.get(i);
            weight = weights.get(i);
            Matrix outputLayerI = bias.add(weight.cross(input));
            layerOutputs.add(outputLayerI);

            prevLayerOutput = outputLayerI;
        }
        setLayerOutputs(layerOutputs);
    }

    private void backpropagation(Pair<Matrix, Matrix> trainingData, double learningRate) {
        // back prop - last layer's calculation is different from hidden layers
        Matrix outputLayerErrorTerm = backpropagationForLastLayer(trainingData, learningRate);
        Matrix nextLayerErrorTerm = outputLayerErrorTerm;
        outputErrorDiff = outputLayerErrorTerm;

        // process the hidden layers
        int i;
        for (i = layers - 2; i > 0; i--) {
            Matrix thisLayerErrorTerm =
                    layerOutputs
                            .get(i)
                            .apply(Functions::differentialSigmoid)
                            .dot(weights.get(i + 1).transpose().cross(nextLayerErrorTerm));
            adjustWeightsAndBiases(learningRate, i, thisLayerErrorTerm);

            nextLayerErrorTerm = thisLayerErrorTerm;
        }

        // for the first hidden layer, previous layer is the input. handle that accordingly
        backpropagationForSecondLayer(trainingData.getA(), nextLayerErrorTerm, learningRate);
    }

    private Matrix backpropagationForLastLayer(
            Pair<Matrix, Matrix> trainingData, double learningRate) {
        int layerInProcessing = layers - 1;
        Matrix outputLayerErrorTerm =
                layerOutputs.get(layerInProcessing).subtract(trainingData.getB());
        adjustWeightsAndBiases(learningRate, layerInProcessing, outputLayerErrorTerm);

        return outputLayerErrorTerm;
    }

    private void adjustWeightsAndBiases(double learningRate, int i, Matrix thisLayerErrorTerm) {
        Matrix deltaWeightI =
                thisLayerErrorTerm.cross(
                        layerOutputs.get(i - 1).apply(Functions::sigmoid).transpose());
        Matrix newWeights = weights.get(i).subtract(deltaWeightI.apply(x -> learningRate * x));
        weights.set(i, newWeights);

        Matrix newBiases = biases.get(i).subtract(thisLayerErrorTerm.apply(x -> learningRate * x));
        biases.set(i, newBiases);
    }

    private void backpropagationForSecondLayer(
            Matrix trainingData, Matrix nextLayerErrorTerm, double learningRate) {
        Matrix thisLayerErrorTerm =
                layerOutputs
                        .getFirst()
                        .apply(Functions::differentialSigmoid)
                        .dot(weights.get(1).transpose().cross(nextLayerErrorTerm));
        Matrix deltaWeightI = thisLayerErrorTerm.cross(trainingData.transpose());
        Matrix newWeights = weights.get(0).subtract(deltaWeightI.apply(x -> learningRate * x));
        weights.set(0, newWeights);

        Matrix newBiases =
                biases.getFirst().subtract(thisLayerErrorTerm.apply(x -> learningRate * x));
        biases.set(0, newBiases);
    }
}
