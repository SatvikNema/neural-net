package com.satvik.ml.core;

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
            int x = 1;
            int index = 0;
            layers = Integer.parseInt(lines.get(index++));
            for (int i = 0; i < layers; i++) {
                String[] dimensions = lines.get(index++).split("\\s+");
                int rows = Integer.parseInt(dimensions[0]);
                int columns = Integer.parseInt(dimensions[1]);

                double[][] weight = new double[rows][columns];
                for (int row = 0; row < rows; row++) {
                    List<Double> doubleList =
                            Stream.of(lines.get(index++).split("\\s+"))
                                    .map(Double::parseDouble)
                                    .toList();
                    double[] thisRow = new double[columns];
                    for (int j = 0; j < columns; j++) {
                        thisRow[j] = doubleList.get(j);
                    }
                    weight[row] = thisRow;
                }
                weights.add(new Matrix(weight));
            }

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
                biases.add(new Matrix(bias));
            }

            double avgError = Double.parseDouble(lines.get(index++));
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
}
