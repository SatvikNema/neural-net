package com.satvik.ml;

import com.satvik.ml.core.MnistTester;
import com.satvik.ml.core.MnistTrainer;
import com.satvik.ml.core.NNBuilder;
import com.satvik.ml.core.NeuralNetwork;
import com.satvik.ml.pojo.Pair;
import com.satvik.ml.reader.MnistReader;
import com.satvik.ml.util.Matrix;
import java.io.File;
import java.io.IOException;
import java.util.List;

public class Main {

    public static final int ITERATIONS = 10;
    static double ALPHA = 0.01;
    private static final String rootPath = "/Users/satvik.nema/Documents/mnist_dataset/";
    private static final String resourcesRoot =
            "/Users/satvik.nema/practise/nerual-net/src/main/resources/";

    public static void main(String[] args) throws IOException {
        String trainImagesPath = rootPath + "train-images.idx3-ubyte";
        String trainLabelsPath = rootPath + "train-labels.idx1-ubyte";

        List<Pair<Matrix, Matrix>> mnistTrainingData =
                MnistReader.getDataForNN(trainImagesPath, trainLabelsPath, 6000);

        List<Integer> hiddenLayersNeuronsCount = List.of(16, 16);

        int inputRows = mnistTrainingData.getFirst().getA().getRows();
        int outputRows = mnistTrainingData.getFirst().getB().getRows();

        MnistTrainer mnistTrainer =
                MnistTrainer.builder()
                        .neuralNetwork(
                                NNBuilder.create(inputRows, outputRows, hiddenLayersNeuronsCount))
                        .iterations(ITERATIONS)
                        .learningRate(ALPHA)
                        .build();
        NeuralNetwork trainedNetwork;

        String modelName =
                resourcesRoot
                        + String.format(
                                "%s-%s-%s.txt", mnistTrainingData.size(), ITERATIONS, ALPHA);
        File f = new File(modelName);
        if (f.isFile()) {
            System.out.println("model for this configuration already exists. loading from memory");
            trainedNetwork = NeuralNetwork.deserialise(modelName);
        } else {
            System.out.println("model for this configuration does NOT exists. Starting training");
            mnistTrainer.train(mnistTrainingData);
            trainedNetwork = mnistTrainer.getNeuralNetwork();
            trainedNetwork.serialise(modelName);
        }

        String testImagesPath = rootPath + "t10k-images.idx3-ubyte";
        String testLabelsPath = rootPath + "t10k-labels.idx1-ubyte";

        List<Pair<Matrix, Matrix>> mnistTestingData =
                MnistReader.getDataForNN(testImagesPath, testLabelsPath, -1);
        MnistTester mnistTester = MnistTester.builder().neuralNetwork(trainedNetwork).build();
        double error = mnistTester.validate(mnistTestingData);
        System.out.println(error);
    }
}
