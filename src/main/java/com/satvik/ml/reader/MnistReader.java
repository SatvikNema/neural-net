package com.satvik.ml.reader;

import com.satvik.ml.Pair;
import com.satvik.ml.util.MathUtils;
import com.satvik.ml.util.Matrix;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class MnistReader {

    public static void main(String[] args) throws IOException {
        System.out.println("hello");
        String trainImagesPath =  "train-images.idx3-ubyte";
        String trainLabelsPath =  "train-labels.idx1-ubyte";

        List<Pair<Matrix, Matrix>> trainingData = getDataForNN(trainImagesPath, trainLabelsPath);
        int x = 1;
    }

    public static List<Pair<Matrix, Matrix>> getDataForNN(String testImagesPath, String testLabelsPath){
        try {
            return getDataForNNHelper(testImagesPath, testLabelsPath);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private static List<Pair<Matrix, Matrix>> getDataForNNHelper(String imagesPath, String labelsPath) throws IOException {
        List<Pair<Matrix, Matrix>> data = new ArrayList<>();
        try(DataInputStream trainingDis = new DataInputStream(new BufferedInputStream(new FileInputStream(imagesPath)))){
            try(DataInputStream labelDis = new DataInputStream(new BufferedInputStream(new FileInputStream(labelsPath)))){
                int magicNumber = trainingDis.readInt();
                int numberOfItems = trainingDis.readInt();
                int nRows = trainingDis.readInt();
                int nCols = trainingDis.readInt();

                int labelMagicNumber = labelDis.readInt();
                int numberOfLabels = labelDis.readInt();

                // only taking 1/10th of data for testing
                for(int t=0;t<10;t++){
                    double[][] imageContent = new double[nRows][nCols];
                    for(int i=0;i<nRows;i++){
                        for(int j=0;j<nCols;j++){
                            imageContent[i][j] = trainingDis.readUnsignedByte();
                        }
                    }
                    Matrix imageData = new Matrix(imageContent).apply(pixel -> MathUtils.scaleValue(pixel, 0, 255, 0, 1)).flatten().transpose();

                    int label = labelDis.readUnsignedByte();
                    double[] output = new double[10];
                    output[label] = 1;
                    Matrix outputMatrix = new Matrix(new double[][]{output}).transpose();
                    data.add(Pair.of(imageData, outputMatrix));


                }
            }

        }
        return data;
    }
}
