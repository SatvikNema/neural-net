package com.satvik.ml;

import com.satvik.ml.util.Matrix;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

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
    private List<Integer> structure;

    @Override
    public String toString(){
        StringBuilder sb = new StringBuilder();
//        getLayerOutputs().forEach(e -> sb.append(e).append("\n"));
        getWeights().forEach(e -> sb.append(e).append("\n"));
        getBiases().forEach(e -> sb.append(e).append("\n"));
        return sb.toString();
    }

}
