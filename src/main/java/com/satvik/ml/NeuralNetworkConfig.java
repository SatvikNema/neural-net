package com.satvik.ml;

import com.satvik.ml.util.Matrix;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class NeuralNetworkConfig {
    private Matrix weights1;
    private Matrix weights2;
    private Matrix weights3;

    private Matrix biases1;
    private Matrix biases2;
    private Matrix biases3;

    private Matrix outputErrorDiff;

}
