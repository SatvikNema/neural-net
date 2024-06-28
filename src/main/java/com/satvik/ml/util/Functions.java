package com.satvik.ml.util;

public class Functions {

    private Functions(){}

    public static double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    public static double differentialSigmoid(double x) {
        return sigmoid(x) * (1 - sigmoid(x));
    }
}
