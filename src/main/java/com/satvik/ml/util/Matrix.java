package com.satvik.ml.util;

import lombok.Data;

import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.function.Function;

import static com.satvik.ml.util.MathUtils.scaleValue;

@Data
public class Matrix {
    private static final Random random = new Random(1);
    private final double[][] content;

    private final int rows;
    private final int columns;

    public Matrix(int i, int j){
        this.rows = i;
        this.columns = j;
        this.content = new double[i][j];
    }

    public Matrix(double[][] arr){
        int r = arr.length, c = arr[0].length;
        content = arr;
        rows = r;
        columns = c;
    }

    public Matrix(int[][] arr){
        int r = arr.length, c = arr[0].length;
        content = new double[r][c];
        rows = r;
        columns = c;
        for(int i=0;i<r;i++){
            for(int j=0;j<c;j++){
                content[i][j] = arr[i][j];
            }
        }
    }

    public Matrix dot(Matrix b){
        if(!dimensionsMatch(b)){
            throw new RuntimeException("cannot do dot product of matrices as the dimensions do not match!");
        }
        double[][] result = new double[b.rows][b.columns];
        for(int i=0;i<rows;i++){
            for(int j=0;j<columns;j++){
                result[i][j] = content[i][j] * b.content[i][j];
            }
        }
        return new Matrix(result);
    }

    public Matrix cross(Matrix b){
        int[] newDims = getCrossProductDimensions(b);
        int r = newDims[0];
        int c = newDims[1];
        double[][] result = new double[r][c];
        double sum;
        for(int i=0;i<rows;i++){
            for(int j=0;j<b.columns;j++){
                sum = 0;
                for(int k=0;k<b.rows;k++){
                    sum += content[i][k] * b.content[k][j];
                }
                result[i][j] = sum;
            }
        }
        return new Matrix(result);
    }

    public Matrix transpose(){
        double[][] result = new double[columns][rows];
        for(int i=0;i<rows;i++){
            for(int j=0;j<columns;j++){
                result[j][i] = content[i][j];
            }
        }
        return new Matrix(result);
    }

    public Matrix apply(Function<Double, Double> f){
        double[][] result = new double[rows][columns];
        for(int i=0;i<rows;i++){
            for(int j=0;j<columns;j++){
                result[i][j] = f.apply(content[i][j]);
            }
        }

        return new Matrix(result);
    }

    private int[] getCrossProductDimensions(Matrix b) {
        if(getColumns() != b.getRows()){
            throw new RuntimeException("Cross product not compatible between the matrices!");
        }
        return new int[]{getRows(), b.getColumns()};
    }

    private boolean dimensionsMatch(Matrix b){
        return getColumns() == b.getColumns() && getRows() == b.getRows();
    }

    @Override
    public String toString(){
        StringBuilder s = new StringBuilder();
        for(double[] i:content){
            s.append(Arrays.toString(i)).append("\n");
        }
        return s.toString();
    }

    public void print(){
        System.out.println(this);
    }

    public Matrix add(Matrix b){
        if(!dimensionsMatch(b)){
            throw new RuntimeException("cannot do matrix addition as the dimensions do not match!");
        }
        double[][] result = new double[rows][columns];
        for(int i=0;i<rows;i++){
            for(int j=0;j<columns;j++){
                result[i][j] = content[i][j] + b.content[i][j];
            }
        }
        return new Matrix(result);
    }

    public Matrix subtract(Matrix b){
        if(!dimensionsMatch(b)){
            throw new RuntimeException("cannot do matrix addition as the dimensions do not match!");
        }
        Matrix b2 = b.apply(x -> -x);
        return add(b2);
    }

    public static Matrix random(int r, int c, double start, double end){

        double[][] result = new double[r][c];
        for(int i=0;i<r;i++){
            for(int j=0;j<c;j++){
                double randomValue = random.nextDouble();
                result[i][j] = scaleValue(randomValue, 0, 1, start, end);
            }
        }
        return new Matrix(result);
    }

    public static Matrix ones(int r, int c){
        double[][] result = new double[r][c];
        for(int i=0;i<r;i++){
            for(int j=0;j<c;j++){
                result[i][j] = 1;
            }
        }
        return new Matrix(result);
    }

    public Matrix multiply(Matrix b) {
        int rows = this.rows;
        int columns = b.rows;
        if(this.columns > 1 || b.columns > 1){
            throw new RuntimeException("Cannot do a multiply with more than one column");
        }
        double[][] result = new double[rows][columns];
        for(int i=0;i<rows;i++){
            for(int j=0;j<columns;j++){
                result[i][j] = this.content[i][0] * b.content[j][0];
            }
        }
        return new Matrix(result);

    }

    public Matrix flatten(){
        double[][] result = new double[1][rows * columns];
        int index = 0;
        for(int i=0;i<rows;i++){
            for(int j=0;j<columns;j++){
                result[0][index++] = content[i][j];
            }

        }
        return new Matrix(result);
    }

    public static Matrix average(List<Matrix> matrices){
        int r = matrices.getFirst().rows;
        int c = matrices.getFirst().columns;
        double[][] result = new double[r][c];
        for(int i=0;i<r;i++){
            for(int j=0;j<c;j++){
                double sum = 0;
                for(Matrix m : matrices){
                    sum += m.content[i][j];
                }
                result[i][j] = sum / matrices.size();
            }
        }
        return new Matrix(result);
    }

    public double sum(){
        double sum = 0;
        for(int i=0;i<rows;i++){
            for(int j=0;j<columns;j++){
                sum += content[i][j];
            }
        }
        return sum;
    }
}
