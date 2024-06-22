package com.satvik.ml.util;

import org.junit.jupiter.api.Test;

class MatrixTest {

  @Test
  void testGetContentToSerialise() {
    Matrix m =
        new Matrix(
            new double[][] {
              {1, 2, 3, 4, 5},
              {6, 7, 8, 9, 10},
              {11, 12, 13, 14, 15},
            });

    String stringContent = m.getContentToSerialise();
  }
}
