package com.satvik.ml.util;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

public class MathUtils {
  private MathUtils() {}

  /**
   * @param n old value which was generated between iL and iR
   * @param iL left limit in which n was generated
   * @param iR right limit in which n was generated
   * @param L new left limit in which n is to be scaled
   * @param R new right limit in which n is to be scaled
   * @return N = new scaled value between L and R proportional to n (between iL to iR)
   */
  public static double scaleValue(double n, double iL, double iR, double L, double R) {
    return L + ((R - L) * ((n - iL) / (iR - iL)));
  }

  private static final Random random = new Random(1);

  public static <T> List<T> shuffle(List<T> arr) {
    int size = arr.size();

    Set<Integer> done = new LinkedHashSet<>();

    for (int i = 0; i < size; i++) {
      int nextIndex = random.nextInt(i, size);
      if (done.contains(nextIndex)) {
        continue;
      }
      done.add(nextIndex);
    }

    if (done.size() != size) {
      boolean startFromStart = random.nextInt(0, 2) % 2 == 0;
      if (startFromStart) {
        for (int i = 0; i < size; i++) {
          done.add(i);
        }
      } else {
        for (int i = size - 1; i >= 0; i--) {
          done.add(i);
        }
      }
    }
    List<T> result = new ArrayList<>();
    done.forEach(index -> result.add(arr.get(index)));
    return result;
  }
}
