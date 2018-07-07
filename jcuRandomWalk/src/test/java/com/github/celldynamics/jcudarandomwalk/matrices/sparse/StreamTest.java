package com.github.celldynamics.jcudarandomwalk.matrices.sparse;

import java.util.stream.IntStream;

import org.apache.commons.lang3.ArrayUtils;

public class StreamTest {

  public static void main(String[] args) {
    int[] a = new int[10];

    IntStream.range(0, 10).forEach(i -> {
      a[i] = i * i;
    });
    System.out.println(ArrayUtils.toString(a));
  }

}
