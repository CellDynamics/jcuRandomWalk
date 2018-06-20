package com.github.celldynamics.jcurandomwalk;

import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Example of MT.
 * 
 * @author baniuk
 *
 */
public class Threading {

  final int numThreads = 3;
  final int cases = 20;
  RandomWalkOptions globalOpts = new RandomWalkOptions();

  public void run(RandomWalkOptions opts) {
    Random ra = new Random();
    int size = ra.nextInt(5000000) + 1;
    System.out.println(
            opts.stack.toString() + " siz: " + size + " TH: " + Thread.currentThread().getName()
                    + " ID: " + Thread.currentThread().getId() % numThreads);
    Double[] t = new Double[size];
    for (int i = 0; i < t.length; i++) {
      t[i] = ra.nextDouble();
    }
    Arrays.sort(t);
  }

  public void runner() {
    ArrayList<MyCallable> opt = new ArrayList<MyCallable>();
    for (int i = 0; i < cases; i++) { // number of items to process
      RandomWalkOptions o = new RandomWalkOptions(globalOpts);
      o.stack = Paths.get("p", "" + i);
      opt.add(new MyCallable(o));
    }

    ExecutorService executor = Executors.newFixedThreadPool(numThreads);
    for (MyCallable mc : opt) {
      executor.submit(mc);
    }
    executor.shutdown();

  }

  class MyCallable implements Callable<Void> {

    private RandomWalkOptions options;

    public MyCallable(RandomWalkOptions options) {
      this.options = options;
    }

    @Override
    public Void call() throws Exception {
      run(options);
      return null;
    }

  }

  public static void main(String[] args) {
    new Threading().runner();

  }

}
