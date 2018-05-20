package com.github.celldynamics.jcurandomwalk;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Main application class.
 */
public class JcuRandomWalk {
  static final Logger LOGGER = LoggerFactory.getLogger(JcuRandomWalk.class.getName());
  private Options options = null;
  private int cliErrorStatus = 0; // returned to main

  /**
   * Constructor - parser initialised by cli.
   * 
   * @param args args passed from cli.
   */
  public JcuRandomWalk(String[] args) {
    CommandLineParser parser = new DefaultParser();
    options = new Options();
    options.addOption(new Option("help", "print this message"));
    try {
      CommandLine cmd = parser.parse(options, args);
      if (cmd.hasOption("help")) { // show help and finish
        showHelp();
        return;
      }
    } catch (org.apache.commons.cli.ParseException pe) {
      System.err.println("Parsing failed. Reason: " + pe.getMessage());
      showHelp();
      cliErrorStatus = 1; // finish with error
      return;
    }
  }

  /**
   * Print help string.
   */
  private void showHelp() {
    HelpFormatter formatter = new HelpFormatter();
    String header = "Random Walk segemntaion on GPU.\n\n";
    String footer = "\n";
    formatter.printHelp("JcuRandomWalk", header, options, footer, true);
  }

  /**
   * Main entry point.
   * 
   * <p>See cmd help for options.
   * 
   * @param args cmd args
   */
  public static void main(String[] args) {
    JcuRandomWalk app = new JcuRandomWalk(args);
    System.exit(app.cliErrorStatus);
  }
}
