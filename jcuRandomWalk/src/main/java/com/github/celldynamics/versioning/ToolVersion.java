package com.github.celldynamics.versioning;

import java.io.IOException;
import java.net.URL;
import java.text.SimpleDateFormat;
import java.time.Instant;
import java.time.format.DateTimeFormatter;
import java.time.temporal.TemporalAccessor;
import java.util.Arrays;
import java.util.Collections;
import java.util.Date;
import java.util.Enumeration;
import java.util.jar.Attributes;
import java.util.jar.Manifest;

import org.apache.commons.lang3.text.WordUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Tool versioning.
 * 
 * @author p.baniukiewicz
 */
public class ToolVersion {

  /**
   * The Constant LOGGER.
   */
  static final Logger LOGGER = LoggerFactory.getLogger(ToolVersion.class.getName());
  /**
   * Message returned by {@link #getQuimPBuildInfo()} if info is not found in jar.
   */
  public static final String defNote = "0.0.0";

  /**
   * Default line length.
   */
  public static int LINE_WRAP = 80;
  private String propertyFileName;

  /**
   * Constructor that links to property file with Maven data.
   * 
   * <p>Maven needs to have enabled resources plugin.
   * 
   * @param propertyFileName name with path of the property file. Path is package name with slashes.
   */
  public ToolVersion(String propertyFileName) {
    this.propertyFileName = propertyFileName;
  }

  /**
   * Prepare info plate for QuimP.
   * 
   * @param authors array of authors and theirs emails. Content will be printed in box.
   * @return QuimP version
   * @see #getFormattedToolVersion(ToolVersionStruct, String[], String)
   */
  public String getToolversion(String[] authors) {
    ToolVersionStruct quimpBuildInfo = getQuimPBuildInfo();
    return getFormattedToolVersion(quimpBuildInfo, authors, propertyFileName);
  }

  /**
   * Prepare info plate for tool.
   * 
   * <p>It contains version, names, etc. By general QuimpToolsCollection class is static. These
   * methods can not be so they must be called:
   * 
   * <pre>
   * <code>LOGGER.debug(new Tool().getQuimPversion());</code>
   * </pre>
   * 
   * @param toolBuildInfo info read from jar
   * @param authors array of authors
   * @param propertyFileName name of property file. It should be in correct package
   * @return Formatted string with QuimP version and authors
   * @see #getQuimPBuildInfo()
   */
  public static String getFormattedToolVersion(ToolVersionStruct toolBuildInfo, String[] authors,
          String propertyFileName) {
    java.util.function.Function<Object[], String> times = (Object[] a) -> {
      String ch = (String) a[0];
      int i = (int) a[1];
      if (i < 0) {
        return "";
      } else {
        return String.join("", Collections.nCopies(i, ch));
      }
    };
    String web = "";
    try {
      web = new PropertyReader().readProperty(propertyFileName, "webPage");
    } catch (Exception e) {
      ;
    }
    // longest string in array
    int longest = Arrays.asList(authors).stream().mapToInt(s -> s.length()).max().getAsInt();
    if (longest > LINE_WRAP) {
      longest = LINE_WRAP;
    }
    String infoPlate = "\u2554" + times.apply(new Object[] { "\u2550", longest + 2 }) + "\u2557";
    infoPlate = infoPlate.concat("\n");
    infoPlate = infoPlate.concat("\u2551" + " " + toolBuildInfo.getName()
            + times.apply(new Object[] { " ", longest - toolBuildInfo.getName().length() }) + " "
            + "\u2551");
    infoPlate = infoPlate.concat("\n");
    infoPlate = infoPlate.concat(
            "\u2551" + " " + times.apply(new Object[] { " ", longest - 0 }) + " " + "\u2551");
    infoPlate = infoPlate.concat("\n");
    for (String s : authors) {
      infoPlate = infoPlate.concat("\u2551" + " " + s
              + times.apply(new Object[] { " ", longest - s.length() }) + " " + "\u2551");
      infoPlate = infoPlate.concat("\n");
    }
    infoPlate = infoPlate
            .concat("\u255F" + times.apply(new Object[] { "\u2500", longest + 2 }) + "\u2562");
    infoPlate = infoPlate.concat("\n");

    infoPlate = infoPlate.concat("\u2551" + " " + web
            + times.apply(new Object[] { " ", longest - web.length() }) + " " + "\u2551");
    infoPlate = infoPlate.concat("\n");
    infoPlate = infoPlate.concat(
            "\u2551" + " " + times.apply(new Object[] { " ", longest - 0 }) + " " + "\u2551");
    infoPlate = infoPlate.concat("\n");
    infoPlate = infoPlate.concat("\u2551" + " " + "Version: " + toolBuildInfo.getVersion()
            + times.apply(new Object[] { " ", longest - toolBuildInfo.getVersion().length() - 9 })
            + " " + "\u2551");
    infoPlate = infoPlate.concat("\n");
    infoPlate = infoPlate.concat("\u2551" + " " + "Build by: " + toolBuildInfo.getBuildstamp()
            + times.apply(
                    new Object[] { " ", longest - toolBuildInfo.getBuildstamp().length() - 10 })
            + " " + "\u2551");
    infoPlate = infoPlate.concat("\n");
    infoPlate = infoPlate
            .concat("\u255A" + times.apply(new Object[] { "\u2550", longest + 2 }) + "\u255D");
    return infoPlate;
  }

  /**
   * Get build info read from jar file.
   * 
   * @return Formatted strings with build info and version.
   */
  public ToolVersionStruct getQuimPBuildInfo() {
    String[] ret = new String[3];
    try {
      Enumeration<URL> resources = getClass().getClassLoader().getResources("META-INF/MANIFEST.MF");
      // get internal name - jar name
      String iname = "not_found";
      try {
        iname = new PropertyReader().readProperty(propertyFileName, "internalName");
      } catch (Exception e) {
        LOGGER.debug("Property not found: " + e.getMessage());
      }
      while (resources.hasMoreElements()) {
        URL reselement = resources.nextElement();
        if (!reselement.toString().contains("/" + iname)) {
          continue;
        }
        Manifest manifest = new Manifest(reselement.openStream());
        Attributes attributes = manifest.getMainAttributes();
        try {
          String date = attributes.getValue("Implementation-Date");

          ret[1] = attributes.getValue("Built-By") + " on: " + implementationDateConverter(date);
          ret[0] = attributes.getValue("Implementation-Version");
          ret[2] = attributes.getValue("Implementation-Title");
          LOGGER.trace(Arrays.toString(ret));
        } catch (Exception e) {
          ; // do not care about problems - just use defaults defined on beginning
        }
      }
    } catch (IOException e) {
      ; // do not care about problems - just use defaults defined on beginning
    }
    // replace possible nulls with default text
    ret[0] = ret[0] == null ? defNote : ret[0];
    ret[1] = ret[1] == null ? defNote : ret[1];
    ret[2] = ret[2] == null ? defNote : ret[2];
    // prepare output
    ToolVersionStruct retmap = new ToolVersionStruct(ret[0], ret[1], ret[2]);
    return retmap;
  }

  /**
   * Reformat date from jar (put there by Maven).
   * 
   * @param dateString string in format "2017-02-24T08:55:44+0000"
   * @return String in format "yyyy-MM-dd hh:mm:ss"
   */
  public static String implementationDateConverter(String dateString) {
    DateTimeFormatter timeFormatter = DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ssZ");
    TemporalAccessor accessor = timeFormatter.parse(dateString);
    Date date = Date.from(Instant.from(accessor));
    SimpleDateFormat dt = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
    return dt.format(date);
  }

  /**
   * Date as string.
   *
   * @return the string
   */
  public static String dateAsString() {
    SimpleDateFormat formatter = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
    Date date = new Date();
    return formatter.format(date);
  }

  /**
   * Insert \n character after given number of chars trying to not break words.
   * 
   * @param in Input string
   * @param len line length
   * @return Wrapped string
   * @see <a href=
   *      "link">http://stackoverflow.com/questions/8314566/splitting-a-string-on-to-several-different-lines-in-java</a>
   */
  public static String stringWrap(String in, int len) {
    return stringWrap(in, len, "\n");
  }

  /**
   * Insert \n character after default number of chars trying to not break words.
   * 
   * @param in Input string
   * @return Wrapped string
   * @see <a href=
   *      "link">http://stackoverflow.com/questions/8314566/splitting-a-string-on-to-several-different-lines-in-java</a>
   */
  public static String stringWrap(String in) {
    return stringWrap(in, LINE_WRAP, "\n");
  }

  /**
   * Insert any symbol after given number of chars trying to not break words.
   * 
   * <p>It preserve position of new line symbol if present in text.
   * 
   * @param in Input string
   * @param len line length
   * @param brek symbol to insert on line break
   * @return Wrapped string
   * @see <a href=
   *      "link">http://stackoverflow.com/questions/8314566/splitting-a-string-on-to-several-different-lines-in-java</a>
   */
  public static String stringWrap(String in, int len, String brek) {
    String[] sp = in.split("\n");
    String str = "";
    for (String s : sp) {
      s = s.concat("\n");
      str = str.concat(WordUtils.wrap(s, len, brek, false, "( |/|\\\\)"));
    }
    str = str.trim();
    // String str = WordUtils.wrap(in, len, brek, false, "( |/|\\\\)");
    return str;
  }

}
