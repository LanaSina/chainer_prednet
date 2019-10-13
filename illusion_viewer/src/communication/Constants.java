package communication;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Random;

/**
 * Global variables
 * @author lana
 *
 */
public class Constants {
	private static boolean shouldLog = true;
	
	/** controls output of all loggers. Set true for verbose mode, false for dry mode.*/
	public static final boolean getShouldLog(){
		return shouldLog;
	}

	public static final boolean setShouldLog(boolean b){
		shouldLog = b;
		return b;
	}

	public final static String IMAGE_OUTPUT_PATH = "/Users/lana/Desktop/prgm/CSL/prednet_chainer_2/generator/images/";
	// public final static String IMAGE_INPUT_PATH = "/Users/lana/Desktop/prgm/CSL/prednet_chainer_2/datasets/";
	public final static String IMAGE_INPUT_PATH = "example_images/";

	/** set to false to save prednet-sized images */
	public final static boolean BIG_SCALE = true;

	public final static int BENHAM_CLASSIC = 0;
	public final static int BENHAM_VAR = 1;
	public final static int BENHAM_RED = 2;
	public final static int BENHAM_DOTS = 3;
	public final static int SIMPLE_SHAPES_0 = 4;
	public final static int SIMPLE_SHAPES_1 = 5;
	public final static int THINNER_LINE = 6;
	public final static int THICKER_LINE = 7;
	public final static int DRAW_X_0 = 8;
	public final static int DRAW_X_1 = 9;
	public final static int SNAKES_0 = 10;
	public final static int SNAKES_1 = 11;
	// public final static int BAD_SNAKES = 12;
	public final static int SNAKES_BW = 13;
	public final static int DRAW_X_PHASE_0 = 14;
	public final static int DRAW_X_PHASE_1 = 15;
	public final static int DRAW_PLUS = 16;
	public final static int CONCENTRIC_0 = 17;
	public final static int CONCENTRIC_1 = 18;
	public final static int FRASER = 19;
	public final static int ROTATING_X_1 = 20;
	public final static int NEW_BENHAM = 21;
	public final static int NO_MASK = 22;
	public final static int CONCURRENT = 23;
	public final static int TIME_WARP = 24;
	public final static int STATIC = 25;
	public final static int CUSTOM_IMAGE = 26;
	public final static int TRAIN = 27;
	public final static int GENERALIZED_BENHAM = 28;
	public final static int CONTRAST = 29;
	public final static int FIRE = 30;
	public final static int ARROW = 31;
	public final static int COLOR_BANDS = 32;
	public final static int DARK_BENHAM = 33;
	public final static int BENHAM_IMAGE = 34;
}
