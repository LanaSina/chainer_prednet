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

	public final static int BENHAM_CLASSIC = 0;
	public final static int BENHAM_VAR = 1;
	public final static int BENHAM_RED = 2;
	public final static int BENHAM_WEIRD= 3;
	public final static int SIMPLE_SHAPES_0= 4;
	public final static int SIMPLE_SHAPES_1= 5;
	public final static int THINNER_LINE= 6;
	public final static int THICKER_LINE= 7;
	public final static int DRAW_X_0= 8;
	public final static int DRAW_X_1= 9;
	public final static int SNAKES_0= 10;
	public final static int SNAKES_1= 11;
	public final static int BAD_SNAKES= 12;
	public final static int SNAKES_BW= 13;

//	/** write data or not*/
//	public static final boolean save = true;
//	/** draw network graph or not*/
//	public static final boolean draw_net = true;
//    public static boolean displayPredictions = false;
//
//
//	//types of experiments
//	public static final int VISION_EXPERIMENT = 0;
//	public static final int SYNECO_EXPERIMENT = 1;
//	public static final int FOOD_EXPERIMENT = 2;
//	public static final int EVEN_EXPERIMENT = 3;
//	public static final int EXOPLANET_EXPERIMENT = 4;
//	public static final int INPUT_TRANSITION_EVEN_EXPERIMENT = 5;
//	public static final int INPUT_TRANSITION_EXOPLANET_EXPERIMENT = 6;

}
