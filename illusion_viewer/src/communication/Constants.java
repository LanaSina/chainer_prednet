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
