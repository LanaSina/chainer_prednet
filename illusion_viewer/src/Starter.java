import communication.Constants;
import communication.MyLog;

import javax.swing.*;

public class Starter {
    static JFrame frame;

    /** set to true to save image frames*/
    static  boolean save = false;

//    static int type = Constants.BENHAM_CLASSIC;
//    static int type = Constants.BENHAM_RED;
//    static int type = Constants.SIMPLE_SHAPES_0;
//    static int type = Constants.DRAW_X_0;
//    static int type = Constants.DRAW_X_1;
//    static int type = Constants.ROTATING_X_1;
//    static int type = Constants.DRAW_X_PHASE_0;
//    static int type = Constants.THICKER_LINE;

//    static int type = Constants.SNAKES_0;
//    static int type = Constants.SNAKES_1;
//    static int type = Constants.SNAKES_BW;
//    static int type = Constants.FRASER;

//    static int type = Constants.CONCENTRIC_0;
//    static int type = Constants.BENHAM_DOTS;
//    static int type = Constants.NO_MASK;
//    static int type = Constants.CONCURRENT;
//    static int type = Constants.TIME_WARP;

//    static int type = Constants.STATIC;
//    static int type = Constants.TRAIN;
//    static int type = Constants.GENERALIZED_BENHAM;
//    static int type = Constants.CONTRAST;
//    static int type = Constants.FIRE;
//    static int type = Constants.ARROW;
//    static int type = Constants.COLOR_BANDS;

//    static int type = Constants.DARK_BENHAM;

    static int type = Constants.BENHAM_DOTS;
//    static int type = Constants.BENHAM_IMAGE;




    /** if you want to input your own image*/
//    static int type = Constants.CUSTOM_IMAGE;
    /** this will be ignored for other types */
    static String customImageName = "kitaoka_tuto/kitaoka_tuto.png";


    public static void main(String[] args) {
        MyLog myLog = new MyLog("Starter", true);

        myLog.say("Launch");

        //main thread

        // Run GUI in the Event Dispatcher Thread (EDT) instead of main thread.
        SwingUtilities.invokeLater(new Runnable() {
            public void run() {

                // Set up main window (using Swing's Jframe)
                frame = new JFrame("");
                frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
                Generator generator = new Generator(type, frame, save);

                int phase = 1;

                String customImage = "";
                switch (type) {
                    case Constants.SNAKES_0: {
                        customImage = "snakes_clean.png";
                        phase = 0;
                        break;
                    }
                    case Constants.SNAKES_1: {
                        customImage = "snakes_clean.png";
                        phase = 0;
                        break;
                    }
                    case Constants.SNAKES_BW: {
                        customImage = "snakes_2_bw.jpg";
                        phase = 0;
                        break;
                    }
                    case Constants.FRASER: {
                        customImage = "fraser.png";
                        phase = 0;
                        break;
                    }
                    case Constants.TRAIN: {
                        customImage = "night_train.png";
                        phase = 0;
                        break;
                    }
                    case Constants.CUSTOM_IMAGE: {
                        customImage = customImageName;
                        phase = 0;
                        break;
                    }
                }

                generator.setImage(customImage, phase);

                frame.setContentPane(generator);
                frame.pack();
                frame.setVisible(true);
            }
        });
    }

}
