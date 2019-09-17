import communication.Constants;
import communication.MyLog;

import javax.swing.*;

public class Starter {
    static JFrame frame;
    static String folderName = "/Users/lana/Desktop/prgm/CSL/prednet_chainer_2/generator/images/";
    static  boolean save = false;

//    static int type = Constants.BENHAM_CLASSIC;
//    static int type = Constants.BENHAM_RED;
//    static int type = Constants.SIMPLE_SHAPES_0;
//    static int type = Constants.DRAW_X_0;
//    static int type = Constants.DRAW_X_1;
//    static int type = Constants.ROTATING_X_1;
//    static int type = Constants.DRAW_X_PHASE_0;

//    static int type = Constants.SNAKES_0;
//    static int type = Constants.BAD_SNAKES;
//    static int type = Constants.SNAKES_BW;
//    static int type = Constants.FRASER;

//    static int type = Constants.CONCENTRIC_0;
//    static int type = Constants.BENHAM_WEIRD;
//    static int type = Constants.NO_MASK;
//    static int type = Constants.CONCURRENT;
//    static int type = Constants.TIME_WARP;

    static int type = Constants.STATIC;



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
                frame.setContentPane(new Generator(type, frame, folderName, save));
                frame.pack();
                frame.setVisible(true);
            }
        });
    }

}
