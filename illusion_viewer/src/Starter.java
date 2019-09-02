import communication.Constants;
import communication.MyLog;

import javax.swing.*;

public class Starter {
    static JFrame frame;
    static String folderName = "/Users/lana/Desktop/prgm/CSL/prednet_chainer_2/generator/images/";

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
                frame.setContentPane(new Generator(Constants.BENHAM_CLASSIC, frame, folderName));
                frame.pack();
                frame.setVisible(true);
            }
        });
    }

}
