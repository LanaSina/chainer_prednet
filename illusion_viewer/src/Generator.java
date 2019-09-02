import communication.MyLog;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import static communication.Constants.BENHAM_CLASSIC;

public class Generator extends JPanel {
    MyLog mlog = new MyLog("Generator", true);

    private int type;
    private int UPDATE_RATE = 50;

    // Container box's width and height
    private static final int BOX_WIDTH = 300;
    private static final int BOX_HEIGHT = 300;

    private int step = 0;
    private boolean save = false;
    boolean readyForSave = false;
    boolean saved = true;

    static JFrame frame;

    String nameFormat = "%02d";
    String folderName;


    public Generator(int type, JFrame frame, String folderName){
        this.type = type;
        this.frame = frame;
        this.folderName = folderName;

        this.setPreferredSize(new Dimension(BOX_WIDTH, BOX_HEIGHT));

        if (save) {
            //create directory
            File theDir = new File(folderName);
            // if the directory does not exist, create it
            if (!theDir.exists()) {
                mlog.say("creating directory: " + folderName);
                boolean result = false;
                try {
                    theDir.mkdirs();
                    result = true;
                } catch (SecurityException se) {
                }
                if (result) {
                    System.out.println("DIR created");
                }
            }
        }

        // Start the motion
        Thread gameThread = new Thread() {
            public void run() {

                while (true) {

                    //mlog.say("timestep " + step);

                    // Refresh the display
                    readyForSave = false;
                    repaint();

                    while(!readyForSave) {
                        try {
                            Thread.sleep(10);  // milliseconds
                        } catch (InterruptedException ex) {
                        }
                    }

                    if (save) {
                        saved = false;
                        screenshot();
                    }


                    step++;
                    try {
                        Thread.sleep(UPDATE_RATE);//1000 / UPDATE_RATE);  // milliseconds
                    } catch (InterruptedException ex) {
                    }

                    while(!saved) {
                        try {
                            Thread.sleep(10);  // milliseconds
                        } catch (InterruptedException ex) {
                        }
                    }
                }
            }
        };
        gameThread.start();
    }

    private void screenshot() {
        Container c = frame.getContentPane();
        BufferedImage image = new BufferedImage(BOX_WIDTH, BOX_HEIGHT, BufferedImage.TYPE_INT_RGB);
        c.paint(image.getGraphics());
        String name = String.format(nameFormat, step) + ".jpg";
        try {
            ImageIO.write(image, "PNG", new File(folderName + name));
        } catch (IOException e) {
            e.printStackTrace();
        }
        saved = true;
    }


    int phase = 0;
    int cirles = 3;
    int center_x = BOX_WIDTH/2;
    int center_y = BOX_HEIGHT/2;
    int separation = 10; //pixels
    int basic_radius = 40;
    private void drawBenham(Graphics g){

        Graphics2D g2 = (Graphics2D) g;
        g2.setStroke(new BasicStroke(3));

        if(phase==0){
            //draw inner circles
            int r = basic_radius;
            for(int i = 0; i<cirles; i++) {
                g.drawOval(center_x-r, center_y-r, r*2, r*2);
                r = r + separation;
            }
        } else if(phase==1){
            //draw outer circles
            int r = basic_radius + 4*separation;
            for(int i = 0; i<cirles; i++) {
                g.drawOval(center_x-r, center_y-r, r*2, r*2);
                r = r + separation;
            }
        } else if(phase==2){
            //draw a big black square
            int r = basic_radius + 7*separation;
            int x = center_x - r;
            int y = center_y - r;
            g.fillRect(x, y, r*2, r*2);

        } else if(phase==3){
            //draw nothing

            phase = -1;
        }

        phase = phase+1;
    }


    /**
     *
     */
    private void drawShapes(Graphics g){
        switch (type){
            case BENHAM_CLASSIC: {
                drawBenham(g);
                break;
            }
        }

//                Graphics2D g2 = (Graphics2D) g;
//                g2.setStroke(new BasicStroke(3));
//                g.drawLine(BOX_WIDTH/2 , BOX_HEIGHT/2-size, BOX_WIDTH/2, BOX_HEIGHT/2+size);
//                g.drawLine(BOX_WIDTH/2-size, BOX_HEIGHT/2, BOX_WIDTH/2+size, BOX_HEIGHT/2);
//
//                //g.fillRect(BOX_WIDTH/2-size, BOX_HEIGHT/2-size, size*2, size*2);
//                //g.fillRect(0, 0, BOX_WIDTH, BOX_HEIGHT);

    }


    /**
     * Custom rendering codes for drawing the JPanel
     */
    @Override
    public void paintComponent(Graphics g) {
        super.paintComponent(g);

        // Paint background
        g.setColor(Color.white);
        g.fillRect(0, 0, BOX_WIDTH, BOX_HEIGHT);

        //draw pattern
        g.setColor(Color.black);

        drawShapes(g);

        readyForSave = true;
    }
}
