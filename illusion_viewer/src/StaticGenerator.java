import communication.Constants;
import communication.MyLog;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class StaticGenerator  extends JPanel {
    static MyLog mlog = new MyLog("StaticGenerator", true);

    // Container box's width and height
    private static final int BOX_WIDTH = 160;
    private static final int BOX_HEIGHT = 120;
    static String nameFormat = "%04d";
    static int step = 0;
    final static String output_path = "/Users/lana/Desktop/prgm/CSL/prednet_chainer_2/generator/static_images/";


    public static void main(String[] args) {

        mlog.say("Launch");

        //create directory
        File theDir = new File(output_path);
        // if the directory does not exist, create it
        if (!theDir.exists()) {
            mlog.say("creating directory: " + output_path);
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

        MyPanel panel = new MyPanel();

    }


    private static class MyPanel  extends JPanel {
        JFrame frame;
        private BufferedImage paintImage;


        MyPanel() {

            frame =new JFrame("TheFrame");
            frame.add(this);
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.setSize(BOX_WIDTH,BOX_HEIGHT);
            frame.setVisible(true);
            setPreferredSize(new Dimension(BOX_WIDTH, BOX_HEIGHT));
            paintImage = new BufferedImage(BOX_WIDTH, BOX_HEIGHT, BufferedImage.TYPE_3BYTE_BGR);



            // Start the motion
            Thread thread = new Thread() {
                public void run() {

                    while (true) {
                        try {
                            Thread.sleep(500);
                        } catch (InterruptedException ex) {
                        }


                        repaint();//*/
                        if(step<10) {
                            screenshot();
                            step++;
                        }
                    }
                }
            };
           thread.start();
        }


        private void screenshot() {
            String name = String.format(nameFormat, step) + ".png";
            mlog.say(name);
            try {
                ImageIO.write(paintImage, "PNG", new File(output_path + name));
            } catch (IOException e) {
                e.printStackTrace();
            }
        }


        private void contrast(Graphics g, int bg_color, int main_color) {

            Color c = new Color(bg_color, bg_color, bg_color);
            g.setColor(c);
            g.fillRect(0, 0, BOX_WIDTH, BOX_HEIGHT);

            if (step >= 5){
                c = new Color(main_color, main_color, main_color);
                g.setColor(c);
                g.fillRect(BOX_WIDTH / 2 - 50, BOX_HEIGHT / 2 - 10, 100, 20);
            }
        }


        /**
         * 20 - 27 : red  (c = 93-34)
         * 27 - 29 : ? (c = 34-17)
         * 29 - 35 : blue (c = 17-33)
         * 35 - 43 : red (c = 33-101)
         * @param g
         * @param bg_color
         */
        private void lines(Graphics g, int bg_color) {

            Color c = new Color(bg_color, bg_color, bg_color);
            g.setColor(c);
            g.fillRect(0, 0, BOX_WIDTH, BOX_HEIGHT);

            if (step >= 5){
                int y = BOX_HEIGHT/5;
                int x = BOX_WIDTH/3;

                int step = 20;
                //for max blue
                int main_color = 255-step;
                int o = 0;
                for (int i = -step; i<step*2; i=i+step){
                    int mc = main_color + i;
                    c = new Color(mc, mc, mc);
                    g.setColor(c);
                    g.drawLine(x, y+o, x*2, y+o);
                    o++;
                }
//                c = new Color(main_color, main_color, main_color);
//                g.setColor(c);
//                g.drawLine(x, y, x*2, y);

                // for max red
                main_color = 70 - step;
                o = 0;
                for (int i = -step; i<step*2; i=i+step){
                    int mc = main_color + i;
                    c = new Color(mc, mc, mc);
                    g.setColor(c);
                    g.drawLine(x, y*2+o, x*2, y*2+o);
                    o++;
                }

                // for weird color
                main_color = step;
                o = 0;
                for (int i = -step; i<step*2; i=i+step){
                    int mc = main_color + i;
                    c = new Color(mc, mc, mc);
                    g.setColor(c);
                    g.drawLine(x, y*3+o, x*2, y*3+o);
                    o++;
                }
            }
        }



        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            g.drawImage(paintImage, 0, 0, null);

            if (paintImage != null){
                Graphics g2 = paintImage.createGraphics();
//                contrast(g2, 180, 255);
                lines(g2, 180);
            }

            g.dispose();
            // repaint panel with new modified paint
            repaint();
        }
    }
}
