import communication.Constants;
import communication.MyLog;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.geom.AffineTransform;
import java.awt.geom.Point2D;
import java.awt.geom.QuadCurve2D;
import java.awt.geom.Rectangle2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;


public class Generator extends JPanel {
    MyLog mlog = new MyLog("Generator", true);

    private BufferedImage paintImage;

    private int type;
    /** flashing rate in milliseconds */
    private int UPDATE_RATE = 50;

    // Container box's width and height
    //private static final
    int BOX_WIDTH;
    //private static final
    int BOX_HEIGHT;

    private int step = 0;
    private boolean save;
    boolean readyForSave = false;
    boolean saved = true;

    JFrame frame;

    String nameFormat = "%04d";
    BufferedImage image = null;
    BufferedImage primaryImage = null;
    BufferedImage secondaryImage = null;
    BufferedImage thirdImage = null;
    boolean do_update = true;

    //graphical parameters
    int phase = 0;
    int cirles = 3;
    int center_x;
    int center_y;
    int separation;
    int basic_radius;
    int timing = 1;


    public Generator(int type, JFrame frame, boolean save){
        this.type = type;

        if(Constants.BIG_SCALE){
            BOX_WIDTH = 800;
            BOX_HEIGHT = 800;
            separation = 8;
            basic_radius = 150;
        } else {
            BOX_WIDTH = 160;
            BOX_HEIGHT = 120;
            separation = 4;
            basic_radius = 15;
        }

        center_x = BOX_WIDTH/2;
        center_y = BOX_HEIGHT/2;

        this.save = save;
        paintImage = new BufferedImage(BOX_WIDTH, BOX_HEIGHT, BufferedImage.TYPE_3BYTE_BGR);

        this.frame = new JFrame("TheFrame");
        this.frame.add(this);
        this.frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(BOX_WIDTH, BOX_HEIGHT);
        this.frame.setVisible(true);

        this.setPreferredSize(new Dimension(BOX_WIDTH, BOX_HEIGHT));

        if (save) {
            //create directory
            File theDir = new File(Constants.IMAGE_OUTPUT_PATH);
            // if the directory does not exist, create it
            if (!theDir.exists()) {
                mlog.say("creating directory: " + Constants.IMAGE_OUTPUT_PATH);
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

        try {
            primaryImage = resize(ImageIO.read(new File(Constants.IMAGE_INPUT_PATH, "rose.jpg")), BOX_HEIGHT);
//            String path = "/Users/lana/Desktop/prgm/CSL/prednet_chainer_2/datasets/kitti/image_00/data/";
//            path = path + String.format("%010d", step) + ".png";
//            primaryImage = resize(ImageIO.read(new File(path)), BOX_HEIGHT);
            secondaryImage = resize(ImageIO.read(new File(Constants.IMAGE_INPUT_PATH, "invert_zebra.jpg")), BOX_HEIGHT);
//            thirdImage = resize(ImageIO.read(new File(Constants.IMAGE_INPUT_PATH, "fire_big.png")), BOX_HEIGHT);
        } catch (IOException e) {
            e.printStackTrace();
        }

        if(type == Constants.CONTRAST){
            try {
                primaryImage  = resize(ImageIO.read(new File(Constants.IMAGE_INPUT_PATH, "rabbit.png")), BOX_HEIGHT);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        // Start the motion
        Thread gameThread = new Thread() {
            public void run() {

                while (true) {

                    if (save) {

                        do_update = true;
                        readyForSave = false;
                        repaint();//*/

                        screenshot();

                    } else {
                        do_update = true;
                        repaint();
                    }


                    step++;
                    try {
                        Thread.sleep(UPDATE_RATE);
                    } catch (InterruptedException ex) {
                    }


                }
            }
        };
        gameThread.start();
    }


    /**
     *
     * @param imageName the name of the image you want to display
     * @param timing 0: the image will be displayed after a white mask. 1: the image will be displayed after a black mask.
     */
    public void setImage(String imageName, int timing){
        if(imageName.length()>0) {
            try {
                image = resize(ImageIO.read(new File(Constants.IMAGE_INPUT_PATH, imageName)), BOX_HEIGHT );
            } catch (IOException e) {
                e.printStackTrace();
            }
            this.timing = timing;
        }
    }

    /**
     * Takes a BufferedImage and resizes it according to the provided targetSize
     *
     * @param src the source BufferedImage
     * @param targetSize maximum height (if portrait) or width (if landscape)
     * @return a resized version of the provided BufferedImage
     */
    private BufferedImage resize(BufferedImage src, int targetSize) {
        if (targetSize <= 0) {
            return src; //this can't be resized
        }
        int targetWidth = targetSize;
        int targetHeight = targetSize;
        float ratio = ((float) src.getHeight() / (float) src.getWidth());
        if (ratio <= 1) { //square or landscape-oriented image
            targetHeight = (int) Math.ceil((float) targetWidth * ratio);
        } else { //portrait image
            targetWidth = Math.round((float) targetHeight / ratio);
        }
        BufferedImage bi = new BufferedImage(targetWidth, targetHeight, src.getTransparency() == Transparency.OPAQUE ? BufferedImage.TYPE_INT_RGB : BufferedImage.TYPE_INT_ARGB);
        Graphics2D g2d = bi.createGraphics();
        g2d.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR); //produces a balanced resizing (fast and decent quality)
        g2d.drawImage(src, 0, 0, targetWidth, targetHeight, null);
        g2d.dispose();
        return bi;
    }

    private void screenshot() {

        while (do_update){
            try {
                Thread.sleep(10);
            } catch (InterruptedException ex) {
            }
        }

        String name = String.format(nameFormat, step) + ".png";
        mlog.say(name);
        try {
            ImageIO.write(paintImage, "PNG", new File(Constants.IMAGE_OUTPUT_PATH + name));
        } catch (IOException e) {
            e.printStackTrace();
        }

        saved = true;
    }

    private void drawBenham_var(Graphics g){

        Graphics2D g2 = (Graphics2D) g;
        g2.setStroke(new BasicStroke(1));
        g.setColor(Color.black);

        if(phase==0){
            //draw inner circles
            int r = basic_radius;
            for(int i = 0; i<cirles; i++) {
                g.drawOval(center_x-r, center_y-r, r*2, r*2);
                r = r + separation;
            }
        } else if(phase==1){
            //draw outer circles
            int r = basic_radius + 3*separation;
            for(int i = 0; i<cirles; i++) {
                g.drawOval(center_x-r, center_y-r, r*2, r*2);
                r = r + separation;
            }
        } else if(phase==2){
            //draw a big black square
            g.setColor(Color.red);
            int r = basic_radius + 6*separation;
            int x = center_x - r;
            int y = center_y - r;
            g.fillRect(x, y, r*2, r*2);

            phase = -1;
//        } else if(phase==3){
//            //draw a big black square
//            g.setColor(Color.red);
//            int r = basic_radius + 6*separation;
//            int x = center_x - r;
//            int y = center_y - r;
//            g.fillRect(x, y, r*2, r*2);
//
//            phase = -1;
        }

        //draw mask
        g.setColor(Color.white);
        //draw left part
        int r = basic_radius + 7*separation;
        int x = center_x - r;
        int y = center_y - r;
        g.fillRect(0, 0, center_x, BOX_HEIGHT);
        //draw bottom right part
        g.fillRect(center_x, center_y, center_x, center_y);


        phase = phase+1;
    }

    int secondary_phase = 0;
    private void simple_shape(Graphics g, int timing){

        Graphics2D g2 = (Graphics2D) g;
        g2.setStroke(new BasicStroke(1));
        g.setColor(Color.black);
        int p_separation = 4;

        if(phase==timing){
            int r =  basic_radius;
            for(int i = 0; i<5; i++) {
                g.drawOval(center_x-r, center_y-r, r*2, r*2);
                r = r + p_separation;
            }

            //check
            g.setColor(Color.black);
            g.drawLine(0, center_y+50, BOX_WIDTH, center_y+50);

        } else if(phase==2){
            //draw a big black square
            g.setColor(Color.black);
            int r = basic_radius + 6*separation;

            int x = center_x - r;
            int y = center_y - r;
//            g.fillRect(x, y, r, r*2);
            g.fillRect(center_x, y, r, r*2);
//
//            //same width
//            g.setColor(Color.white);
//            r =  basic_radius;
//            for(int i = 0; i<5; i++) {
//                g.drawOval(center_x-r, center_y-r, r*2, r*2);
//                r = r + p_separation;
//            }
//
//            r = basic_radius + 6*separation;
//            int x = center_x - r;
//            int y = center_y - r;
//            g.setColor(Color.white);
//            g.fillRect(center_x, y, r, r*2);

            phase = -1;

        }


        //check
        g.setColor(Color.black);
        g.drawLine(0, center_y, BOX_WIDTH, center_y);

        phase = phase+1;
    }

    //blue black green white
    private void concentric_alt(Graphics g, int timing){

        Graphics2D g2 = (Graphics2D) g;
        int p_separation = 10;
        g2.setStroke(new BasicStroke(p_separation));
        g.setColor(Color.black);
        //int p_separation = 4;
        int r =  basic_radius/2;

        if(phase==timing){

            int angle = 0;
            int sep_angle = 10;
            int x = center_x - 100;
            int y = center_y - p_separation/2;
            int h = 30;

            for (int i =0; i<5 ; i++) {
                g.setColor(Color.blue);
                g.fillRect(x, y, p_separation/2, h);
                x+=p_separation/2;

//                g.setColor(Color.black);
//                g.fillRect(x, y, p_separation/2, h);
                x+=p_separation/2;

                g.setColor(Color.green);
                g.fillRect(x, y, p_separation, h);
                x+=p_separation;

                //white
                x+=p_separation/2;
            }

        } else if(phase==2){
            //draw a big black square
            g.setColor(Color.black);
            r = basic_radius + 6*separation;
            int x = center_x - r;
            int y = center_y - r;
            g.fillRect(x, y, r*2, r*2);
            secondary_phase = secondary_phase + 30;
            phase = -1;
        } else {
            int x = center_x - 100;
            int y = center_y - p_separation/2;
            int h = 30;

            for (int i =0; i<5 ; i++) {
//                g.setColor(Color.blue);
//                g.fillRect(x, y, p_separation, h);
                x+=p_separation/2;

                g.setColor(Color.black);
                g.fillRect(x, y, p_separation/2, h);
                x+=p_separation/2;

//                g.setColor(Color.green);
//                g.fillRect(x, y, p_separation, h);
                x+=p_separation;

                //white
                x+=p_separation/2;
            }
        }

        phase = phase+1;
    }

    //blue black green white
    private void concentric(Graphics g, int timing){

        Graphics2D g2 = (Graphics2D) g;
        int p_separation = 4;
        g2.setStroke(new BasicStroke(p_separation));
        g.setColor(Color.black);
        //int p_separation = 4;
        int r =  basic_radius/2;

        if(phase==timing){

            int angle = 0;
            int sep_angle = 10;
            int x = center_x - 100;
            int y = center_y - p_separation/2;
            int h = 30;

            for (int i =0; i<3 ; i++) {
                g.setColor(Color.blue);
                g.fillRect(x, y, p_separation/2, h);
                x+=p_separation/2;

                g.setColor(Color.black);
//                g.fillRect(x, y, p_separation/2, h);
                x+=p_separation/2;

//                g.setColor(Color.green);
//                g.fillRect(x, y, p_separation, h);
                x+=p_separation/2;

                //white
                x+=p_separation/2;
            }

            for (int i =3; i<6 ; i++) {
                g.setColor(Color.green);
//                g.fillRect(x, y, p_separation, h);
                x+=p_separation/2;

                g.setColor(Color.black);
                g.fillRect(x, y, p_separation/2, h);
                x+=p_separation/2;

                g.setColor(Color.blue);
//                g.fillRect(x, y, p_separation, h);
                x+=p_separation/2;

                //white
                x+=p_separation/2;
            }

        } else if(phase==2){
            //draw a big black square
            g.setColor(Color.black);
            r = basic_radius + 6*separation;
            int x = center_x - r;
            int y = center_y - r;
            g.fillRect(x, y, r*2, r*2);
            secondary_phase = secondary_phase + 30;
            phase = -1;
        } else {
            //phase = -1;
            int x = center_x - 100;
            int y = center_y - p_separation/2;
            int h = 30;

            for (int i =0; i<3 ; i++) {
                g.setColor(Color.blue);
//                g.fillRect(x, y, p_separation, h);
                x+=p_separation/2;

                g.setColor(Color.black);
                g.fillRect(x, y, p_separation/2, h);
                x+=p_separation/2;

                g.setColor(Color.green);
//                g.fillRect(x, y, p_separation, h);
                x+=p_separation/2;

                //white
                x+=p_separation/2;
            }

            for (int i=3; i<6 ; i++) {
                g.setColor(Color.blue);
                g.fillRect(x, y, p_separation/2, h);
                x+=p_separation/2;

                g.setColor(Color.black);
//                g.fillRect(x, y, p_separation/2, h);
                x+=p_separation/2;

//                g.setColor(Color.green);
//                g.fillRect(x, y, p_separation, h);
                x+=p_separation/2;

                //white
                x+=p_separation/2;
            }
        }

        phase = phase+1;
    }



    private void smooth_concurrent(Graphics g) {

        int p_separation = 8;
        //g2.setStroke(new BasicStroke(p_separation));
        g.setColor(Color.black);

        mlog.say("repaint " + phase);
        int x = center_x;
        int y = center_y;
        int h = 60;

        if (phase == 0) {
            int r = p_separation;

            //big  line
            for (int i = 0; i < 4; i++){

                g.setColor(Color.black);
                g.fillRect(x, y, r, h);
                x += r*2;
            }
//            g.setColor(Color.black);
//            g.fillRect(x, y, r, h);

        } else if(phase==1){
            int r = p_separation;

//            for (int i = 0; i < 10; i++){
//                g.setColor(Color.blue);
//                g.fillRect(x, y, r, h);
//                x += r * 2;
//            }

//            g.setColor(Color.black);
//            g.fillRect(x, y, r, h);

//            x += p_separation;
//            g.setColor(Color.black);
//            g.fillRect(x, y, r, h);

        } else if (phase == 2) {
            int r = p_separation;
            x += r;

            for (int i = 0; i < 4; i++) {

                g.setColor(Color.black);
                g.fillRect(x, y, r, h);
                x += r * 2;
            }

//            g.setColor(Color.black);
//            int r = basic_radius + 6 * separation;
//            x = center_x - r;
//            y = center_y - r;
//            g.fillRect(x, y, r * 2, r * 2);

//            phase = -1;
        } else if (phase == 3) {

            g.setColor(Color.black);
            int r = basic_radius + 6 * separation;
            x = center_x - r;
            y = center_y - r;
            g.fillRect(x, y, r * 2, r * 2);
//
            phase = -1;
        }

        phase = phase+1;
    }

    private void time_warp(Graphics g){

        Graphics2D g2 = (Graphics2D) g;
        int p_separation = 8;
        //g2.setStroke(new BasicStroke(p_separation));
        g.setColor(Color.black);

        mlog.say("repaint " + phase);
        int x = center_x;
        int y = center_y;
        int h = 60;

        if(phase==0){
            UPDATE_RATE = 200;

            int r = p_separation;

            //big  line
            g.setColor(Color.black);
            g.fillRect(x, y, r, h);

        } else if(phase==1){
            UPDATE_RATE = 40; //next one
            int r = p_separation;

            x += p_separation;
            g.setColor(Color.black);
            g.fillRect(x, y, r, h);

//        } else if (phase == 2){
//
//            g.setColor(Color.black);
//            g.fillRect(x, y, p_separation, h);

//            g.setColor(Color.black);
//            int r = basic_radius + 6*separation;
//            x = center_x - r;
//            y = center_y - r;
//            g.fillRect(x, y, r*2, r*2);

            phase = -1;
        }

        phase = phase+1;
    }


    private void smooth_growing(Graphics g){

        Graphics2D g2 = (Graphics2D) g;
        int p_separation = 12;
        //g2.setStroke(new BasicStroke(p_separation));
        g.setColor(Color.black);

        int x = center_x;
        int y = center_y;
        int h = 30;

        if(phase==0){



        } else if(phase==1){

            //thin black line
            int r = 2;//p_separation/2;

            g.setColor(Color.red);
            g.fillRect(x-r/2, y, r, h);

        } else if (phase == 2){

            //thin red line
//            int r = p_separation/2;
//            g.setColor(Color.red);
//            g.fillRect(x-r/2, y, r, h);

            //thin red lines
//            int r = 1;
//            g.setColor(Color.red);
//            g.fillRect(x - p_separation/4, y, r, h);
//            g.fillRect(x + p_separation/4 - r, y, r, h);

            int r = p_separation/2;

            //big black line

            g.setColor(Color.black);
            g.fillRect(x-r/2, y, r, h);
        } else {

            //thick  line
            int r = p_separation;
            g.setColor(Color.black);
            g.fillRect(x-r/2, y, r, h);
            phase = -1;
        }

        phase = phase+1;
    }

    private void no_mask(Graphics g, int timing){

        Graphics2D g2 = (Graphics2D) g;
        int p_separation = 12;
        g2.setStroke(new BasicStroke(p_separation));
        g.setColor(Color.black);

        mlog.say("repaint " + phase);

        if(phase==timing){

            int x = center_x - 100;
            int y = center_y - p_separation/2;
            int h = 30;

            for (int i =0; i<6 ; i++) {
                g.setColor(Color.black);
                g.fillRect(x, y, p_separation, h);
                x+=p_separation;

                //black
                x+=p_separation;

                //white
                x+=p_separation;
            }

        } else if(phase==1){
            //draw a big black square
            g.setColor(Color.black);
            int r = basic_radius + 6*separation;
            int x = center_x - r;
            int y = center_y - r;
            g.fillRect(x, y, r*2, r*2);
//            phase = -1;
        } else {

            int x = center_x - 100;
            int y = center_y - p_separation/2;
            int h = 30;

            for (int i =0; i<6 ; i++) {
                x+=p_separation;


                g.setColor(Color.black);
                g.fillRect(x, y, p_separation, h);
                x+=p_separation;

                //white
                x+=p_separation;
            }
            phase = -1;
        }

        phase = phase+1;

    }

    private void new_benham(Graphics g, int timing){

//        UPDATE_RATE = (int) (Math.cos(secondary_phase/10.0)*25 + 70);

        Graphics2D g2 = (Graphics2D) g;
        int p_separation = 12;
        g2.setStroke(new BasicStroke(p_separation));
        g.setColor(Color.black);

        if(phase==timing){

            int x = center_x - 100;
            int y = center_y - p_separation/2;
            int h = 30;

            for (int i =0; i<6 ; i++) {
                g.setColor(Color.black);
                g.fillRect(x, y, p_separation, h);
                x+=p_separation;

                g.setColor(Color.black);
//                g.fillRect(x, y, p_separation, h);
                x+=p_separation;
                //white
//                x+=p_separation;
            }

        } else if(phase==2){
            //draw a big black square
            g.setColor(Color.black);
            int r = basic_radius + 6*separation;
            int x = center_x - r;
            int y = center_y - r;
            g.fillRect(x, y, r*2, r*2);
            phase = -1;
            secondary_phase = secondary_phase+1;
        } else {
            int x = center_x - 100;
            int y = center_y - p_separation/2;
            int h = 30;

            for (int i =0; i<6 ; i++) {
                g.setColor(Color.black);
//                g.fillRect(x, y, p_separation, h);
                x+=p_separation;

                g.setColor(Color.black);
                g.fillRect(x, y, p_separation, h);
                x+=p_separation;
                //white
//                x+=p_separation;
            }
        }

        phase = phase+1;
    }

    private void draw_x(Graphics g, int timing){

        Graphics2D g2 = (Graphics2D) g;
        g2.setStroke(new BasicStroke(1));
        g.setColor(Color.black);

        if(phase==timing){
            int r =  basic_radius;
            g.drawLine(center_x-r, center_y-r, center_x+r, center_y+r);
            g.drawLine(center_x+r, center_y-r, center_x-r, center_y+r);
        } /*else if(phase==1){

        }*/ else if(phase==2){
            //draw a big black square
            g.setColor(Color.black);
            int r = basic_radius + 6*separation;
            int x = center_x - r;
            int y = center_y - r;
            g.fillRect(x, y, r*2, r*2);
            phase = -1;
        }

        phase = phase+1;
    }


    double rotating_speed = -Math.PI/48;
    double rad_angle = 0;

    private void draw_rotating_x(Graphics g, int timing){

        Graphics2D g2 = (Graphics2D) g;
        g2.setStroke(new BasicStroke(1));
        g.setColor(Color.black);

        if(phase==timing){
            int r =  basic_radius;

            int x1 = (int)(Math.cos(rad_angle)*r);
            int y1 = (int)(Math.sin(rad_angle)*r);
            int x2 = (int)(Math.cos(rad_angle + Math.PI)*r);
            int y2 = (int)(Math.sin(rad_angle + Math.PI)*r);

            g.drawLine(center_x+x1, center_y-y1, center_x+x2, center_y-y2);

            double inv_angle = rad_angle + Math.PI/2;
            x1 = (int)(Math.cos(inv_angle)*r);
            y1 = (int)(Math.sin(inv_angle)*r);
            x2 = (int)(Math.cos(inv_angle + Math.PI)*r);
            y2 = (int)(Math.sin(inv_angle + Math.PI)*r);

            g.drawLine(center_x+x1, center_y-y1, center_x+x2, center_y-y2);
            rad_angle = rad_angle + rotating_speed;
        } else if(phase==2){
            //draw a big black square
            g.setColor(Color.black);
            int r = basic_radius + 6*separation;
            int x = center_x - r;
            int y = center_y - r;
            g.fillRect(x, y, r*2, r*2);
            phase = -1;
        }

        phase = phase+1;
    }

    private void draw_plus(Graphics g, int timing){

        Graphics2D g2 = (Graphics2D) g;
        g2.setStroke(new BasicStroke(1));
        g.setColor(Color.black);

        if(phase==timing){
            int r =  basic_radius;
            g.drawLine(center_x, center_y-r, center_x, center_y+r);
            g.drawLine(center_x-r, center_y, center_x+r, center_y);
        } /*else if(phase==1){

        }*/ else if(phase==2){
            //draw a big black square
            g.setColor(Color.black);
            int r = basic_radius + 6*separation;
            int x = center_x - r;
            int y = center_y - r;
            g.fillRect(x, y, r*2, r*2);
            secondary_phase = secondary_phase + 30;
            phase = -1;
        }

        phase = phase+1;
    }

    private void draw_x_dephased(Graphics g, int timing){

        Graphics2D g2 = (Graphics2D) g;
        g2.setStroke(new BasicStroke(1));
        Color c = new Color(150, 200, 0);
        //c = Color.GREEN;
        g.setColor(Color.black);
        int r =  basic_radius;

        if(phase==timing){
            //g.drawLine(center_x-r, center_y-r, center_x+r, center_y+r);

            g.drawLine(100, center_y-r/6, BOX_WIDTH-100, center_y-r/6);
        } else if(phase==2){

            r = r/2;// + 6*separation;
            int x = 100; //center_x - r;
            int y = center_y - r;

            Color[] colors = {Color.black, Color.darkGray, Color.blue, Color.red};

            int width = 50;
            for (int i=0; i<colors.length; i++){
                g.setColor(colors[i]);
                int xx = i*2*width;
                g.fillRect(x + xx, y, width, r*2);
            }

            phase = -1;
        } else{
            g.drawLine(100, center_y+r/6, BOX_WIDTH-100, center_y+r/6);
        }

        phase = phase+1;
    }

    private void drawColorBands(Graphics g){

        Graphics2D g2 = (Graphics2D) g;
        //g2.setStroke(new BasicStroke(1));
        Color c = new Color(150, 200, 0);
        //c = Color.GREEN;
        g.setColor(Color.black);


        if(phase==0){
            //g.drawLine(center_x-r, center_y-r, center_x+r, center_y+r);

            g.drawLine(100, 100, BOX_WIDTH-100, 100);
        } else if(phase==2){
//            g.setColor(Color.black);
//
//            g.fillRect(0, 0, BOX_WIDTH, BOX_HEIGHT);

            int startX = 0, startY = 0, endX = BOX_WIDTH, endY = 0;

            Color startColor = Color.black;
            Color endColor = Color.white;

            GradientPaint gradient = new GradientPaint(startX, startY, startColor, endX, endY, endColor);
            g2.setPaint(gradient);

            g2.fill(new Rectangle2D.Double(0,0,BOX_WIDTH,BOX_HEIGHT));
            phase = -1;
        }

        phase = phase+1;
    }

    private void thicker_line(Graphics g){

        Graphics2D g2 = (Graphics2D) g;
        g2.setStroke(new BasicStroke(1));
        g.setColor(Color.black);

        if(phase==0){

        } else if(phase==1){
            g.fillRect(center_x+basic_radius/2, center_y-basic_radius, separation, separation);
            g.drawLine(center_x+basic_radius/2, center_y-basic_radius, center_x+5*basic_radius, center_y-basic_radius);
        } else if(phase==2){
            //draw a big black square
            g.setColor(Color.black);
            int r = basic_radius + 6*separation;
            int x = center_x - r;
            int y = center_y - r;
            g.fillRect(x, y, r*2, r*2);

            phase = -1;
        }

        //draw mask
        g.setColor(Color.white);
        //draw left part
        int r = basic_radius + 7*separation;
        int x = center_x - r;
        int y = center_y - r;
        g.fillRect(0, 0, center_x, BOX_HEIGHT);
        //draw bottom right part
        g.fillRect(center_x, center_y, center_x, center_y);


        phase = phase+1;
    }

    private void thinner_line(Graphics g){

        Graphics2D g2 = (Graphics2D) g;
        g2.setStroke(new BasicStroke(1));
        g.setColor(Color.black);

        if(phase==0){
            g.fillRect(center_x+basic_radius/2, center_y-basic_radius, separation, separation);
            g.drawLine(center_x+basic_radius/2, center_y-basic_radius, center_x+5*basic_radius, center_y-basic_radius);
        } else if(phase==2){
            //draw a big black square
            g.setColor(Color.black);
            int r = basic_radius + 6*separation;
            int x = center_x - r;
            int y = center_y - r;
            g.fillRect(x, y, r*2, r*2);

            phase = -1;
        }

        //draw mask
        g.setColor(Color.white);
        //draw left part
        int r = basic_radius + 7*separation;
        int x = center_x - r;
        int y = center_y - r;
        g.fillRect(0, 0, center_x, BOX_HEIGHT);
        //draw bottom right part
        g.fillRect(center_x, center_y, center_x, center_y);


        phase = phase+1;
    }

    private void drawContrast(Graphics g){

        Graphics2D g2 = (Graphics2D) g;
        g2.setStroke(new BasicStroke(1));

        int contrast;
        Color c;

//        //background
        c = new Color(130,50,50);
        g.setColor(c);
        g.fillRect(0, 0, BOX_WIDTH, BOX_HEIGHT);


        int size_ = 20;
        int offset = 200;

        int x = center_x - primaryImage.getWidth()/2;
        int y = center_y - primaryImage.getHeight()/2;
        g.drawImage(primaryImage, x, y, null);

        if(phase==0){
            contrast = 0;
            c = new Color(contrast,contrast,contrast);
            g.setColor(c);

            g.fillOval(center_x-offset, center_y-size_, size_*2, size_*2);
            g.fillRect(center_x-offset, center_y-size_, size_*2, size_*2);

            contrast = 255;
            c = new Color(contrast,contrast,contrast);
            g.setColor(c);
            g.fillRect(center_x+150, center_y-size_, size_*2, size_*2);
        } else if (phase == 1){
            contrast = 255;
            phase = -1;
            c = new Color(contrast,contrast,contrast);
            g.setColor(c);
//            g.fillOval(center_x-offset, center_y-size_, size_*2, size_*2);
            g.fillRect(center_x-offset, center_y-size_, size_*2, size_*2);

            contrast = 0;
            c = new Color(contrast,contrast,contrast);
            g.setColor(c);
            g.fillRect(center_x+150, center_y-size_, size_*2, size_*2);

        }

        size_ = 60;

        //nose
        if(secondary_phase==0){
            contrast = 255;
            c = new Color(contrast,contrast,contrast);
            g.setColor(c);
//            g.fillRect(center_x-35, center_y+80, size_, size_*2);
            int n=3;
            int[] xpoints = new int[n];
            x = center_x - size_ -1;
            xpoints[0] = x;
            xpoints[1] = x+size_;
            xpoints[2] = x+size_;

            int[] ypoints = new int[n];
            y = center_y + 100;
            ypoints[0] = y;
            ypoints[1] = y;
            ypoints[2] = y+30*2-20;

            g.fillPolygon(xpoints, ypoints, n);

            contrast = 0;
            c = new Color(contrast,contrast,contrast);
            g.setColor(c);
//            g.fillRect(center_x-35+size_, center_y+80, size_, size_*2);

            x = center_x-1;
            xpoints[0] = x;
            xpoints[1] = x+size_;
            xpoints[2] = x;

            ypoints[0] = y;
            ypoints[1] = y;
            ypoints[2] = y+30*2-20;
            g.fillPolygon(xpoints, ypoints, n);

        } else if (secondary_phase == 2){
            contrast = 0;
            c = new Color(contrast,contrast,contrast);
            g.setColor(c);
//            g.fillRect(center_x-35, center_y+80, size_*2, size_*2);
            int n=3;
            int[] xpoints = new int[n];
            x = center_x - size_ -1;
            xpoints[0] = x;
            xpoints[1] = x+size_;
            xpoints[2] = x+size_;

            int[] ypoints = new int[n];
            y = center_y + 100;
            ypoints[0] = y;
            ypoints[1] = y;
            ypoints[2] = y+30*2-20;

            g.fillPolygon(xpoints, ypoints, n);

            contrast = 255;
            c = new Color(contrast,contrast,contrast);
            g.setColor(c);
//            g.fillRect(center_x-35+size_, center_y+80, size_, size_*2);

            x = center_x-1;
            xpoints[0] = x;
            xpoints[1] = x+size_;
            xpoints[2] = x;

            ypoints[0] = y;
            ypoints[1] = y;
            ypoints[2] = y+30*2-20;
            g.fillPolygon(xpoints, ypoints, n);

            secondary_phase = -1;

        } else {

            //background
            contrast = 255;
            c = new Color(contrast,contrast,contrast);
            g.setColor(c);
//            g.fillRect(center_x-35, center_y+80, size_*2, size_*2);

            int n=3;
            int[] xpoints = new int[n];
            x = center_x - size_ -1;
            xpoints[0] = x;
            xpoints[1] = x+size_*2;
            xpoints[2] = center_x-1;
            int[] ypoints = new int[n];
            y = center_y + 100;
            ypoints[0] = y;
            ypoints[1] = y;
            ypoints[2] = y+30*2-20;
            g.fillPolygon(xpoints, ypoints, n);


            //draw nose lines
            contrast = 0;
            c = new Color(contrast,contrast,contrast);
            g.setColor(c);

            y = center_y+80+2;
            g2.setStroke(new BasicStroke(1));
//            g.drawLine(center_x-31, y, center_x-31 + size_*2, y);
//            g.drawLine(center_x-1, y, center_x-1, y + size_*2);

            x = center_x-31;
            y = y + 40;
//            g.drawLine(x -100, y, x + size_*2 + 100, y);
            // create new QuadCurve2D.Float
            QuadCurve2D q = new QuadCurve2D.Float();
            // draw QuadCurve2D.Float with set coordinates
            q.setCurve(x-120, y+20, x, y-50, x+170, y+20);
            g2.draw(q);
            q.setCurve(x+150, y+35, x, y-35, x-100, y+35);
            g2.draw(q);

        }

        phase = phase+1;
        secondary_phase = secondary_phase+1;
    }

    private void drawBenham_red(Graphics g){

        Graphics2D g2 = (Graphics2D) g;
        g2.setStroke(new BasicStroke(1));
        g.setColor(Color.blue);

        if(phase==0){
            //draw inner arcs
            int r = basic_radius;
            for(int i = 0; i<cirles; i++) {
                g.drawArc(center_x-r, center_y-r, r*2, r*2, 0, 45);
                r = r + separation;
            }
            //r = basic_radius + 4*separation;
            for(int i = 0; i<cirles; i++) {
                g.drawArc(center_x-r, center_y-r, r*2, r*2, 45, 90);
                r = r + separation;
            }
        } else if(phase==1){
            //draw outer circles
            int r = basic_radius + 6*separation;
            for(int i = 0; i<cirles; i++) {
                g.drawArc(center_x-r, center_y-r, r*2, r*2, 0, 45);
                r = r + separation;
            }
            for(int i = 0; i<cirles; i++) {
                g.drawArc(center_x-r, center_y-r, r*2, r*2, 45, 90);
                r = r + separation;
            }
        } else if(phase==2){
            //draw a big square
            g.setColor(Color.black);
            int r = basic_radius + 12*separation;
            int x = center_x - r;
            int y = center_y - r;
            g.fillRect(x, y, r*2, r*2);

            phase = -1;
        }/* else if(phase==3){
            //draw nothing

            phase = -1;
        }*/

        //draw mask
        g.setColor(Color.white);
        //draw left part
        int r = basic_radius + 7*separation;
        int x = center_x - r;
        int y = center_y - r;
        g.fillRect(0, 0, center_x, BOX_HEIGHT);
        //draw bottom right part
        g.fillRect(center_x, center_y, center_x, center_y);


        phase = phase+1;
    }

    private void drawDarkBenham(Graphics g){

        Color mainColor = Color.red;//Color.darkGray;
        Color lineColor = Color.blue;//Color.gray;

        Graphics2D g2 = (Graphics2D) g;
        g2.setStroke(new BasicStroke(1));
        g.setColor(mainColor);


        //background
        int r = basic_radius + 12*separation;
        int x = center_x;// - r;
        int y = center_y - r;
//        g.setColor(mainColor); //darkgray
//        g.fillRect(x, y, r, r*2);
//        g.setColor(Color.lightGray);
//        g.fillRect(center_x+r/2, y, r/2, r*2);

        g.setColor(mainColor);
        //background
        g.fillRect(center_x, y, r*2, r*2);

        if(phase==0){
            //draw inner arcs
            g.setColor(lineColor);

//            g.fillRect(center_x+10, center_y-100+10, 20, 20);
//            g.fillRect(center_x+50, center_y-100+10, 25, 25);


            r = basic_radius;
            for(int i = 0; i<cirles; i++) {
                g.drawArc(center_x-r, center_y-r, r*2, r*2, 0, 45);
                r = r + separation;
            }
            //r = basic_radius + 4*separation;
            for(int i = 0; i<cirles; i++) {
                g.drawArc(center_x-r, center_y-r, r*2, r*2, 45, 90);
                r = r + separation;
            }
        } else if(phase==1){
            //draw outer circles
            g.setColor(lineColor);

//            g.fillRect(center_x+10, center_y-100+10, 25, 25);

//            g.fillRect(center_x+50, center_y-100+10, 20, 20);


            r = basic_radius + 6*separation;
            for(int i = 0; i<cirles; i++) {
                g.drawArc(center_x-r, center_y-r, r*2, r*2, 0, 45);
                r = r + separation;
            }
            for(int i = 0; i<cirles; i++) {
                g.drawArc(center_x-r, center_y-r, r*2, r*2, 45, 90);
                r = r + separation;
            }
        } else if(phase==2){
            //draw a big square
            g.setColor(lineColor);
            r = basic_radius + 12*separation;
            x = center_x - r;
            y = center_y - r;
            g.fillRect(x, y, r*2, r*2);

            phase = -1;
        }

        //draw mask
        g.setColor(Color.white);
        //draw left part
        r = basic_radius + 7*separation;
        x = center_x - r;
        y = center_y - r;
        g.fillRect(0, 0, center_x, BOX_HEIGHT);
        //draw bottom right part
        g.fillRect(center_x, center_y, center_x, center_y);
        phase = phase+1;
    }

    private void drawBenham_weird(Graphics g){

        Graphics2D g2 = (Graphics2D) g;
        g2.setStroke(new BasicStroke(1));
        //darker
        Color c = Color.black; //new Color(0,80,0);
        g.setColor(c);//black

        if(phase==0){
            g.setColor(Color.black);
            for (int i = center_x-100; i<center_x+100; i = i+2 ){
                for (int j = center_y-50; j<center_y; j = j+2 ){
                    g.drawLine(i,j,i+1,j+1);
                }
            }

//            g.drawRect(center_x-100, center_y-20, 150, 2);
        } else if(phase==1){
            g.setColor(Color.black);
            //draw outer circles

            for (int i = center_x-100; i<center_x+100; i = i+2 ){
                for (int j = center_y+1; j<center_y+50; j = j+2 ){
                    g.drawLine(i,j,i+1,j+1);
                }
            }
//            g.drawRect(center_x-20, center_y, 150, 2);
        } else if(phase==2){

            g.setColor(Color.black);//black

            int r = 150;
            int x = center_x - r;
            int y = center_y - r;
            g.fillRect(x, y, r*2, r*2);

            phase = -1;
        }

        phase = phase+1;
    }


    private void drawBenhamImage(Graphics g){

        UPDATE_RATE = 40;

        Graphics2D g2 = (Graphics2D) g;
        g2.setStroke(new BasicStroke(1));
        g.setColor(Color.black);//black

        int w = primaryImage.getWidth();
        int h = primaryImage.getHeight();

        int phases = 4;
        int n_colors = phases - 1;


        int i_step = BOX_WIDTH/phases;
        int j_step = BOX_HEIGHT/phases;
        int xstart = phase*i_step;
        Color startColor = Color.black;
        Color endColor = Color.white;
        int gr_size = i_step/2;
        GradientPaint gradient;// = new GradientPaint(xstart, 0, endColor, xstart+gr_size, 0, startColor);

        for(int pseudo_y=0; pseudo_y<phases; pseudo_y++) {

            // pseudo_phase is == to the phase when current row == 0
            // pseudo_phase is shifted by 1 column for every subsequent row
            int pseudo_phase = phase;//(pseudo_y+phase)%phases;
            int ystart = pseudo_y * j_step;

            //mask
            xstart = pseudo_phase * i_step;
            Point2D center = new Point2D.Float(xstart+i_step/2, ystart+j_step/2);
            float radius = j_step;
            float r_max = 0.9f;//j_step*1.0f/i_step;
            float[] dist = {0.0f,r_max};
            Color[] colors = {Color.BLACK, Color.WHITE};
            RadialGradientPaint p = new RadialGradientPaint(center, radius, dist, colors);
//            g2.setPaint(p);
//            g2.fillRect(xstart, ystart, i_step, j_step);

//
//            Color c0 = Color.black;
//            Color c1 = Color.white;
//
//            double x0 = xstart;
//            double y0 = ystart;
//            double x1 = xstart+i_step;
//            double y1 = ystart+j_step;
//            double ww = i_step;
//            double hh = j_step;
//            int s = 10;
//            g2.setColor(Color.black);
//            g2.fillRect(xstart+s, ystart+s, i_step-2*s, j_step-2*s);
//
//            // Left
//            g2.setPaint(new GradientPaint(
//                    new Point2D.Double(x0, y0), c1,
//                    new Point2D.Double(x0 + s, y0), c0));
//            g2.fill(new Rectangle2D.Double(x0, y0, s, hh));
//            // Right
//            g2.setPaint(new GradientPaint(
//                    new Point2D.Double(x1-s, y0), c0,
//                    new Point2D.Double(x1, y0), c1));
//            g2.fill(new Rectangle2D.Double(x1-s, y0, s, hh));
//            // Top
//            g2.setPaint(new GradientPaint(
//                    new Point2D.Double(x0, y0), c1,
//                    new Point2D.Double(x0, y0 + s), c0));
//            g2.fill(new Rectangle2D.Double(x0, y0, ww, s));
//            // Bottom
//            g2.setPaint(new GradientPaint(
//                    new Point2D.Double(x0, y1-s), c0,
//                    new Point2D.Double(x0, y1), c1));
//            g2.fill(new Rectangle2D.Double(x0, y1-s, ww, s));
//
//            float fractions[] = new float[] { 0.0f, 1.0f };
//            // Top Left
//            g2.setPaint(new RadialGradientPaint(
//                    new Rectangle2D.Double(x0, y0, s + s, s + s),
//                    fractions, colors, MultipleGradientPaint.CycleMethod.NO_CYCLE));
//            g2.fill(new Rectangle2D.Double(x0, y0, s, s));
//            // Top Right
//            g2.setPaint(new RadialGradientPaint(
//                    new Rectangle2D.Double(x1-2*s, y0, s + s, s + s),
//                    fractions, colors, MultipleGradientPaint.CycleMethod.NO_CYCLE));
//            g2.fill(new Rectangle2D.Double(x1-s, y0, s, s));
//            // Bottom Left
//            g2.setPaint(new RadialGradientPaint(
//                    new Rectangle2D.Double(x0, y1 - 2*s, s + s, s + s),
//                    fractions, colors, MultipleGradientPaint.CycleMethod.NO_CYCLE));
//            g2.fill(new Rectangle2D.Double(x0, y1-s, s, s));
//            // Bottom Right
//            g2.setPaint(new RadialGradientPaint(
//                    new Rectangle2D.Double(x1-2*s, y1 - 2*s, s + s, s + s),
//                    fractions, colors, MultipleGradientPaint.CycleMethod.NO_CYCLE));
//            g2.fill(new Rectangle2D.Double(x1-s, y1-s, s, s));

            g.setColor(Color.black);

        for (int pseudo_x = 0; pseudo_x < phases; pseudo_x++) {

            if(phase==0){
                //mask
                xstart = pseudo_x * i_step;
                center = new Point2D.Float(xstart+i_step/2, ystart+j_step/2);
                radius = j_step;
                p = new RadialGradientPaint(center, radius, dist, colors);
                g2.setPaint(p);
                g2.fillRect(xstart, ystart, i_step, j_step);
            }
                xstart = pseudo_x * i_step;

                center = new Point2D.Float(xstart+i_step/2, ystart+j_step/2);
                p = new RadialGradientPaint(center, radius, dist, colors);
                g2.setPaint(p);

                for (int i = xstart; i < xstart + i_step; i = i + 3) {

                    if(i>w) break;

                    double temp = i * 1.0 / (w + 1);
                    int relative_x = (int) (temp * phases);
                    relative_x = relative_x + 1; //1 to phases

                    for (int j = ystart; j < ystart+j_step; j = j + 3) {

                        if(j>h) break;

                        Color pix = new Color(primaryImage.getRGB(i, j));

                        // what degree of grayscale should this pixel be
                        temp = pix.getRed() * 1.0 / 256;
                        int relative_color = (int) (temp * n_colors) + 1; //1 to n_colors
                        // how far from the current mask should it be
                        // high rgb = pale = right after the mask
                        int a = (phases - relative_color);
                        //a = (a + relative_x + n_colors) % phases;
                        if (a == pseudo_phase) {
                            g2.fillRect(i, j, 2, 2);
                        }
                    }
                }
            }
        }

        //mask
        g.setColor(Color.white);
        g.fillRect(0, h, BOX_WIDTH, BOX_HEIGHT-h);

        if(phase+1>=phases){
//            readNextImage();
            phase = -1;
        }

        phase = phase+1;
    }


    private void drawBenhamImage_old(Graphics g){

        UPDATE_RATE = 50; //50

        Graphics2D g2 = (Graphics2D) g;
        g2.setStroke(new BasicStroke(1));
        g.setColor(Color.black);//black

        int w = primaryImage.getWidth();
        int h = primaryImage.getHeight();

        int phases = 4; //4
        int n_colors = phases - 1;
        int color_step = 256/n_colors;
        int max = color_step*(n_colors-phase);

        //draw progressive mask
//        if(phase==phases-2 || phase==0){
//            //grey
//            g.setColor(Color.gray);
//            g.fillRect(0, 0, w, h);
//        }

//        //draw progressive mask
//        if(phase==phases-3 || phase==1){
//            //grey
//            g.setColor(Color.lightGray);
//            g.fillRect(0, 0, w, h);
//        }

        g.setColor(Color.black);
        //more time == darker appearance == small rgb
        for (int i = 0; i<w; i = i+3 ){
            for (int j = 0; j<h; j = j+3 ){
                Color pix = new Color(primaryImage.getRGB(i,j));
                if(pix.getRed()>=max-color_step && pix.getRed()<max) {
                    g.drawLine(i, j, i + 1, j + 1);
                }
            }
        }

        if(phase+1>=phases){

            //mask
            g.setColor(Color.black);
            g.fillRect(0, 0, w, h);
            phase = -1;

//            readNextImage();
        }

        phase = phase+1;
    }


    private void readNextImage(){
        if(step>82) step = 0;

        String path = "/Users/lana/Desktop/prgm/CSL/prednet_chainer_2/datasets/kitti/image_00/data/";
        path = path + String.format("%010d", step) + ".png";
        try {
            primaryImage = resize(ImageIO.read(new File(path)), BOX_HEIGHT);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void drawBenham(Graphics g){

        Graphics2D g2 = (Graphics2D) g;
        g2.setStroke(new BasicStroke(1));
        //darker
        Color c = Color.black; //new Color(0,80,0);
        g.setColor(c);//black

        if(phase==0){
//
//            g.setColor(Color.black);//black
//
//            int r = basic_radius + 12*separation;
//            int x = center_x - r;
//            int y = center_y - r;
//            g.fillRect(x, y, r*2, r*2/3);

            //draw inner arcs
            g.setColor(c);
            int r = basic_radius/2;
            for(int i = 0; i<cirles; i++) {
                g.drawArc(center_x-r, center_y-r, r*2, r*2, 0, 45);
                r = r + separation;
            }
            //r = basic_radius + 4*separation;
            for(int i = 0; i<cirles; i++) {
                g.drawArc(center_x-r, center_y-r, r*2, r*2, 45, 90);
                r = r + separation;
            }
        } else if(phase==1){
            //draw outer circles
            int r = basic_radius + 6*separation;
            for(int i = 0; i<cirles; i++) {
                g.drawArc(center_x-r, center_y-r, r*2, r*2, 0, 45);
                r = r + separation;
            }
            for(int i = 0; i<cirles; i++) {
                g.drawArc(center_x-r, center_y-r, r*2, r*2, 45, 90);
                r = r + separation;
            }
        } else if(phase==2){
            //draw a big black square
            //even darker
            // c = new Color(0,60,0);
            g.setColor(Color.black);//black

            int r = basic_radius + 12*separation;
            int x = center_x - r;
            int y = center_y - r;
            g.fillRect(x, y, r*2, r*2);


//            //draw inner arcs
//            g.setColor(Color.black);//black
//            r = basic_radius/2;
//            for(int i = 0; i<cirles; i++) {
//                g.drawArc(center_x-r, center_y-r, r*2, r*2, 0, 45);
//                r = r + separation;
//            }
//            //r = basic_radius + 4*separation;
//            for(int i = 0; i<cirles; i++) {
//                g.drawArc(center_x-r, center_y-r, r*2, r*2, 45, 90);
//                r = r + separation;
//            }

            phase = -1;
        }/* else if(phase==3){
            //draw nothing

            phase = -1;
        }*/

        //draw mask
        g.setColor(Color.white);
        //draw left part
        int r = basic_radius + 7*separation;
        g.fillRect(0, 0, center_x, BOX_HEIGHT);
        //draw bottom right part
        g.fillRect(center_x, center_y, center_x, center_y);

        phase = phase+1;
    }


    private void drawFlashCustom(Graphics g, int timing){

        if(phase==timing){
            //draw black mask
            g.setColor(Color.black);
            g.fillRect(0, 0, BOX_WIDTH, BOX_HEIGHT);

        } else if(phase==1){
            //draw image
            drawStaticSnakes(g);
        } else if(phase==2){
            //draw white
            phase = -1;
        }

        phase = phase+1;
    }



    // height

    /**
     *
     * @param r is radius
     * @param g2
     * @param shift_angle
     * @param h_0
     */
    private void drawBands(int r, Graphics2D g2, int shift_angle, int h_0){

        AffineTransform old = g2.getTransform();

        g2.rotate(Math.toRadians(shift_angle));

        //draw black
        g2.setColor(Color.black);//black
        int w_0 = h_0/4;//2*diff;
        int density = 20;
        for(int i = 0; i<360; i=i+density) {
            int x = 0 ;
            int y = r ;

            Rectangle rect = new Rectangle(x, y, w_0, h_0);

            g2.rotate(Math.toRadians(density));
            g2.draw(rect);
            g2.fill(rect);
        }
        g2.setTransform(old);
        g2.rotate(Math.toRadians(shift_angle));

        //thin gray/red
        Color red = new Color(200,100,100);
        //dark gray/blue
        Color blue = new Color(0,0,150);

        g2.setColor(red);
        int w_1 = w_0-2;//diff;
        for(int i = 0; i<360; i=i+density) {
            int x = w_0;
            int y = r ;

            Rectangle rect = new Rectangle(x, y, w_1, h_0);

            g2.rotate(Math.toRadians(density));
            g2.draw(rect);
            g2.fill(rect);
        }

        g2.setTransform(old);
        g2.rotate(Math.toRadians(shift_angle));

        g2.setColor(blue);
        int w_2 = w_0+2;//3*diff;
        for(int i = 0; i<360; i=i+density) {
            int x = -w_2;
            int y = r ;

            Rectangle rect = new Rectangle(x, y, w_2, h_0);

            g2.rotate(Math.toRadians(density));
            g2.draw(rect);
            g2.fill(rect);
        }

        g2.setTransform(old);
    }

    private void showDephasedImage(Graphics g, int timing, BufferedImage image){
        Graphics2D g2 = (Graphics2D) g;
        g2.setStroke(new BasicStroke(1));
        g.setColor(Color.black);

        if(phase==0){
            int x = center_x - image.getWidth()/2;
            int y = center_y - image.getHeight()/2;
            g.drawImage(image, x, y, null);
        } else if(phase==2){//2
            //draw a big black square
            g.setColor(Color.black);
            g.fillRect(0, 0, BOX_WIDTH, BOX_HEIGHT);

//            int startX = 0, startY = 0, endX = BOX_WIDTH, endY = 0;
//
//            Color startColor = Color.black;
//            Color endColor = Color.white;
//
//            if(secondary_phase==1){
//                startColor = Color.white;
//                endColor = Color.black;
//                secondary_phase = -1;
//            }
//
//            GradientPaint gradient = new GradientPaint(startX, startY, startColor, endX, endY, endColor);
//            g2.setPaint(gradient);
//
//            g2.fill(new Rectangle2D.Double(0,0,BOX_WIDTH,BOX_HEIGHT));
//
//            secondary_phase = secondary_phase+1;
            phase = -1;
        } else {
            //draw white
            int contrast = 100;
            Color c = new Color(contrast,contrast,contrast);
//            g.setColor(c);
//            g.fillRect(0, 0, BOX_WIDTH, BOX_HEIGHT);
        }

        phase = phase+1;
    }


    private void drawStaticSnakes(Graphics g) {

        Graphics2D g2 = (Graphics2D) g;
        //change origins
        g2.translate(center_x, center_y);
        g2.scale(1.0, -1.0);

        int r = basic_radius;///2;
        int h = 40;
        drawBands(r, g2, 0, h);
        r = r - h;
        h = h*3/4;
        drawBands(r, g2, 30, h);
        r = r - h;
        h = h*3/4;
        drawBands(r, g2, 30, h);
        r = r - h;
        h = h*3/4;
        drawBands(r, g2, 30, h);
    }


    /**
     *
     * @param g
     * @param timing 0 to show inner first
     */
    private void drawGeneralizedBenham(Graphics g, int timing){
        Graphics2D g2 = (Graphics2D) g;
        g2.setStroke(new BasicStroke(1));
        g.setColor(Color.black);

        mlog.say("phase " + phase);


        if(phase==timing){
            int x = center_x - primaryImage.getWidth()/2;
            int y = center_y - primaryImage.getHeight()/2;
            g.drawImage(primaryImage, x, y, null);
        } else if(phase==2){//2
            //draw a big black square
            g.setColor(Color.black);
            g.fillRect(0, 0, BOX_WIDTH, BOX_HEIGHT);
            phase = -1;
        } else {
            int x = center_x - secondaryImage.getWidth()/2;
            int y = center_y - secondaryImage.getHeight()/2;
            g.drawImage(secondaryImage, x, y, null);
            //phase = -1;
        }

        phase = phase+1;
    }


    private void drawFire(Graphics g){
        Graphics2D g2 = (Graphics2D) g;
        g2.setStroke(new BasicStroke(1));
        g.setColor(Color.black);

        mlog.say("phase " + phase);


        if(phase==0){
            int x = center_x - primaryImage.getWidth()/2;
            int y = center_y - primaryImage.getHeight()/2;
            g.drawImage(primaryImage, x, y, null);
        } else if(phase==1){
            int x = center_x - secondaryImage.getWidth()/2;
            int y = center_y - secondaryImage.getHeight()/2;
            g.drawImage(secondaryImage, x, y, null);
        } else if(phase==2){
            int x = center_x - thirdImage.getWidth()/2;
            int y = center_y - thirdImage.getHeight()/2;
            g.drawImage(thirdImage, x, y, null);
        } else {
            //draw a big black square
            g.setColor(Color.black);
            g.fillRect(0, 0, BOX_WIDTH, BOX_HEIGHT);
            phase = -1;
        }

        phase = phase+1;
    }


    private void drawArrows(Graphics g){
        Graphics2D g2 = (Graphics2D) g;
        g2.setStroke(new BasicStroke(1));
        g.setColor(Color.black);



        int y = center_y - primaryImage.getHeight()/2;
        int x = center_x + phase*50 - primaryImage.getWidth()/2;

        if(phase<4){
            g.drawImage(primaryImage, x, y, null);
        } else {
            //draw a big black square
            g.setColor(Color.black);
            g.fillRect(0, 0, BOX_WIDTH, BOX_HEIGHT);
            phase = -1;
        }

        phase = phase+1;
    }

    /**
     *
     */
    private void drawShapes(){//(Graphics g){

        Graphics g = paintImage.createGraphics();

//        int contrast = 100;
//        Color c = new Color(contrast,contrast,contrast);
//        g.setColor(c);

        g.setColor(Color.white);
        g.fillRect(0, 0, BOX_WIDTH, BOX_HEIGHT);



        // draw on paintImage using Graphics

        if(image!=null){
            showDephasedImage(g, timing, image);
        } else {

            switch (type) {
                case Constants.BENHAM_CLASSIC: {
                    drawBenham(g);
                    break;
                }
                case Constants.BENHAM_VAR: {
                    drawBenham_var(g);
                    break;
                }
                case Constants.BENHAM_RED: {
                    drawBenham_red(g);
                    break;
                }

                case Constants.BENHAM_WEIRD: {
                    drawBenham_weird(g);
                    break;
                }

                case Constants.SIMPLE_SHAPES_0: {
                    simple_shape(g, 0);
                    break;
                }
                case Constants.SIMPLE_SHAPES_1: {
                    simple_shape(g, 1);
                    break;
                }

                case Constants.THINNER_LINE: {
                    thinner_line(g);
                    break;
                }

                case Constants.THICKER_LINE: {
                    thicker_line(g);
                    break;
                }

                case Constants.DRAW_X_0: {
                    draw_x(g, 0);
                    break;
                }
                case Constants.DRAW_X_1: {
                    draw_x(g, 1);
                    break;
                }
                case Constants.ROTATING_X_1: {
                    draw_rotating_x(g, 1);
                    break;
                }
                case Constants.DRAW_X_PHASE_0: {
                    draw_x_dephased(g, 0);
                    break;
                }
                case Constants.DRAW_X_PHASE_1: {
                    draw_x_dephased(g, 1);
                    break;
                }

                case Constants.DRAW_PLUS: {
                    draw_plus(g, 0);
                    break;
                }

                case Constants.CONCENTRIC_0: {
                    concentric(g, 0);
                    break;
                }
                case Constants.CONCENTRIC_1: {
                    concentric(g, 1);
                    break;
                }

                case Constants.NEW_BENHAM: {
                    new_benham(g, 0);
                    break;
                }

                case Constants.NO_MASK: {
                    no_mask(g, 0);
                    break;
                }

                case Constants.CONCURRENT: {
                    smooth_concurrent(g);
                    break;
                }
                case Constants.TIME_WARP: {
                    time_warp(g);
                    break;
                }

                case Constants.STATIC: {
                    drawStaticSnakes(g);
                    break;
                }

                case Constants.GENERALIZED_BENHAM: {
                    drawGeneralizedBenham(g,0);
                    break;
                }

                case Constants.CONTRAST: {
                    drawContrast(g);
                    break;
                }

                case Constants.FIRE: {
                    drawFire(g);
                    break;
                }

                case Constants.ARROW: {
                    drawArrows(g);
                    break;
                }

                case Constants.COLOR_BANDS: {
                    drawColorBands(g);
                    break;
                }

                case Constants.DARK_BENHAM: {
                    drawDarkBenham(g);
                    break;
                }

                case Constants.BENHAM_IMAGE: {
                    drawBenhamImage(g);
                    break;
                }
            }
        }

        do_update = false;

        g.dispose();
        // repaint panel with new modified paint
        repaint();

    }


    @Override
    protected void paintComponent(Graphics g){
        super.paintComponent(g);
        g.drawImage(paintImage, 0, 0, null);

        if(do_update) {
            drawShapes();
        }
    }
}
