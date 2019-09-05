import communication.Constants;
import communication.MyLog;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.color.ColorSpace;
import java.awt.image.BufferedImage;
import java.awt.image.ColorConvertOp;
import java.io.File;
import java.io.IOException;


public class Generator extends JPanel {
    MyLog mlog = new MyLog("Generator", true);

    private BufferedImage paintImage;

    private int type;
    private int UPDATE_RATE = 50;

    // Container box's width and height
    private static final int BOX_WIDTH = 800;
    private static final int BOX_HEIGHT = 800;

    private int step = 0;
    private boolean save;
    boolean readyForSave = false;
    boolean saved = true;

    JFrame frame;

    String nameFormat = "%04d";
    String folderName;
    BufferedImage snakesImage;
    BufferedImage bwSnakesImage;
    BufferedImage fraserImage;
    boolean do_update = true;


    public Generator(int type, JFrame frame, String folderName, boolean save){
        this.type = type;
        //this.frame = frame;
        //frame.add(this);
        this.folderName = folderName;
        this.save = save;
        paintImage = new BufferedImage(BOX_WIDTH, BOX_HEIGHT, BufferedImage.TYPE_3BYTE_BGR);

        this.frame = new JFrame("TheFrame");
        this.frame.add(this);
        this.frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(BOX_WIDTH, BOX_HEIGHT);
        this.frame.setVisible(true);


        String path = "/Users/lana/Desktop/prgm/CSL/prednet_chainer_2/datasets/";
        try {
            snakesImage = ImageIO.read(new File(path, "snakes_1.jpg"));
            bwSnakesImage = ImageIO.read(new File(path, "snakes_2.jpg"));
            fraserImage = ImageIO.read(new File(path, "fraser.png"));
        } catch (IOException e) {
            e.printStackTrace();
        }

        ColorConvertOp op = new ColorConvertOp(ColorSpace.getInstance(ColorSpace.CS_GRAY), null);
        op.filter(bwSnakesImage, bwSnakesImage);

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
            ImageIO.write(paintImage, "PNG", new File(folderName + name));
        } catch (IOException e) {
            e.printStackTrace();
        }

        saved = true;
    }


    int phase = 0;
    int cirles = 3;
    int center_x = BOX_WIDTH/2;
    int center_y = BOX_HEIGHT/2;
    int separation = 8; //pixels
    int basic_radius = 150;

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
//            g.drawOval(center_x-r, center_y-r, r*2, r*2);
            for(int i = 0; i<cirles; i++) {
                //g.drawArc(center_x-r, center_y-r, r*2, r*2, 0, 45);
                g.drawOval(center_x-r, center_y-r, r*2, r*2);
                r = r + p_separation;
            }

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
                g.fillRect(x, y, p_separation, h);
                x+=p_separation;

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
                x+=p_separation;

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
//                g.setColor(Color.blue);
//                g.fillRect(x, y, p_separation, h);
                x+=p_separation;

                g.setColor(Color.black);
                g.fillRect(x, y, p_separation/2, h);
                x+=p_separation/2;

//                g.setColor(Color.green);
//                g.fillRect(x, y, p_separation, h);
                x+=p_separation;

                //white
                x+=p_separation/2;
            }

            for (int i =3; i<6 ; i++) {
                g.setColor(Color.green);
//                g.fillRect(x, y, p_separation, h);
                x+=p_separation;

//                g.setColor(Color.black);
//                g.fillRect(x, y, p_separation/2, h);
                x+=p_separation/2;

                g.setColor(Color.blue);
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

            for (int i =0; i<3 ; i++) {
                g.setColor(Color.blue);
                g.fillRect(x, y, p_separation, h);
                x+=p_separation;

//                g.setColor(Color.black);
//                g.fillRect(x, y, p_separation/2, h);
                x+=p_separation/2;

                g.setColor(Color.green);
//                g.fillRect(x, y, p_separation, h);
                x+=p_separation;

                //white
                x+=p_separation/2;
            }

            for (int i=3; i<6 ; i++) {
//                g.setColor(Color.blue);
//                g.fillRect(x, y, p_separation, h);
                x+=p_separation;

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
        g.setColor(Color.black);
        int r =  basic_radius;

        if(phase==timing){
            g.drawLine(center_x-r, center_y-r, center_x+r, center_y+r);
        } else if(phase==2){
            //draw a big black square
            g.setColor(Color.black);
            r = basic_radius + 6*separation;
            int x = center_x - r;
            int y = center_y - r;
            g.fillRect(x, y, r*2, r*2);
            secondary_phase = secondary_phase + 30;
            phase = -1;
        } else{
            g.drawLine(center_x+r, center_y-r, center_x-r, center_y+r);
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

    private void drawBenham_red(Graphics g){

        Graphics2D g2 = (Graphics2D) g;
        g2.setStroke(new BasicStroke(1));
        g.setColor(Color.green);

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

    private void draw_snakes(Graphics g, int timing, boolean bw){

        Graphics2D g2 = (Graphics2D) g;
        g2.setStroke(new BasicStroke(1));
        g.setColor(Color.black);

        if(phase==timing){//timing
            int x = center_x - snakesImage.getWidth()/2;
            int y = center_y - snakesImage.getHeight()/2;
            if(bw){
                g.drawImage(bwSnakesImage, x, y, null);
            } else {
                g.drawImage(snakesImage, x, y, null);
            }
        } else if(phase==2){//2
            //draw a big black square
            g.setColor(Color.black);
            int r = 350;
            int x = center_x - r;
            int y = center_y - r;
            g.fillRect(x, y, r*2, r*2);
            secondary_phase = secondary_phase + 30;
            phase = -1;
        }

        phase = phase+1;
    }

    private void draw_frazer(Graphics g, int timing){

        Graphics2D g2 = (Graphics2D) g;
        g2.setStroke(new BasicStroke(1));
        g.setColor(Color.black);

        if(phase==timing){//timing
            int x = center_x - snakesImage.getWidth()/2;
            int y = center_y - snakesImage.getHeight()/2;
            g.drawImage(fraserImage, x, y, null);
        } else if(phase==2){//2
            //draw a big black square
            g.setColor(Color.black);
            int r = 350;
            int x = center_x - r;
            int y = center_y - r;
            g.fillRect(x, y, r*2, r*2);
            secondary_phase = secondary_phase + 30;
            phase = -1;
        }

        phase = phase+1;
    }

    private void draw_bad_snakes(Graphics g){

        Graphics2D g2 = (Graphics2D) g;
        g2.setStroke(new BasicStroke(1));
        g.setColor(Color.black);

        if(phase==0){//timing
            int x = center_x - snakesImage.getWidth()/2;
            int y = center_y - snakesImage.getHeight()/2;
            g.drawImage(snakesImage, x, y, null);
        } else if(phase==1){//2
            //draw a big black square
            g.setColor(Color.black);
            int r = 350;
            int x = center_x - r;
            int y = center_y - r;
            g.fillRect(x, y, r*2, r*2);
            secondary_phase = secondary_phase + 30;
            phase = -1;
        }

        phase = phase+1;
    }


    private void drawBenham_weird(Graphics g){

        Graphics2D g2 = (Graphics2D) g;
        g2.setStroke(new BasicStroke(1));
        g.setColor(Color.black);

        if(phase==0){
            //draw a big square
            g.setColor(Color.black);
//            int r = basic_radius + 12*separation;
//            int x = center_x - r;
//            int y = center_y - r;
//            g.fillRect(x, y, r*2, r*2);
//
//            g.setColor(Color.white);

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
            //draw a big  square
            g.setColor(Color.red);
            int r = basic_radius + 12*separation;
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

    private void drawBenham(Graphics g){

        Graphics2D g2 = (Graphics2D) g;
        g2.setStroke(new BasicStroke(1));
        g.setColor(Color.black);

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
            //draw a big black square
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

    /**
     *
     */
    private void drawShapes(){//(Graphics g){

        Graphics g = paintImage.createGraphics();
        // Paint background
        g.setColor(Color.white);
        g.fillRect(0, 0, BOX_WIDTH, BOX_HEIGHT);

        // draw on paintImage using Graphics



        switch (type){
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
                draw_plus(g,0);
                break;
            }

            case Constants.SNAKES_0: {
                draw_snakes(g, 0, false);
                break;
            }
            case Constants.SNAKES_1: {
                draw_snakes(g, 1, false);
                break;
            }
            case Constants.SNAKES_BW: {
                draw_snakes(g, 0, true);
                break;
            }
            case Constants.BAD_SNAKES: {
                draw_bad_snakes(g);
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

            case Constants.FRASER: {
                draw_frazer(g, 1);
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
