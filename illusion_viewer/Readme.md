# Illusion viewer

Developed in Java.

## Install


Install Java JDK: https://java.com/en/download/
(Walkthrough: https://www.wikihow.com/Install-the-Java-Software-Development-Kit)

Install the IntelliJ IDE: https://www.jetbrains.com/idea/

## Setup

Open IntelliJ and import the sources: 

Import project -> select the illusion_viewer folder; "Create project from Existing Sources"

or 

File -> New -> Project from sources , select illusion_viewer

Click "next" until setup is finished.

## Modify

Source files are in the `src` folder.

In `Constants` change the name of the output folder (where image can be saved) and of the input folder (where the images that are displayed are).

In `Starter` you can change the type of illusion that is displayed.


## Run

Cick Build -> Build Project or click the green hammer in the top right corner.


Click Add Configuration on the right of the green hammer. Click '+' -> Application

Enter a name, eg `Default`

For `Main class` select `Starter`

Apply -> OK

Click `run` (the green play button to the right of the green hammer)

## Make video from frames (high quality, quickTime compatible)

`ffmpeg -framerate 20 -i "%04d.png" -pix_fmt yuv420p -vb 20M _out.mp4`

Make gif: put the frames you want to loop in a folder.

```
ffmpeg -framerate 20 -i "%04d.png" -pix_fmt yuv420p -vb 20M _out.avi
ffmpeg -i _out.avi -pix_fmt rgb24 -loop 0 _out.gif
```

select all files for ffmpeg
`ffmpeg -framerate 10 -pattern_type glob -i '*.png' -pix_fmt yuv420p -vb 20M _out.avi`

## Resize images for prednet

resize
`mogrify -path . -resize 160x400 -quality 100 -format png *.png`

crop: w x h + x offset + y offset
`mogrify -crop 160x120+50+100 *.png`


eg

mogrify -path . -resize 400x120 -quality 100 -format png *.png
