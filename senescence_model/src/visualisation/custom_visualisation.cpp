//
// Created by cwels on 13/10/2018.
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <time.h>

#include "custom_visualisation.h"
#include <GLUTInputController.h>
#include <flame_globals.h>

int window_width = 800;
int window_height = 600;

void windowResize(int width, int height);


//full screen mode
int fullScreenMode;

//light
GLfloat lightPosition[] = {25.0, 25.0f, 25.0f, 1.0f};

//framerate
float start_time;
float end_time;
float frame_time;
float fps;
int frames;
int av_frames;

extern void initVisualisation();
extern void runVisualisation();


//extern void set_EYE_X(float* eye_x);
//extern void set_EYE_Y(float* eye_y);
//extern void set_EYE_Z(float* eye_z);



extern void initVisualisation()
{


    // Create GL context
    int   argc   = 1;
    char glutString[] = "GLUT application";
    char *argv[] = {glutString, NULL};
    //char *argv[] = {"GLUT application", NULL};

    glutInit(&argc, argv);
    glewInit();
    if (! glewIsSupported( "GL_VERSION_2_0 "
                           "GL_ARB_pixel_buffer_object"
    )) {
        fprintf( stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush( stderr);
        return;
    }

    // default initialization
    glClearColor( 1.0, 1.0, 1.0, 1.0);


    glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize( window_width, window_height);
    glutCreateWindow( "FLAME GPU Visualiser");
    glutReshapeFunc(windowResize);



    // initialize GL
//    if(FALSE == initGL()) {
//        return;
//    }

    //load pedestrians
//    initPedestrianPopulation();

    //initialise input control
//    initInputConroller();

    //init menu
//    initMenuItems();


    //FPS
    start_time = 0;
    end_time = 0;
    frame_time = 0;
    fps = 0;
    frames = 0;
    av_frames = 25;

    // register callbacks
//    glutDisplayFunc( display);
//    glutCloseFunc( close);
//    glutKeyboardFunc( keyboard);
//    glutSpecialFunc( specialKeyboard);
//    glutMouseFunc( mouse);

    // Set the closing behaviour
    glutSetOption( GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS );



}

extern void runVisualisation()
{
    //init FLAME GPU globals controller - after initial states have been loaded
    initGlobalsController();
    // Update all menu texts
//    updateAllMenuTexts();
    // Flush outputs prior to glut main loop.
    fflush(stdout);
    fflush(stderr);
    // start rendering mainloop
    glutMainLoop();
}

////////////////////////////////////////////////////////////////////////////////
//! Close callback
////////////////////////////////////////////////////////////////////////////////
//void close(){
//     Call exit functions and clean up after simulation
//    cleanupFLAMESimulation();
//}

//void updateSimulationConstants(){
//    set_EYE_X(&eye[0]);
//    set_EYE_Y(&eye[1]);
//    set_EYE_Z(&eye[2]);
//}

void windowResize(int width, int height){
    window_width = width;
    window_height = height;
}


void toggleFullScreenMode()
{
    fullScreenMode = !fullScreenMode;
    glutFullScreen();
}

float getFPS()
{
    return fps;
}
















