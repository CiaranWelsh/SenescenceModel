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
#include <iostream>
#include <wingdi.h>

#include "custom_visualisation.h"
#include "GLUTInputController.h"
#include "flame_globals.h"
#include "cube.h"

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


extern void renderCube(
        GLuint* instances_data1_tbo,
        cudaGraphicsResource_t* p_instances_data1_cgr
        );

extern void initVisualisation();
extern void runVisualisation();


//extern void set_EYE_X(float* eye_x);
//extern void set_EYE_Y(float* eye_y);
//extern void set_EYE_Z(float* eye_z);

void renderScene(void);

void renderScene(void) {

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

//    glBegin(GL_TRIANGLES);
//    glVertex3f(-0.5,-0.5,0.0);
//    glVertex3f(0.5,0.0,0.0);
//    glVertex3f(0.0,0.5,0.0);
//    glEnd();

    glutSwapBuffers();
}

extern void initVisualisation()
{


    // Create GL context
    int   argc   = 1;
    char glutString[] = "GLUT application";
    char *argv[] = {glutString, NULL};
    //char *argv[] = {"GLUT application", NULL};

    glutInit(&argc, argv);
    glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize( window_width, window_height);
    glutInitContextVersion(4, 3);
    glutInitContextProfile(GLUT_CORE_PROFILE);
    glutCreateWindow( "FLAME GPU Visualiser");

    if (glewInit()){
        std::cerr << "no GLEW avaiable   " << std::endl;
        exit(EXIT_FAILURE);
    }

//    glutDisplayFunc(renderScene);
//    glutReshapeFunc(windowResize);

//    glewInit();
//
//    GLenum err = glewInit();
//
//    if (GLEW_OK != err)
//    {
//        /* Problem: glewInit failed, something is seriously wrong. */
//        fprintf(stderr, "GLEW/GLUT init Error: %s\n", glewGetErrorString(err));
//    }
//
//    glClearColor( 1.0, 1.0, 1.0, 1.0);
//
//
//    glutReshapeFunc(windowResize);

    //my code here

//    cudaEventRecord(start);


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


//    //FPS
//    start_time = 0;
//    end_time = 0;
//    frame_time = 0;
//    fps = 0;
//    frames = 0;
//    av_frames = 25;

    // register callbacks
//    glutDisplayFunc( display);
//    glutCloseFunc( close);
//    glutKeyboardFunc( keyboard);
//    glutSpecialFunc( specialKeyboard);
//    glutMouseFunc( mouse);

    // Set the closing behaviour
//    glutSetOption( GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS );
}

extern void runVisualisation()
{
    //init FLAME GPU globals controller - after initial states have been loaded
//    initGlobalsController();
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


void display(void)
{

    /* clear window */
    glClear(GL_COLOR_BUFFER_BIT);

    /* draw scene */
//    glutSolidTeapot(.5);

    /* flush drawing routines to the window */
    glFlush();
    //start timing
    //glFinish();
//    start_time = clock();
//
//    glEnable( GL_DEPTH_TEST);
//    glEnable(GL_LIGHTING);
//    glEnable(GL_LIGHT0);
//
//    //lookat
//    gluLookAt(eye[0], eye[1], eye[2], look[0], look[1], look[2], up[0], up[1], up[2]);
//
//    //lighting
//    glLightfv(GL_LIGHT0, GL_POSITION, lightPosition);
////
////    updateSimulationConstants();
//
//    stepFLAMESimulation();
//    renderCube();
//
////    drawInfoDisplay(window_width, window_height);
////    drawMenuDisplay(window_width, window_height);
//
//    //end timing
//    glFinish();
//    end_time = clock();
//    frame_time += (end_time - start_time);
//    if (frames == av_frames){
//        fps = (float)av_frames/(frame_time/(float)CLOCKS_PER_SEC);
//        frames = 0;
//        frame_time = 0.0f;
//    }else{
//        frames++;
//    }
//
//    //redraw
//    glutSwapBuffers();
//    glutPostRedisplay();
//
//    // If an early exit has been requested, close the visualisation by leaving the main loop.
//    if(getExitFLAMESimulation()){
//        glutLeaveMainLoop();
//    }

}












