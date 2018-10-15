/*
 * Copyright 2011 University of Sheffield.
 * Author: Dr Paul Richmond 
 * Contact: p.richmond@sheffield.ac.uk (http://www.paulrichmond.staff.shef.ac.uk)
 *
 * University of Sheffield retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * University of Sheffield is strictly prohibited.
 *
 * For terms of licence agreement please attached licence or view licence 
 * on www.flamegpu.com website.
 * 
 */
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cmath>
#include "custom_visualisation.h"

#include "GLUTInputController.h"
#include <flame_globals.h>
//#include "MenuDisplay.h"
//#include "GlobalsController.h"
//#include "MenuDisplay.h"

#define ENV_MAX 2.0f
#define ENV_MIN -ENV_MAX
#define ENV_WIDTH (2*ENV_MAX)

//viewpoint vectors and eye distance
float eye[3];
float up[3];
float look[3];
float eye_distance;

float theta;
float phi;
float cos_theta;
float sin_theta;
float cos_phi;
float sin_phi;

int mouse_old_x, mouse_old_y;

int zoom_key = 0;

#define TRANSLATION_SCALE 0.005f
#define ROTATION_SCALE 0.01f
#define ZOOM_SCALE 0.01f

#define MAX_ZOOM 0.01f
#define MIN_PHI 0.0f

#define PI 3.14
#define rad(x) (PI / 180) * x

//prototypes
void updateRotationComponents();
//mouse motion funcs
void rotate(int x, int y);
void zoom(int x, int y);
void translate(int x, int y);

void initInputConroller()
{
	//init view
	eye_distance = ENV_MAX*1.75f;
	up[0] = 0.0f;
	up[1] = 1.0f;
	up[2] = 0.0f;
	eye[0] = 0.0f;
	eye[1] = 0.0f;
	eye[2] = eye_distance;
	look[0] = 0.0f;
	look[1] = 0.0f;
	look[2] = 0.0f;

	theta = 3.14159265f;
	phi = 1.57079633f;
}


void mouse(int button, int state, int x, int y)
{
	if (zoom_key)
		button  = GLUT_MIDDLE_BUTTON;

    if (state == GLUT_DOWN) {
        switch(button)
		{
			case(GLUT_LEFT_BUTTON):
			{
				glutMotionFunc(translate);
				break;	
			}
			case(GLUT_RIGHT_BUTTON):
			{
				glutMotionFunc(rotate);
				break;	
			}
			case(GLUT_MIDDLE_BUTTON):
			{
				glutMotionFunc(zoom);
				break;	
			}
		}
    } else if (state == GLUT_UP) {
		glutMotionFunc(NULL);
    }

    mouse_old_x = x;
    mouse_old_y = y;
    glutPostRedisplay();
}

void updateRotationComponents()
{
	cos_theta = (float) cos(theta);
	sin_theta = (float) sin(theta);
	cos_phi = (float) cos(phi);
	sin_phi = (float) sin(phi);
}

void rotate(int x, int y)
{
	//calc change in mouse movement
	float dx = (float) (x - mouse_old_x);
	float dy = (float) (y - mouse_old_y);

	//update rotation component values
	updateRotationComponents();

	//update eye distance
	theta-=dx*ROTATION_SCALE;
	phi+=dy*ROTATION_SCALE;

	phi = (phi<MIN_PHI)?0.0f:phi;

	//update eye and and up vectors
	eye[0]= look[0] + -eye_distance*sin_theta*cos_phi;
	eye[1]= look[1] + eye_distance*cos_theta*cos_phi;
	eye[2]= look[2] + eye_distance*sin_phi;
	up[0]= sin_theta*sin_phi;
	up[1]= -cos_theta*sin_phi;
	up[2]= cos_phi;
	//update prev positions
	mouse_old_x = x;
	mouse_old_y = y;
}

void zoom(int x, int y)
{
	//calc change in mouse movement
	// float dx = (float) (x - mouse_old_x);
	float dy = (float) (y - mouse_old_y);

	//update rotation component values
	updateRotationComponents();

	//update eye distance
	eye_distance -= dy*ZOOM_SCALE;
	eye_distance = (eye_distance<MAX_ZOOM)?MAX_ZOOM:eye_distance;

	//update eye vector
	eye[0]= look[0] + -eye_distance*sin_theta*cos_phi;
	eye[1]= look[1] + eye_distance*cos_theta*cos_phi;
	eye[2]= look[2] + eye_distance*sin_phi;

	//update prev positions
	mouse_old_x = x;
	mouse_old_y = y;
}

void translate(int x, int y)
{
	//calc change in mouse movement
	float dx = (float) (x - mouse_old_x);
	float dy = (float) (y - mouse_old_y);

	//update rotation component values
	updateRotationComponents();

	//translate look and eye vector position
	look[0] += ((dx*cos_theta) + (dy*sin_theta))*TRANSLATION_SCALE;
	look[1] += ((dx*sin_theta) - (dy*cos_theta))*TRANSLATION_SCALE;
	look[2] += 0.0;
	eye[0]= look[0] + -eye_distance*sin_theta*cos_phi;
	eye[1]= look[1] + eye_distance*cos_theta*cos_phi;
	eye[2]= look[2] + eye_distance*sin_phi;


	//update prev positions
	mouse_old_x = x;
	mouse_old_y = y;
}

////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void keyboard( unsigned char key, int /*x*/, int /*y*/)
{
	const float* current_var_float;
	float new_float_var;

	const int* current_var_int;
	int new_var_int;

	float increment_float = 0.001f;
	int increment_int = 1;
//    113/119, q/w TISSUE_DAMAGE_PROB
//    101/114, e/r QUIESCENT_MIGRATION_SCALE
//    116, 121, t/new_float_var REPAIR_RANGE
//    117, 105, u/i DAMAGE_DETECTION_RANGE
	switch( key) {
		// Space == 32

		case('q') :
			current_var_float = get_TISSUE_DAMAGE_PROB();
			new_float_var = *current_var_float - increment_float;
			if (new_float_var < 0.0f){
				new_float_var = *current_var_float;
			}
			set_TISSUE_DAMAGE_PROB(&new_float_var);
			std::cout<<"q: TISSUE_DAMAGE_PROB: start="<<*current_var_float<<":New value="<< new_float_var<<std::endl;
			break;

		case('w') :
			current_var_float = get_TISSUE_DAMAGE_PROB();
			new_float_var = *current_var_float +increment_float;
			if (new_float_var > 1.0f){
				new_float_var = *current_var_float;
			}
			set_TISSUE_DAMAGE_PROB(&new_float_var);
			std::cout<<"w: TISSUE_DAMAGE_PROB: start="<<*current_var_float<<":New value="<< new_float_var<<std::endl;
			break;

			// 113 = q
		case('e') :
			current_var_float = get_QUIESCENT_MIGRATION_SCALE();
			new_float_var = *current_var_float - increment_float;
			if (new_float_var < 0.0f){
				new_float_var = *current_var_float;
			}
			set_QUIESCENT_MIGRATION_SCALE(&new_float_var);
			std::cout<<"e: QUIESCENT_MIGRATION_SCALE: start="<<*current_var_float<<":New value="<< new_float_var<<std::endl;
			break;

		case('r') :
			current_var_float = get_QUIESCENT_MIGRATION_SCALE();
			new_float_var = *current_var_float +increment_float;
			if (new_float_var > 1.0f){
				new_float_var = *current_var_float;
			}
			set_QUIESCENT_MIGRATION_SCALE(&new_float_var);
			std::cout<<"r: QUIESCENT_MIGRATION_SCALE: start="<<*current_var_float<<":New value="<< new_float_var<<std::endl;
			break;

		case('t') :
			current_var_float = get_REPAIR_RANGE();
			new_float_var = *current_var_float -increment_float;
			if (new_float_var < 0.0f){
				new_float_var = *current_var_float;
			}
			set_REPAIR_RANGE(&new_float_var);
			std::cout<<"t: REPAIR_RANGE: start="<<*current_var_float<<":New value="<< new_float_var<<std::endl;
			break;

		case('y') :
			current_var_float = get_REPAIR_RANGE();
			new_float_var = *current_var_float +increment_float;
			if (new_float_var > 1.0f){
				new_float_var = *current_var_float;
			}
			set_REPAIR_RANGE(&new_float_var);
			std::cout<<"y: REPAIR_RANGE: start="<<*current_var_float<<":New value="<< new_float_var<<std::endl;
			break;

		case('u') :
			current_var_float = get_DAMAGE_DETECTION_RANGE();
			new_float_var = *current_var_float -increment_float;
			if (new_float_var < 0.0f){
				new_float_var = *current_var_float;
			}
			set_DAMAGE_DETECTION_RANGE(&new_float_var);
			std::cout<<"u: DAMAGE_DETECTION_RANGE: start="<<*current_var_float<<":New value="<< new_float_var<<std::endl;
			break;

		case('i') :
			current_var_float = get_DAMAGE_DETECTION_RANGE();
			new_float_var = *current_var_float +increment_float;
			if (new_float_var > 1.0f){
				new_float_var = *current_var_float;
			}
			set_DAMAGE_DETECTION_RANGE(&new_float_var);
			std::cout<<"i: DAMAGE_DETECTION_RANGE: start="<<*current_var_float<<":New value="<< new_float_var<<std::endl;
			break;

		case('o') :

			current_var_int = get_REPAIR_RATE();
			new_var_int = *current_var_int - increment_int;
			if (new_float_var < 0.0f){
				new_float_var = *current_var_int;
			}
			set_REPAIR_RATE(&new_var_int);
			std::cout<<"o: REPAIR_RATE: start="<<*current_var_int<<":New value="<< new_var_int<<std::endl;
			break;

		case('p') :
			current_var_int = get_REPAIR_RATE();
			new_var_int = *current_var_int + increment_int;
			if (new_float_var > 1.0f){
				new_float_var = *current_var_int;
			}
			set_REPAIR_RATE(&new_var_int);
			std::cout<<"p: REPAIR_RATE: start="<<*current_var_int<<":New value="<< new_var_int<<std::endl;
			break;

		case('h') :
			std::cout<<"\nHelp for keyboard options:\n"<<std::endl;
			std::cout<<"q: TISSUE_DAMAGE_PROB - "<<increment_float<<std::endl;
			std::cout<<"w: TISSUE_DAMAGE_PROB + "<<increment_float<<std::endl;
			std::cout<<"e: QUIESCENT_MIGRATION_SCALE - "<<increment_float<<std::endl;
			std::cout<<"r: QUIESCENT_MIGRATION_SCALE + "<<increment_float<<std::endl;
			std::cout<<"t: REPAIR_RANGE - "<<increment_float<<std::endl;
			std::cout<<"y: REPAIR_RANGE + "<<increment_float<<std::endl;
			std::cout<<"u: DAMAGE_DETECTION_RANGE - "<<increment_float<<std::endl;
			std::cout<<"i: DAMAGE_DETECTION_RANGE + "<<increment_float<<std::endl;
			std::cout<<"o: REPAIR_RATE - "<<increment_int<<std::endl;
			std::cout<<"p: REPAIR_RATE + "<<increment_int<<std::endl;
			break;
	}

}


//void specialKeyboard(int key, int x, int y)
//{
//	if (menuDisplayed())
//	{
//		switch(key) {
//			case(GLUT_KEY_DOWN):
//			{
//				handleDownKey();
//				break;
//			}
//			case(GLUT_KEY_UP):
//			{
//				handleUpKey();
//				break;
//			}
//			case(GLUT_KEY_LEFT):
//			{
//				handleLeftKey();
//				break;
//			}
//			case(GLUT_KEY_RIGHT):
//			{
//				handleRightKey();
//				break;
//			}
//			default:
//			{
//				break;
//			}
//		}
//	}
//}
