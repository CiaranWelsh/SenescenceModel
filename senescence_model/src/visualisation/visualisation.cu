// modified by Ciaran welsh
/*
 * FLAME GPU v 1.5.X for CUDA 9
 * Copyright University of Sheffield.
 * Original Author: Dr Paul Richmond (user contributions tracked on https://github.com/FLAMEGPU/FLAMEGPU)
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


// includes, project
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <iostream>

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_gl_interop.h>
	    
#include "header.h"

#define FOVY 45.0


#ifndef __VISUALISATION_H
#define __VISUALISATION_H

// constants
const unsigned int WINDOW_WIDTH = 1000;
const unsigned int WINDOW_HEIGHT = 750;

//frustrum
const float NEAR_CLIP = 0.1;
const float FAR_CLIP = 100;

//Circle model fidelity
const int SPHERE_SLICES = 20;
const int SPHERE_STACKS = 20;
const float SPHERE_RADIUS = 0.025f;

const int CUBE_SLICES = 20;
const int CUBE_STACKS = 20;


//Viewing Distance
const float VIEW_DISTANCE = 1.25;

//light position
GLfloat LIGHT_POSITION[] = {10.0f, 10.0f, 10.0f, 1.0f};

#endif //__VISUALISATION_H



// bo variables
GLuint cubeVerts;
GLuint sphereVerts;

GLuint sphereNormals;
GLuint cubeNormals;

//Simulation output buffers/textures

cudaGraphicsResource_t TissueBlock_default_cgr;
GLuint TissueBlock_default_tbo;
GLuint TissueBlock_default_displacementTex;

cudaGraphicsResource_t Fibroblast_Quiescent_cgr;
GLuint Fibroblast_Quiescent_tbo;
GLuint Fibroblast_Quiescent_displacementTex;

cudaGraphicsResource_t Fibroblast_Repair_cgr;
GLuint Fibroblast_Repair_tbo;
GLuint Fibroblast_Repair_displacementTex;


// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -VIEW_DISTANCE;

// keyboard controls
#if defined(PAUSE_ON_START)
bool paused = true;
#else
bool paused = false;
#endif

// vertex Shader
GLuint vertexShader;
GLuint fragmentShader;
GLuint shaderProgram;
GLuint vs_displacementMap;
GLuint vs_mapIndex;



//timer
cudaEvent_t start, stop;
const int display_rate = 50;
int frame_count;
float frame_time = 0.0;

#ifdef SIMULATION_DELAY
//delay
int delay_count = 0;
#endif

// prototypes
int initGL();
void initShader(const char*, const char*);
void createVBO( GLuint* vbo, GLuint size);
void deleteVBO( GLuint* vbo);
void createTBO( cudaGraphicsResource_t* cudaResource, GLuint* tbo, GLuint* tex, GLuint size);
void deleteTBO( cudaGraphicsResource_t* cudaResource, GLuint* tbo);
void setVertexBufferData();
void reshape(int width, int height);
void display();
void close();
void keyboard( unsigned char key, int x, int y);
void special(int key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void runCuda();
void checkGLError();

/* Error check function for safe CUDA API calling */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/* Error check function for post CUDA Kernel calling */
#define gpuErrchkLaunch() { gpuLaunchAssert(__FILE__, __LINE__); }
inline void gpuLaunchAssert(const char *file, int line, bool abort=true)
{
	gpuAssert( cudaPeekAtLastError(), file, line );
#ifdef _DEBUG
	gpuAssert( cudaDeviceSynchronize(), file, line );
#endif

}
const char fibroblastVertexShaderSource[] =
{
	"#extension GL_EXT_gpu_shader4 : enable										\n"
	"uniform samplerBuffer displacementMap;										\n"
	"attribute in float mapIndex;												\n"
	"varying vec3 normal, lightDir;												\n"
	"varying vec4 colour;														\n"
    "void main()																\n"
    "{																			\n"
	"	vec4 position = gl_Vertex;											    \n"
	"	vec4 lookup = texelFetchBuffer(displacementMap, (int)mapIndex);		    \n"
    "	if (lookup.w > 0)   	                								\n"
	"		colour = vec4(1.0, 1.0, 1.0, 0.0);								    \n"
    "	else if (lookup.w > 6.5)	               								\n"
	"		colour = vec4(0.0, 0.0, 1.0, 0.0);								    \n"
    "	else if (lookup.w > 5.5)	                							\n"
	"		colour = vec4(1.0, 0.0, 1.0, 0.0);								    \n"
	"	else if (lookup.w > 4.5)	                							\n"
	"		colour = vec4(0.0, 1.0, 1.0, 0.0);								    \n"
    "	else if (lookup.w > 3.5)	                							\n"
	"		colour = vec4(1.0, 1.0, 0.0, 0.0);								    \n"
	"	else if (lookup.w > 2.5)	                							\n"
	"		colour = vec4(0.518, 0.353, 0.02, 0.0);						    	\n"
	"	else if (lookup.w > 1.5)	                							\n"
	"		colour = vec4(0.0, 1.0, 0.0, 0.0);								    \n"
    "	else if (lookup.w > 1.0)	                							\n"
	"		colour = vec4(1.0, 0.0, 0.0, 0.0);								    \n"
    "	                	                							\n"
//	"	colour = vec4(1 - (lookup.w / 100.0), 0.0, 0.0, 0.0);								    \n"
	"																    		\n"
	"	lookup.w = 1.0;												    		\n"
	"	position += lookup;											    		\n"
	"   gl_Position = gl_ModelViewProjectionMatrix * position;		    		\n"
	"																			\n"
	"	vec3 mvVertex = vec3(gl_ModelViewMatrix * position);			    	\n"
	"	lightDir = vec3(gl_LightSource[0].position.xyz - mvVertex);				\n"
	"	normal = gl_NormalMatrix * gl_Normal;									\n"
    "}																			\n"
};

const char tissueVertexShaderSource[] =
{
	"#extension GL_EXT_gpu_shader4 : enable										\n"
	"uniform samplerBuffer displacementMap;										\n"
	"attribute in float mapIndex;												\n"
	"varying vec3 normal, lightDir;												\n"
	"varying vec4 colour;														\n"
    "void main()																\n"
    "{																			\n"
	"	vec4 position = gl_Vertex;											    \n"
	"	vec4 lookup = texelFetchBuffer(displacementMap, (int)mapIndex);		    \n"
//    "	if (lookup.w > 7.5)	                								\n"
//	"		colour = vec4(0.518, 0.353, 0.02, 0.0);						    	\n"
//    "	else if (lookup.w > 6.5)	               								\n"
//	"		colour = vec4(1.0, 1.0, 1.0, 0.0);								    \n"
//    "	else if (lookup.w > 5.5)	                							\n"
//	"		colour = vec4(1.0, 0.0, 1.0, 0.0);								    \n"
//	"	else if (lookup.w > 4.5)	                							\n"
//	"		colour = vec4(0.0, 1.0, 1.0, 0.0);								    \n"
//    "	else if (lookup.w > 3.5)	                							\n"
//	"		colour = vec4(1.0, 1.0, 0.0, 0.0);								    \n"
//	"	else if (lookup.w > 2.5)	                							\n"
//	"		colour = vec4(0.0, 0.0, 1.0, 0.0);								    \n"
//	"	else if (lookup.w > 1.5)	                							\n"
//	"		colour = vec4(0.0, 1.0, 0.0, 0.0);								    \n"
//    "	else if (lookup.w > 1.0)	                							\n"
//	"		colour = vec4(1.0, 0.0, 0.0, 0.0);								    \n"
//    "	else                      	                							\n"
	"	colour = vec4(1 - (lookup.w / 100.0), 0.0, 0.0, 0.0);								    \n"
	"																    		\n"
	"	lookup.w = 1.0;												    		\n"
	"	position += lookup;											    		\n"
	"   gl_Position = gl_ModelViewProjectionMatrix * position;		    		\n"
	"																			\n"
	"	vec3 mvVertex = vec3(gl_ModelViewMatrix * position);			    	\n"
	"	lightDir = vec3(gl_LightSource[0].position.xyz - mvVertex);				\n"
	"	normal = gl_NormalMatrix * gl_Normal;									\n"
    "}																			\n"
};

const char fragmentShaderSource[] =
{
	"varying vec3 normal, lightDir;												\n"
	"varying vec4 colour;														\n"
	"void main (void)															\n"
	"{																			\n"
	"	// Defining The Material Colors											\n"
	"	vec4 AmbientColor = vec4(0.25, 0.0, 0.0, 1.0);							\n"
	"	vec4 DiffuseColor = colour;					            		    	\n"
	"																			\n"
	"	// Scaling The Input Vector To Length 1									\n"
	"	vec3 n_normal = normalize(normal);							        	\n"
	"	vec3 n_lightDir = normalize(lightDir);	                                \n"
	"																			\n"
	"	// Calculating The Diffuse Term And Clamping It To [0;1]				\n"
	"	float DiffuseTerm = clamp(dot(n_normal, n_lightDir), 0.0, 1.0);\n"
	"																			\n"
	"	// Calculating The Final Color											\n"
	"	gl_FragColor = AmbientColor + DiffuseColor * DiffuseTerm;				\n"
	"																			\n"
	"}																			\n"
};

//GPU Kernels

__global__ void output_TissueBlock_agent_to_VBO(xmachine_memory_TissueBlock_list* agents, glm::vec4* vbo, glm::vec3 centralise){

	//global thread index
	int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

	vbo[index].x = 0.0;
	vbo[index].y = 0.0;
	vbo[index].z = 0.0;

    vbo[index].x = agents->x[index] - centralise.x;
    vbo[index].y = agents->y[index] - centralise.y;
    vbo[index].z = agents->z[index] - centralise.z;
    vbo[index].w = agents->damage[index];
}

__global__ void output_Fibroblast_agent_to_VBO(xmachine_memory_Fibroblast_list* agents, glm::vec4* vbo, glm::vec3 centralise){

	//global thread index
	int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

	vbo[index].x = 0.0;
	vbo[index].y = 0.0;
	vbo[index].z = 0.0;

    vbo[index].x = agents->x[index] - centralise.x;
    vbo[index].y = agents->y[index] - centralise.y;
    vbo[index].z = agents->z[index] - centralise.z;
    vbo[index].w = 1.0;
}


void initVisualisation()
{
	// Create GL context
	int   argc   = 1;
        char glutString[] = "GLUT application";
	char *argv[] = {glutString, NULL};
	//char *argv[] = {"GLUT application", NULL};
	glutInit( &argc, argv);
	glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize( WINDOW_WIDTH, WINDOW_HEIGHT);
	glutCreateWindow( "FLAME GPU Visualiser");

	// initialize GL
	if( !initGL()) {
			return;
	}
	initShader(tissueVertexShaderSource, fragmentShaderSource);

	// register callbacks
	glutReshapeFunc( reshape);
	glutDisplayFunc( display);
    glutCloseFunc( close);
	glutKeyboardFunc( keyboard);
	glutSpecialFunc( special);
	glutMouseFunc( mouse);
	glutMotionFunc( motion);

	// Set the closing behaviour
    glutSetOption( GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS );


	// create VBO's
	createVBO( &sphereVerts, SPHERE_SLICES* (SPHERE_STACKS+1) * sizeof(glm::vec3));
	createVBO( &sphereNormals, SPHERE_SLICES* (SPHERE_STACKS+1) * sizeof (glm::vec3));
	setVertexBufferData();

	// create TBO
	createTBO(&TissueBlock_default_cgr, &TissueBlock_default_tbo, &TissueBlock_default_displacementTex, xmachine_memory_TissueBlock_MAX * sizeof( glm::vec4));

	createTBO(&Fibroblast_Quiescent_cgr, &Fibroblast_Quiescent_tbo, &Fibroblast_Quiescent_displacementTex, xmachine_memory_Fibroblast_MAX * sizeof( glm::vec4));

	createTBO(&Fibroblast_Repair_cgr, &Fibroblast_Repair_tbo, &Fibroblast_Repair_displacementTex, xmachine_memory_Fibroblast_MAX * sizeof( glm::vec4));


	//set shader uniforms
	glUseProgram(shaderProgram);

	//create a events for timer
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
}

void runVisualisation(){
	// Flush outputs prior to simulation loop.
	fflush(stdout);
	fflush(stderr);
	// start rendering mainloop
	glutMainLoop();
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runCuda()
{
	if(!paused){
#ifdef SIMULATION_DELAY
	delay_count++;
	if (delay_count == SIMULATION_DELAY){
		delay_count = 0;
		singleIteration();
	}
#else
	singleIteration();
#endif
	}

	//kernals sizes
	int threads_per_tile = 256;
	int tile_size;
	dim3 grid;
	dim3 threads;
	glm::vec3 centralise;

	//pointer
	glm::vec4 *dptr;


	if (get_agent_TissueBlock_default_count() > 0)
	{
		// map OpenGL buffer object for writing from CUDA
        size_t accessibleBufferSize = 0;
        gpuErrchk(cudaGraphicsMapResources(1, &TissueBlock_default_cgr));
		gpuErrchk(cudaGraphicsResourceGetMappedPointer( (void**)&dptr, &accessibleBufferSize, TissueBlock_default_cgr));
		//cuda block size
		tile_size = (int) ceil((float)get_agent_TissueBlock_default_count()/threads_per_tile);
		grid = dim3(tile_size, 1, 1);
		threads = dim3(threads_per_tile, 1, 1);

        //continuous variables
        centralise = getMaximumBounds() + getMinimumBounds();
        centralise /= 2;

		output_TissueBlock_agent_to_VBO<<< grid, threads>>>(get_device_TissueBlock_default_agents(), dptr, centralise);
		gpuErrchkLaunch();
		// unmap buffer object
        gpuErrchk(cudaGraphicsUnmapResources(1, &TissueBlock_default_cgr));
	}

	if (get_agent_Fibroblast_Quiescent_count() > 0)
	{
		// map OpenGL buffer object for writing from CUDA
        size_t accessibleBufferSize = 0;
        gpuErrchk(cudaGraphicsMapResources(1, &Fibroblast_Quiescent_cgr));
		gpuErrchk(cudaGraphicsResourceGetMappedPointer( (void**)&dptr, &accessibleBufferSize, Fibroblast_Quiescent_cgr));
		//cuda block size
		tile_size = (int) ceil((float)get_agent_Fibroblast_Quiescent_count()/threads_per_tile);
		grid = dim3(tile_size, 1, 1);
		threads = dim3(threads_per_tile, 1, 1);

        //continuous variables
        centralise = getMaximumBounds() + getMinimumBounds();
        centralise /= 2;

		output_Fibroblast_agent_to_VBO<<< grid, threads>>>(get_device_Fibroblast_Quiescent_agents(), dptr, centralise);
		gpuErrchkLaunch();
		// unmap buffer object
        gpuErrchk(cudaGraphicsUnmapResources(1, &Fibroblast_Quiescent_cgr));
	}

	if (get_agent_Fibroblast_Repair_count() > 0)
	{
		// map OpenGL buffer object for writing from CUDA
        size_t accessibleBufferSize = 0;
        gpuErrchk(cudaGraphicsMapResources(1, &Fibroblast_Repair_cgr));
		gpuErrchk(cudaGraphicsResourceGetMappedPointer( (void**)&dptr, &accessibleBufferSize, Fibroblast_Repair_cgr));
		//cuda block size
		tile_size = (int) ceil((float)get_agent_Fibroblast_Repair_count()/threads_per_tile);
		grid = dim3(tile_size, 1, 1);
		threads = dim3(threads_per_tile, 1, 1);

        //continuous variables
        centralise = getMaximumBounds() + getMinimumBounds();
        centralise /= 2;

		output_Fibroblast_agent_to_VBO<<< grid, threads>>>(get_device_Fibroblast_Repair_agents(), dptr, centralise);
		gpuErrchkLaunch();
		// unmap buffer object
        gpuErrchk(cudaGraphicsUnmapResources(1, &Fibroblast_Repair_cgr));
	}

}

////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////
int initGL()
{
	// initialize necessary OpenGL extensions
	glewInit();
	if (! glewIsSupported( "GL_VERSION_2_0 "
		"GL_ARB_pixel_buffer_object")) {
		fprintf( stderr, "ERROR: Support for necessary OpenGL extensions missing.\n");
		fflush( stderr);
		return 1;
	}

	// default initialization
	glClearColor( 1.0, 1.0, 1.0, 1.0);
	glEnable( GL_DEPTH_TEST);

	reshape(WINDOW_WIDTH, WINDOW_HEIGHT);
	checkGLError();

	//lighting
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);

	return 1;
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize GLSL Vertex Shader
////////////////////////////////////////////////////////////////////////////////
void initShader(const char* v, const char*f)
{
//	const char* v = tissueVertexShaderSource;
//	const char* f = fragmentShaderSource;

	//vertex shader
	vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShader, 1, &v, 0);
	glCompileShader(vertexShader);

	//fragment shader
	fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShader, 1, &f, 0);
	glCompileShader(fragmentShader);

	//program
	shaderProgram = glCreateProgram();
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);
	glLinkProgram(shaderProgram);

	// check for errors
	GLint status;
	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &status);
	if (status == GL_FALSE){
		printf("ERROR: Shader Compilation Error\n");
		char data[262144];
		int len;
		glGetShaderInfoLog(vertexShader, 262144, &len, data); 
		printf("%s", data);
	}
	glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &status);
	if (status == GL_FALSE){
		printf("ERROR: Shader Compilation Error\n");
		char data[262144];
		int len;
		glGetShaderInfoLog(fragmentShader, 262144, &len, data); 
		printf("%s", data);
	}
	glGetProgramiv(shaderProgram, GL_LINK_STATUS, &status);
	if (status == GL_FALSE){
		printf("ERROR: Shader Program Link Error\n");
	}

	// get shader variables
	vs_displacementMap = glGetUniformLocation(shaderProgram, "displacementMap");
	vs_mapIndex = glGetAttribLocation(shaderProgram, "mapIndex"); 
}

////////////////////////////////////////////////////////////////////////////////
//! Create VBO
////////////////////////////////////////////////////////////////////////////////
void createVBO(GLuint* vbo, GLuint size)
{
	// create buffer object
	glGenBuffers( 1, vbo);
	glBindBuffer( GL_ARRAY_BUFFER, *vbo);

	// initialize buffer object
	glBufferData( GL_ARRAY_BUFFER, size, 0, GL_STATIC_DRAW);

	glBindBuffer( GL_ARRAY_BUFFER, 0);

	checkGLError();
}

////////////////////////////////////////////////////////////////////////////////
//! Delete VBO
////////////////////////////////////////////////////////////////////////////////
void deleteVBO( GLuint* vbo)
{
	glBindBuffer( 1, *vbo);
	glDeleteBuffers( 1, vbo);

	*vbo = 0;
}

////////////////////////////////////////////////////////////////////////////////
//! Create TBO
////////////////////////////////////////////////////////////////////////////////
void createTBO(cudaGraphicsResource_t* cudaResource, GLuint* tbo, GLuint* tex, GLuint size)
{
	// create buffer object
	glGenBuffers( 1, tbo);
	glBindBuffer( GL_TEXTURE_BUFFER_EXT, *tbo);

	// initialize buffer object
	glBufferData( GL_TEXTURE_BUFFER_EXT, size, 0, GL_DYNAMIC_DRAW);

	//tex
	glGenTextures(1, tex);
	glBindTexture(GL_TEXTURE_BUFFER_EXT, *tex);
	glTexBufferEXT(GL_TEXTURE_BUFFER_EXT, GL_RGBA32F_ARB, *tbo); 
	glBindBuffer(GL_TEXTURE_BUFFER_EXT, 0);

    // register buffer object with CUDA
    gpuErrchk(cudaGraphicsGLRegisterBuffer(cudaResource, *tbo, cudaGraphicsMapFlagsWriteDiscard)); 

    checkGLError();
}

////////////////////////////////////////////////////////////////////////////////
//! Delete TBO
////////////////////////////////////////////////////////////////////////////////
void deleteTBO(cudaGraphicsResource_t* cudaResource,  GLuint* tbo)
{
    gpuErrchk(cudaGraphicsUnregisterResource(*cudaResource));
    *cudaResource = 0;

    glBindBuffer( 1, *tbo);
    glDeleteBuffers( 1, tbo);

	*tbo = 0;
}

////////////////////////////////////////////////////////////////////////////////
//! Set Sphere Vertex Data
////////////////////////////////////////////////////////////////////////////////

static void setSphereVertex(glm::vec3* data, int slice, int stack) {
	float PI = 3.14159265358f;
    
	float sl = 2*PI*slice/SPHERE_SLICES;
	float st = 2*PI*stack/SPHERE_STACKS;
 
	data->x = cos(st) * sin(sl) * SPHERE_RADIUS;
	data->y = sin(st) * sin(sl) * SPHERE_RADIUS;
	data->z = cos(sl) * SPHERE_RADIUS;
}


////////////////////////////////////////////////////////////////////////////////
//! Set Sphere Normal Data
////////////////////////////////////////////////////////////////////////////////

static void setSphereNormal(glm::vec3* data, int slice, int stack) {
	float PI = 3.14159265358f;
    
	float sl = 2*PI*slice/SPHERE_SLICES;
	float st = 2*PI*stack/SPHERE_STACKS;
 
	data->x = cos(st)*sin(sl);
	data->y = sin(st)*sin(sl);
	data->z = cos(sl);
}


////////////////////////////////////////////////////////////////////////////////
//! Set Vertex Buffer Data
////////////////////////////////////////////////////////////////////////////////
void setVertexBufferData()
{
	int slice, stack;
	int i;

	// upload vertex points data
	glBindBuffer(GL_ARRAY_BUFFER, sphereVerts);
	glm::vec3* verts =( glm::vec3*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
	i = 0;
	for (slice=0; slice<SPHERE_SLICES/2; slice++) {
		for (stack=0; stack<=SPHERE_STACKS; stack++) {
			setSphereVertex(&verts[i++], slice, stack);
			setSphereVertex(&verts[i++], slice+1, stack);
		}
    }
	glUnmapBuffer(GL_ARRAY_BUFFER);

	// upload vertex normal data
	glBindBuffer(GL_ARRAY_BUFFER, sphereNormals);
	glm::vec3* normals =( glm::vec3*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
	i = 0;
	for (slice=0; slice<SPHERE_SLICES/2; slice++) {
		for (stack=0; stack<=SPHERE_STACKS; stack++) {
			setSphereNormal(&normals[i++], slice, stack);
			setSphereNormal(&normals[i++], slice+1, stack);
		}
    }
	glUnmapBuffer(GL_ARRAY_BUFFER);
}




////////////////////////////////////////////////////////////////////////////////
//! Reshape callback
////////////////////////////////////////////////////////////////////////////////

void reshape(int width, int height){
	// viewport
	glViewport( 0, 0, width, height);

	// projection
	glMatrixMode( GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(FOVY, (GLfloat)width / (GLfloat) height, NEAR_CLIP, FAR_CLIP);

	checkGLError();
}


////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display()
{
	float millis;
	
	//CUDA start Timing
	cudaEventRecord(start);

	// run CUDA kernel to generate vertex positions
	runCuda();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// set view matrix
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();


	//zoom
	glTranslatef(0.0, 0.0, translate_z); 
	//move
	glRotatef(rotate_x, 1.0, 0.0, 0.0);
	glRotatef(rotate_y, 0.0, 0.0, 1.0);


	//Set light position
	glLightfv(GL_LIGHT0, GL_POSITION, LIGHT_POSITION);

	
	//Draw TissueBlock Agents in default state
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_BUFFER_EXT, TissueBlock_default_displacementTex);
	//loop
	for (int i=0; i< get_agent_TissueBlock_default_count(); i++){
		glVertexAttrib1f(vs_mapIndex, (float)i);
		
		//draw using vertex and attribute data on the gpu (fast)
		glEnableClientState(GL_VERTEX_ARRAY);
		glEnableClientState(GL_NORMAL_ARRAY);

		glBindBuffer(GL_ARRAY_BUFFER, sphereVerts);
		glVertexPointer(3, GL_FLOAT, 0, 0);

		glBindBuffer(GL_ARRAY_BUFFER, sphereNormals);
		glNormalPointer(GL_FLOAT, 0, 0);

		glDrawArrays(GL_TRIANGLE_STRIP, 0, SPHERE_SLICES * (SPHERE_STACKS+1));

		glDisableClientState(GL_NORMAL_ARRAY);
		glDisableClientState(GL_VERTEX_ARRAY);
	}
	
	//Draw Fibroblast Agents in Quiescent state
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_BUFFER_EXT, Fibroblast_Quiescent_displacementTex);
	//loop
	for (int i=0; i< get_agent_Fibroblast_Quiescent_count(); i++){
		glVertexAttrib1f(vs_mapIndex, (float)i);
		
		//draw using vertex and attribute data on the gpu (fast)
		glEnableClientState(GL_VERTEX_ARRAY);
		glEnableClientState(GL_NORMAL_ARRAY);

		glBindBuffer(GL_ARRAY_BUFFER, sphereVerts);
		glVertexPointer(3, GL_FLOAT, 0, 0);

		glBindBuffer(GL_ARRAY_BUFFER, sphereNormals);
		glNormalPointer(GL_FLOAT, 0, 0);

		glDrawArrays(GL_TRIANGLE_STRIP, 0, SPHERE_SLICES * (SPHERE_STACKS+1));

		glDisableClientState(GL_NORMAL_ARRAY);
		glDisableClientState(GL_VERTEX_ARRAY);
	}
	
	//Draw Fibroblast Agents in Repair state
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_BUFFER_EXT, Fibroblast_Repair_displacementTex);
	//loop
	for (int i=0; i< get_agent_Fibroblast_Repair_count(); i++){
		glVertexAttrib1f(vs_mapIndex, (float)i);
		
		//draw using vertex and attribute data on the gpu (fast)
		glEnableClientState(GL_VERTEX_ARRAY);
		glEnableClientState(GL_NORMAL_ARRAY);

		glBindBuffer(GL_ARRAY_BUFFER, sphereVerts);
		glVertexPointer(3, GL_FLOAT, 0, 0);

		glBindBuffer(GL_ARRAY_BUFFER, sphereNormals);
		glNormalPointer(GL_FLOAT, 0, 0);

		glDrawArrays(GL_TRIANGLE_STRIP, 0, SPHERE_SLICES * (SPHERE_STACKS+1));

		glDisableClientState(GL_NORMAL_ARRAY);
		glDisableClientState(GL_VERTEX_ARRAY);
	}
	

	//CUDA stop timing
	cudaEventRecord(stop);
	glFlush();
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&millis, start, stop);
  frame_time += millis;

	if(frame_count == display_rate){
		char title [100];
		sprintf(title, "Execution & Rendering Total: %f (FPS), %f milliseconds per frame", display_rate/(frame_time/1000.0f), frame_time/display_rate);
		glutSetWindowTitle(title);

		//reset
		frame_count = 0;
    frame_time = 0.0;
	}else{
		frame_count++;
	}


	glutSwapBuffers();
	glutPostRedisplay();

    // If an early exit has been requested, close the visualisation by leaving the main loop.
    if(get_exit_early()){
        glutLeaveMainLoop();
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Window close callback
////////////////////////////////////////////////////////////////////////////////
void close()
{
    // Cleanup visualisation memory
    deleteVBO( &sphereVerts);
    deleteVBO( &sphereNormals);
    
    deleteTBO( &TissueBlock_default_cgr, &TissueBlock_default_tbo);
    
    deleteTBO( &Fibroblast_Quiescent_cgr, &Fibroblast_Quiescent_tbo);
    
    deleteTBO( &Fibroblast_Repair_cgr, &Fibroblast_Repair_tbo);
    
    // Destroy cuda events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    // Call exit functions and clean up simulation memory
    cleanup();
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
    case(32):
        paused = !paused;
        break;
    // Esc == 27
	case(27) :
        // Set the flag indicating we wish to exit the simulation.
        set_exit_early();
        break;


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




void special(int key, int x, int y){
    switch (key)
    {
    case(GLUT_KEY_RIGHT) :
        singleIteration();
        fflush(stdout);
        break;
    }
}


////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////
void mouse(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN) {
		mouse_buttons |= 1<<button;
	} else if (state == GLUT_UP) {
		mouse_buttons = 0;
	}

	mouse_old_x = x;
	mouse_old_y = y;
	glutPostRedisplay();
}

void motion(int x, int y)
{
	float dx = (float) x - mouse_old_x;
	float dy = (float) y - mouse_old_y;

	if (mouse_buttons & 1) {
		rotate_x += dy * 0.2f;
		rotate_y += dx * 0.2f;
	} else if (mouse_buttons & 4) {
		translate_z += dy * VIEW_DISTANCE * 0.001f;
	}

  mouse_old_x = x;
  mouse_old_y = y;
}

void checkGLError(){
  int Error;
  if((Error = glGetError()) != GL_NO_ERROR)
  {
    const char* Message = (const char*)gluErrorString(Error);
    fprintf(stderr, "OpenGL Error : %s\n", Message);
  }
}
