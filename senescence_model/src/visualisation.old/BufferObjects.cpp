//
// Created by cwels on 13/10/2018.
//

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
#include <stdlib.h>
#include <stdio.h>

#include "BufferObjects.h"
#include "custom_visualisation.h"



void createVBO(GLuint* vbo, GLenum target, GLuint size)
{
    glGenBuffers( 1, vbo);
    glBindBuffer( target, *vbo);

    glBufferData( target, size, 0, GL_STATIC_DRAW);

    glBindBuffer( target, 0);

    checkGLError();
}

void deleteVBO( GLuint* vbo)
{
    glBindBuffer( 1, *vbo);
    glDeleteBuffers( 1, vbo);

    *vbo = 0;
}

void createTBO(
        cudaGraphicsResource_t* cudaResource,
        GLuint* tbo,
        GLuint* tex,
        GLuint size)
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
    cudaGraphicsGLRegisterBuffer(cudaResource, *tbo, cudaGraphicsMapFlagsWriteDiscard);

    checkGLError();
}

void deleteTBO( GLuint* tbo)
{
    glBindBuffer( 1, *tbo);
    glDeleteBuffers( 1, tbo);

    *tbo = 0;
}
void checkGLError(){
    int Error;
    if((Error = glGetError()) != GL_NO_ERROR)
    {
        const char* Message = (const char*)gluErrorString(Error);
        fprintf(stderr, "OpenGL Error : %s\n", Message);
    }
}
