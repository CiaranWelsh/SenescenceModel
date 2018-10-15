//
// Created by cwels on 14/10/2018.
//

#ifndef FIRSTFLAMEPROJECT_CUBE_H
#define FIRSTFLAMEPROJECT_CUBE_H
#include <cuda_gl_interop.h>
#include "BufferObjects.h"

extern int get_agent_TissueBlock_MAX_count();
extern void generate_cube_instances(
        GLuint* instances_data1_tbo,
        cudaGraphicsResource_t * p_instances_data1_cgr
        );

struct cubeShader
{
    GLuint program;
    GLuint colourLocation;
    GLuint positionLocation;
};

struct bufferObjects
{
    GLuint vertexBuffer;
    GLuint fibroblastRepairTexBuffer;
    GLuint fibroblastQuiescentTexBuffer;
    GLuint tissueBlockTex;
};

cubeShader createCubeShader();
extern void renderCube();
void initCubes();

extern bufferObjects createCubeBufferObjects();


static const float vertices[] =
        {
                -1.0f,-1.0f,-1.0f, // triangle 1 : begin
                -1.0f,-1.0f, 1.0f,
                -1.0f, 1.0f, 1.0f, // triangle 1 : end
                1.0f, 1.0f,-1.0f, // triangle 2 : begin
                -1.0f,-1.0f,-1.0f,
                -1.0f, 1.0f,-1.0f, // triangle 2 : end
                1.0f,-1.0f, 1.0f,
                -1.0f,-1.0f,-1.0f,
                1.0f,-1.0f,-1.0f,
                1.0f, 1.0f,-1.0f,
                1.0f,-1.0f,-1.0f,
                -1.0f,-1.0f,-1.0f,
                -1.0f,-1.0f,-1.0f,
                -1.0f, 1.0f, 1.0f,
                -1.0f, 1.0f,-1.0f,
                1.0f,-1.0f, 1.0f,
                -1.0f,-1.0f, 1.0f,
                -1.0f,-1.0f,-1.0f,
                -1.0f, 1.0f, 1.0f,
                -1.0f,-1.0f, 1.0f,
                1.0f,-1.0f, 1.0f,
                1.0f, 1.0f, 1.0f,
                1.0f,-1.0f,-1.0f,
                1.0f, 1.0f,-1.0f,
                1.0f,-1.0f,-1.0f,
                1.0f, 1.0f, 1.0f,
                1.0f,-1.0f, 1.0f,
                1.0f, 1.0f, 1.0f,
                1.0f, 1.0f,-1.0f,
                -1.0f, 1.0f,-1.0f,
                1.0f, 1.0f, 1.0f,
                -1.0f, 1.0f,-1.0f,
                -1.0f, 1.0f, 1.0f,
                1.0f, 1.0f, 1.0f,
                -1.0f, 1.0f, 1.0f,
                1.0f,-1.0f, 1.0f
        };

static const GLchar* cubeShaderSource =
        {
        R"glsl(
            #version 420 core
            #extension EXT_gpu_shader4 : require
            in vec3 position;
            in float mapIndex;
            //in vec3 colour;

            uniform samplerBuffer displacementMap;

            uniform mat4 model;
            uniform mat4 view;
            uniform mat4 proj;

            out vec4 colour;
            out vec3 position;

            void main()
            {
                vec4 position = gl_Vertex;
                vec4 lookup = texelFetchBuffers(displacementMap, (int)mapIndex);
                colour = vec4(1 - (lookup.w / 100.0), 0.0, 0.0, 0.15);

                lookup.w = 1.0;
                position += lookup;
                gl_Position = proj * view * model * vec4(position, 0.3);

            };
            )glsl"
        };

static const  GLchar* fragmentSource =
        {
        R"glsl(
            #version 420 core
            in vec3 Colour;

            out vec4 outColour;
            void main()
            {

                outColour= vec4(Colour, 1.0);
            }
            )glsl"
        };















#endif //FIRSTFLAMEPROJECT_CUBE_H
