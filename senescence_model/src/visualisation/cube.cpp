

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <GL/glew.h>
#include <GL/freeglut.h>

#include <glm/gtc/type_ptr.hpp>
#include "header.h"


#include "BufferObjects.h"
//#include "PedestrianPopulation.h"
//#include "OBJModel.h"
//#include "CustomVisualisation.h"

//cuda graphics resources
cudaGraphicsResource_t TissueBlock_default_cgr;
cudaGraphicsResource_t Fibroblast_Quiescent_cgr;
cudaGraphicsResource_t Fibroblast_Repair_cgr;


extern int get_agent_TissueBlock_MAX_count();

struct cubeShader
{
    GLuint program;
    glm::vec3 colourLocation;
    glm::vec3 positionLocation;
};

struct bufferObjects
{
    GLuint vertexBuffer;
    GLuint fibroblastRepairTexBuffer;
    GLuint fibroblastQuiescentTexBuffer;
    GLuint tissueBlockTex;
};

cubeShader createCubeShader();
void renderCube();
void initCubes();




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

static const char cubeShaderSource[] =
        {
            R"glsl(
            #version 420 core
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

static const char* fragmentSource =
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

bufferObjects createCubeBufferObjects()
{
    GLuint cube_vbo;
    createVBO(&cube_vbo, GL_ARRAY_BUFFER, sizeof(vertices));
    glBindBuffer(GL_ARRAY_BUFFER, cube_vbo);

    // create tbo
    GLuint cube_tbo;
    GLuint instances_data_tex;

    GLuint TissueBlock_default_tbo;
    GLuint Fibroblast_Quiescent_tbo;
    GLuint Fibroblast_Repair_tbo;

    GLuint TissueBlock_default_displacementTex;
    GLuint Fibroblast_Quiescent_displacementTex;
    GLuint Fibroblast_Repair_displacementTex;


    createTBO(&TissueBlock_default_cgr, &TissueBlock_default_tbo, &TissueBlock_default_displacementTex, 1000 * sizeof( glm::vec4));

    createTBO(&Fibroblast_Quiescent_cgr, &Fibroblast_Quiescent_tbo, &Fibroblast_Quiescent_displacementTex, 1000 * sizeof( glm::vec4));

    createTBO(&Fibroblast_Repair_cgr, &Fibroblast_Repair_tbo, &Fibroblast_Repair_displacementTex, 1000 * sizeof( glm::vec4));

//    registerBO(*instances_data_cgr, &instances_data_cgr);

    glBindBuffer(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    bufferObjects bo = bufferObjects();
    bo.fibroblastQuiescentTexBuffer = Fibroblast_Quiescent_tbo;
    bo.fibroblastRepairTexBuffer = Fibroblast_Repair_tbo;
    bo.tissueBlockTex = TissueBlock_default_tbo;


    return bo;
}


cubeShader createCubeShader()
{
    int status;

    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &cubeShaderSource, NULL0);
    glCompileShader(vertexShader);

    GLuint fragShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragShader, 1, &fragmentSource, NULL)
    glCompileShader(fragShader);

    GLuint program = glCreateProgram();
    glAttachShader(vertexShader);
    glAttachShader(fragShader);
    glLinkProgram(program);

    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &status);
    if (status == GL_FALSE){
        char data[1024];
        int len;
        printf("ERROR: Vertex Shader Compilation Error\n");
        glGetShaderInfoLog(vertexShader, 1024, &len, data);
        printf("%s", data);
    }
    glGetShaderiv(fragShader, GL_COMPILE_STATUS, &status);
    if (status == GL_FALSE){
        char data[1024];
        int len;
        printf("ERROR: Fragment Shader Compilation Error\n");
        glGetShaderInfoLog(fragShader, 1024, &len, data);
        printf("%s", data);
    }

    glGetProgramiv(program, GL_LINK_STATUS, &status);
    if (status == GL_FALSE){
        printf("ERROR: Shader Program Link Error\n");
    }

    shader = cubeShader;
    shader.program = program;
    shader.colourLocation = glGetUniformLocation(program, "colour");
    shader.positionLocation = glGetUniformLocation(program, "position");
    return shader
}



void renderCube(
        GLuint* instances_data1_tbo,
        cudaGraphicsResource_t* p_instances_data1_cgr)
{

    bufferObjects bo = createCubeBufferObjects();
    cubeShader shader = createCubeShader();

    int i;
    int count=0;
    // args: tbo and cgr objs
    generate_cube_instances(instances_data1_tbo, p_instances_data1_cgr)
    glUseProgram(shader.program)

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(
            GL_TEXTURE_BUFFER_EXT,
            bo.tissueBlockTex);
//    glActiveTexture(GL_TEXTURE1);
//    glBindTexture(
//            GL_TEXTURE_BUFFER_EXT,
//            bo.fibroblastQuiescentTexBuffer);
//    glActiveTexture(GL_TEXTURE2);
//    glBindTexture(
//            GL_TEXTURE_BUFFER_EXT,
//            bo.fibroblastRepairTexBuffer);

    GLuint mapIndexLoc = glGetAttribLocation(shader.program, "mapIndex");
    GLuint positionLoc = glGetAttribLocation(shader.program, "position");

    GLuint displacementMapLoc = glGetUniformLocation(shader.program, "displacementMap");
    GLuint modelLoc = glGetUniformLocation(shader.program, 'model');
    GLuint viewLoc = glGetUniformLocation(shader.program, 'view');
    GLuint projLoc = glGetUniformLocation(shader.program, 'proj');

    glm::mat4 model = glm::mat4(1.0f);
    glm::mat4 view = glm::mat4(1.0f);
    glm::mat4 proj = glm::mat4(1.0f);

    for (i=0; i<get_agent_TissueBlock_MAX_count(), i++)
    {
        glVertexAttrib1f(mapIndexLoc, float(i))
        count++

        glBindBuffer(GL_ARRAY_BUFFER, bo.vertexBuffer);
        glEnableVertexAttribArray(positionLoc);
        glVertexAttribPointer(positionLoc, 3, GLfloat, GL_False, 0, 0)

        //uniforma
        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(proj));


        glEnableClientState(GL_VERTEX_ARRAY);

        glDrawElements(GL_TRIANGLES, 0, 36);
    }

}


















