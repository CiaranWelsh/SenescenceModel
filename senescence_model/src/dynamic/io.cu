
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


#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <limits.h>
#include <algorithm>
#include <string>
#include <vector>



#ifdef _WIN32
#define strtok_r strtok_s
#endif

// include header
#include "header.h"

glm::vec3 agent_maximum;
glm::vec3 agent_minimum;

int fpgu_strtol(const char* str){
    return (int)strtol(str, NULL, 0);
}

unsigned int fpgu_strtoul(const char* str){
    return (unsigned int)strtoul(str, NULL, 0);
}

long long int fpgu_strtoll(const char* str){
    return strtoll(str, NULL, 0);
}

unsigned long long int fpgu_strtoull(const char* str){
    return strtoull(str, NULL, 0);
}

double fpgu_strtod(const char* str){
    return strtod(str, NULL);
}

float fgpu_atof(const char* str){
    return (float)atof(str);
}


//templated class function to read array inputs from supported types
template <class T>
void readArrayInput( T (*parseFunc)(const char*), char* buffer, T *array, unsigned int expected_items){
    unsigned int i = 0;
    const char s[2] = ",";
    char * token;
    char * end_str;

    token = strtok_r(buffer, s, &end_str);
    while (token != NULL){
        if (i>=expected_items){
            printf("Error: variable array has too many items, expected %d!\n", expected_items);
            exit(EXIT_FAILURE);
        }
        
        array[i++] = (T)parseFunc(token);
        
        token = strtok_r(NULL, s, &end_str);
    }
    if (i != expected_items){
        printf("Error: variable array has %d items, expected %d!\n", i, expected_items);
        exit(EXIT_FAILURE);
    }
}

//templated class function to read array inputs from supported types
template <class T, class BASE_T, unsigned int D>
void readArrayInputVectorType( BASE_T (*parseFunc)(const char*), char* buffer, T *array, unsigned int expected_items){
    unsigned int i = 0;
    const char s[2] = "|";
    char * token;
    char * end_str;

    token = strtok_r(buffer, s, &end_str);
    while (token != NULL){
        if (i>=expected_items){
            printf("Error: Agent memory array has too many items, expected %d!\n", expected_items);
            exit(EXIT_FAILURE);
        }
        
        //read vector type as an array
        T vec;
        readArrayInput<BASE_T>(parseFunc, token, (BASE_T*) &vec, D);
        array[i++] = vec;
        
        token = strtok_r(NULL, s, &end_str);
    }
    if (i != expected_items){
        printf("Error: Agent memory array has %d items, expected %d!\n", i, expected_items);
        exit(EXIT_FAILURE);
    }
}

void saveIterationData(char* outputpath, int iteration_number, xmachine_memory_A_list* h_As_moving_A, xmachine_memory_A_list* d_As_moving_A, int h_xmachine_memory_A_moving_A_count,xmachine_memory_A_list* h_As_change_direction_A, xmachine_memory_A_list* d_As_change_direction_A, int h_xmachine_memory_A_change_direction_A_count,xmachine_memory_A_list* h_As_get_going_again_A, xmachine_memory_A_list* d_As_get_going_again_A, int h_xmachine_memory_A_get_going_again_A_count,xmachine_memory_B_list* h_Bs_moving_B, xmachine_memory_B_list* d_Bs_moving_B, int h_xmachine_memory_B_moving_B_count,xmachine_memory_B_list* h_Bs_change_direction_B, xmachine_memory_B_list* d_Bs_change_direction_B, int h_xmachine_memory_B_change_direction_B_count,xmachine_memory_B_list* h_Bs_get_going_again_B, xmachine_memory_B_list* d_Bs_get_going_again_B, int h_xmachine_memory_B_get_going_again_B_count)
{
    PROFILE_SCOPED_RANGE("saveIterationData");
	cudaError_t cudaStatus;
	
	//Device to host memory transfer
	
	cudaStatus = cudaMemcpy( h_As_moving_A, d_As_moving_A, sizeof(xmachine_memory_A_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr,"Error Copying A Agent moving_A State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
	cudaStatus = cudaMemcpy( h_As_change_direction_A, d_As_change_direction_A, sizeof(xmachine_memory_A_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr,"Error Copying A Agent change_direction_A State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
	cudaStatus = cudaMemcpy( h_As_get_going_again_A, d_As_get_going_again_A, sizeof(xmachine_memory_A_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr,"Error Copying A Agent get_going_again_A State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
	cudaStatus = cudaMemcpy( h_Bs_moving_B, d_Bs_moving_B, sizeof(xmachine_memory_B_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr,"Error Copying B Agent moving_B State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
	cudaStatus = cudaMemcpy( h_Bs_change_direction_B, d_Bs_change_direction_B, sizeof(xmachine_memory_B_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr,"Error Copying B Agent change_direction_B State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
	cudaStatus = cudaMemcpy( h_Bs_get_going_again_B, d_Bs_get_going_again_B, sizeof(xmachine_memory_B_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr,"Error Copying B Agent get_going_again_B State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
	
	/* Pointer to file */
	FILE *file;
	char data[100];

	sprintf(data, "%s%i.xml", outputpath, iteration_number);
	//printf("Writing iteration %i data to %s\n", iteration_number, data);
	file = fopen(data, "w");
    if(file == nullptr){
        printf("Error: Could not open file `%s` for output. Aborting.\n", data);
        exit(EXIT_FAILURE);
    }
    fputs("<states>\n<itno>", file);
    sprintf(data, "%i", iteration_number);
    fputs(data, file);
    fputs("</itno>\n", file);
    fputs("<environment>\n" , file);
    
	fputs("</environment>\n" , file);

	//Write each A agent to xml
	for (int i=0; i<h_xmachine_memory_A_moving_A_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>A</name>\n", file);
        
		fputs("<id>", file);
        sprintf(data, "%d", h_As_moving_A->id[i]);
		fputs(data, file);
		fputs("</id>\n", file);
        
		fputs("<x>", file);
        sprintf(data, "%f", h_As_moving_A->x[i]);
		fputs(data, file);
		fputs("</x>\n", file);
        
		fputs("<y>", file);
        sprintf(data, "%f", h_As_moving_A->y[i]);
		fputs(data, file);
		fputs("</y>\n", file);
        
		fputs("<z>", file);
        sprintf(data, "%f", h_As_moving_A->z[i]);
		fputs(data, file);
		fputs("</z>\n", file);
        
		fputs("<fx>", file);
        sprintf(data, "%f", h_As_moving_A->fx[i]);
		fputs(data, file);
		fputs("</fx>\n", file);
        
		fputs("<fy>", file);
        sprintf(data, "%f", h_As_moving_A->fy[i]);
		fputs(data, file);
		fputs("</fy>\n", file);
        
		fputs("<fz>", file);
        sprintf(data, "%f", h_As_moving_A->fz[i]);
		fputs(data, file);
		fputs("</fz>\n", file);
        
		fputs("<colour>", file);
        sprintf(data, "%d", h_As_moving_A->colour[i]);
		fputs(data, file);
		fputs("</colour>\n", file);
        
		fputs("</xagent>\n", file);
	}
	//Write each A agent to xml
	for (int i=0; i<h_xmachine_memory_A_change_direction_A_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>A</name>\n", file);
        
		fputs("<id>", file);
        sprintf(data, "%d", h_As_change_direction_A->id[i]);
		fputs(data, file);
		fputs("</id>\n", file);
        
		fputs("<x>", file);
        sprintf(data, "%f", h_As_change_direction_A->x[i]);
		fputs(data, file);
		fputs("</x>\n", file);
        
		fputs("<y>", file);
        sprintf(data, "%f", h_As_change_direction_A->y[i]);
		fputs(data, file);
		fputs("</y>\n", file);
        
		fputs("<z>", file);
        sprintf(data, "%f", h_As_change_direction_A->z[i]);
		fputs(data, file);
		fputs("</z>\n", file);
        
		fputs("<fx>", file);
        sprintf(data, "%f", h_As_change_direction_A->fx[i]);
		fputs(data, file);
		fputs("</fx>\n", file);
        
		fputs("<fy>", file);
        sprintf(data, "%f", h_As_change_direction_A->fy[i]);
		fputs(data, file);
		fputs("</fy>\n", file);
        
		fputs("<fz>", file);
        sprintf(data, "%f", h_As_change_direction_A->fz[i]);
		fputs(data, file);
		fputs("</fz>\n", file);
        
		fputs("<colour>", file);
        sprintf(data, "%d", h_As_change_direction_A->colour[i]);
		fputs(data, file);
		fputs("</colour>\n", file);
        
		fputs("</xagent>\n", file);
	}
	//Write each A agent to xml
	for (int i=0; i<h_xmachine_memory_A_get_going_again_A_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>A</name>\n", file);
        
		fputs("<id>", file);
        sprintf(data, "%d", h_As_get_going_again_A->id[i]);
		fputs(data, file);
		fputs("</id>\n", file);
        
		fputs("<x>", file);
        sprintf(data, "%f", h_As_get_going_again_A->x[i]);
		fputs(data, file);
		fputs("</x>\n", file);
        
		fputs("<y>", file);
        sprintf(data, "%f", h_As_get_going_again_A->y[i]);
		fputs(data, file);
		fputs("</y>\n", file);
        
		fputs("<z>", file);
        sprintf(data, "%f", h_As_get_going_again_A->z[i]);
		fputs(data, file);
		fputs("</z>\n", file);
        
		fputs("<fx>", file);
        sprintf(data, "%f", h_As_get_going_again_A->fx[i]);
		fputs(data, file);
		fputs("</fx>\n", file);
        
		fputs("<fy>", file);
        sprintf(data, "%f", h_As_get_going_again_A->fy[i]);
		fputs(data, file);
		fputs("</fy>\n", file);
        
		fputs("<fz>", file);
        sprintf(data, "%f", h_As_get_going_again_A->fz[i]);
		fputs(data, file);
		fputs("</fz>\n", file);
        
		fputs("<colour>", file);
        sprintf(data, "%d", h_As_get_going_again_A->colour[i]);
		fputs(data, file);
		fputs("</colour>\n", file);
        
		fputs("</xagent>\n", file);
	}
	//Write each B agent to xml
	for (int i=0; i<h_xmachine_memory_B_moving_B_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>B</name>\n", file);
        
		fputs("<id>", file);
        sprintf(data, "%d", h_Bs_moving_B->id[i]);
		fputs(data, file);
		fputs("</id>\n", file);
        
		fputs("<x>", file);
        sprintf(data, "%f", h_Bs_moving_B->x[i]);
		fputs(data, file);
		fputs("</x>\n", file);
        
		fputs("<y>", file);
        sprintf(data, "%f", h_Bs_moving_B->y[i]);
		fputs(data, file);
		fputs("</y>\n", file);
        
		fputs("<z>", file);
        sprintf(data, "%f", h_Bs_moving_B->z[i]);
		fputs(data, file);
		fputs("</z>\n", file);
        
		fputs("<fx>", file);
        sprintf(data, "%f", h_Bs_moving_B->fx[i]);
		fputs(data, file);
		fputs("</fx>\n", file);
        
		fputs("<fy>", file);
        sprintf(data, "%f", h_Bs_moving_B->fy[i]);
		fputs(data, file);
		fputs("</fy>\n", file);
        
		fputs("<fz>", file);
        sprintf(data, "%f", h_Bs_moving_B->fz[i]);
		fputs(data, file);
		fputs("</fz>\n", file);
        
		fputs("<colour>", file);
        sprintf(data, "%d", h_Bs_moving_B->colour[i]);
		fputs(data, file);
		fputs("</colour>\n", file);
        
		fputs("</xagent>\n", file);
	}
	//Write each B agent to xml
	for (int i=0; i<h_xmachine_memory_B_change_direction_B_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>B</name>\n", file);
        
		fputs("<id>", file);
        sprintf(data, "%d", h_Bs_change_direction_B->id[i]);
		fputs(data, file);
		fputs("</id>\n", file);
        
		fputs("<x>", file);
        sprintf(data, "%f", h_Bs_change_direction_B->x[i]);
		fputs(data, file);
		fputs("</x>\n", file);
        
		fputs("<y>", file);
        sprintf(data, "%f", h_Bs_change_direction_B->y[i]);
		fputs(data, file);
		fputs("</y>\n", file);
        
		fputs("<z>", file);
        sprintf(data, "%f", h_Bs_change_direction_B->z[i]);
		fputs(data, file);
		fputs("</z>\n", file);
        
		fputs("<fx>", file);
        sprintf(data, "%f", h_Bs_change_direction_B->fx[i]);
		fputs(data, file);
		fputs("</fx>\n", file);
        
		fputs("<fy>", file);
        sprintf(data, "%f", h_Bs_change_direction_B->fy[i]);
		fputs(data, file);
		fputs("</fy>\n", file);
        
		fputs("<fz>", file);
        sprintf(data, "%f", h_Bs_change_direction_B->fz[i]);
		fputs(data, file);
		fputs("</fz>\n", file);
        
		fputs("<colour>", file);
        sprintf(data, "%d", h_Bs_change_direction_B->colour[i]);
		fputs(data, file);
		fputs("</colour>\n", file);
        
		fputs("</xagent>\n", file);
	}
	//Write each B agent to xml
	for (int i=0; i<h_xmachine_memory_B_get_going_again_B_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>B</name>\n", file);
        
		fputs("<id>", file);
        sprintf(data, "%d", h_Bs_get_going_again_B->id[i]);
		fputs(data, file);
		fputs("</id>\n", file);
        
		fputs("<x>", file);
        sprintf(data, "%f", h_Bs_get_going_again_B->x[i]);
		fputs(data, file);
		fputs("</x>\n", file);
        
		fputs("<y>", file);
        sprintf(data, "%f", h_Bs_get_going_again_B->y[i]);
		fputs(data, file);
		fputs("</y>\n", file);
        
		fputs("<z>", file);
        sprintf(data, "%f", h_Bs_get_going_again_B->z[i]);
		fputs(data, file);
		fputs("</z>\n", file);
        
		fputs("<fx>", file);
        sprintf(data, "%f", h_Bs_get_going_again_B->fx[i]);
		fputs(data, file);
		fputs("</fx>\n", file);
        
		fputs("<fy>", file);
        sprintf(data, "%f", h_Bs_get_going_again_B->fy[i]);
		fputs(data, file);
		fputs("</fy>\n", file);
        
		fputs("<fz>", file);
        sprintf(data, "%f", h_Bs_get_going_again_B->fz[i]);
		fputs(data, file);
		fputs("</fz>\n", file);
        
		fputs("<colour>", file);
        sprintf(data, "%d", h_Bs_get_going_again_B->colour[i]);
		fputs(data, file);
		fputs("</colour>\n", file);
        
		fputs("</xagent>\n", file);
	}
	
	

	fputs("</states>\n" , file);
	
	/* Close the file */
	fclose(file);

}

void readInitialStates(char* inputpath, xmachine_memory_A_list* h_As, int* h_xmachine_memory_A_count,xmachine_memory_B_list* h_Bs, int* h_xmachine_memory_B_count)
{
    PROFILE_SCOPED_RANGE("readInitialStates");

	int temp = 0;
	int* itno = &temp;

	/* Pointer to file */
	FILE *file;
	/* Char and char buffer for reading file to */
	char c = ' ';
	const int bufferSize = 10000;
	char buffer[bufferSize];
	char agentname[1000];

	/* Pointer to x-memory for initial state data */
	/*xmachine * current_xmachine;*/
	/* Variables for checking tags */
	int reading, i;
	int in_tag, in_itno, in_xagent, in_name, in_comment;
    int in_A_id;
    int in_A_x;
    int in_A_y;
    int in_A_z;
    int in_A_fx;
    int in_A_fy;
    int in_A_fz;
    int in_A_colour;
    int in_B_id;
    int in_B_x;
    int in_B_y;
    int in_B_z;
    int in_B_fx;
    int in_B_fy;
    int in_B_fz;
    int in_B_colour;
    
    /* tags for environment global variables */
    int in_env;
	/* set agent count to zero */
	*h_xmachine_memory_A_count = 0;
	*h_xmachine_memory_B_count = 0;
	
	/* Variables for initial state data */
	int A_id;
	float A_x;
	float A_y;
	float A_z;
	float A_fx;
	float A_fy;
	float A_fz;
	int A_colour;
	int B_id;
	float B_x;
	float B_y;
	float B_z;
	float B_fx;
	float B_fy;
	float B_fz;
	int B_colour;

    /* Variables for environment variables */
    


	/* Initialise variables */
    agent_maximum.x = 0;
    agent_maximum.y = 0;
    agent_maximum.z = 0;
    agent_minimum.x = 0;
    agent_minimum.y = 0;
    agent_minimum.z = 0;
	reading = 1;
    in_comment = 0;
	in_tag = 0;
	in_itno = 0;
    in_env = 0;
    in_xagent = 0;
	in_name = 0;
	in_A_id = 0;
	in_A_x = 0;
	in_A_y = 0;
	in_A_z = 0;
	in_A_fx = 0;
	in_A_fy = 0;
	in_A_fz = 0;
	in_A_colour = 0;
	in_B_id = 0;
	in_B_x = 0;
	in_B_y = 0;
	in_B_z = 0;
	in_B_fx = 0;
	in_B_fy = 0;
	in_B_fz = 0;
	in_B_colour = 0;
	//set all A values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_A_MAX; k++)
	{	
		h_As->id[k] = 0;
		h_As->x[k] = 0;
		h_As->y[k] = 0;
		h_As->z[k] = 0;
		h_As->fx[k] = 0;
		h_As->fy[k] = 0;
		h_As->fz[k] = 0;
		h_As->colour[k] = 0;
	}
	
	//set all B values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_B_MAX; k++)
	{	
		h_Bs->id[k] = 0;
		h_Bs->x[k] = 0;
		h_Bs->y[k] = 0;
		h_Bs->z[k] = 0;
		h_Bs->fx[k] = 0;
		h_Bs->fy[k] = 0;
		h_Bs->fz[k] = 0;
		h_Bs->colour[k] = 0;
	}
	

	/* Default variables for memory */
    A_id = 0;
    A_x = 0;
    A_y = 0;
    A_z = 0;
    A_fx = 0;
    A_fy = 0;
    A_fz = 0;
    A_colour = 0;
    B_id = 0;
    B_x = 0;
    B_y = 0;
    B_z = 0;
    B_fx = 0;
    B_fy = 0;
    B_fz = 0;
    B_colour = 0;

    /* Default variables for environment variables */
    
    
    // If no input path was specified, issue a message and return.
    if(inputpath[0] == '\0'){
        printf("No initial states file specified. Using default values.\n");
        return;
    }
    
    // Otherwise an input path was specified, and we have previously checked that it is (was) not a directory. 
    
	// Attempt to open the non directory path as read only.
	file = fopen(inputpath, "r");
    
    // If the file could not be opened, issue a message and return.
    if(file == nullptr)
    {
      printf("Could not open input file %s. Continuing with default values\n", inputpath);
      return;
    }
    // Otherwise we can iterate the file until the end of XML is reached.
    size_t bytesRead = 0;
    i = 0;
	while(reading==1)
	{
        // If I exceeds our buffer size we must abort
        if(i >= bufferSize){
            fprintf(stderr, "Error: XML Parsing failed Tag name or content too long (> %d characters)\n", bufferSize);
            exit(EXIT_FAILURE);
        }

		/* Get the next char from the file */
		c = (char)fgetc(file);

        // Check if we reached the end of the file.
        if(c == EOF){
            // Break out of the loop. This allows for empty files(which may or may not be)
            break;
        }
        // Increment byte counter.
        bytesRead++;

        /*If in a  comment, look for the end of a comment */
        if(in_comment){

            /* Look for an end tag following two (or more) hyphens.
               To support very long comments, we use the minimal amount of buffer we can. 
               If we see a hyphen, store it and increment i (but don't increment i)
               If we see a > check if we have a correct terminating comment
               If we see any other characters, reset i.
            */

            if(c == '-'){
                buffer[i] = c;
                i++;
            } else if(c == '>' && i >= 2){
                in_comment = 0;
                i = 0;
            } else {
                i = 0;
            }

            /*// If we see the end tag, check the preceding two characters for a close comment, if enough characters have been read for -->
            if(c == '>' && i >= 2 && buffer[i-1] == '-' && buffer[i-2] == '-'){
                in_comment = 0;
                buffer[0] = 0;
                i = 0;
            } else {
                // Otherwise just store it in the buffer so we can keep checking for close tags
                buffer[i] = c;
                i++;
            }*/
        }
		/* If the end of a tag */
		else if(c == '>')
		{
			/* Place 0 at end of buffer to make chars a string */
			buffer[i] = 0;

			if(strcmp(buffer, "states") == 0) reading = 1;
			if(strcmp(buffer, "/states") == 0) reading = 0;
			if(strcmp(buffer, "itno") == 0) in_itno = 1;
			if(strcmp(buffer, "/itno") == 0) in_itno = 0;
            if(strcmp(buffer, "environment") == 0) in_env = 1;
            if(strcmp(buffer, "/environment") == 0) in_env = 0;
			if(strcmp(buffer, "name") == 0) in_name = 1;
			if(strcmp(buffer, "/name") == 0) in_name = 0;
            if(strcmp(buffer, "xagent") == 0) in_xagent = 1;
			if(strcmp(buffer, "/xagent") == 0)
			{
				if(strcmp(agentname, "A") == 0)
				{
					if (*h_xmachine_memory_A_count > xmachine_memory_A_MAX){
						printf("ERROR: MAX Buffer size (%i) for agent A exceeded whilst reading data\n", xmachine_memory_A_MAX);
						// Close the file and stop reading
						fclose(file);
						exit(EXIT_FAILURE);
					}
                    
					h_As->id[*h_xmachine_memory_A_count] = A_id;
					h_As->x[*h_xmachine_memory_A_count] = A_x;//Check maximum x value
                    if(agent_maximum.x < A_x)
                        agent_maximum.x = (float)A_x;
                    //Check minimum x value
                    if(agent_minimum.x > A_x)
                        agent_minimum.x = (float)A_x;
                    
					h_As->y[*h_xmachine_memory_A_count] = A_y;//Check maximum y value
                    if(agent_maximum.y < A_y)
                        agent_maximum.y = (float)A_y;
                    //Check minimum y value
                    if(agent_minimum.y > A_y)
                        agent_minimum.y = (float)A_y;
                    
					h_As->z[*h_xmachine_memory_A_count] = A_z;//Check maximum z value
                    if(agent_maximum.z < A_z)
                        agent_maximum.z = (float)A_z;
                    //Check minimum z value
                    if(agent_minimum.z > A_z)
                        agent_minimum.z = (float)A_z;
                    
					h_As->fx[*h_xmachine_memory_A_count] = A_fx;
					h_As->fy[*h_xmachine_memory_A_count] = A_fy;
					h_As->fz[*h_xmachine_memory_A_count] = A_fz;
					h_As->colour[*h_xmachine_memory_A_count] = A_colour;
					(*h_xmachine_memory_A_count) ++;	
				}
				else if(strcmp(agentname, "B") == 0)
				{
					if (*h_xmachine_memory_B_count > xmachine_memory_B_MAX){
						printf("ERROR: MAX Buffer size (%i) for agent B exceeded whilst reading data\n", xmachine_memory_B_MAX);
						// Close the file and stop reading
						fclose(file);
						exit(EXIT_FAILURE);
					}
                    
					h_Bs->id[*h_xmachine_memory_B_count] = B_id;
					h_Bs->x[*h_xmachine_memory_B_count] = B_x;//Check maximum x value
                    if(agent_maximum.x < B_x)
                        agent_maximum.x = (float)B_x;
                    //Check minimum x value
                    if(agent_minimum.x > B_x)
                        agent_minimum.x = (float)B_x;
                    
					h_Bs->y[*h_xmachine_memory_B_count] = B_y;//Check maximum y value
                    if(agent_maximum.y < B_y)
                        agent_maximum.y = (float)B_y;
                    //Check minimum y value
                    if(agent_minimum.y > B_y)
                        agent_minimum.y = (float)B_y;
                    
					h_Bs->z[*h_xmachine_memory_B_count] = B_z;//Check maximum z value
                    if(agent_maximum.z < B_z)
                        agent_maximum.z = (float)B_z;
                    //Check minimum z value
                    if(agent_minimum.z > B_z)
                        agent_minimum.z = (float)B_z;
                    
					h_Bs->fx[*h_xmachine_memory_B_count] = B_fx;
					h_Bs->fy[*h_xmachine_memory_B_count] = B_fy;
					h_Bs->fz[*h_xmachine_memory_B_count] = B_fz;
					h_Bs->colour[*h_xmachine_memory_B_count] = B_colour;
					(*h_xmachine_memory_B_count) ++;	
				}
				else
				{
					printf("Warning: agent name undefined - '%s'\n", agentname);
				}



				/* Reset xagent variables */
                A_id = 0;
                A_x = 0;
                A_y = 0;
                A_z = 0;
                A_fx = 0;
                A_fy = 0;
                A_fz = 0;
                A_colour = 0;
                B_id = 0;
                B_x = 0;
                B_y = 0;
                B_z = 0;
                B_fx = 0;
                B_fy = 0;
                B_fz = 0;
                B_colour = 0;
                
                in_xagent = 0;
			}
			if(strcmp(buffer, "id") == 0) in_A_id = 1;
			if(strcmp(buffer, "/id") == 0) in_A_id = 0;
			if(strcmp(buffer, "x") == 0) in_A_x = 1;
			if(strcmp(buffer, "/x") == 0) in_A_x = 0;
			if(strcmp(buffer, "y") == 0) in_A_y = 1;
			if(strcmp(buffer, "/y") == 0) in_A_y = 0;
			if(strcmp(buffer, "z") == 0) in_A_z = 1;
			if(strcmp(buffer, "/z") == 0) in_A_z = 0;
			if(strcmp(buffer, "fx") == 0) in_A_fx = 1;
			if(strcmp(buffer, "/fx") == 0) in_A_fx = 0;
			if(strcmp(buffer, "fy") == 0) in_A_fy = 1;
			if(strcmp(buffer, "/fy") == 0) in_A_fy = 0;
			if(strcmp(buffer, "fz") == 0) in_A_fz = 1;
			if(strcmp(buffer, "/fz") == 0) in_A_fz = 0;
			if(strcmp(buffer, "colour") == 0) in_A_colour = 1;
			if(strcmp(buffer, "/colour") == 0) in_A_colour = 0;
			if(strcmp(buffer, "id") == 0) in_B_id = 1;
			if(strcmp(buffer, "/id") == 0) in_B_id = 0;
			if(strcmp(buffer, "x") == 0) in_B_x = 1;
			if(strcmp(buffer, "/x") == 0) in_B_x = 0;
			if(strcmp(buffer, "y") == 0) in_B_y = 1;
			if(strcmp(buffer, "/y") == 0) in_B_y = 0;
			if(strcmp(buffer, "z") == 0) in_B_z = 1;
			if(strcmp(buffer, "/z") == 0) in_B_z = 0;
			if(strcmp(buffer, "fx") == 0) in_B_fx = 1;
			if(strcmp(buffer, "/fx") == 0) in_B_fx = 0;
			if(strcmp(buffer, "fy") == 0) in_B_fy = 1;
			if(strcmp(buffer, "/fy") == 0) in_B_fy = 0;
			if(strcmp(buffer, "fz") == 0) in_B_fz = 1;
			if(strcmp(buffer, "/fz") == 0) in_B_fz = 0;
			if(strcmp(buffer, "colour") == 0) in_B_colour = 1;
			if(strcmp(buffer, "/colour") == 0) in_B_colour = 0;
			
            /* environment variables */
            

			/* End of tag and reset buffer */
			in_tag = 0;
			i = 0;
		}
		/* If start of tag */
		else if(c == '<')
		{
			/* Place /0 at end of buffer to end numbers */
			buffer[i] = 0;
			/* Flag in tag */
			in_tag = 1;

			if(in_itno) *itno = atoi(buffer);
			if(in_name) strcpy(agentname, buffer);
			else if (in_xagent)
			{
				if(in_A_id){
                    A_id = (int) fpgu_strtol(buffer); 
                }
				if(in_A_x){
                    A_x = (float) fgpu_atof(buffer); 
                }
				if(in_A_y){
                    A_y = (float) fgpu_atof(buffer); 
                }
				if(in_A_z){
                    A_z = (float) fgpu_atof(buffer); 
                }
				if(in_A_fx){
                    A_fx = (float) fgpu_atof(buffer); 
                }
				if(in_A_fy){
                    A_fy = (float) fgpu_atof(buffer); 
                }
				if(in_A_fz){
                    A_fz = (float) fgpu_atof(buffer); 
                }
				if(in_A_colour){
                    A_colour = (int) fpgu_strtol(buffer); 
                }
				if(in_B_id){
                    B_id = (int) fpgu_strtol(buffer); 
                }
				if(in_B_x){
                    B_x = (float) fgpu_atof(buffer); 
                }
				if(in_B_y){
                    B_y = (float) fgpu_atof(buffer); 
                }
				if(in_B_z){
                    B_z = (float) fgpu_atof(buffer); 
                }
				if(in_B_fx){
                    B_fx = (float) fgpu_atof(buffer); 
                }
				if(in_B_fy){
                    B_fy = (float) fgpu_atof(buffer); 
                }
				if(in_B_fz){
                    B_fz = (float) fgpu_atof(buffer); 
                }
				if(in_B_colour){
                    B_colour = (int) fpgu_strtol(buffer); 
                }
				
            }
            else if (in_env){
            
            }
		/* Reset buffer */
			i = 0;
		}
		/* If in tag put read char into buffer */
		else if(in_tag)
		{
            // Check if we are a comment, when we are in a tag and buffer[0:2] == "!--"
            if(i == 2 && c == '-' && buffer[1] == '-' && buffer[0] == '!'){
                in_comment = 1;
                // Reset the buffer and i.
                buffer[0] = 0;
                i = 0;
            }

            // Store the character and increment the counter
            buffer[i] = c;
            i++;

		}
		/* If in data read char into buffer */
		else
		{
			buffer[i] = c;
			i++;
		}
	}
    // If no bytes were read, raise a warning.
    if(bytesRead == 0){
        fprintf(stdout, "Warning: %s is an empty file\n", inputpath);
        fflush(stdout);
    }

    // If the in_comment flag is still marked, issue a warning.
    if(in_comment){
        fprintf(stdout, "Warning: Un-terminated comment in %s\n", inputpath);
        fflush(stdout);
    }    

	/* Close the file */
	fclose(file);
}

glm::vec3 getMaximumBounds(){
    return agent_maximum;
}

glm::vec3 getMinimumBounds(){
    return agent_minimum;
}


/* Methods to load static networks from disk */
