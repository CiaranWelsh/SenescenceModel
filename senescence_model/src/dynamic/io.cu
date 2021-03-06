
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

void saveIterationData(char* outputpath, int iteration_number, xmachine_memory_TissueBlock_list* h_TissueBlocks_default, xmachine_memory_TissueBlock_list* d_TissueBlocks_default, int h_xmachine_memory_TissueBlock_default_count,xmachine_memory_Fibroblast_list* h_Fibroblasts_Quiescent, xmachine_memory_Fibroblast_list* d_Fibroblasts_Quiescent, int h_xmachine_memory_Fibroblast_Quiescent_count,xmachine_memory_Fibroblast_list* h_Fibroblasts_Repair, xmachine_memory_Fibroblast_list* d_Fibroblasts_Repair, int h_xmachine_memory_Fibroblast_Repair_count)
{
    PROFILE_SCOPED_RANGE("saveIterationData");
	cudaError_t cudaStatus;
	
	//Device to host memory transfer
	
	cudaStatus = cudaMemcpy( h_TissueBlocks_default, d_TissueBlocks_default, sizeof(xmachine_memory_TissueBlock_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr,"Error Copying TissueBlock Agent default State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
	cudaStatus = cudaMemcpy( h_Fibroblasts_Quiescent, d_Fibroblasts_Quiescent, sizeof(xmachine_memory_Fibroblast_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr,"Error Copying Fibroblast Agent Quiescent State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
	cudaStatus = cudaMemcpy( h_Fibroblasts_Repair, d_Fibroblasts_Repair, sizeof(xmachine_memory_Fibroblast_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr,"Error Copying Fibroblast Agent Repair State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
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
    
    fputs("\t<TISSUE_DAMAGE_PROB>", file);
    sprintf(data, "%f", (*get_TISSUE_DAMAGE_PROB()));
    fputs(data, file);
    fputs("</TISSUE_DAMAGE_PROB>\n", file);
    fputs("\t<QUIESCENT_MIGRATION_SCALE>", file);
    sprintf(data, "%f", (*get_QUIESCENT_MIGRATION_SCALE()));
    fputs(data, file);
    fputs("</QUIESCENT_MIGRATION_SCALE>\n", file);
    fputs("\t<REPAIR_RANGE>", file);
    sprintf(data, "%f", (*get_REPAIR_RANGE()));
    fputs(data, file);
    fputs("</REPAIR_RANGE>\n", file);
    fputs("\t<DAMAGE_DETECTION_RANGE>", file);
    sprintf(data, "%f", (*get_DAMAGE_DETECTION_RANGE()));
    fputs(data, file);
    fputs("</DAMAGE_DETECTION_RANGE>\n", file);
    fputs("\t<REPAIR_RATE>", file);
    sprintf(data, "%d", (*get_REPAIR_RATE()));
    fputs(data, file);
    fputs("</REPAIR_RATE>\n", file);
	fputs("</environment>\n" , file);

	//Write each TissueBlock agent to xml
	for (int i=0; i<h_xmachine_memory_TissueBlock_default_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>TissueBlock</name>\n", file);
        
		fputs("<id>", file);
        sprintf(data, "%d", h_TissueBlocks_default->id[i]);
		fputs(data, file);
		fputs("</id>\n", file);
        
		fputs("<x>", file);
        sprintf(data, "%f", h_TissueBlocks_default->x[i]);
		fputs(data, file);
		fputs("</x>\n", file);
        
		fputs("<y>", file);
        sprintf(data, "%f", h_TissueBlocks_default->y[i]);
		fputs(data, file);
		fputs("</y>\n", file);
        
		fputs("<z>", file);
        sprintf(data, "%f", h_TissueBlocks_default->z[i]);
		fputs(data, file);
		fputs("</z>\n", file);
        
		fputs("<damage>", file);
        sprintf(data, "%d", h_TissueBlocks_default->damage[i]);
		fputs(data, file);
		fputs("</damage>\n", file);
        
		fputs("</xagent>\n", file);
	}
	//Write each Fibroblast agent to xml
	for (int i=0; i<h_xmachine_memory_Fibroblast_Quiescent_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>Fibroblast</name>\n", file);
        
		fputs("<id>", file);
        sprintf(data, "%d", h_Fibroblasts_Quiescent->id[i]);
		fputs(data, file);
		fputs("</id>\n", file);
        
		fputs("<x>", file);
        sprintf(data, "%f", h_Fibroblasts_Quiescent->x[i]);
		fputs(data, file);
		fputs("</x>\n", file);
        
		fputs("<y>", file);
        sprintf(data, "%f", h_Fibroblasts_Quiescent->y[i]);
		fputs(data, file);
		fputs("</y>\n", file);
        
		fputs("<z>", file);
        sprintf(data, "%f", h_Fibroblasts_Quiescent->z[i]);
		fputs(data, file);
		fputs("</z>\n", file);
        
		fputs("<damage>", file);
        sprintf(data, "%d", h_Fibroblasts_Quiescent->damage[i]);
		fputs(data, file);
		fputs("</damage>\n", file);
        
		fputs("<current_state>", file);
        sprintf(data, "%d", h_Fibroblasts_Quiescent->current_state[i]);
		fputs(data, file);
		fputs("</current_state>\n", file);
        
		fputs("<go_to_state>", file);
        sprintf(data, "%d", h_Fibroblasts_Quiescent->go_to_state[i]);
		fputs(data, file);
		fputs("</go_to_state>\n", file);
        
		fputs("</xagent>\n", file);
	}
	//Write each Fibroblast agent to xml
	for (int i=0; i<h_xmachine_memory_Fibroblast_Repair_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>Fibroblast</name>\n", file);
        
		fputs("<id>", file);
        sprintf(data, "%d", h_Fibroblasts_Repair->id[i]);
		fputs(data, file);
		fputs("</id>\n", file);
        
		fputs("<x>", file);
        sprintf(data, "%f", h_Fibroblasts_Repair->x[i]);
		fputs(data, file);
		fputs("</x>\n", file);
        
		fputs("<y>", file);
        sprintf(data, "%f", h_Fibroblasts_Repair->y[i]);
		fputs(data, file);
		fputs("</y>\n", file);
        
		fputs("<z>", file);
        sprintf(data, "%f", h_Fibroblasts_Repair->z[i]);
		fputs(data, file);
		fputs("</z>\n", file);
        
		fputs("<damage>", file);
        sprintf(data, "%d", h_Fibroblasts_Repair->damage[i]);
		fputs(data, file);
		fputs("</damage>\n", file);
        
		fputs("<current_state>", file);
        sprintf(data, "%d", h_Fibroblasts_Repair->current_state[i]);
		fputs(data, file);
		fputs("</current_state>\n", file);
        
		fputs("<go_to_state>", file);
        sprintf(data, "%d", h_Fibroblasts_Repair->go_to_state[i]);
		fputs(data, file);
		fputs("</go_to_state>\n", file);
        
		fputs("</xagent>\n", file);
	}
	
	

	fputs("</states>\n" , file);
	
	/* Close the file */
	fclose(file);

}

void initEnvVars()
{
PROFILE_SCOPED_RANGE("initEnvVars");

    float t_TISSUE_DAMAGE_PROB = (float)0.001;
    set_TISSUE_DAMAGE_PROB(&t_TISSUE_DAMAGE_PROB);
    float t_QUIESCENT_MIGRATION_SCALE = (float)0.001;
    set_QUIESCENT_MIGRATION_SCALE(&t_QUIESCENT_MIGRATION_SCALE);
    float t_REPAIR_RANGE = (float)0.001;
    set_REPAIR_RANGE(&t_REPAIR_RANGE);
    float t_DAMAGE_DETECTION_RANGE = (float)0.001;
    set_DAMAGE_DETECTION_RANGE(&t_DAMAGE_DETECTION_RANGE);
    int t_REPAIR_RATE = (int)1;
    set_REPAIR_RATE(&t_REPAIR_RATE);
}

void readInitialStates(char* inputpath, xmachine_memory_TissueBlock_list* h_TissueBlocks, int* h_xmachine_memory_TissueBlock_count,xmachine_memory_Fibroblast_list* h_Fibroblasts, int* h_xmachine_memory_Fibroblast_count)
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
    int in_TissueBlock_id;
    int in_TissueBlock_x;
    int in_TissueBlock_y;
    int in_TissueBlock_z;
    int in_TissueBlock_damage;
    int in_Fibroblast_id;
    int in_Fibroblast_x;
    int in_Fibroblast_y;
    int in_Fibroblast_z;
    int in_Fibroblast_damage;
    int in_Fibroblast_current_state;
    int in_Fibroblast_go_to_state;
    
    /* tags for environment global variables */
    int in_env;
    int in_env_TISSUE_DAMAGE_PROB;
    
    int in_env_QUIESCENT_MIGRATION_SCALE;
    
    int in_env_REPAIR_RANGE;
    
    int in_env_DAMAGE_DETECTION_RANGE;
    
    int in_env_REPAIR_RATE;
    
	/* set agent count to zero */
	*h_xmachine_memory_TissueBlock_count = 0;
	*h_xmachine_memory_Fibroblast_count = 0;
	
	/* Variables for initial state data */
	int TissueBlock_id;
	float TissueBlock_x;
	float TissueBlock_y;
	float TissueBlock_z;
	int TissueBlock_damage;
	int Fibroblast_id;
	float Fibroblast_x;
	float Fibroblast_y;
	float Fibroblast_z;
	int Fibroblast_damage;
	int Fibroblast_current_state;
	int Fibroblast_go_to_state;

    /* Variables for environment variables */
    float env_TISSUE_DAMAGE_PROB;
    float env_QUIESCENT_MIGRATION_SCALE;
    float env_REPAIR_RANGE;
    float env_DAMAGE_DETECTION_RANGE;
    int env_REPAIR_RATE;
    


	/* Initialise variables */
    initEnvVars();
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
	in_TissueBlock_id = 0;
	in_TissueBlock_x = 0;
	in_TissueBlock_y = 0;
	in_TissueBlock_z = 0;
	in_TissueBlock_damage = 0;
	in_Fibroblast_id = 0;
	in_Fibroblast_x = 0;
	in_Fibroblast_y = 0;
	in_Fibroblast_z = 0;
	in_Fibroblast_damage = 0;
	in_Fibroblast_current_state = 0;
	in_Fibroblast_go_to_state = 0;
    in_env_TISSUE_DAMAGE_PROB = 0;
    in_env_QUIESCENT_MIGRATION_SCALE = 0;
    in_env_REPAIR_RANGE = 0;
    in_env_DAMAGE_DETECTION_RANGE = 0;
    in_env_REPAIR_RATE = 0;
	//set all TissueBlock values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_TissueBlock_MAX; k++)
	{	
		h_TissueBlocks->id[k] = 0;
		h_TissueBlocks->x[k] = 0;
		h_TissueBlocks->y[k] = 0;
		h_TissueBlocks->z[k] = 0;
		h_TissueBlocks->damage[k] = 0;
	}
	
	//set all Fibroblast values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_Fibroblast_MAX; k++)
	{	
		h_Fibroblasts->id[k] = 0;
		h_Fibroblasts->x[k] = 0;
		h_Fibroblasts->y[k] = 0;
		h_Fibroblasts->z[k] = 0;
		h_Fibroblasts->damage[k] = 0;
		h_Fibroblasts->current_state[k] = 0;
		h_Fibroblasts->go_to_state[k] = 0;
	}
	

	/* Default variables for memory */
    TissueBlock_id = 0;
    TissueBlock_x = 0;
    TissueBlock_y = 0;
    TissueBlock_z = 0;
    TissueBlock_damage = 0;
    Fibroblast_id = 0;
    Fibroblast_x = 0;
    Fibroblast_y = 0;
    Fibroblast_z = 0;
    Fibroblast_damage = 0;
    Fibroblast_current_state = 0;
    Fibroblast_go_to_state = 0;

    /* Default variables for environment variables */
    env_TISSUE_DAMAGE_PROB = 0.001;
    env_QUIESCENT_MIGRATION_SCALE = 0.001;
    env_REPAIR_RANGE = 0.001;
    env_DAMAGE_DETECTION_RANGE = 0.001;
    env_REPAIR_RATE = 1;
    
    
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
				if(strcmp(agentname, "TissueBlock") == 0)
				{
					if (*h_xmachine_memory_TissueBlock_count > xmachine_memory_TissueBlock_MAX){
						printf("ERROR: MAX Buffer size (%i) for agent TissueBlock exceeded whilst reading data\n", xmachine_memory_TissueBlock_MAX);
						// Close the file and stop reading
						fclose(file);
						exit(EXIT_FAILURE);
					}
                    
					h_TissueBlocks->id[*h_xmachine_memory_TissueBlock_count] = TissueBlock_id;
					h_TissueBlocks->x[*h_xmachine_memory_TissueBlock_count] = TissueBlock_x;//Check maximum x value
                    if(agent_maximum.x < TissueBlock_x)
                        agent_maximum.x = (float)TissueBlock_x;
                    //Check minimum x value
                    if(agent_minimum.x > TissueBlock_x)
                        agent_minimum.x = (float)TissueBlock_x;
                    
					h_TissueBlocks->y[*h_xmachine_memory_TissueBlock_count] = TissueBlock_y;//Check maximum y value
                    if(agent_maximum.y < TissueBlock_y)
                        agent_maximum.y = (float)TissueBlock_y;
                    //Check minimum y value
                    if(agent_minimum.y > TissueBlock_y)
                        agent_minimum.y = (float)TissueBlock_y;
                    
					h_TissueBlocks->z[*h_xmachine_memory_TissueBlock_count] = TissueBlock_z;//Check maximum z value
                    if(agent_maximum.z < TissueBlock_z)
                        agent_maximum.z = (float)TissueBlock_z;
                    //Check minimum z value
                    if(agent_minimum.z > TissueBlock_z)
                        agent_minimum.z = (float)TissueBlock_z;
                    
					h_TissueBlocks->damage[*h_xmachine_memory_TissueBlock_count] = TissueBlock_damage;
					(*h_xmachine_memory_TissueBlock_count) ++;	
				}
				else if(strcmp(agentname, "Fibroblast") == 0)
				{
					if (*h_xmachine_memory_Fibroblast_count > xmachine_memory_Fibroblast_MAX){
						printf("ERROR: MAX Buffer size (%i) for agent Fibroblast exceeded whilst reading data\n", xmachine_memory_Fibroblast_MAX);
						// Close the file and stop reading
						fclose(file);
						exit(EXIT_FAILURE);
					}
                    
					h_Fibroblasts->id[*h_xmachine_memory_Fibroblast_count] = Fibroblast_id;
					h_Fibroblasts->x[*h_xmachine_memory_Fibroblast_count] = Fibroblast_x;//Check maximum x value
                    if(agent_maximum.x < Fibroblast_x)
                        agent_maximum.x = (float)Fibroblast_x;
                    //Check minimum x value
                    if(agent_minimum.x > Fibroblast_x)
                        agent_minimum.x = (float)Fibroblast_x;
                    
					h_Fibroblasts->y[*h_xmachine_memory_Fibroblast_count] = Fibroblast_y;//Check maximum y value
                    if(agent_maximum.y < Fibroblast_y)
                        agent_maximum.y = (float)Fibroblast_y;
                    //Check minimum y value
                    if(agent_minimum.y > Fibroblast_y)
                        agent_minimum.y = (float)Fibroblast_y;
                    
					h_Fibroblasts->z[*h_xmachine_memory_Fibroblast_count] = Fibroblast_z;//Check maximum z value
                    if(agent_maximum.z < Fibroblast_z)
                        agent_maximum.z = (float)Fibroblast_z;
                    //Check minimum z value
                    if(agent_minimum.z > Fibroblast_z)
                        agent_minimum.z = (float)Fibroblast_z;
                    
					h_Fibroblasts->damage[*h_xmachine_memory_Fibroblast_count] = Fibroblast_damage;
					h_Fibroblasts->current_state[*h_xmachine_memory_Fibroblast_count] = Fibroblast_current_state;
					h_Fibroblasts->go_to_state[*h_xmachine_memory_Fibroblast_count] = Fibroblast_go_to_state;
					(*h_xmachine_memory_Fibroblast_count) ++;	
				}
				else
				{
					printf("Warning: agent name undefined - '%s'\n", agentname);
				}



				/* Reset xagent variables */
                TissueBlock_id = 0;
                TissueBlock_x = 0;
                TissueBlock_y = 0;
                TissueBlock_z = 0;
                TissueBlock_damage = 0;
                Fibroblast_id = 0;
                Fibroblast_x = 0;
                Fibroblast_y = 0;
                Fibroblast_z = 0;
                Fibroblast_damage = 0;
                Fibroblast_current_state = 0;
                Fibroblast_go_to_state = 0;
                
                in_xagent = 0;
			}
			if(strcmp(buffer, "id") == 0) in_TissueBlock_id = 1;
			if(strcmp(buffer, "/id") == 0) in_TissueBlock_id = 0;
			if(strcmp(buffer, "x") == 0) in_TissueBlock_x = 1;
			if(strcmp(buffer, "/x") == 0) in_TissueBlock_x = 0;
			if(strcmp(buffer, "y") == 0) in_TissueBlock_y = 1;
			if(strcmp(buffer, "/y") == 0) in_TissueBlock_y = 0;
			if(strcmp(buffer, "z") == 0) in_TissueBlock_z = 1;
			if(strcmp(buffer, "/z") == 0) in_TissueBlock_z = 0;
			if(strcmp(buffer, "damage") == 0) in_TissueBlock_damage = 1;
			if(strcmp(buffer, "/damage") == 0) in_TissueBlock_damage = 0;
			if(strcmp(buffer, "id") == 0) in_Fibroblast_id = 1;
			if(strcmp(buffer, "/id") == 0) in_Fibroblast_id = 0;
			if(strcmp(buffer, "x") == 0) in_Fibroblast_x = 1;
			if(strcmp(buffer, "/x") == 0) in_Fibroblast_x = 0;
			if(strcmp(buffer, "y") == 0) in_Fibroblast_y = 1;
			if(strcmp(buffer, "/y") == 0) in_Fibroblast_y = 0;
			if(strcmp(buffer, "z") == 0) in_Fibroblast_z = 1;
			if(strcmp(buffer, "/z") == 0) in_Fibroblast_z = 0;
			if(strcmp(buffer, "damage") == 0) in_Fibroblast_damage = 1;
			if(strcmp(buffer, "/damage") == 0) in_Fibroblast_damage = 0;
			if(strcmp(buffer, "current_state") == 0) in_Fibroblast_current_state = 1;
			if(strcmp(buffer, "/current_state") == 0) in_Fibroblast_current_state = 0;
			if(strcmp(buffer, "go_to_state") == 0) in_Fibroblast_go_to_state = 1;
			if(strcmp(buffer, "/go_to_state") == 0) in_Fibroblast_go_to_state = 0;
			
            /* environment variables */
            if(strcmp(buffer, "TISSUE_DAMAGE_PROB") == 0) in_env_TISSUE_DAMAGE_PROB = 1;
            if(strcmp(buffer, "/TISSUE_DAMAGE_PROB") == 0) in_env_TISSUE_DAMAGE_PROB = 0;
			if(strcmp(buffer, "QUIESCENT_MIGRATION_SCALE") == 0) in_env_QUIESCENT_MIGRATION_SCALE = 1;
            if(strcmp(buffer, "/QUIESCENT_MIGRATION_SCALE") == 0) in_env_QUIESCENT_MIGRATION_SCALE = 0;
			if(strcmp(buffer, "REPAIR_RANGE") == 0) in_env_REPAIR_RANGE = 1;
            if(strcmp(buffer, "/REPAIR_RANGE") == 0) in_env_REPAIR_RANGE = 0;
			if(strcmp(buffer, "DAMAGE_DETECTION_RANGE") == 0) in_env_DAMAGE_DETECTION_RANGE = 1;
            if(strcmp(buffer, "/DAMAGE_DETECTION_RANGE") == 0) in_env_DAMAGE_DETECTION_RANGE = 0;
			if(strcmp(buffer, "REPAIR_RATE") == 0) in_env_REPAIR_RATE = 1;
            if(strcmp(buffer, "/REPAIR_RATE") == 0) in_env_REPAIR_RATE = 0;
			

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
				if(in_TissueBlock_id){
                    TissueBlock_id = (int) fpgu_strtol(buffer); 
                }
				if(in_TissueBlock_x){
                    TissueBlock_x = (float) fgpu_atof(buffer); 
                }
				if(in_TissueBlock_y){
                    TissueBlock_y = (float) fgpu_atof(buffer); 
                }
				if(in_TissueBlock_z){
                    TissueBlock_z = (float) fgpu_atof(buffer); 
                }
				if(in_TissueBlock_damage){
                    TissueBlock_damage = (int) fpgu_strtol(buffer); 
                }
				if(in_Fibroblast_id){
                    Fibroblast_id = (int) fpgu_strtol(buffer); 
                }
				if(in_Fibroblast_x){
                    Fibroblast_x = (float) fgpu_atof(buffer); 
                }
				if(in_Fibroblast_y){
                    Fibroblast_y = (float) fgpu_atof(buffer); 
                }
				if(in_Fibroblast_z){
                    Fibroblast_z = (float) fgpu_atof(buffer); 
                }
				if(in_Fibroblast_damage){
                    Fibroblast_damage = (int) fpgu_strtol(buffer); 
                }
				if(in_Fibroblast_current_state){
                    Fibroblast_current_state = (int) fpgu_strtol(buffer); 
                }
				if(in_Fibroblast_go_to_state){
                    Fibroblast_go_to_state = (int) fpgu_strtol(buffer); 
                }
				
            }
            else if (in_env){
            if(in_env_TISSUE_DAMAGE_PROB){
              
                    env_TISSUE_DAMAGE_PROB = (float) fgpu_atof(buffer);
                    
                    set_TISSUE_DAMAGE_PROB(&env_TISSUE_DAMAGE_PROB);
                  
              }
            if(in_env_QUIESCENT_MIGRATION_SCALE){
              
                    env_QUIESCENT_MIGRATION_SCALE = (float) fgpu_atof(buffer);
                    
                    set_QUIESCENT_MIGRATION_SCALE(&env_QUIESCENT_MIGRATION_SCALE);
                  
              }
            if(in_env_REPAIR_RANGE){
              
                    env_REPAIR_RANGE = (float) fgpu_atof(buffer);
                    
                    set_REPAIR_RANGE(&env_REPAIR_RANGE);
                  
              }
            if(in_env_DAMAGE_DETECTION_RANGE){
              
                    env_DAMAGE_DETECTION_RANGE = (float) fgpu_atof(buffer);
                    
                    set_DAMAGE_DETECTION_RANGE(&env_DAMAGE_DETECTION_RANGE);
                  
              }
            if(in_env_REPAIR_RATE){
              
                    env_REPAIR_RATE = (int) fpgu_strtol(buffer);
                    
                    set_REPAIR_RATE(&env_REPAIR_RATE);
                  
              }
            
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
