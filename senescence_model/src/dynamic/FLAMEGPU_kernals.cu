
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


#ifndef _FLAMEGPU_KERNELS_H_
#define _FLAMEGPU_KERNELS_H_

#include "header.h"


/* Agent count constants */

__constant__ int d_xmachine_memory_TissueBlock_count;

__constant__ int d_xmachine_memory_Fibroblast_count;

/* Agent state count constants */

__constant__ int d_xmachine_memory_TissueBlock_default_count;

__constant__ int d_xmachine_memory_Fibroblast_Quiescent_count;

__constant__ int d_xmachine_memory_Fibroblast_EarlySenescent_count;

__constant__ int d_xmachine_memory_Fibroblast_Senescent_count;

__constant__ int d_xmachine_memory_Fibroblast_Proliferating_count;

__constant__ int d_xmachine_memory_Fibroblast_Repair_count;


/* Message constants */

/* fibroblast_damage_report Message variables */
/* Non partitioned, spatial partitioned and on-graph partitioned message variables  */
__constant__ int d_message_fibroblast_damage_report_count;         /**< message list counter*/
__constant__ int d_message_fibroblast_damage_report_output_type;   /**< message output type (single or optional)*/
//Spatial Partitioning Variables
__constant__ glm::vec3 d_message_fibroblast_damage_report_min_bounds;           /**< min bounds (x,y,z) of partitioning environment */
__constant__ glm::vec3 d_message_fibroblast_damage_report_max_bounds;           /**< max bounds (x,y,z) of partitioning environment */
__constant__ glm::ivec3 d_message_fibroblast_damage_report_partitionDim;           /**< partition dimensions (x,y,z) of partitioning environment */
__constant__ float d_message_fibroblast_damage_report_radius;                 /**< partition radius (used to determin the size of the partitions) */

/* tissue_damage_report Message variables */
/* Non partitioned, spatial partitioned and on-graph partitioned message variables  */
__constant__ int d_message_tissue_damage_report_count;         /**< message list counter*/
__constant__ int d_message_tissue_damage_report_output_type;   /**< message output type (single or optional)*/
//Spatial Partitioning Variables
__constant__ glm::vec3 d_message_tissue_damage_report_min_bounds;           /**< min bounds (x,y,z) of partitioning environment */
__constant__ glm::vec3 d_message_tissue_damage_report_max_bounds;           /**< max bounds (x,y,z) of partitioning environment */
__constant__ glm::ivec3 d_message_tissue_damage_report_partitionDim;           /**< partition dimensions (x,y,z) of partitioning environment */
__constant__ float d_message_tissue_damage_report_radius;                 /**< partition radius (used to determin the size of the partitions) */

/* doublings Message variables */
/* Non partitioned, spatial partitioned and on-graph partitioned message variables  */
__constant__ int d_message_doublings_count;         /**< message list counter*/
__constant__ int d_message_doublings_output_type;   /**< message output type (single or optional)*/

/* count Message variables */
/* Non partitioned, spatial partitioned and on-graph partitioned message variables  */
__constant__ int d_message_count_count;         /**< message list counter*/
__constant__ int d_message_count_output_type;   /**< message output type (single or optional)*/

/* quiescent_location_report Message variables */
/* Non partitioned, spatial partitioned and on-graph partitioned message variables  */
__constant__ int d_message_quiescent_location_report_count;         /**< message list counter*/
__constant__ int d_message_quiescent_location_report_output_type;   /**< message output type (single or optional)*/
//Spatial Partitioning Variables
__constant__ glm::vec3 d_message_quiescent_location_report_min_bounds;           /**< min bounds (x,y,z) of partitioning environment */
__constant__ glm::vec3 d_message_quiescent_location_report_max_bounds;           /**< max bounds (x,y,z) of partitioning environment */
__constant__ glm::ivec3 d_message_quiescent_location_report_partitionDim;           /**< partition dimensions (x,y,z) of partitioning environment */
__constant__ float d_message_quiescent_location_report_radius;                 /**< partition radius (used to determin the size of the partitions) */

/* senescent_location_report Message variables */
/* Non partitioned, spatial partitioned and on-graph partitioned message variables  */
__constant__ int d_message_senescent_location_report_count;         /**< message list counter*/
__constant__ int d_message_senescent_location_report_output_type;   /**< message output type (single or optional)*/
//Spatial Partitioning Variables
__constant__ glm::vec3 d_message_senescent_location_report_min_bounds;           /**< min bounds (x,y,z) of partitioning environment */
__constant__ glm::vec3 d_message_senescent_location_report_max_bounds;           /**< max bounds (x,y,z) of partitioning environment */
__constant__ glm::ivec3 d_message_senescent_location_report_partitionDim;           /**< partition dimensions (x,y,z) of partitioning environment */
__constant__ float d_message_senescent_location_report_radius;                 /**< partition radius (used to determin the size of the partitions) */

	

/* Graph Constants */


/* Graph device array pointer(s) */


/* Graph host array pointer(s) */

    
//include each function file

#include "functions.c"
    
/* Texture bindings */
/* fibroblast_damage_report Message Bindings */texture<int, 1, cudaReadModeElementType> tex_xmachine_message_fibroblast_damage_report_id;
__constant__ int d_tex_xmachine_message_fibroblast_damage_report_id_offset;texture<float, 1, cudaReadModeElementType> tex_xmachine_message_fibroblast_damage_report_x;
__constant__ int d_tex_xmachine_message_fibroblast_damage_report_x_offset;texture<float, 1, cudaReadModeElementType> tex_xmachine_message_fibroblast_damage_report_y;
__constant__ int d_tex_xmachine_message_fibroblast_damage_report_y_offset;texture<float, 1, cudaReadModeElementType> tex_xmachine_message_fibroblast_damage_report_z;
__constant__ int d_tex_xmachine_message_fibroblast_damage_report_z_offset;texture<int, 1, cudaReadModeElementType> tex_xmachine_message_fibroblast_damage_report_damage;
__constant__ int d_tex_xmachine_message_fibroblast_damage_report_damage_offset;
texture<int, 1, cudaReadModeElementType> tex_xmachine_message_fibroblast_damage_report_pbm_start;
__constant__ int d_tex_xmachine_message_fibroblast_damage_report_pbm_start_offset;
texture<int, 1, cudaReadModeElementType> tex_xmachine_message_fibroblast_damage_report_pbm_end_or_count;
__constant__ int d_tex_xmachine_message_fibroblast_damage_report_pbm_end_or_count_offset;


/* tissue_damage_report Message Bindings */texture<int, 1, cudaReadModeElementType> tex_xmachine_message_tissue_damage_report_id;
__constant__ int d_tex_xmachine_message_tissue_damage_report_id_offset;texture<float, 1, cudaReadModeElementType> tex_xmachine_message_tissue_damage_report_x;
__constant__ int d_tex_xmachine_message_tissue_damage_report_x_offset;texture<float, 1, cudaReadModeElementType> tex_xmachine_message_tissue_damage_report_y;
__constant__ int d_tex_xmachine_message_tissue_damage_report_y_offset;texture<float, 1, cudaReadModeElementType> tex_xmachine_message_tissue_damage_report_z;
__constant__ int d_tex_xmachine_message_tissue_damage_report_z_offset;texture<int, 1, cudaReadModeElementType> tex_xmachine_message_tissue_damage_report_damage;
__constant__ int d_tex_xmachine_message_tissue_damage_report_damage_offset;
texture<int, 1, cudaReadModeElementType> tex_xmachine_message_tissue_damage_report_pbm_start;
__constant__ int d_tex_xmachine_message_tissue_damage_report_pbm_start_offset;
texture<int, 1, cudaReadModeElementType> tex_xmachine_message_tissue_damage_report_pbm_end_or_count;
__constant__ int d_tex_xmachine_message_tissue_damage_report_pbm_end_or_count_offset;




/* quiescent_location_report Message Bindings */texture<int, 1, cudaReadModeElementType> tex_xmachine_message_quiescent_location_report_id;
__constant__ int d_tex_xmachine_message_quiescent_location_report_id_offset;texture<float, 1, cudaReadModeElementType> tex_xmachine_message_quiescent_location_report_x;
__constant__ int d_tex_xmachine_message_quiescent_location_report_x_offset;texture<float, 1, cudaReadModeElementType> tex_xmachine_message_quiescent_location_report_y;
__constant__ int d_tex_xmachine_message_quiescent_location_report_y_offset;texture<float, 1, cudaReadModeElementType> tex_xmachine_message_quiescent_location_report_z;
__constant__ int d_tex_xmachine_message_quiescent_location_report_z_offset;
texture<int, 1, cudaReadModeElementType> tex_xmachine_message_quiescent_location_report_pbm_start;
__constant__ int d_tex_xmachine_message_quiescent_location_report_pbm_start_offset;
texture<int, 1, cudaReadModeElementType> tex_xmachine_message_quiescent_location_report_pbm_end_or_count;
__constant__ int d_tex_xmachine_message_quiescent_location_report_pbm_end_or_count_offset;


/* senescent_location_report Message Bindings */texture<int, 1, cudaReadModeElementType> tex_xmachine_message_senescent_location_report_id;
__constant__ int d_tex_xmachine_message_senescent_location_report_id_offset;texture<float, 1, cudaReadModeElementType> tex_xmachine_message_senescent_location_report_x;
__constant__ int d_tex_xmachine_message_senescent_location_report_x_offset;texture<float, 1, cudaReadModeElementType> tex_xmachine_message_senescent_location_report_y;
__constant__ int d_tex_xmachine_message_senescent_location_report_y_offset;texture<float, 1, cudaReadModeElementType> tex_xmachine_message_senescent_location_report_z;
__constant__ int d_tex_xmachine_message_senescent_location_report_z_offset;
texture<int, 1, cudaReadModeElementType> tex_xmachine_message_senescent_location_report_pbm_start;
__constant__ int d_tex_xmachine_message_senescent_location_report_pbm_start_offset;
texture<int, 1, cudaReadModeElementType> tex_xmachine_message_senescent_location_report_pbm_end_or_count;
__constant__ int d_tex_xmachine_message_senescent_location_report_pbm_end_or_count_offset;


    
#define WRAP(x,m) (((x)<m)?(x):(x%m)) /**< Simple wrap */
#define sWRAP(x,m) (((x)<m)?(((x)<0)?(m+(x)):(x)):(m-(x))) /**<signed integer wrap (no modulus) for negatives where 2m > |x| > m */

//PADDING WILL ONLY AVOID SM CONFLICTS FOR 32BIT
//SM_OFFSET REQUIRED AS FERMI STARTS INDEXING MEMORY FROM LOCATION 0 (i.e. NULL)??
__constant__ int d_SM_START;
__constant__ int d_PADDING;

//SM addressing macro to avoid conflicts (32 bit only)
#define SHARE_INDEX(i, s) ((((s) + d_PADDING)* (i))+d_SM_START) /**<offset struct size by padding to avoid bank conflicts */

//if doubel support is needed then define the following function which requires sm_13 or later
#ifdef _DOUBLE_SUPPORT_REQUIRED_
__inline__ __device__ double tex1DfetchDouble(texture<int2, 1, cudaReadModeElementType> tex, int i)
{
	int2 v = tex1Dfetch(tex, i);
  //IF YOU HAVE AN ERROR HERE THEN YOU ARE USING DOUBLE VALUES IN AGENT MEMORY AND NOT COMPILING FOR DOUBLE SUPPORTED HARDWARE
  //To compile for double supported hardware change the CUDA Build rule property "Use sm_13 Architecture (double support)" on the CUDA-Specific Propert Page of the CUDA Build Rule for simulation.cu
	return __hiloint2double(v.y, v.x);
}
#endif

/* Helper functions */
/** next_cell
 * Function used for finding the next cell when using spatial partitioning
 * Upddates the relative cell variable which can have value of -1, 0 or +1
 * @param relative_cell pointer to the relative cell position
 * @return boolean if there is a next cell. True unless relative_Cell value was 1,1,1
 */
__device__ bool next_cell3D(glm::ivec3* relative_cell)
{
	if (relative_cell->x < 1)
	{
		relative_cell->x++;
		return true;
	}
	relative_cell->x = -1;

	if (relative_cell->y < 1)
	{
		relative_cell->y++;
		return true;
	}
	relative_cell->y = -1;
	
	if (relative_cell->z < 1)
	{
		relative_cell->z++;
		return true;
	}
	relative_cell->z = -1;
	
	return false;
}

/** next_cell2D
 * Function used for finding the next cell when using spatial partitioning. Z component is ignored
 * Upddates the relative cell variable which can have value of -1, 0 or +1
 * @param relative_cell pointer to the relative cell position
 * @return boolean if there is a next cell. True unless relative_Cell value was 1,1
 */
__device__ bool next_cell2D(glm::ivec3* relative_cell)
{
	if (relative_cell->x < 1)
	{
		relative_cell->x++;
		return true;
	}
	relative_cell->x = -1;

	if (relative_cell->y < 1)
	{
		relative_cell->y++;
		return true;
	}
	relative_cell->y = -1;
	
	return false;
}


/** Quiescent2Proliferating_function_filter
 *	Standard agent condition function. Filters agents from one state list to the next depending on the condition
 * @param currentState xmachine_memory_Fibroblast_list representing agent i the current state
 * @param nextState xmachine_memory_Fibroblast_list representing agent i the next state
 */
 __global__ void Quiescent2Proliferating_function_filter(xmachine_memory_Fibroblast_list* currentState, xmachine_memory_Fibroblast_list* nextState)
 {
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;
	
	//check thread max
	if (index < d_xmachine_memory_Fibroblast_count){
	
		//apply the filter
		if (currentState->proliferate_bool[index]==1)
		{	//copy agent data to newstate list
			nextState->id[index] = currentState->id[index];
			nextState->x[index] = currentState->x[index];
			nextState->y[index] = currentState->y[index];
			nextState->z[index] = currentState->z[index];
			nextState->doublings[index] = currentState->doublings[index];
			nextState->damage[index] = currentState->damage[index];
			nextState->proliferate_bool[index] = currentState->proliferate_bool[index];
			nextState->transition_to_early_sen[index] = currentState->transition_to_early_sen[index];
			//set scan input flag to 1
			nextState->_scan_input[index] = 1;
		}
		else
		{
			//set scan input flag of current state to 1 (keep agent)
			currentState->_scan_input[index] = 1;
		}
	
	}
 }

/** TransitionToEarlySen_function_filter
 *	Standard agent condition function. Filters agents from one state list to the next depending on the condition
 * @param currentState xmachine_memory_Fibroblast_list representing agent i the current state
 * @param nextState xmachine_memory_Fibroblast_list representing agent i the next state
 */
 __global__ void TransitionToEarlySen_function_filter(xmachine_memory_Fibroblast_list* currentState, xmachine_memory_Fibroblast_list* nextState)
 {
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;
	
	//check thread max
	if (index < d_xmachine_memory_Fibroblast_count){
	
		//apply the filter
		if (currentState->transition_to_early_sen[index]==1)
		{	//copy agent data to newstate list
			nextState->id[index] = currentState->id[index];
			nextState->x[index] = currentState->x[index];
			nextState->y[index] = currentState->y[index];
			nextState->z[index] = currentState->z[index];
			nextState->doublings[index] = currentState->doublings[index];
			nextState->damage[index] = currentState->damage[index];
			nextState->proliferate_bool[index] = currentState->proliferate_bool[index];
			nextState->transition_to_early_sen[index] = currentState->transition_to_early_sen[index];
			//set scan input flag to 1
			nextState->_scan_input[index] = 1;
		}
		else
		{
			//set scan input flag of current state to 1 (keep agent)
			currentState->_scan_input[index] = 1;
		}
	
	}
 }

////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created TissueBlock agent functions */

/** reset_TissueBlock_scan_input
 * TissueBlock agent reset scan input function
 * @param agents The xmachine_memory_TissueBlock_list agent list
 */
__global__ void reset_TissueBlock_scan_input(xmachine_memory_TissueBlock_list* agents){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	agents->_position[index] = 0;
	agents->_scan_input[index] = 0;
}



/** scatter_TissueBlock_Agents
 * TissueBlock scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_TissueBlock_list agent list destination
 * @param agents_src xmachine_memory_TissueBlock_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void scatter_TissueBlock_Agents(xmachine_memory_TissueBlock_list* agents_dst, xmachine_memory_TissueBlock_list* agents_src, int dst_agent_count, int number_to_scatter){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = agents_src->_scan_input[index];

	//if optional message is to be written. 
	//must check agent is within number to scatter as unused threads may have scan input = 1
	if ((_scan_input == 1)&&(index < number_to_scatter)){
		int output_index = agents_src->_position[index] + dst_agent_count;

		//AoS - xmachine_message_location Un-Coalesced scattered memory write     
        agents_dst->_position[output_index] = output_index;        
		agents_dst->id[output_index] = agents_src->id[index];        
		agents_dst->x[output_index] = agents_src->x[index];        
		agents_dst->y[output_index] = agents_src->y[index];        
		agents_dst->z[output_index] = agents_src->z[index];        
		agents_dst->damage[output_index] = agents_src->damage[index];
	}
}

/** append_TissueBlock_Agents
 * TissueBlock scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_TissueBlock_list agent list destination
 * @param agents_src xmachine_memory_TissueBlock_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void append_TissueBlock_Agents(xmachine_memory_TissueBlock_list* agents_dst, xmachine_memory_TissueBlock_list* agents_src, int dst_agent_count, int number_to_append){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//must check agent is within number to append as unused threads may have scan input = 1
    if (index < number_to_append){
	    int output_index = index + dst_agent_count;

	    //AoS - xmachine_message_location Un-Coalesced scattered memory write
	    agents_dst->_position[output_index] = output_index;
	    agents_dst->id[output_index] = agents_src->id[index];
	    agents_dst->x[output_index] = agents_src->x[index];
	    agents_dst->y[output_index] = agents_src->y[index];
	    agents_dst->z[output_index] = agents_src->z[index];
	    agents_dst->damage[output_index] = agents_src->damage[index];
    }
}

/** add_TissueBlock_agent
 * Continuous TissueBlock agent add agent function writes agent data to agent swap
 * @param agents xmachine_memory_TissueBlock_list to add agents to 
 * @param id agent variable of type int
 * @param x agent variable of type float
 * @param y agent variable of type float
 * @param z agent variable of type float
 * @param damage agent variable of type int
 */
template <int AGENT_TYPE>
__device__ void add_TissueBlock_agent(xmachine_memory_TissueBlock_list* agents, int id, float x, float y, float z, int damage){
	
	int index;
    
    //calculate the agents index in global agent list (depends on agent type)
	if (AGENT_TYPE == DISCRETE_2D){
		int width = (blockDim.x* gridDim.x);
		glm::ivec2 global_position;
		global_position.x = (blockIdx.x*blockDim.x) + threadIdx.x;
		global_position.y = (blockIdx.y*blockDim.y) + threadIdx.y;
		index = global_position.x + (global_position.y* width);
	}else//AGENT_TYPE == CONTINOUS
		index = threadIdx.x + blockIdx.x*blockDim.x;

	//for prefix sum
	agents->_position[index] = 0;
	agents->_scan_input[index] = 1;

	//write data to new buffer
	agents->id[index] = id;
	agents->x[index] = x;
	agents->y[index] = y;
	agents->z[index] = z;
	agents->damage[index] = damage;

}

//non templated version assumes DISCRETE_2D but works also for CONTINUOUS
__device__ void add_TissueBlock_agent(xmachine_memory_TissueBlock_list* agents, int id, float x, float y, float z, int damage){
    add_TissueBlock_agent<DISCRETE_2D>(agents, id, x, y, z, damage);
}

/** reorder_TissueBlock_agents
 * Continuous TissueBlock agent areorder function used after key value pairs have been sorted
 * @param values sorted index values
 * @param unordered_agents list of unordered agents
 * @ param ordered_agents list used to output ordered agents
 */
__global__ void reorder_TissueBlock_agents(unsigned int* values, xmachine_memory_TissueBlock_list* unordered_agents, xmachine_memory_TissueBlock_list* ordered_agents)
{
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	uint old_pos = values[index];

	//reorder agent data
	ordered_agents->id[index] = unordered_agents->id[old_pos];
	ordered_agents->x[index] = unordered_agents->x[old_pos];
	ordered_agents->y[index] = unordered_agents->y[old_pos];
	ordered_agents->z[index] = unordered_agents->z[old_pos];
	ordered_agents->damage[index] = unordered_agents->damage[old_pos];
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created Fibroblast agent functions */

/** reset_Fibroblast_scan_input
 * Fibroblast agent reset scan input function
 * @param agents The xmachine_memory_Fibroblast_list agent list
 */
__global__ void reset_Fibroblast_scan_input(xmachine_memory_Fibroblast_list* agents){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	agents->_position[index] = 0;
	agents->_scan_input[index] = 0;
}



/** scatter_Fibroblast_Agents
 * Fibroblast scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_Fibroblast_list agent list destination
 * @param agents_src xmachine_memory_Fibroblast_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void scatter_Fibroblast_Agents(xmachine_memory_Fibroblast_list* agents_dst, xmachine_memory_Fibroblast_list* agents_src, int dst_agent_count, int number_to_scatter){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = agents_src->_scan_input[index];

	//if optional message is to be written. 
	//must check agent is within number to scatter as unused threads may have scan input = 1
	if ((_scan_input == 1)&&(index < number_to_scatter)){
		int output_index = agents_src->_position[index] + dst_agent_count;

		//AoS - xmachine_message_location Un-Coalesced scattered memory write     
        agents_dst->_position[output_index] = output_index;        
		agents_dst->id[output_index] = agents_src->id[index];        
		agents_dst->x[output_index] = agents_src->x[index];        
		agents_dst->y[output_index] = agents_src->y[index];        
		agents_dst->z[output_index] = agents_src->z[index];        
		agents_dst->doublings[output_index] = agents_src->doublings[index];        
		agents_dst->damage[output_index] = agents_src->damage[index];        
		agents_dst->proliferate_bool[output_index] = agents_src->proliferate_bool[index];        
		agents_dst->transition_to_early_sen[output_index] = agents_src->transition_to_early_sen[index];
	}
}

/** append_Fibroblast_Agents
 * Fibroblast scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_Fibroblast_list agent list destination
 * @param agents_src xmachine_memory_Fibroblast_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void append_Fibroblast_Agents(xmachine_memory_Fibroblast_list* agents_dst, xmachine_memory_Fibroblast_list* agents_src, int dst_agent_count, int number_to_append){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//must check agent is within number to append as unused threads may have scan input = 1
    if (index < number_to_append){
	    int output_index = index + dst_agent_count;

	    //AoS - xmachine_message_location Un-Coalesced scattered memory write
	    agents_dst->_position[output_index] = output_index;
	    agents_dst->id[output_index] = agents_src->id[index];
	    agents_dst->x[output_index] = agents_src->x[index];
	    agents_dst->y[output_index] = agents_src->y[index];
	    agents_dst->z[output_index] = agents_src->z[index];
	    agents_dst->doublings[output_index] = agents_src->doublings[index];
	    agents_dst->damage[output_index] = agents_src->damage[index];
	    agents_dst->proliferate_bool[output_index] = agents_src->proliferate_bool[index];
	    agents_dst->transition_to_early_sen[output_index] = agents_src->transition_to_early_sen[index];
    }
}

/** add_Fibroblast_agent
 * Continuous Fibroblast agent add agent function writes agent data to agent swap
 * @param agents xmachine_memory_Fibroblast_list to add agents to 
 * @param id agent variable of type int
 * @param x agent variable of type float
 * @param y agent variable of type float
 * @param z agent variable of type float
 * @param doublings agent variable of type float
 * @param damage agent variable of type int
 * @param proliferate_bool agent variable of type int
 * @param transition_to_early_sen agent variable of type int
 */
template <int AGENT_TYPE>
__device__ void add_Fibroblast_agent(xmachine_memory_Fibroblast_list* agents, int id, float x, float y, float z, float doublings, int damage, int proliferate_bool, int transition_to_early_sen){
	
	int index;
    
    //calculate the agents index in global agent list (depends on agent type)
	if (AGENT_TYPE == DISCRETE_2D){
		int width = (blockDim.x* gridDim.x);
		glm::ivec2 global_position;
		global_position.x = (blockIdx.x*blockDim.x) + threadIdx.x;
		global_position.y = (blockIdx.y*blockDim.y) + threadIdx.y;
		index = global_position.x + (global_position.y* width);
	}else//AGENT_TYPE == CONTINOUS
		index = threadIdx.x + blockIdx.x*blockDim.x;

	//for prefix sum
	agents->_position[index] = 0;
	agents->_scan_input[index] = 1;

	//write data to new buffer
	agents->id[index] = id;
	agents->x[index] = x;
	agents->y[index] = y;
	agents->z[index] = z;
	agents->doublings[index] = doublings;
	agents->damage[index] = damage;
	agents->proliferate_bool[index] = proliferate_bool;
	agents->transition_to_early_sen[index] = transition_to_early_sen;

}

//non templated version assumes DISCRETE_2D but works also for CONTINUOUS
__device__ void add_Fibroblast_agent(xmachine_memory_Fibroblast_list* agents, int id, float x, float y, float z, float doublings, int damage, int proliferate_bool, int transition_to_early_sen){
    add_Fibroblast_agent<DISCRETE_2D>(agents, id, x, y, z, doublings, damage, proliferate_bool, transition_to_early_sen);
}

/** reorder_Fibroblast_agents
 * Continuous Fibroblast agent areorder function used after key value pairs have been sorted
 * @param values sorted index values
 * @param unordered_agents list of unordered agents
 * @ param ordered_agents list used to output ordered agents
 */
__global__ void reorder_Fibroblast_agents(unsigned int* values, xmachine_memory_Fibroblast_list* unordered_agents, xmachine_memory_Fibroblast_list* ordered_agents)
{
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	uint old_pos = values[index];

	//reorder agent data
	ordered_agents->id[index] = unordered_agents->id[old_pos];
	ordered_agents->x[index] = unordered_agents->x[old_pos];
	ordered_agents->y[index] = unordered_agents->y[old_pos];
	ordered_agents->z[index] = unordered_agents->z[old_pos];
	ordered_agents->doublings[index] = unordered_agents->doublings[old_pos];
	ordered_agents->damage[index] = unordered_agents->damage[old_pos];
	ordered_agents->proliferate_bool[index] = unordered_agents->proliferate_bool[old_pos];
	ordered_agents->transition_to_early_sen[index] = unordered_agents->transition_to_early_sen[old_pos];
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created fibroblast_damage_report message functions */


/** add_fibroblast_damage_report_message
 * Add non partitioned or spatially partitioned fibroblast_damage_report message
 * @param messages xmachine_message_fibroblast_damage_report_list message list to add too
 * @param id agent variable of type int
 * @param x agent variable of type float
 * @param y agent variable of type float
 * @param z agent variable of type float
 * @param damage agent variable of type int
 */
__device__ void add_fibroblast_damage_report_message(xmachine_message_fibroblast_damage_report_list* messages, int id, float x, float y, float z, int damage){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x + d_message_fibroblast_damage_report_count;

	int _position;
	int _scan_input;

	//decide output position
	if(d_message_fibroblast_damage_report_output_type == single_message){
		_position = index; //same as agent position
		_scan_input = 0;
	}else if (d_message_fibroblast_damage_report_output_type == optional_message){
		_position = 0;	   //to be calculated using Prefix sum
		_scan_input = 1;
	}

	//AoS - xmachine_message_fibroblast_damage_report Coalesced memory write
	messages->_scan_input[index] = _scan_input;	
	messages->_position[index] = _position;
	messages->id[index] = id;
	messages->x[index] = x;
	messages->y[index] = y;
	messages->z[index] = z;
	messages->damage[index] = damage;

}

/**
 * Scatter non partitioned or spatially partitioned fibroblast_damage_report message (for optional messages)
 * @param messages scatter_optional_fibroblast_damage_report_messages Sparse xmachine_message_fibroblast_damage_report_list message list
 * @param message_swap temp xmachine_message_fibroblast_damage_report_list message list to scatter sparse messages to
 */
__global__ void scatter_optional_fibroblast_damage_report_messages(xmachine_message_fibroblast_damage_report_list* messages, xmachine_message_fibroblast_damage_report_list* messages_swap){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = messages_swap->_scan_input[index];

	//if optional message is to be written
	if (_scan_input == 1){
		int output_index = messages_swap->_position[index] + d_message_fibroblast_damage_report_count;

		//AoS - xmachine_message_fibroblast_damage_report Un-Coalesced scattered memory write
		messages->_position[output_index] = output_index;
		messages->id[output_index] = messages_swap->id[index];
		messages->x[output_index] = messages_swap->x[index];
		messages->y[output_index] = messages_swap->y[index];
		messages->z[output_index] = messages_swap->z[index];
		messages->damage[output_index] = messages_swap->damage[index];				
	}
}

/** reset_fibroblast_damage_report_swaps
 * Reset non partitioned or spatially partitioned fibroblast_damage_report message swaps (for scattering optional messages)
 * @param message_swap message list to reset _position and _scan_input values back to 0
 */
__global__ void reset_fibroblast_damage_report_swaps(xmachine_message_fibroblast_damage_report_list* messages_swap){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	messages_swap->_position[index] = 0;
	messages_swap->_scan_input[index] = 0;
}

/* Message functions */

/** message_fibroblast_damage_report_grid_position
 * Calculates the grid cell position given an glm::vec3 vector
 * @param position glm::vec3 vector representing a position
 */
__device__ glm::ivec3 message_fibroblast_damage_report_grid_position(glm::vec3 position)
{
    glm::ivec3 gridPos;
    gridPos.x = floor((position.x - d_message_fibroblast_damage_report_min_bounds.x) * (float)d_message_fibroblast_damage_report_partitionDim.x / (d_message_fibroblast_damage_report_max_bounds.x - d_message_fibroblast_damage_report_min_bounds.x));
    gridPos.y = floor((position.y - d_message_fibroblast_damage_report_min_bounds.y) * (float)d_message_fibroblast_damage_report_partitionDim.y / (d_message_fibroblast_damage_report_max_bounds.y - d_message_fibroblast_damage_report_min_bounds.y));
    gridPos.z = floor((position.z - d_message_fibroblast_damage_report_min_bounds.z) * (float)d_message_fibroblast_damage_report_partitionDim.z / (d_message_fibroblast_damage_report_max_bounds.z - d_message_fibroblast_damage_report_min_bounds.z));

	//do wrapping or bounding
	

    return gridPos;
}

/** message_fibroblast_damage_report_hash
 * Given the grid position in partition space this function calculates a hash value
 * @param gridPos The position in partition space
 */
__device__ unsigned int message_fibroblast_damage_report_hash(glm::ivec3 gridPos)
{
	//cheap bounding without mod (within range +- partition dimension)
	gridPos.x = (gridPos.x<0)? d_message_fibroblast_damage_report_partitionDim.x-1: gridPos.x; 
	gridPos.x = (gridPos.x>=d_message_fibroblast_damage_report_partitionDim.x)? 0 : gridPos.x; 
	gridPos.y = (gridPos.y<0)? d_message_fibroblast_damage_report_partitionDim.y-1 : gridPos.y; 
	gridPos.y = (gridPos.y>=d_message_fibroblast_damage_report_partitionDim.y)? 0 : gridPos.y; 
	gridPos.z = (gridPos.z<0)? d_message_fibroblast_damage_report_partitionDim.z-1: gridPos.z; 
	gridPos.z = (gridPos.z>=d_message_fibroblast_damage_report_partitionDim.z)? 0 : gridPos.z; 

	//unique id
	return ((gridPos.z * d_message_fibroblast_damage_report_partitionDim.y) * d_message_fibroblast_damage_report_partitionDim.x) + (gridPos.y * d_message_fibroblast_damage_report_partitionDim.x) + gridPos.x;
}

#ifdef FAST_ATOMIC_SORTING
	/** hist_fibroblast_damage_report_messages
		 * Kernal function for performing a histogram (count) on each partition bin and saving the hash and index of a message within that bin
		 * @param local_bin_index output index of the message within the calculated bin
		 * @param unsorted_index output bin index (hash) value
		 * @param messages the message list used to generate the hash value outputs
		 * @param agent_count the current number of agents outputting messages
		 */
	__global__ void hist_fibroblast_damage_report_messages(uint* local_bin_index, uint* unsorted_index, int* global_bin_count, xmachine_message_fibroblast_damage_report_list* messages, int agent_count)
	{
		unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;

		if (index >= agent_count)
			return;
        glm::vec3 position = glm::vec3(messages->x[index], messages->y[index], messages->z[index]);
		glm::ivec3 grid_position = message_fibroblast_damage_report_grid_position(position);
		unsigned int hash = message_fibroblast_damage_report_hash(grid_position);
		unsigned int bin_idx = atomicInc((unsigned int*) &global_bin_count[hash], 0xFFFFFFFF);
		local_bin_index[index] = bin_idx;
		unsorted_index[index] = hash;
	}
	
	/** reorder_fibroblast_damage_report_messages
	 * Reorders the messages accoring to the partition boundary matrix start indices of each bin
	 * @param local_bin_index index of the message within the desired bin
	 * @param unsorted_index bin index (hash) value
	 * @param pbm_start_index the start indices of the partition boundary matrix
	 * @param unordered_messages the original unordered message data
	 * @param ordered_messages buffer used to scatter messages into the correct order
	  @param agent_count the current number of agents outputting messages
	 */
	 __global__ void reorder_fibroblast_damage_report_messages(uint* local_bin_index, uint* unsorted_index, int* pbm_start_index, xmachine_message_fibroblast_damage_report_list* unordered_messages, xmachine_message_fibroblast_damage_report_list* ordered_messages, int agent_count)
	{
		int index = (blockIdx.x *blockDim.x) + threadIdx.x;

		if (index >= agent_count)
			return;

		uint i = unsorted_index[index];
		unsigned int sorted_index = local_bin_index[index] + pbm_start_index[i];

		//finally reorder agent data
		ordered_messages->id[sorted_index] = unordered_messages->id[index];
		ordered_messages->x[sorted_index] = unordered_messages->x[index];
		ordered_messages->y[sorted_index] = unordered_messages->y[index];
		ordered_messages->z[sorted_index] = unordered_messages->z[index];
		ordered_messages->damage[sorted_index] = unordered_messages->damage[index];
	}
	 
#else

	/** hash_fibroblast_damage_report_messages
	 * Kernal function for calculating a hash value for each messahe depending on its position
	 * @param keys output for the hash key
	 * @param values output for the index value
	 * @param messages the message list used to generate the hash value outputs
	 */
	__global__ void hash_fibroblast_damage_report_messages(uint* keys, uint* values, xmachine_message_fibroblast_damage_report_list* messages)
	{
		unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;
        glm::vec3 position = glm::vec3(messages->x[index], messages->y[index], messages->z[index]);
		glm::ivec3 grid_position = message_fibroblast_damage_report_grid_position(position);
		unsigned int hash = message_fibroblast_damage_report_hash(grid_position);

		keys[index] = hash;
		values[index] = index;
	}

	/** reorder_fibroblast_damage_report_messages
	 * Reorders the messages accoring to the ordered sort identifiers and builds a Partition Boundary Matrix by looking at the previosu threads sort id.
	 * @param keys the sorted hash keys
	 * @param values the sorted index values
	 * @param matrix the PBM
	 * @param unordered_messages the original unordered message data
	 * @param ordered_messages buffer used to scatter messages into the correct order
	 */
	__global__ void reorder_fibroblast_damage_report_messages(uint* keys, uint* values, xmachine_message_fibroblast_damage_report_PBM* matrix, xmachine_message_fibroblast_damage_report_list* unordered_messages, xmachine_message_fibroblast_damage_report_list* ordered_messages)
	{
		extern __shared__ int sm_data [];

		int index = (blockIdx.x * blockDim.x) + threadIdx.x;

		//load threads sort key into sm
		uint key = keys[index];
		uint old_pos = values[index];

		sm_data[threadIdx.x] = key;
		__syncthreads();
	
		unsigned int prev_key;

		//if first thread then no prev sm value so get prev from global memory 
		if (threadIdx.x == 0)
		{
			//first thread has no prev value so ignore
			if (index != 0)
				prev_key = keys[index-1];
		}
		//get previous ident from sm
		else	
		{
			prev_key = sm_data[threadIdx.x-1];
		}

		//TODO: Check key is not out of bounds

		//set partition boundaries
		if (index < d_message_fibroblast_damage_report_count)
		{
			//if first thread then set first partition cell start
			if (index == 0)
			{
				matrix->start[key] = index;
			}

			//if edge of a boundr update start and end of partition
			else if (prev_key != key)
			{
				//set start for key
				matrix->start[key] = index;

				//set end for key -1
				matrix->end_or_count[prev_key] = index;
			}

			//if last thread then set final partition cell end
			if (index == d_message_fibroblast_damage_report_count-1)
			{
				matrix->end_or_count[key] = index+1;
			}
		}
	
		//finally reorder agent data
		ordered_messages->id[index] = unordered_messages->id[old_pos];
		ordered_messages->x[index] = unordered_messages->x[old_pos];
		ordered_messages->y[index] = unordered_messages->y[old_pos];
		ordered_messages->z[index] = unordered_messages->z[old_pos];
		ordered_messages->damage[index] = unordered_messages->damage[old_pos];
	}

#endif

/** load_next_fibroblast_damage_report_message
 * Used to load the next message data to shared memory
 * Idea is check the current cell index to see if we can simply get a message from the current cell
 * If we are at the end of the current cell then loop till we find the next cell with messages (this way we ignore cells with no messages)
 * @param messages the message list
 * @param partition_matrix the PBM
 * @param relative_cell the relative partition cell position from the agent position
 * @param cell_index_max the maximum index of the current partition cell
 * @param agent_grid_cell the agents partition cell position
 * @param cell_index the current cell index in agent_grid_cell+relative_cell
 * @return true if a message has been loaded into sm false otherwise
 */
__device__ bool load_next_fibroblast_damage_report_message(xmachine_message_fibroblast_damage_report_list* messages, xmachine_message_fibroblast_damage_report_PBM* partition_matrix, glm::ivec3 relative_cell, int cell_index_max, glm::ivec3 agent_grid_cell, int cell_index)
{
	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];

	int move_cell = true;
	cell_index ++;

	//see if we need to move to a new partition cell
	if(cell_index < cell_index_max)
		move_cell = false;

	while(move_cell)
	{
		//get the next relative grid position 
        if (next_cell3D(&relative_cell))
		{
			//calculate the next cells grid position and hash
			glm::ivec3 next_cell_position = agent_grid_cell + relative_cell;
			int next_cell_hash = message_fibroblast_damage_report_hash(next_cell_position);
			//use the hash to calculate the start index
			int cell_index_min = tex1Dfetch(tex_xmachine_message_fibroblast_damage_report_pbm_start, next_cell_hash + d_tex_xmachine_message_fibroblast_damage_report_pbm_start_offset);
			cell_index_max = tex1Dfetch(tex_xmachine_message_fibroblast_damage_report_pbm_end_or_count, next_cell_hash + d_tex_xmachine_message_fibroblast_damage_report_pbm_end_or_count_offset);
			//check for messages in the cell (cell index max is the count for atomic sorting)
#ifdef FAST_ATOMIC_SORTING
			if (cell_index_max > 0)
			{
				//when using fast atomics value represents bin count not last index!
				cell_index_max += cell_index_min; //when using fast atomics value represents bin count not last index!
#else
			if (cell_index_min != 0xffffffff)
			{
#endif
				//start from the cell index min
				cell_index = cell_index_min;
				//exit the loop as we have found a valid cell with message data
				move_cell = false;
			}
		}
		else
		{
			//we have exhausted all the neighbouring cells so there are no more messages
			return false;
		}
	}
	
	//get message data using texture fetch
	xmachine_message_fibroblast_damage_report temp_message;
	temp_message._relative_cell = relative_cell;
	temp_message._cell_index_max = cell_index_max;
	temp_message._cell_index = cell_index;
	temp_message._agent_grid_cell = agent_grid_cell;

	//Using texture cache
  temp_message.id = tex1Dfetch(tex_xmachine_message_fibroblast_damage_report_id, cell_index + d_tex_xmachine_message_fibroblast_damage_report_id_offset); temp_message.x = tex1Dfetch(tex_xmachine_message_fibroblast_damage_report_x, cell_index + d_tex_xmachine_message_fibroblast_damage_report_x_offset); temp_message.y = tex1Dfetch(tex_xmachine_message_fibroblast_damage_report_y, cell_index + d_tex_xmachine_message_fibroblast_damage_report_y_offset); temp_message.z = tex1Dfetch(tex_xmachine_message_fibroblast_damage_report_z, cell_index + d_tex_xmachine_message_fibroblast_damage_report_z_offset); temp_message.damage = tex1Dfetch(tex_xmachine_message_fibroblast_damage_report_damage, cell_index + d_tex_xmachine_message_fibroblast_damage_report_damage_offset); 

	//load it into shared memory (no sync as no sharing between threads)
	int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_fibroblast_damage_report));
	xmachine_message_fibroblast_damage_report* sm_message = ((xmachine_message_fibroblast_damage_report*)&message_share[message_index]);
	sm_message[0] = temp_message;

	return true;
}


/*
 * get first spatial partitioned fibroblast_damage_report message (first batch load into shared memory)
 */
__device__ xmachine_message_fibroblast_damage_report* get_first_fibroblast_damage_report_message(xmachine_message_fibroblast_damage_report_list* messages, xmachine_message_fibroblast_damage_report_PBM* partition_matrix, float x, float y, float z){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];

	// If there are no messages, do not load any messages
	if(d_message_fibroblast_damage_report_count == 0){
		return nullptr;
	}

	glm::ivec3 relative_cell = glm::ivec3(-2, -1, -1);
	int cell_index_max = 0;
	int cell_index = 0;
	glm::vec3 position = glm::vec3(x, y, z);
	glm::ivec3 agent_grid_cell = message_fibroblast_damage_report_grid_position(position);
	
	if (load_next_fibroblast_damage_report_message(messages, partition_matrix, relative_cell, cell_index_max, agent_grid_cell, cell_index))
	{
		int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_fibroblast_damage_report));
		return ((xmachine_message_fibroblast_damage_report*)&message_share[message_index]);
	}
	else
	{
		return nullptr;
	}
}

/*
 * get next spatial partitioned fibroblast_damage_report message (either from SM or next batch load)
 */
__device__ xmachine_message_fibroblast_damage_report* get_next_fibroblast_damage_report_message(xmachine_message_fibroblast_damage_report* message, xmachine_message_fibroblast_damage_report_list* messages, xmachine_message_fibroblast_damage_report_PBM* partition_matrix){
	
	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	// If there are no messages, do not load any messages
	if(d_message_fibroblast_damage_report_count == 0){
		return nullptr;
	}
	
	if (load_next_fibroblast_damage_report_message(messages, partition_matrix, message->_relative_cell, message->_cell_index_max, message->_agent_grid_cell, message->_cell_index))
	{
		//get conflict free address of 
		int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_fibroblast_damage_report));
		return ((xmachine_message_fibroblast_damage_report*)&message_share[message_index]);
	}
	else
		return nullptr;
	
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created tissue_damage_report message functions */


/** add_tissue_damage_report_message
 * Add non partitioned or spatially partitioned tissue_damage_report message
 * @param messages xmachine_message_tissue_damage_report_list message list to add too
 * @param id agent variable of type int
 * @param x agent variable of type float
 * @param y agent variable of type float
 * @param z agent variable of type float
 * @param damage agent variable of type int
 */
__device__ void add_tissue_damage_report_message(xmachine_message_tissue_damage_report_list* messages, int id, float x, float y, float z, int damage){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x + d_message_tissue_damage_report_count;

	int _position;
	int _scan_input;

	//decide output position
	if(d_message_tissue_damage_report_output_type == single_message){
		_position = index; //same as agent position
		_scan_input = 0;
	}else if (d_message_tissue_damage_report_output_type == optional_message){
		_position = 0;	   //to be calculated using Prefix sum
		_scan_input = 1;
	}

	//AoS - xmachine_message_tissue_damage_report Coalesced memory write
	messages->_scan_input[index] = _scan_input;	
	messages->_position[index] = _position;
	messages->id[index] = id;
	messages->x[index] = x;
	messages->y[index] = y;
	messages->z[index] = z;
	messages->damage[index] = damage;

}

/**
 * Scatter non partitioned or spatially partitioned tissue_damage_report message (for optional messages)
 * @param messages scatter_optional_tissue_damage_report_messages Sparse xmachine_message_tissue_damage_report_list message list
 * @param message_swap temp xmachine_message_tissue_damage_report_list message list to scatter sparse messages to
 */
__global__ void scatter_optional_tissue_damage_report_messages(xmachine_message_tissue_damage_report_list* messages, xmachine_message_tissue_damage_report_list* messages_swap){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = messages_swap->_scan_input[index];

	//if optional message is to be written
	if (_scan_input == 1){
		int output_index = messages_swap->_position[index] + d_message_tissue_damage_report_count;

		//AoS - xmachine_message_tissue_damage_report Un-Coalesced scattered memory write
		messages->_position[output_index] = output_index;
		messages->id[output_index] = messages_swap->id[index];
		messages->x[output_index] = messages_swap->x[index];
		messages->y[output_index] = messages_swap->y[index];
		messages->z[output_index] = messages_swap->z[index];
		messages->damage[output_index] = messages_swap->damage[index];				
	}
}

/** reset_tissue_damage_report_swaps
 * Reset non partitioned or spatially partitioned tissue_damage_report message swaps (for scattering optional messages)
 * @param message_swap message list to reset _position and _scan_input values back to 0
 */
__global__ void reset_tissue_damage_report_swaps(xmachine_message_tissue_damage_report_list* messages_swap){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	messages_swap->_position[index] = 0;
	messages_swap->_scan_input[index] = 0;
}

/* Message functions */

/** message_tissue_damage_report_grid_position
 * Calculates the grid cell position given an glm::vec3 vector
 * @param position glm::vec3 vector representing a position
 */
__device__ glm::ivec3 message_tissue_damage_report_grid_position(glm::vec3 position)
{
    glm::ivec3 gridPos;
    gridPos.x = floor((position.x - d_message_tissue_damage_report_min_bounds.x) * (float)d_message_tissue_damage_report_partitionDim.x / (d_message_tissue_damage_report_max_bounds.x - d_message_tissue_damage_report_min_bounds.x));
    gridPos.y = floor((position.y - d_message_tissue_damage_report_min_bounds.y) * (float)d_message_tissue_damage_report_partitionDim.y / (d_message_tissue_damage_report_max_bounds.y - d_message_tissue_damage_report_min_bounds.y));
    gridPos.z = floor((position.z - d_message_tissue_damage_report_min_bounds.z) * (float)d_message_tissue_damage_report_partitionDim.z / (d_message_tissue_damage_report_max_bounds.z - d_message_tissue_damage_report_min_bounds.z));

	//do wrapping or bounding
	

    return gridPos;
}

/** message_tissue_damage_report_hash
 * Given the grid position in partition space this function calculates a hash value
 * @param gridPos The position in partition space
 */
__device__ unsigned int message_tissue_damage_report_hash(glm::ivec3 gridPos)
{
	//cheap bounding without mod (within range +- partition dimension)
	gridPos.x = (gridPos.x<0)? d_message_tissue_damage_report_partitionDim.x-1: gridPos.x; 
	gridPos.x = (gridPos.x>=d_message_tissue_damage_report_partitionDim.x)? 0 : gridPos.x; 
	gridPos.y = (gridPos.y<0)? d_message_tissue_damage_report_partitionDim.y-1 : gridPos.y; 
	gridPos.y = (gridPos.y>=d_message_tissue_damage_report_partitionDim.y)? 0 : gridPos.y; 
	gridPos.z = (gridPos.z<0)? d_message_tissue_damage_report_partitionDim.z-1: gridPos.z; 
	gridPos.z = (gridPos.z>=d_message_tissue_damage_report_partitionDim.z)? 0 : gridPos.z; 

	//unique id
	return ((gridPos.z * d_message_tissue_damage_report_partitionDim.y) * d_message_tissue_damage_report_partitionDim.x) + (gridPos.y * d_message_tissue_damage_report_partitionDim.x) + gridPos.x;
}

#ifdef FAST_ATOMIC_SORTING
	/** hist_tissue_damage_report_messages
		 * Kernal function for performing a histogram (count) on each partition bin and saving the hash and index of a message within that bin
		 * @param local_bin_index output index of the message within the calculated bin
		 * @param unsorted_index output bin index (hash) value
		 * @param messages the message list used to generate the hash value outputs
		 * @param agent_count the current number of agents outputting messages
		 */
	__global__ void hist_tissue_damage_report_messages(uint* local_bin_index, uint* unsorted_index, int* global_bin_count, xmachine_message_tissue_damage_report_list* messages, int agent_count)
	{
		unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;

		if (index >= agent_count)
			return;
        glm::vec3 position = glm::vec3(messages->x[index], messages->y[index], messages->z[index]);
		glm::ivec3 grid_position = message_tissue_damage_report_grid_position(position);
		unsigned int hash = message_tissue_damage_report_hash(grid_position);
		unsigned int bin_idx = atomicInc((unsigned int*) &global_bin_count[hash], 0xFFFFFFFF);
		local_bin_index[index] = bin_idx;
		unsorted_index[index] = hash;
	}
	
	/** reorder_tissue_damage_report_messages
	 * Reorders the messages accoring to the partition boundary matrix start indices of each bin
	 * @param local_bin_index index of the message within the desired bin
	 * @param unsorted_index bin index (hash) value
	 * @param pbm_start_index the start indices of the partition boundary matrix
	 * @param unordered_messages the original unordered message data
	 * @param ordered_messages buffer used to scatter messages into the correct order
	  @param agent_count the current number of agents outputting messages
	 */
	 __global__ void reorder_tissue_damage_report_messages(uint* local_bin_index, uint* unsorted_index, int* pbm_start_index, xmachine_message_tissue_damage_report_list* unordered_messages, xmachine_message_tissue_damage_report_list* ordered_messages, int agent_count)
	{
		int index = (blockIdx.x *blockDim.x) + threadIdx.x;

		if (index >= agent_count)
			return;

		uint i = unsorted_index[index];
		unsigned int sorted_index = local_bin_index[index] + pbm_start_index[i];

		//finally reorder agent data
		ordered_messages->id[sorted_index] = unordered_messages->id[index];
		ordered_messages->x[sorted_index] = unordered_messages->x[index];
		ordered_messages->y[sorted_index] = unordered_messages->y[index];
		ordered_messages->z[sorted_index] = unordered_messages->z[index];
		ordered_messages->damage[sorted_index] = unordered_messages->damage[index];
	}
	 
#else

	/** hash_tissue_damage_report_messages
	 * Kernal function for calculating a hash value for each messahe depending on its position
	 * @param keys output for the hash key
	 * @param values output for the index value
	 * @param messages the message list used to generate the hash value outputs
	 */
	__global__ void hash_tissue_damage_report_messages(uint* keys, uint* values, xmachine_message_tissue_damage_report_list* messages)
	{
		unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;
        glm::vec3 position = glm::vec3(messages->x[index], messages->y[index], messages->z[index]);
		glm::ivec3 grid_position = message_tissue_damage_report_grid_position(position);
		unsigned int hash = message_tissue_damage_report_hash(grid_position);

		keys[index] = hash;
		values[index] = index;
	}

	/** reorder_tissue_damage_report_messages
	 * Reorders the messages accoring to the ordered sort identifiers and builds a Partition Boundary Matrix by looking at the previosu threads sort id.
	 * @param keys the sorted hash keys
	 * @param values the sorted index values
	 * @param matrix the PBM
	 * @param unordered_messages the original unordered message data
	 * @param ordered_messages buffer used to scatter messages into the correct order
	 */
	__global__ void reorder_tissue_damage_report_messages(uint* keys, uint* values, xmachine_message_tissue_damage_report_PBM* matrix, xmachine_message_tissue_damage_report_list* unordered_messages, xmachine_message_tissue_damage_report_list* ordered_messages)
	{
		extern __shared__ int sm_data [];

		int index = (blockIdx.x * blockDim.x) + threadIdx.x;

		//load threads sort key into sm
		uint key = keys[index];
		uint old_pos = values[index];

		sm_data[threadIdx.x] = key;
		__syncthreads();
	
		unsigned int prev_key;

		//if first thread then no prev sm value so get prev from global memory 
		if (threadIdx.x == 0)
		{
			//first thread has no prev value so ignore
			if (index != 0)
				prev_key = keys[index-1];
		}
		//get previous ident from sm
		else	
		{
			prev_key = sm_data[threadIdx.x-1];
		}

		//TODO: Check key is not out of bounds

		//set partition boundaries
		if (index < d_message_tissue_damage_report_count)
		{
			//if first thread then set first partition cell start
			if (index == 0)
			{
				matrix->start[key] = index;
			}

			//if edge of a boundr update start and end of partition
			else if (prev_key != key)
			{
				//set start for key
				matrix->start[key] = index;

				//set end for key -1
				matrix->end_or_count[prev_key] = index;
			}

			//if last thread then set final partition cell end
			if (index == d_message_tissue_damage_report_count-1)
			{
				matrix->end_or_count[key] = index+1;
			}
		}
	
		//finally reorder agent data
		ordered_messages->id[index] = unordered_messages->id[old_pos];
		ordered_messages->x[index] = unordered_messages->x[old_pos];
		ordered_messages->y[index] = unordered_messages->y[old_pos];
		ordered_messages->z[index] = unordered_messages->z[old_pos];
		ordered_messages->damage[index] = unordered_messages->damage[old_pos];
	}

#endif

/** load_next_tissue_damage_report_message
 * Used to load the next message data to shared memory
 * Idea is check the current cell index to see if we can simply get a message from the current cell
 * If we are at the end of the current cell then loop till we find the next cell with messages (this way we ignore cells with no messages)
 * @param messages the message list
 * @param partition_matrix the PBM
 * @param relative_cell the relative partition cell position from the agent position
 * @param cell_index_max the maximum index of the current partition cell
 * @param agent_grid_cell the agents partition cell position
 * @param cell_index the current cell index in agent_grid_cell+relative_cell
 * @return true if a message has been loaded into sm false otherwise
 */
__device__ bool load_next_tissue_damage_report_message(xmachine_message_tissue_damage_report_list* messages, xmachine_message_tissue_damage_report_PBM* partition_matrix, glm::ivec3 relative_cell, int cell_index_max, glm::ivec3 agent_grid_cell, int cell_index)
{
	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];

	int move_cell = true;
	cell_index ++;

	//see if we need to move to a new partition cell
	if(cell_index < cell_index_max)
		move_cell = false;

	while(move_cell)
	{
		//get the next relative grid position 
        if (next_cell3D(&relative_cell))
		{
			//calculate the next cells grid position and hash
			glm::ivec3 next_cell_position = agent_grid_cell + relative_cell;
			int next_cell_hash = message_tissue_damage_report_hash(next_cell_position);
			//use the hash to calculate the start index
			int cell_index_min = tex1Dfetch(tex_xmachine_message_tissue_damage_report_pbm_start, next_cell_hash + d_tex_xmachine_message_tissue_damage_report_pbm_start_offset);
			cell_index_max = tex1Dfetch(tex_xmachine_message_tissue_damage_report_pbm_end_or_count, next_cell_hash + d_tex_xmachine_message_tissue_damage_report_pbm_end_or_count_offset);
			//check for messages in the cell (cell index max is the count for atomic sorting)
#ifdef FAST_ATOMIC_SORTING
			if (cell_index_max > 0)
			{
				//when using fast atomics value represents bin count not last index!
				cell_index_max += cell_index_min; //when using fast atomics value represents bin count not last index!
#else
			if (cell_index_min != 0xffffffff)
			{
#endif
				//start from the cell index min
				cell_index = cell_index_min;
				//exit the loop as we have found a valid cell with message data
				move_cell = false;
			}
		}
		else
		{
			//we have exhausted all the neighbouring cells so there are no more messages
			return false;
		}
	}
	
	//get message data using texture fetch
	xmachine_message_tissue_damage_report temp_message;
	temp_message._relative_cell = relative_cell;
	temp_message._cell_index_max = cell_index_max;
	temp_message._cell_index = cell_index;
	temp_message._agent_grid_cell = agent_grid_cell;

	//Using texture cache
  temp_message.id = tex1Dfetch(tex_xmachine_message_tissue_damage_report_id, cell_index + d_tex_xmachine_message_tissue_damage_report_id_offset); temp_message.x = tex1Dfetch(tex_xmachine_message_tissue_damage_report_x, cell_index + d_tex_xmachine_message_tissue_damage_report_x_offset); temp_message.y = tex1Dfetch(tex_xmachine_message_tissue_damage_report_y, cell_index + d_tex_xmachine_message_tissue_damage_report_y_offset); temp_message.z = tex1Dfetch(tex_xmachine_message_tissue_damage_report_z, cell_index + d_tex_xmachine_message_tissue_damage_report_z_offset); temp_message.damage = tex1Dfetch(tex_xmachine_message_tissue_damage_report_damage, cell_index + d_tex_xmachine_message_tissue_damage_report_damage_offset); 

	//load it into shared memory (no sync as no sharing between threads)
	int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_tissue_damage_report));
	xmachine_message_tissue_damage_report* sm_message = ((xmachine_message_tissue_damage_report*)&message_share[message_index]);
	sm_message[0] = temp_message;

	return true;
}


/*
 * get first spatial partitioned tissue_damage_report message (first batch load into shared memory)
 */
__device__ xmachine_message_tissue_damage_report* get_first_tissue_damage_report_message(xmachine_message_tissue_damage_report_list* messages, xmachine_message_tissue_damage_report_PBM* partition_matrix, float x, float y, float z){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];

	// If there are no messages, do not load any messages
	if(d_message_tissue_damage_report_count == 0){
		return nullptr;
	}

	glm::ivec3 relative_cell = glm::ivec3(-2, -1, -1);
	int cell_index_max = 0;
	int cell_index = 0;
	glm::vec3 position = glm::vec3(x, y, z);
	glm::ivec3 agent_grid_cell = message_tissue_damage_report_grid_position(position);
	
	if (load_next_tissue_damage_report_message(messages, partition_matrix, relative_cell, cell_index_max, agent_grid_cell, cell_index))
	{
		int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_tissue_damage_report));
		return ((xmachine_message_tissue_damage_report*)&message_share[message_index]);
	}
	else
	{
		return nullptr;
	}
}

/*
 * get next spatial partitioned tissue_damage_report message (either from SM or next batch load)
 */
__device__ xmachine_message_tissue_damage_report* get_next_tissue_damage_report_message(xmachine_message_tissue_damage_report* message, xmachine_message_tissue_damage_report_list* messages, xmachine_message_tissue_damage_report_PBM* partition_matrix){
	
	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	// If there are no messages, do not load any messages
	if(d_message_tissue_damage_report_count == 0){
		return nullptr;
	}
	
	if (load_next_tissue_damage_report_message(messages, partition_matrix, message->_relative_cell, message->_cell_index_max, message->_agent_grid_cell, message->_cell_index))
	{
		//get conflict free address of 
		int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_tissue_damage_report));
		return ((xmachine_message_tissue_damage_report*)&message_share[message_index]);
	}
	else
		return nullptr;
	
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created doublings message functions */


/** add_doublings_message
 * Add non partitioned or spatially partitioned doublings message
 * @param messages xmachine_message_doublings_list message list to add too
 * @param number agent variable of type int
 */
__device__ void add_doublings_message(xmachine_message_doublings_list* messages, int number){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x + d_message_doublings_count;

	int _position;
	int _scan_input;

	//decide output position
	if(d_message_doublings_output_type == single_message){
		_position = index; //same as agent position
		_scan_input = 0;
	}else if (d_message_doublings_output_type == optional_message){
		_position = 0;	   //to be calculated using Prefix sum
		_scan_input = 1;
	}

	//AoS - xmachine_message_doublings Coalesced memory write
	messages->_scan_input[index] = _scan_input;	
	messages->_position[index] = _position;
	messages->number[index] = number;

}

/**
 * Scatter non partitioned or spatially partitioned doublings message (for optional messages)
 * @param messages scatter_optional_doublings_messages Sparse xmachine_message_doublings_list message list
 * @param message_swap temp xmachine_message_doublings_list message list to scatter sparse messages to
 */
__global__ void scatter_optional_doublings_messages(xmachine_message_doublings_list* messages, xmachine_message_doublings_list* messages_swap){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = messages_swap->_scan_input[index];

	//if optional message is to be written
	if (_scan_input == 1){
		int output_index = messages_swap->_position[index] + d_message_doublings_count;

		//AoS - xmachine_message_doublings Un-Coalesced scattered memory write
		messages->_position[output_index] = output_index;
		messages->number[output_index] = messages_swap->number[index];				
	}
}

/** reset_doublings_swaps
 * Reset non partitioned or spatially partitioned doublings message swaps (for scattering optional messages)
 * @param message_swap message list to reset _position and _scan_input values back to 0
 */
__global__ void reset_doublings_swaps(xmachine_message_doublings_list* messages_swap){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	messages_swap->_position[index] = 0;
	messages_swap->_scan_input[index] = 0;
}

/* Message functions */

__device__ xmachine_message_doublings* get_first_doublings_message(xmachine_message_doublings_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = (ceil((float)d_message_doublings_count/ blockDim.x)* blockDim.x);

	//if no messages then return a null pointer (false)
	if (wrap_size == 0)
		return nullptr;

	//global thread index
	int global_index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//global thread index
	int index = WRAP(global_index, wrap_size);

	//SoA to AoS - xmachine_message_doublings Coalesced memory read
	xmachine_message_doublings temp_message;
	temp_message._position = messages->_position[index];
	temp_message.number = messages->number[index];

	//AoS to shared memory
	int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_doublings));
	xmachine_message_doublings* sm_message = ((xmachine_message_doublings*)&message_share[message_index]);
	sm_message[0] = temp_message;

	__syncthreads();

  //HACK FOR 64 bit addressing issue in sm
	return ((xmachine_message_doublings*)&message_share[d_SM_START]);
}

__device__ xmachine_message_doublings* get_next_doublings_message(xmachine_message_doublings* message, xmachine_message_doublings_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = ceil((float)d_message_doublings_count/ blockDim.x)*blockDim.x;

	int i = WRAP((message->_position + 1),wrap_size);

	//If end of messages (last message not multiple of gridsize) go to 0 index
	if (i >= d_message_doublings_count)
		i = 0;

	//Check if back to start position of first message
	if (i == WRAP((blockDim.x* blockIdx.x), wrap_size))
		return nullptr;

	int tile = floor((float)i/(blockDim.x)); //tile is round down position over blockDim
	i = i % blockDim.x;						 //mod i for shared memory index

	//if count == Block Size load next tile int shared memory values
	if (i == 0){
		__syncthreads();					//make sure we don't change shared memory until all threads are here (important for emu-debug mode)
		
		//SoA to AoS - xmachine_message_doublings Coalesced memory read
		int index = (tile* blockDim.x) + threadIdx.x;
		xmachine_message_doublings temp_message;
		temp_message._position = messages->_position[index];
		temp_message.number = messages->number[index];

		//AoS to shared memory
		int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_doublings));
		xmachine_message_doublings* sm_message = ((xmachine_message_doublings*)&message_share[message_index]);
		sm_message[0] = temp_message;

		__syncthreads();					//make sure we don't start returning messages until all threads have updated shared memory
	}

	int message_index = SHARE_INDEX(i, sizeof(xmachine_message_doublings));
	return ((xmachine_message_doublings*)&message_share[message_index]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created count message functions */


/** add_count_message
 * Add non partitioned or spatially partitioned count message
 * @param messages xmachine_message_count_list message list to add too
 * @param number agent variable of type int
 */
__device__ void add_count_message(xmachine_message_count_list* messages, int number){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x + d_message_count_count;

	int _position;
	int _scan_input;

	//decide output position
	if(d_message_count_output_type == single_message){
		_position = index; //same as agent position
		_scan_input = 0;
	}else if (d_message_count_output_type == optional_message){
		_position = 0;	   //to be calculated using Prefix sum
		_scan_input = 1;
	}

	//AoS - xmachine_message_count Coalesced memory write
	messages->_scan_input[index] = _scan_input;	
	messages->_position[index] = _position;
	messages->number[index] = number;

}

/**
 * Scatter non partitioned or spatially partitioned count message (for optional messages)
 * @param messages scatter_optional_count_messages Sparse xmachine_message_count_list message list
 * @param message_swap temp xmachine_message_count_list message list to scatter sparse messages to
 */
__global__ void scatter_optional_count_messages(xmachine_message_count_list* messages, xmachine_message_count_list* messages_swap){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = messages_swap->_scan_input[index];

	//if optional message is to be written
	if (_scan_input == 1){
		int output_index = messages_swap->_position[index] + d_message_count_count;

		//AoS - xmachine_message_count Un-Coalesced scattered memory write
		messages->_position[output_index] = output_index;
		messages->number[output_index] = messages_swap->number[index];				
	}
}

/** reset_count_swaps
 * Reset non partitioned or spatially partitioned count message swaps (for scattering optional messages)
 * @param message_swap message list to reset _position and _scan_input values back to 0
 */
__global__ void reset_count_swaps(xmachine_message_count_list* messages_swap){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	messages_swap->_position[index] = 0;
	messages_swap->_scan_input[index] = 0;
}

/* Message functions */

__device__ xmachine_message_count* get_first_count_message(xmachine_message_count_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = (ceil((float)d_message_count_count/ blockDim.x)* blockDim.x);

	//if no messages then return a null pointer (false)
	if (wrap_size == 0)
		return nullptr;

	//global thread index
	int global_index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//global thread index
	int index = WRAP(global_index, wrap_size);

	//SoA to AoS - xmachine_message_count Coalesced memory read
	xmachine_message_count temp_message;
	temp_message._position = messages->_position[index];
	temp_message.number = messages->number[index];

	//AoS to shared memory
	int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_count));
	xmachine_message_count* sm_message = ((xmachine_message_count*)&message_share[message_index]);
	sm_message[0] = temp_message;

	__syncthreads();

  //HACK FOR 64 bit addressing issue in sm
	return ((xmachine_message_count*)&message_share[d_SM_START]);
}

__device__ xmachine_message_count* get_next_count_message(xmachine_message_count* message, xmachine_message_count_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = ceil((float)d_message_count_count/ blockDim.x)*blockDim.x;

	int i = WRAP((message->_position + 1),wrap_size);

	//If end of messages (last message not multiple of gridsize) go to 0 index
	if (i >= d_message_count_count)
		i = 0;

	//Check if back to start position of first message
	if (i == WRAP((blockDim.x* blockIdx.x), wrap_size))
		return nullptr;

	int tile = floor((float)i/(blockDim.x)); //tile is round down position over blockDim
	i = i % blockDim.x;						 //mod i for shared memory index

	//if count == Block Size load next tile int shared memory values
	if (i == 0){
		__syncthreads();					//make sure we don't change shared memory until all threads are here (important for emu-debug mode)
		
		//SoA to AoS - xmachine_message_count Coalesced memory read
		int index = (tile* blockDim.x) + threadIdx.x;
		xmachine_message_count temp_message;
		temp_message._position = messages->_position[index];
		temp_message.number = messages->number[index];

		//AoS to shared memory
		int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_count));
		xmachine_message_count* sm_message = ((xmachine_message_count*)&message_share[message_index]);
		sm_message[0] = temp_message;

		__syncthreads();					//make sure we don't start returning messages until all threads have updated shared memory
	}

	int message_index = SHARE_INDEX(i, sizeof(xmachine_message_count));
	return ((xmachine_message_count*)&message_share[message_index]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created quiescent_location_report message functions */


/** add_quiescent_location_report_message
 * Add non partitioned or spatially partitioned quiescent_location_report message
 * @param messages xmachine_message_quiescent_location_report_list message list to add too
 * @param id agent variable of type int
 * @param x agent variable of type float
 * @param y agent variable of type float
 * @param z agent variable of type float
 */
__device__ void add_quiescent_location_report_message(xmachine_message_quiescent_location_report_list* messages, int id, float x, float y, float z){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x + d_message_quiescent_location_report_count;

	int _position;
	int _scan_input;

	//decide output position
	if(d_message_quiescent_location_report_output_type == single_message){
		_position = index; //same as agent position
		_scan_input = 0;
	}else if (d_message_quiescent_location_report_output_type == optional_message){
		_position = 0;	   //to be calculated using Prefix sum
		_scan_input = 1;
	}

	//AoS - xmachine_message_quiescent_location_report Coalesced memory write
	messages->_scan_input[index] = _scan_input;	
	messages->_position[index] = _position;
	messages->id[index] = id;
	messages->x[index] = x;
	messages->y[index] = y;
	messages->z[index] = z;

}

/**
 * Scatter non partitioned or spatially partitioned quiescent_location_report message (for optional messages)
 * @param messages scatter_optional_quiescent_location_report_messages Sparse xmachine_message_quiescent_location_report_list message list
 * @param message_swap temp xmachine_message_quiescent_location_report_list message list to scatter sparse messages to
 */
__global__ void scatter_optional_quiescent_location_report_messages(xmachine_message_quiescent_location_report_list* messages, xmachine_message_quiescent_location_report_list* messages_swap){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = messages_swap->_scan_input[index];

	//if optional message is to be written
	if (_scan_input == 1){
		int output_index = messages_swap->_position[index] + d_message_quiescent_location_report_count;

		//AoS - xmachine_message_quiescent_location_report Un-Coalesced scattered memory write
		messages->_position[output_index] = output_index;
		messages->id[output_index] = messages_swap->id[index];
		messages->x[output_index] = messages_swap->x[index];
		messages->y[output_index] = messages_swap->y[index];
		messages->z[output_index] = messages_swap->z[index];				
	}
}

/** reset_quiescent_location_report_swaps
 * Reset non partitioned or spatially partitioned quiescent_location_report message swaps (for scattering optional messages)
 * @param message_swap message list to reset _position and _scan_input values back to 0
 */
__global__ void reset_quiescent_location_report_swaps(xmachine_message_quiescent_location_report_list* messages_swap){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	messages_swap->_position[index] = 0;
	messages_swap->_scan_input[index] = 0;
}

/* Message functions */

/** message_quiescent_location_report_grid_position
 * Calculates the grid cell position given an glm::vec3 vector
 * @param position glm::vec3 vector representing a position
 */
__device__ glm::ivec3 message_quiescent_location_report_grid_position(glm::vec3 position)
{
    glm::ivec3 gridPos;
    gridPos.x = floor((position.x - d_message_quiescent_location_report_min_bounds.x) * (float)d_message_quiescent_location_report_partitionDim.x / (d_message_quiescent_location_report_max_bounds.x - d_message_quiescent_location_report_min_bounds.x));
    gridPos.y = floor((position.y - d_message_quiescent_location_report_min_bounds.y) * (float)d_message_quiescent_location_report_partitionDim.y / (d_message_quiescent_location_report_max_bounds.y - d_message_quiescent_location_report_min_bounds.y));
    gridPos.z = floor((position.z - d_message_quiescent_location_report_min_bounds.z) * (float)d_message_quiescent_location_report_partitionDim.z / (d_message_quiescent_location_report_max_bounds.z - d_message_quiescent_location_report_min_bounds.z));

	//do wrapping or bounding
	

    return gridPos;
}

/** message_quiescent_location_report_hash
 * Given the grid position in partition space this function calculates a hash value
 * @param gridPos The position in partition space
 */
__device__ unsigned int message_quiescent_location_report_hash(glm::ivec3 gridPos)
{
	//cheap bounding without mod (within range +- partition dimension)
	gridPos.x = (gridPos.x<0)? d_message_quiescent_location_report_partitionDim.x-1: gridPos.x; 
	gridPos.x = (gridPos.x>=d_message_quiescent_location_report_partitionDim.x)? 0 : gridPos.x; 
	gridPos.y = (gridPos.y<0)? d_message_quiescent_location_report_partitionDim.y-1 : gridPos.y; 
	gridPos.y = (gridPos.y>=d_message_quiescent_location_report_partitionDim.y)? 0 : gridPos.y; 
	gridPos.z = (gridPos.z<0)? d_message_quiescent_location_report_partitionDim.z-1: gridPos.z; 
	gridPos.z = (gridPos.z>=d_message_quiescent_location_report_partitionDim.z)? 0 : gridPos.z; 

	//unique id
	return ((gridPos.z * d_message_quiescent_location_report_partitionDim.y) * d_message_quiescent_location_report_partitionDim.x) + (gridPos.y * d_message_quiescent_location_report_partitionDim.x) + gridPos.x;
}

#ifdef FAST_ATOMIC_SORTING
	/** hist_quiescent_location_report_messages
		 * Kernal function for performing a histogram (count) on each partition bin and saving the hash and index of a message within that bin
		 * @param local_bin_index output index of the message within the calculated bin
		 * @param unsorted_index output bin index (hash) value
		 * @param messages the message list used to generate the hash value outputs
		 * @param agent_count the current number of agents outputting messages
		 */
	__global__ void hist_quiescent_location_report_messages(uint* local_bin_index, uint* unsorted_index, int* global_bin_count, xmachine_message_quiescent_location_report_list* messages, int agent_count)
	{
		unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;

		if (index >= agent_count)
			return;
        glm::vec3 position = glm::vec3(messages->x[index], messages->y[index], messages->z[index]);
		glm::ivec3 grid_position = message_quiescent_location_report_grid_position(position);
		unsigned int hash = message_quiescent_location_report_hash(grid_position);
		unsigned int bin_idx = atomicInc((unsigned int*) &global_bin_count[hash], 0xFFFFFFFF);
		local_bin_index[index] = bin_idx;
		unsorted_index[index] = hash;
	}
	
	/** reorder_quiescent_location_report_messages
	 * Reorders the messages accoring to the partition boundary matrix start indices of each bin
	 * @param local_bin_index index of the message within the desired bin
	 * @param unsorted_index bin index (hash) value
	 * @param pbm_start_index the start indices of the partition boundary matrix
	 * @param unordered_messages the original unordered message data
	 * @param ordered_messages buffer used to scatter messages into the correct order
	  @param agent_count the current number of agents outputting messages
	 */
	 __global__ void reorder_quiescent_location_report_messages(uint* local_bin_index, uint* unsorted_index, int* pbm_start_index, xmachine_message_quiescent_location_report_list* unordered_messages, xmachine_message_quiescent_location_report_list* ordered_messages, int agent_count)
	{
		int index = (blockIdx.x *blockDim.x) + threadIdx.x;

		if (index >= agent_count)
			return;

		uint i = unsorted_index[index];
		unsigned int sorted_index = local_bin_index[index] + pbm_start_index[i];

		//finally reorder agent data
		ordered_messages->id[sorted_index] = unordered_messages->id[index];
		ordered_messages->x[sorted_index] = unordered_messages->x[index];
		ordered_messages->y[sorted_index] = unordered_messages->y[index];
		ordered_messages->z[sorted_index] = unordered_messages->z[index];
	}
	 
#else

	/** hash_quiescent_location_report_messages
	 * Kernal function for calculating a hash value for each messahe depending on its position
	 * @param keys output for the hash key
	 * @param values output for the index value
	 * @param messages the message list used to generate the hash value outputs
	 */
	__global__ void hash_quiescent_location_report_messages(uint* keys, uint* values, xmachine_message_quiescent_location_report_list* messages)
	{
		unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;
        glm::vec3 position = glm::vec3(messages->x[index], messages->y[index], messages->z[index]);
		glm::ivec3 grid_position = message_quiescent_location_report_grid_position(position);
		unsigned int hash = message_quiescent_location_report_hash(grid_position);

		keys[index] = hash;
		values[index] = index;
	}

	/** reorder_quiescent_location_report_messages
	 * Reorders the messages accoring to the ordered sort identifiers and builds a Partition Boundary Matrix by looking at the previosu threads sort id.
	 * @param keys the sorted hash keys
	 * @param values the sorted index values
	 * @param matrix the PBM
	 * @param unordered_messages the original unordered message data
	 * @param ordered_messages buffer used to scatter messages into the correct order
	 */
	__global__ void reorder_quiescent_location_report_messages(uint* keys, uint* values, xmachine_message_quiescent_location_report_PBM* matrix, xmachine_message_quiescent_location_report_list* unordered_messages, xmachine_message_quiescent_location_report_list* ordered_messages)
	{
		extern __shared__ int sm_data [];

		int index = (blockIdx.x * blockDim.x) + threadIdx.x;

		//load threads sort key into sm
		uint key = keys[index];
		uint old_pos = values[index];

		sm_data[threadIdx.x] = key;
		__syncthreads();
	
		unsigned int prev_key;

		//if first thread then no prev sm value so get prev from global memory 
		if (threadIdx.x == 0)
		{
			//first thread has no prev value so ignore
			if (index != 0)
				prev_key = keys[index-1];
		}
		//get previous ident from sm
		else	
		{
			prev_key = sm_data[threadIdx.x-1];
		}

		//TODO: Check key is not out of bounds

		//set partition boundaries
		if (index < d_message_quiescent_location_report_count)
		{
			//if first thread then set first partition cell start
			if (index == 0)
			{
				matrix->start[key] = index;
			}

			//if edge of a boundr update start and end of partition
			else if (prev_key != key)
			{
				//set start for key
				matrix->start[key] = index;

				//set end for key -1
				matrix->end_or_count[prev_key] = index;
			}

			//if last thread then set final partition cell end
			if (index == d_message_quiescent_location_report_count-1)
			{
				matrix->end_or_count[key] = index+1;
			}
		}
	
		//finally reorder agent data
		ordered_messages->id[index] = unordered_messages->id[old_pos];
		ordered_messages->x[index] = unordered_messages->x[old_pos];
		ordered_messages->y[index] = unordered_messages->y[old_pos];
		ordered_messages->z[index] = unordered_messages->z[old_pos];
	}

#endif

/** load_next_quiescent_location_report_message
 * Used to load the next message data to shared memory
 * Idea is check the current cell index to see if we can simply get a message from the current cell
 * If we are at the end of the current cell then loop till we find the next cell with messages (this way we ignore cells with no messages)
 * @param messages the message list
 * @param partition_matrix the PBM
 * @param relative_cell the relative partition cell position from the agent position
 * @param cell_index_max the maximum index of the current partition cell
 * @param agent_grid_cell the agents partition cell position
 * @param cell_index the current cell index in agent_grid_cell+relative_cell
 * @return true if a message has been loaded into sm false otherwise
 */
__device__ bool load_next_quiescent_location_report_message(xmachine_message_quiescent_location_report_list* messages, xmachine_message_quiescent_location_report_PBM* partition_matrix, glm::ivec3 relative_cell, int cell_index_max, glm::ivec3 agent_grid_cell, int cell_index)
{
	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];

	int move_cell = true;
	cell_index ++;

	//see if we need to move to a new partition cell
	if(cell_index < cell_index_max)
		move_cell = false;

	while(move_cell)
	{
		//get the next relative grid position 
        if (next_cell3D(&relative_cell))
		{
			//calculate the next cells grid position and hash
			glm::ivec3 next_cell_position = agent_grid_cell + relative_cell;
			int next_cell_hash = message_quiescent_location_report_hash(next_cell_position);
			//use the hash to calculate the start index
			int cell_index_min = tex1Dfetch(tex_xmachine_message_quiescent_location_report_pbm_start, next_cell_hash + d_tex_xmachine_message_quiescent_location_report_pbm_start_offset);
			cell_index_max = tex1Dfetch(tex_xmachine_message_quiescent_location_report_pbm_end_or_count, next_cell_hash + d_tex_xmachine_message_quiescent_location_report_pbm_end_or_count_offset);
			//check for messages in the cell (cell index max is the count for atomic sorting)
#ifdef FAST_ATOMIC_SORTING
			if (cell_index_max > 0)
			{
				//when using fast atomics value represents bin count not last index!
				cell_index_max += cell_index_min; //when using fast atomics value represents bin count not last index!
#else
			if (cell_index_min != 0xffffffff)
			{
#endif
				//start from the cell index min
				cell_index = cell_index_min;
				//exit the loop as we have found a valid cell with message data
				move_cell = false;
			}
		}
		else
		{
			//we have exhausted all the neighbouring cells so there are no more messages
			return false;
		}
	}
	
	//get message data using texture fetch
	xmachine_message_quiescent_location_report temp_message;
	temp_message._relative_cell = relative_cell;
	temp_message._cell_index_max = cell_index_max;
	temp_message._cell_index = cell_index;
	temp_message._agent_grid_cell = agent_grid_cell;

	//Using texture cache
  temp_message.id = tex1Dfetch(tex_xmachine_message_quiescent_location_report_id, cell_index + d_tex_xmachine_message_quiescent_location_report_id_offset); temp_message.x = tex1Dfetch(tex_xmachine_message_quiescent_location_report_x, cell_index + d_tex_xmachine_message_quiescent_location_report_x_offset); temp_message.y = tex1Dfetch(tex_xmachine_message_quiescent_location_report_y, cell_index + d_tex_xmachine_message_quiescent_location_report_y_offset); temp_message.z = tex1Dfetch(tex_xmachine_message_quiescent_location_report_z, cell_index + d_tex_xmachine_message_quiescent_location_report_z_offset); 

	//load it into shared memory (no sync as no sharing between threads)
	int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_quiescent_location_report));
	xmachine_message_quiescent_location_report* sm_message = ((xmachine_message_quiescent_location_report*)&message_share[message_index]);
	sm_message[0] = temp_message;

	return true;
}


/*
 * get first spatial partitioned quiescent_location_report message (first batch load into shared memory)
 */
__device__ xmachine_message_quiescent_location_report* get_first_quiescent_location_report_message(xmachine_message_quiescent_location_report_list* messages, xmachine_message_quiescent_location_report_PBM* partition_matrix, float x, float y, float z){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];

	// If there are no messages, do not load any messages
	if(d_message_quiescent_location_report_count == 0){
		return nullptr;
	}

	glm::ivec3 relative_cell = glm::ivec3(-2, -1, -1);
	int cell_index_max = 0;
	int cell_index = 0;
	glm::vec3 position = glm::vec3(x, y, z);
	glm::ivec3 agent_grid_cell = message_quiescent_location_report_grid_position(position);
	
	if (load_next_quiescent_location_report_message(messages, partition_matrix, relative_cell, cell_index_max, agent_grid_cell, cell_index))
	{
		int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_quiescent_location_report));
		return ((xmachine_message_quiescent_location_report*)&message_share[message_index]);
	}
	else
	{
		return nullptr;
	}
}

/*
 * get next spatial partitioned quiescent_location_report message (either from SM or next batch load)
 */
__device__ xmachine_message_quiescent_location_report* get_next_quiescent_location_report_message(xmachine_message_quiescent_location_report* message, xmachine_message_quiescent_location_report_list* messages, xmachine_message_quiescent_location_report_PBM* partition_matrix){
	
	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	// If there are no messages, do not load any messages
	if(d_message_quiescent_location_report_count == 0){
		return nullptr;
	}
	
	if (load_next_quiescent_location_report_message(messages, partition_matrix, message->_relative_cell, message->_cell_index_max, message->_agent_grid_cell, message->_cell_index))
	{
		//get conflict free address of 
		int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_quiescent_location_report));
		return ((xmachine_message_quiescent_location_report*)&message_share[message_index]);
	}
	else
		return nullptr;
	
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created senescent_location_report message functions */


/** add_senescent_location_report_message
 * Add non partitioned or spatially partitioned senescent_location_report message
 * @param messages xmachine_message_senescent_location_report_list message list to add too
 * @param id agent variable of type int
 * @param x agent variable of type float
 * @param y agent variable of type float
 * @param z agent variable of type float
 */
__device__ void add_senescent_location_report_message(xmachine_message_senescent_location_report_list* messages, int id, float x, float y, float z){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x + d_message_senescent_location_report_count;

	int _position;
	int _scan_input;

	//decide output position
	if(d_message_senescent_location_report_output_type == single_message){
		_position = index; //same as agent position
		_scan_input = 0;
	}else if (d_message_senescent_location_report_output_type == optional_message){
		_position = 0;	   //to be calculated using Prefix sum
		_scan_input = 1;
	}

	//AoS - xmachine_message_senescent_location_report Coalesced memory write
	messages->_scan_input[index] = _scan_input;	
	messages->_position[index] = _position;
	messages->id[index] = id;
	messages->x[index] = x;
	messages->y[index] = y;
	messages->z[index] = z;

}

/**
 * Scatter non partitioned or spatially partitioned senescent_location_report message (for optional messages)
 * @param messages scatter_optional_senescent_location_report_messages Sparse xmachine_message_senescent_location_report_list message list
 * @param message_swap temp xmachine_message_senescent_location_report_list message list to scatter sparse messages to
 */
__global__ void scatter_optional_senescent_location_report_messages(xmachine_message_senescent_location_report_list* messages, xmachine_message_senescent_location_report_list* messages_swap){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = messages_swap->_scan_input[index];

	//if optional message is to be written
	if (_scan_input == 1){
		int output_index = messages_swap->_position[index] + d_message_senescent_location_report_count;

		//AoS - xmachine_message_senescent_location_report Un-Coalesced scattered memory write
		messages->_position[output_index] = output_index;
		messages->id[output_index] = messages_swap->id[index];
		messages->x[output_index] = messages_swap->x[index];
		messages->y[output_index] = messages_swap->y[index];
		messages->z[output_index] = messages_swap->z[index];				
	}
}

/** reset_senescent_location_report_swaps
 * Reset non partitioned or spatially partitioned senescent_location_report message swaps (for scattering optional messages)
 * @param message_swap message list to reset _position and _scan_input values back to 0
 */
__global__ void reset_senescent_location_report_swaps(xmachine_message_senescent_location_report_list* messages_swap){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	messages_swap->_position[index] = 0;
	messages_swap->_scan_input[index] = 0;
}

/* Message functions */

/** message_senescent_location_report_grid_position
 * Calculates the grid cell position given an glm::vec3 vector
 * @param position glm::vec3 vector representing a position
 */
__device__ glm::ivec3 message_senescent_location_report_grid_position(glm::vec3 position)
{
    glm::ivec3 gridPos;
    gridPos.x = floor((position.x - d_message_senescent_location_report_min_bounds.x) * (float)d_message_senescent_location_report_partitionDim.x / (d_message_senescent_location_report_max_bounds.x - d_message_senescent_location_report_min_bounds.x));
    gridPos.y = floor((position.y - d_message_senescent_location_report_min_bounds.y) * (float)d_message_senescent_location_report_partitionDim.y / (d_message_senescent_location_report_max_bounds.y - d_message_senescent_location_report_min_bounds.y));
    gridPos.z = floor((position.z - d_message_senescent_location_report_min_bounds.z) * (float)d_message_senescent_location_report_partitionDim.z / (d_message_senescent_location_report_max_bounds.z - d_message_senescent_location_report_min_bounds.z));

	//do wrapping or bounding
	

    return gridPos;
}

/** message_senescent_location_report_hash
 * Given the grid position in partition space this function calculates a hash value
 * @param gridPos The position in partition space
 */
__device__ unsigned int message_senescent_location_report_hash(glm::ivec3 gridPos)
{
	//cheap bounding without mod (within range +- partition dimension)
	gridPos.x = (gridPos.x<0)? d_message_senescent_location_report_partitionDim.x-1: gridPos.x; 
	gridPos.x = (gridPos.x>=d_message_senescent_location_report_partitionDim.x)? 0 : gridPos.x; 
	gridPos.y = (gridPos.y<0)? d_message_senescent_location_report_partitionDim.y-1 : gridPos.y; 
	gridPos.y = (gridPos.y>=d_message_senescent_location_report_partitionDim.y)? 0 : gridPos.y; 
	gridPos.z = (gridPos.z<0)? d_message_senescent_location_report_partitionDim.z-1: gridPos.z; 
	gridPos.z = (gridPos.z>=d_message_senescent_location_report_partitionDim.z)? 0 : gridPos.z; 

	//unique id
	return ((gridPos.z * d_message_senescent_location_report_partitionDim.y) * d_message_senescent_location_report_partitionDim.x) + (gridPos.y * d_message_senescent_location_report_partitionDim.x) + gridPos.x;
}

#ifdef FAST_ATOMIC_SORTING
	/** hist_senescent_location_report_messages
		 * Kernal function for performing a histogram (count) on each partition bin and saving the hash and index of a message within that bin
		 * @param local_bin_index output index of the message within the calculated bin
		 * @param unsorted_index output bin index (hash) value
		 * @param messages the message list used to generate the hash value outputs
		 * @param agent_count the current number of agents outputting messages
		 */
	__global__ void hist_senescent_location_report_messages(uint* local_bin_index, uint* unsorted_index, int* global_bin_count, xmachine_message_senescent_location_report_list* messages, int agent_count)
	{
		unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;

		if (index >= agent_count)
			return;
        glm::vec3 position = glm::vec3(messages->x[index], messages->y[index], messages->z[index]);
		glm::ivec3 grid_position = message_senescent_location_report_grid_position(position);
		unsigned int hash = message_senescent_location_report_hash(grid_position);
		unsigned int bin_idx = atomicInc((unsigned int*) &global_bin_count[hash], 0xFFFFFFFF);
		local_bin_index[index] = bin_idx;
		unsorted_index[index] = hash;
	}
	
	/** reorder_senescent_location_report_messages
	 * Reorders the messages accoring to the partition boundary matrix start indices of each bin
	 * @param local_bin_index index of the message within the desired bin
	 * @param unsorted_index bin index (hash) value
	 * @param pbm_start_index the start indices of the partition boundary matrix
	 * @param unordered_messages the original unordered message data
	 * @param ordered_messages buffer used to scatter messages into the correct order
	  @param agent_count the current number of agents outputting messages
	 */
	 __global__ void reorder_senescent_location_report_messages(uint* local_bin_index, uint* unsorted_index, int* pbm_start_index, xmachine_message_senescent_location_report_list* unordered_messages, xmachine_message_senescent_location_report_list* ordered_messages, int agent_count)
	{
		int index = (blockIdx.x *blockDim.x) + threadIdx.x;

		if (index >= agent_count)
			return;

		uint i = unsorted_index[index];
		unsigned int sorted_index = local_bin_index[index] + pbm_start_index[i];

		//finally reorder agent data
		ordered_messages->id[sorted_index] = unordered_messages->id[index];
		ordered_messages->x[sorted_index] = unordered_messages->x[index];
		ordered_messages->y[sorted_index] = unordered_messages->y[index];
		ordered_messages->z[sorted_index] = unordered_messages->z[index];
	}
	 
#else

	/** hash_senescent_location_report_messages
	 * Kernal function for calculating a hash value for each messahe depending on its position
	 * @param keys output for the hash key
	 * @param values output for the index value
	 * @param messages the message list used to generate the hash value outputs
	 */
	__global__ void hash_senescent_location_report_messages(uint* keys, uint* values, xmachine_message_senescent_location_report_list* messages)
	{
		unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;
        glm::vec3 position = glm::vec3(messages->x[index], messages->y[index], messages->z[index]);
		glm::ivec3 grid_position = message_senescent_location_report_grid_position(position);
		unsigned int hash = message_senescent_location_report_hash(grid_position);

		keys[index] = hash;
		values[index] = index;
	}

	/** reorder_senescent_location_report_messages
	 * Reorders the messages accoring to the ordered sort identifiers and builds a Partition Boundary Matrix by looking at the previosu threads sort id.
	 * @param keys the sorted hash keys
	 * @param values the sorted index values
	 * @param matrix the PBM
	 * @param unordered_messages the original unordered message data
	 * @param ordered_messages buffer used to scatter messages into the correct order
	 */
	__global__ void reorder_senescent_location_report_messages(uint* keys, uint* values, xmachine_message_senescent_location_report_PBM* matrix, xmachine_message_senescent_location_report_list* unordered_messages, xmachine_message_senescent_location_report_list* ordered_messages)
	{
		extern __shared__ int sm_data [];

		int index = (blockIdx.x * blockDim.x) + threadIdx.x;

		//load threads sort key into sm
		uint key = keys[index];
		uint old_pos = values[index];

		sm_data[threadIdx.x] = key;
		__syncthreads();
	
		unsigned int prev_key;

		//if first thread then no prev sm value so get prev from global memory 
		if (threadIdx.x == 0)
		{
			//first thread has no prev value so ignore
			if (index != 0)
				prev_key = keys[index-1];
		}
		//get previous ident from sm
		else	
		{
			prev_key = sm_data[threadIdx.x-1];
		}

		//TODO: Check key is not out of bounds

		//set partition boundaries
		if (index < d_message_senescent_location_report_count)
		{
			//if first thread then set first partition cell start
			if (index == 0)
			{
				matrix->start[key] = index;
			}

			//if edge of a boundr update start and end of partition
			else if (prev_key != key)
			{
				//set start for key
				matrix->start[key] = index;

				//set end for key -1
				matrix->end_or_count[prev_key] = index;
			}

			//if last thread then set final partition cell end
			if (index == d_message_senescent_location_report_count-1)
			{
				matrix->end_or_count[key] = index+1;
			}
		}
	
		//finally reorder agent data
		ordered_messages->id[index] = unordered_messages->id[old_pos];
		ordered_messages->x[index] = unordered_messages->x[old_pos];
		ordered_messages->y[index] = unordered_messages->y[old_pos];
		ordered_messages->z[index] = unordered_messages->z[old_pos];
	}

#endif

/** load_next_senescent_location_report_message
 * Used to load the next message data to shared memory
 * Idea is check the current cell index to see if we can simply get a message from the current cell
 * If we are at the end of the current cell then loop till we find the next cell with messages (this way we ignore cells with no messages)
 * @param messages the message list
 * @param partition_matrix the PBM
 * @param relative_cell the relative partition cell position from the agent position
 * @param cell_index_max the maximum index of the current partition cell
 * @param agent_grid_cell the agents partition cell position
 * @param cell_index the current cell index in agent_grid_cell+relative_cell
 * @return true if a message has been loaded into sm false otherwise
 */
__device__ bool load_next_senescent_location_report_message(xmachine_message_senescent_location_report_list* messages, xmachine_message_senescent_location_report_PBM* partition_matrix, glm::ivec3 relative_cell, int cell_index_max, glm::ivec3 agent_grid_cell, int cell_index)
{
	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];

	int move_cell = true;
	cell_index ++;

	//see if we need to move to a new partition cell
	if(cell_index < cell_index_max)
		move_cell = false;

	while(move_cell)
	{
		//get the next relative grid position 
        if (next_cell3D(&relative_cell))
		{
			//calculate the next cells grid position and hash
			glm::ivec3 next_cell_position = agent_grid_cell + relative_cell;
			int next_cell_hash = message_senescent_location_report_hash(next_cell_position);
			//use the hash to calculate the start index
			int cell_index_min = tex1Dfetch(tex_xmachine_message_senescent_location_report_pbm_start, next_cell_hash + d_tex_xmachine_message_senescent_location_report_pbm_start_offset);
			cell_index_max = tex1Dfetch(tex_xmachine_message_senescent_location_report_pbm_end_or_count, next_cell_hash + d_tex_xmachine_message_senescent_location_report_pbm_end_or_count_offset);
			//check for messages in the cell (cell index max is the count for atomic sorting)
#ifdef FAST_ATOMIC_SORTING
			if (cell_index_max > 0)
			{
				//when using fast atomics value represents bin count not last index!
				cell_index_max += cell_index_min; //when using fast atomics value represents bin count not last index!
#else
			if (cell_index_min != 0xffffffff)
			{
#endif
				//start from the cell index min
				cell_index = cell_index_min;
				//exit the loop as we have found a valid cell with message data
				move_cell = false;
			}
		}
		else
		{
			//we have exhausted all the neighbouring cells so there are no more messages
			return false;
		}
	}
	
	//get message data using texture fetch
	xmachine_message_senescent_location_report temp_message;
	temp_message._relative_cell = relative_cell;
	temp_message._cell_index_max = cell_index_max;
	temp_message._cell_index = cell_index;
	temp_message._agent_grid_cell = agent_grid_cell;

	//Using texture cache
  temp_message.id = tex1Dfetch(tex_xmachine_message_senescent_location_report_id, cell_index + d_tex_xmachine_message_senescent_location_report_id_offset); temp_message.x = tex1Dfetch(tex_xmachine_message_senescent_location_report_x, cell_index + d_tex_xmachine_message_senescent_location_report_x_offset); temp_message.y = tex1Dfetch(tex_xmachine_message_senescent_location_report_y, cell_index + d_tex_xmachine_message_senescent_location_report_y_offset); temp_message.z = tex1Dfetch(tex_xmachine_message_senescent_location_report_z, cell_index + d_tex_xmachine_message_senescent_location_report_z_offset); 

	//load it into shared memory (no sync as no sharing between threads)
	int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_senescent_location_report));
	xmachine_message_senescent_location_report* sm_message = ((xmachine_message_senescent_location_report*)&message_share[message_index]);
	sm_message[0] = temp_message;

	return true;
}


/*
 * get first spatial partitioned senescent_location_report message (first batch load into shared memory)
 */
__device__ xmachine_message_senescent_location_report* get_first_senescent_location_report_message(xmachine_message_senescent_location_report_list* messages, xmachine_message_senescent_location_report_PBM* partition_matrix, float x, float y, float z){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];

	// If there are no messages, do not load any messages
	if(d_message_senescent_location_report_count == 0){
		return nullptr;
	}

	glm::ivec3 relative_cell = glm::ivec3(-2, -1, -1);
	int cell_index_max = 0;
	int cell_index = 0;
	glm::vec3 position = glm::vec3(x, y, z);
	glm::ivec3 agent_grid_cell = message_senescent_location_report_grid_position(position);
	
	if (load_next_senescent_location_report_message(messages, partition_matrix, relative_cell, cell_index_max, agent_grid_cell, cell_index))
	{
		int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_senescent_location_report));
		return ((xmachine_message_senescent_location_report*)&message_share[message_index]);
	}
	else
	{
		return nullptr;
	}
}

/*
 * get next spatial partitioned senescent_location_report message (either from SM or next batch load)
 */
__device__ xmachine_message_senescent_location_report* get_next_senescent_location_report_message(xmachine_message_senescent_location_report* message, xmachine_message_senescent_location_report_list* messages, xmachine_message_senescent_location_report_PBM* partition_matrix){
	
	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	// If there are no messages, do not load any messages
	if(d_message_senescent_location_report_count == 0){
		return nullptr;
	}
	
	if (load_next_senescent_location_report_message(messages, partition_matrix, message->_relative_cell, message->_cell_index_max, message->_agent_grid_cell, message->_cell_index))
	{
		//get conflict free address of 
		int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_senescent_location_report));
		return ((xmachine_message_senescent_location_report*)&message_share[message_index]);
	}
	else
		return nullptr;
	
}


	
/////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created GPU kernels  */



/**
 *
 */
__global__ void GPUFLAME_TissueTakesDamage(xmachine_memory_TissueBlock_list* agents, RNG_rand48* rand48){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_TissueBlock_count)
        return;
    

	//SoA to AoS - xmachine_memory_TissueTakesDamage Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_TissueBlock agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.id = agents->id[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.z = agents->z[index];
	agent.damage = agents->damage[index];

	//FLAME function call
	int dead = !TissueTakesDamage(&agent, rand48);
	

	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_TissueTakesDamage Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->z[index] = agent.z;
	agents->damage[index] = agent.damage;
}

/**
 *
 */
__global__ void GPUFLAME_TissueSendDamageReport(xmachine_memory_TissueBlock_list* agents, xmachine_message_tissue_damage_report_list* tissue_damage_report_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_TissueBlock_count)
        return;
    

	//SoA to AoS - xmachine_memory_TissueSendDamageReport Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_TissueBlock agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.id = agents->id[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.z = agents->z[index];
	agent.damage = agents->damage[index];

	//FLAME function call
	int dead = !TissueSendDamageReport(&agent, tissue_damage_report_messages	);
	

	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_TissueSendDamageReport Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->z[index] = agent.z;
	agents->damage[index] = agent.damage;
}

/**
 *
 */
__global__ void GPUFLAME_QuiescentMigration(xmachine_memory_Fibroblast_list* agents, xmachine_message_tissue_damage_report_list* tissue_damage_report_messages, xmachine_message_tissue_damage_report_PBM* partition_matrix){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_Fibroblast_count)
        return;
    

	//SoA to AoS - xmachine_memory_QuiescentMigration Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_Fibroblast agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.id = agents->id[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.z = agents->z[index];
	agent.doublings = agents->doublings[index];
	agent.damage = agents->damage[index];
	agent.proliferate_bool = agents->proliferate_bool[index];
	agent.transition_to_early_sen = agents->transition_to_early_sen[index];

	//FLAME function call
	int dead = !QuiescentMigration(&agent, tissue_damage_report_messages, partition_matrix);
	

	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_QuiescentMigration Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->z[index] = agent.z;
	agents->doublings[index] = agent.doublings;
	agents->damage[index] = agent.damage;
	agents->proliferate_bool[index] = agent.proliferate_bool;
	agents->transition_to_early_sen[index] = agent.transition_to_early_sen;
}

/**
 *
 */
__global__ void GPUFLAME_SenescentMigration(xmachine_memory_Fibroblast_list* agents, xmachine_message_tissue_damage_report_list* tissue_damage_report_messages, xmachine_message_tissue_damage_report_PBM* partition_matrix){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_Fibroblast_count)
        return;
    

	//SoA to AoS - xmachine_memory_SenescentMigration Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_Fibroblast agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.id = agents->id[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.z = agents->z[index];
	agent.doublings = agents->doublings[index];
	agent.damage = agents->damage[index];
	agent.proliferate_bool = agents->proliferate_bool[index];
	agent.transition_to_early_sen = agents->transition_to_early_sen[index];

	//FLAME function call
	int dead = !SenescentMigration(&agent, tissue_damage_report_messages, partition_matrix);
	

	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_SenescentMigration Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->z[index] = agent.z;
	agents->doublings[index] = agent.doublings;
	agents->damage[index] = agent.damage;
	agents->proliferate_bool[index] = agent.proliferate_bool;
	agents->transition_to_early_sen[index] = agent.transition_to_early_sen;
}

/**
 *
 */
__global__ void GPUFLAME_EarlySenescentMigration(xmachine_memory_Fibroblast_list* agents, xmachine_message_tissue_damage_report_list* tissue_damage_report_messages, xmachine_message_tissue_damage_report_PBM* partition_matrix){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_Fibroblast_count)
        return;
    

	//SoA to AoS - xmachine_memory_EarlySenescentMigration Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_Fibroblast agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.id = agents->id[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.z = agents->z[index];
	agent.doublings = agents->doublings[index];
	agent.damage = agents->damage[index];
	agent.proliferate_bool = agents->proliferate_bool[index];
	agent.transition_to_early_sen = agents->transition_to_early_sen[index];

	//FLAME function call
	int dead = !EarlySenescentMigration(&agent, tissue_damage_report_messages, partition_matrix);
	

	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_EarlySenescentMigration Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->z[index] = agent.z;
	agents->doublings[index] = agent.doublings;
	agents->damage[index] = agent.damage;
	agents->proliferate_bool[index] = agent.proliferate_bool;
	agents->transition_to_early_sen[index] = agent.transition_to_early_sen;
}

/**
 *
 */
__global__ void GPUFLAME_QuiescentTakesDamage(xmachine_memory_Fibroblast_list* agents, xmachine_message_fibroblast_damage_report_list* fibroblast_damage_report_messages, xmachine_message_fibroblast_damage_report_PBM* partition_matrix, RNG_rand48* rand48){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_Fibroblast_count)
        return;
    

	//SoA to AoS - xmachine_memory_QuiescentTakesDamage Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_Fibroblast agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.id = agents->id[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.z = agents->z[index];
	agent.doublings = agents->doublings[index];
	agent.damage = agents->damage[index];
	agent.proliferate_bool = agents->proliferate_bool[index];
	agent.transition_to_early_sen = agents->transition_to_early_sen[index];

	//FLAME function call
	int dead = !QuiescentTakesDamage(&agent, fibroblast_damage_report_messages, partition_matrix, rand48);
	

	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_QuiescentTakesDamage Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->z[index] = agent.z;
	agents->doublings[index] = agent.doublings;
	agents->damage[index] = agent.damage;
	agents->proliferate_bool[index] = agent.proliferate_bool;
	agents->transition_to_early_sen[index] = agent.transition_to_early_sen;
}

/**
 *
 */
__global__ void GPUFLAME_QuiescentSendDamageReport(xmachine_memory_Fibroblast_list* agents, xmachine_message_fibroblast_damage_report_list* fibroblast_damage_report_messages, RNG_rand48* rand48){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_Fibroblast_count)
        return;
    

	//SoA to AoS - xmachine_memory_QuiescentSendDamageReport Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_Fibroblast agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.id = agents->id[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.z = agents->z[index];
	agent.doublings = agents->doublings[index];
	agent.damage = agents->damage[index];
	agent.proliferate_bool = agents->proliferate_bool[index];
	agent.transition_to_early_sen = agents->transition_to_early_sen[index];

	//FLAME function call
	int dead = !QuiescentSendDamageReport(&agent, fibroblast_damage_report_messages	, rand48);
	

	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_QuiescentSendDamageReport Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->z[index] = agent.z;
	agents->doublings[index] = agent.doublings;
	agents->damage[index] = agent.damage;
	agents->proliferate_bool[index] = agent.proliferate_bool;
	agents->transition_to_early_sen[index] = agent.transition_to_early_sen;
}

/**
 *
 */
__global__ void GPUFLAME_Quiescent2Proliferating(xmachine_memory_Fibroblast_list* agents, RNG_rand48* rand48){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_Fibroblast_count)
        return;
    

	//SoA to AoS - xmachine_memory_Quiescent2Proliferating Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_Fibroblast agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.id = agents->id[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.z = agents->z[index];
	agent.doublings = agents->doublings[index];
	agent.damage = agents->damage[index];
	agent.proliferate_bool = agents->proliferate_bool[index];
	agent.transition_to_early_sen = agents->transition_to_early_sen[index];

	//FLAME function call
	int dead = !Quiescent2Proliferating(&agent, rand48);
	

	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_Quiescent2Proliferating Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->z[index] = agent.z;
	agents->doublings[index] = agent.doublings;
	agents->damage[index] = agent.damage;
	agents->proliferate_bool[index] = agent.proliferate_bool;
	agents->transition_to_early_sen[index] = agent.transition_to_early_sen;
}

/**
 *
 */
__global__ void GPUFLAME_Proliferation(xmachine_memory_Fibroblast_list* agents, xmachine_memory_Fibroblast_list* Fibroblast_agents){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_Fibroblast_count)
        return;
    

	//SoA to AoS - xmachine_memory_Proliferation Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_Fibroblast agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.id = agents->id[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.z = agents->z[index];
	agent.doublings = agents->doublings[index];
	agent.damage = agents->damage[index];
	agent.proliferate_bool = agents->proliferate_bool[index];
	agent.transition_to_early_sen = agents->transition_to_early_sen[index];

	//FLAME function call
	int dead = !Proliferation(&agent, Fibroblast_agents);
	

	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_Proliferation Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->z[index] = agent.z;
	agents->doublings[index] = agent.doublings;
	agents->damage[index] = agent.damage;
	agents->proliferate_bool[index] = agent.proliferate_bool;
	agents->transition_to_early_sen[index] = agent.transition_to_early_sen;
}

/**
 *
 */
__global__ void GPUFLAME_BystanderEffect(xmachine_memory_Fibroblast_list* agents, xmachine_message_senescent_location_report_list* senescent_location_report_messages, xmachine_message_senescent_location_report_PBM* partition_matrix, RNG_rand48* rand48){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_Fibroblast_count)
        return;
    

	//SoA to AoS - xmachine_memory_BystanderEffect Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_Fibroblast agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.id = agents->id[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.z = agents->z[index];
	agent.doublings = agents->doublings[index];
	agent.damage = agents->damage[index];
	agent.proliferate_bool = agents->proliferate_bool[index];
	agent.transition_to_early_sen = agents->transition_to_early_sen[index];

	//FLAME function call
	int dead = !BystanderEffect(&agent, senescent_location_report_messages, partition_matrix, rand48);
	

	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_BystanderEffect Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->z[index] = agent.z;
	agents->doublings[index] = agent.doublings;
	agents->damage[index] = agent.damage;
	agents->proliferate_bool[index] = agent.proliferate_bool;
	agents->transition_to_early_sen[index] = agent.transition_to_early_sen;
}

/**
 *
 */
__global__ void GPUFLAME_ExcessiveDamage(xmachine_memory_Fibroblast_list* agents, xmachine_message_fibroblast_damage_report_list* fibroblast_damage_report_messages, xmachine_message_fibroblast_damage_report_PBM* partition_matrix, RNG_rand48* rand48){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_Fibroblast_count)
        return;
    

	//SoA to AoS - xmachine_memory_ExcessiveDamage Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_Fibroblast agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.id = agents->id[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.z = agents->z[index];
	agent.doublings = agents->doublings[index];
	agent.damage = agents->damage[index];
	agent.proliferate_bool = agents->proliferate_bool[index];
	agent.transition_to_early_sen = agents->transition_to_early_sen[index];

	//FLAME function call
	int dead = !ExcessiveDamage(&agent, fibroblast_damage_report_messages, partition_matrix, rand48);
	

	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_ExcessiveDamage Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->z[index] = agent.z;
	agents->doublings[index] = agent.doublings;
	agents->damage[index] = agent.damage;
	agents->proliferate_bool[index] = agent.proliferate_bool;
	agents->transition_to_early_sen[index] = agent.transition_to_early_sen;
}

/**
 *
 */
__global__ void GPUFLAME_ReplicativeSenescence(xmachine_memory_Fibroblast_list* agents, xmachine_message_doublings_list* doublings_messages, RNG_rand48* rand48){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    
    //No partitioned input requires threads to be launched beyond the agent count to ensure full block sizes
    

	//SoA to AoS - xmachine_memory_ReplicativeSenescence Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_Fibroblast agent;
    //No partitioned input may launch more threads than required - only load agent data within bounds. 
    if (index < d_xmachine_memory_Fibroblast_count){
    
	agent.id = agents->id[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.z = agents->z[index];
	agent.doublings = agents->doublings[index];
	agent.damage = agents->damage[index];
	agent.proliferate_bool = agents->proliferate_bool[index];
	agent.transition_to_early_sen = agents->transition_to_early_sen[index];
	} else {
	
	agent.id = 0;
	agent.x = 0;
	agent.y = 0;
	agent.z = 0;
	agent.doublings = 0;
	agent.damage = 0;
	agent.proliferate_bool = 0;
	agent.transition_to_early_sen = 0;
	}

	//FLAME function call
	int dead = !ReplicativeSenescence(&agent, doublings_messages, rand48);
	

	
    //No partitioned input may launch more threads than required - only write agent data within bounds. 
    if (index < d_xmachine_memory_Fibroblast_count){
    //continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_ReplicativeSenescence Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->z[index] = agent.z;
	agents->doublings[index] = agent.doublings;
	agents->damage[index] = agent.damage;
	agents->proliferate_bool[index] = agent.proliferate_bool;
	agents->transition_to_early_sen[index] = agent.transition_to_early_sen;
	}
}

/**
 *
 */
__global__ void GPUFLAME_TransitionToEarlySen(xmachine_memory_Fibroblast_list* agents){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_Fibroblast_count)
        return;
    

	//SoA to AoS - xmachine_memory_TransitionToEarlySen Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_Fibroblast agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.id = agents->id[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.z = agents->z[index];
	agent.doublings = agents->doublings[index];
	agent.damage = agents->damage[index];
	agent.proliferate_bool = agents->proliferate_bool[index];
	agent.transition_to_early_sen = agents->transition_to_early_sen[index];

	//FLAME function call
	int dead = !TransitionToEarlySen(&agent);
	

	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_TransitionToEarlySen Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->z[index] = agent.z;
	agents->doublings[index] = agent.doublings;
	agents->damage[index] = agent.damage;
	agents->proliferate_bool[index] = agent.proliferate_bool;
	agents->transition_to_early_sen[index] = agent.transition_to_early_sen;
}

/**
 *
 */
__global__ void GPUFLAME_FullSenescence(xmachine_memory_Fibroblast_list* agents, xmachine_message_count_list* count_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    
    //No partitioned input requires threads to be launched beyond the agent count to ensure full block sizes
    

	//SoA to AoS - xmachine_memory_FullSenescence Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_Fibroblast agent;
    //No partitioned input may launch more threads than required - only load agent data within bounds. 
    if (index < d_xmachine_memory_Fibroblast_count){
    
	agent.id = agents->id[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.z = agents->z[index];
	agent.doublings = agents->doublings[index];
	agent.damage = agents->damage[index];
	agent.proliferate_bool = agents->proliferate_bool[index];
	agent.transition_to_early_sen = agents->transition_to_early_sen[index];
	} else {
	
	agent.id = 0;
	agent.x = 0;
	agent.y = 0;
	agent.z = 0;
	agent.doublings = 0;
	agent.damage = 0;
	agent.proliferate_bool = 0;
	agent.transition_to_early_sen = 0;
	}

	//FLAME function call
	int dead = !FullSenescence(&agent, count_messages);
	

	
    //No partitioned input may launch more threads than required - only write agent data within bounds. 
    if (index < d_xmachine_memory_Fibroblast_count){
    //continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_FullSenescence Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->z[index] = agent.z;
	agents->doublings[index] = agent.doublings;
	agents->damage[index] = agent.damage;
	agents->proliferate_bool[index] = agent.proliferate_bool;
	agents->transition_to_early_sen[index] = agent.transition_to_early_sen;
	}
}

/**
 *
 */
__global__ void GPUFLAME_ClearanceOfEarlySenescent(xmachine_memory_Fibroblast_list* agents){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_Fibroblast_count)
        return;
    

	//SoA to AoS - xmachine_memory_ClearanceOfEarlySenescent Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_Fibroblast agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.id = agents->id[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.z = agents->z[index];
	agent.doublings = agents->doublings[index];
	agent.damage = agents->damage[index];
	agent.proliferate_bool = agents->proliferate_bool[index];
	agent.transition_to_early_sen = agents->transition_to_early_sen[index];

	//FLAME function call
	int dead = !ClearanceOfEarlySenescent(&agent);
	

	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_ClearanceOfEarlySenescent Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->z[index] = agent.z;
	agents->doublings[index] = agent.doublings;
	agents->damage[index] = agent.damage;
	agents->proliferate_bool[index] = agent.proliferate_bool;
	agents->transition_to_early_sen[index] = agent.transition_to_early_sen;
}

/**
 *
 */
__global__ void GPUFLAME_ClearanceOfSenescent(xmachine_memory_Fibroblast_list* agents){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_Fibroblast_count)
        return;
    

	//SoA to AoS - xmachine_memory_ClearanceOfSenescent Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_Fibroblast agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.id = agents->id[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.z = agents->z[index];
	agent.doublings = agents->doublings[index];
	agent.damage = agents->damage[index];
	agent.proliferate_bool = agents->proliferate_bool[index];
	agent.transition_to_early_sen = agents->transition_to_early_sen[index];

	//FLAME function call
	int dead = !ClearanceOfSenescent(&agent);
	

	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_ClearanceOfSenescent Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->z[index] = agent.z;
	agents->doublings[index] = agent.doublings;
	agents->damage[index] = agent.damage;
	agents->proliferate_bool[index] = agent.proliferate_bool;
	agents->transition_to_early_sen[index] = agent.transition_to_early_sen;
}

/**
 *
 */
__global__ void GPUFLAME_DetectDamage(xmachine_memory_Fibroblast_list* agents, xmachine_message_tissue_damage_report_list* tissue_damage_report_messages, xmachine_message_tissue_damage_report_PBM* partition_matrix){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_Fibroblast_count)
        return;
    

	//SoA to AoS - xmachine_memory_DetectDamage Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_Fibroblast agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.id = agents->id[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.z = agents->z[index];
	agent.doublings = agents->doublings[index];
	agent.damage = agents->damage[index];
	agent.proliferate_bool = agents->proliferate_bool[index];
	agent.transition_to_early_sen = agents->transition_to_early_sen[index];

	//FLAME function call
	int dead = !DetectDamage(&agent, tissue_damage_report_messages, partition_matrix);
	

	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_DetectDamage Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->z[index] = agent.z;
	agents->doublings[index] = agent.doublings;
	agents->damage[index] = agent.damage;
	agents->proliferate_bool[index] = agent.proliferate_bool;
	agents->transition_to_early_sen[index] = agent.transition_to_early_sen;
}

/**
 *
 */
__global__ void GPUFLAME_RepairDamage(xmachine_memory_Fibroblast_list* agents){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_Fibroblast_count)
        return;
    

	//SoA to AoS - xmachine_memory_RepairDamage Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_Fibroblast agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.id = agents->id[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.z = agents->z[index];
	agent.doublings = agents->doublings[index];
	agent.damage = agents->damage[index];
	agent.proliferate_bool = agents->proliferate_bool[index];
	agent.transition_to_early_sen = agents->transition_to_early_sen[index];

	//FLAME function call
	int dead = !RepairDamage(&agent);
	

	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_RepairDamage Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->z[index] = agent.z;
	agents->doublings[index] = agent.doublings;
	agents->damage[index] = agent.damage;
	agents->proliferate_bool[index] = agent.proliferate_bool;
	agents->transition_to_early_sen[index] = agent.transition_to_early_sen;
}

/**
 *
 */
__global__ void GPUFLAME_DamageRepaired(xmachine_memory_Fibroblast_list* agents){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_Fibroblast_count)
        return;
    

	//SoA to AoS - xmachine_memory_DamageRepaired Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_Fibroblast agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.id = agents->id[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.z = agents->z[index];
	agent.doublings = agents->doublings[index];
	agent.damage = agents->damage[index];
	agent.proliferate_bool = agents->proliferate_bool[index];
	agent.transition_to_early_sen = agents->transition_to_early_sen[index];

	//FLAME function call
	int dead = !DamageRepaired(&agent);
	

	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_DamageRepaired Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->z[index] = agent.z;
	agents->doublings[index] = agent.doublings;
	agents->damage[index] = agent.damage;
	agents->proliferate_bool[index] = agent.proliferate_bool;
	agents->transition_to_early_sen[index] = agent.transition_to_early_sen;
}

	
	
/* Graph utility functions */



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Rand48 functions */

__device__ static glm::uvec2 RNG_rand48_iterate_single(glm::uvec2 Xn, glm::uvec2 A, glm::uvec2 C)
{
	unsigned int R0, R1;

	// low 24-bit multiplication
	const unsigned int lo00 = __umul24(Xn.x, A.x);
	const unsigned int hi00 = __umulhi(Xn.x, A.x);

	// 24bit distribution of 32bit multiplication results
	R0 = (lo00 & 0xFFFFFF);
	R1 = (lo00 >> 24) | (hi00 << 8);

	R0 += C.x; R1 += C.y;

	// transfer overflows
	R1 += (R0 >> 24);
	R0 &= 0xFFFFFF;

	// cross-terms, low/hi 24-bit multiplication
	R1 += __umul24(Xn.y, A.x);
	R1 += __umul24(Xn.x, A.y);

	R1 &= 0xFFFFFF;

	return glm::uvec2(R0, R1);
}

//Templated function
template <int AGENT_TYPE>
__device__ float rnd(RNG_rand48* rand48){

	int index;
	
	//calculate the agents index in global agent list
	if (AGENT_TYPE == DISCRETE_2D){
		int width = (blockDim.x * gridDim.x);
		glm::ivec2 global_position;
		global_position.x = (blockIdx.x * blockDim.x) + threadIdx.x;
		global_position.y = (blockIdx.y * blockDim.y) + threadIdx.y;
		index = global_position.x + (global_position.y * width);
	}else//AGENT_TYPE == CONTINOUS
		index = threadIdx.x + blockIdx.x*blockDim.x;

	glm::uvec2 state = rand48->seeds[index];
	glm::uvec2 A = rand48->A;
	glm::uvec2 C = rand48->C;

	int rand = ( state.x >> 17 ) | ( state.y << 7);

	// this actually iterates the RNG
	state = RNG_rand48_iterate_single(state, A, C);

	rand48->seeds[index] = state;

	return (float)rand/2147483647;
}

__device__ float rnd(RNG_rand48* rand48){
	return rnd<DISCRETE_2D>(rand48);
}

#endif //_FLAMEGPU_KERNELS_H_
