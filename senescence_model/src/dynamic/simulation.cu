
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


  //Disable internal thrust warnings about conversions
  #ifdef _MSC_VER
  #pragma warning(push)
  #pragma warning (disable : 4267)
  #pragma warning (disable : 4244)
  #endif
  #ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wunused-parameter"
  #endif

  // includes
  #include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/extrema.h>
#include <thrust/system/cuda/execution_policy.h>
#include <cub/cub.cuh>

// include FLAME kernels
#include "FLAMEGPU_kernals.cu"


#ifdef _MSC_VER
#pragma warning(pop)
#endif
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

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

/* SM padding and offset variables */
int SM_START;
int PADDING;

unsigned int g_iterationNumber;

/* Agent Memory */

/* TissueBlock Agent variables these lists are used in the agent function where as the other lists are used only outside the agent functions*/
xmachine_memory_TissueBlock_list* d_TissueBlocks;      /**< Pointer to agent list (population) on the device*/
xmachine_memory_TissueBlock_list* d_TissueBlocks_swap; /**< Pointer to agent list swap on the device (used when killing agents)*/
xmachine_memory_TissueBlock_list* d_TissueBlocks_new;  /**< Pointer to new agent list on the device (used to hold new agents before they are appended to the population)*/
int h_xmachine_memory_TissueBlock_count;   /**< Agent population size counter */ 
uint * d_xmachine_memory_TissueBlock_keys;	  /**< Agent sort identifiers keys*/
uint * d_xmachine_memory_TissueBlock_values;  /**< Agent sort identifiers value */

/* TissueBlock state variables */
xmachine_memory_TissueBlock_list* h_TissueBlocks_default;      /**< Pointer to agent list (population) on host*/
xmachine_memory_TissueBlock_list* d_TissueBlocks_default;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_TissueBlock_default_count;   /**< Agent population size counter */ 

/* Fibroblast Agent variables these lists are used in the agent function where as the other lists are used only outside the agent functions*/
xmachine_memory_Fibroblast_list* d_Fibroblasts;      /**< Pointer to agent list (population) on the device*/
xmachine_memory_Fibroblast_list* d_Fibroblasts_swap; /**< Pointer to agent list swap on the device (used when killing agents)*/
xmachine_memory_Fibroblast_list* d_Fibroblasts_new;  /**< Pointer to new agent list on the device (used to hold new agents before they are appended to the population)*/
int h_xmachine_memory_Fibroblast_count;   /**< Agent population size counter */ 
uint * d_xmachine_memory_Fibroblast_keys;	  /**< Agent sort identifiers keys*/
uint * d_xmachine_memory_Fibroblast_values;  /**< Agent sort identifiers value */

/* Fibroblast state variables */
xmachine_memory_Fibroblast_list* h_Fibroblasts_Quiescent;      /**< Pointer to agent list (population) on host*/
xmachine_memory_Fibroblast_list* d_Fibroblasts_Quiescent;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_Fibroblast_Quiescent_count;   /**< Agent population size counter */ 

/* Fibroblast state variables */
xmachine_memory_Fibroblast_list* h_Fibroblasts_Repair;      /**< Pointer to agent list (population) on host*/
xmachine_memory_Fibroblast_list* d_Fibroblasts_Repair;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_Fibroblast_Repair_count;   /**< Agent population size counter */ 


/* Variables to track the state of host copies of state lists, for the purposes of host agent data access.
 * @future - if the host data is current it may be possible to avoid duplicating memcpy in xml output.
 */
unsigned int h_TissueBlocks_default_variable_id_data_iteration;
unsigned int h_TissueBlocks_default_variable_x_data_iteration;
unsigned int h_TissueBlocks_default_variable_y_data_iteration;
unsigned int h_TissueBlocks_default_variable_z_data_iteration;
unsigned int h_TissueBlocks_default_variable_damage_data_iteration;
unsigned int h_Fibroblasts_Quiescent_variable_id_data_iteration;
unsigned int h_Fibroblasts_Quiescent_variable_x_data_iteration;
unsigned int h_Fibroblasts_Quiescent_variable_y_data_iteration;
unsigned int h_Fibroblasts_Quiescent_variable_z_data_iteration;
unsigned int h_Fibroblasts_Quiescent_variable_damage_data_iteration;
unsigned int h_Fibroblasts_Quiescent_variable_current_state_data_iteration;
unsigned int h_Fibroblasts_Quiescent_variable_go_to_state_data_iteration;
unsigned int h_Fibroblasts_Repair_variable_id_data_iteration;
unsigned int h_Fibroblasts_Repair_variable_x_data_iteration;
unsigned int h_Fibroblasts_Repair_variable_y_data_iteration;
unsigned int h_Fibroblasts_Repair_variable_z_data_iteration;
unsigned int h_Fibroblasts_Repair_variable_damage_data_iteration;
unsigned int h_Fibroblasts_Repair_variable_current_state_data_iteration;
unsigned int h_Fibroblasts_Repair_variable_go_to_state_data_iteration;


/* Message Memory */

/* tissue_damage_report Message variables */
xmachine_message_tissue_damage_report_list* h_tissue_damage_reports;         /**< Pointer to message list on host*/
xmachine_message_tissue_damage_report_list* d_tissue_damage_reports;         /**< Pointer to message list on device*/
xmachine_message_tissue_damage_report_list* d_tissue_damage_reports_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Non partitioned and spatial partitioned message variables  */
int h_message_tissue_damage_report_count;         /**< message list counter*/
int h_message_tissue_damage_report_output_type;   /**< message output type (single or optional)*/
/* Spatial Partitioning Variables*/
#ifdef FAST_ATOMIC_SORTING
	uint * d_xmachine_message_tissue_damage_report_local_bin_index;	  /**< index offset within the assigned bin */
	uint * d_xmachine_message_tissue_damage_report_unsorted_index;		/**< unsorted index (hash) value for message */
    // Values for CUB exclusive scan of spatially partitioned variables
    void * d_temp_scan_storage_xmachine_message_tissue_damage_report;
    size_t temp_scan_bytes_xmachine_message_tissue_damage_report;
#else
	uint * d_xmachine_message_tissue_damage_report_keys;	  /**< message sort identifier keys*/
	uint * d_xmachine_message_tissue_damage_report_values;  /**< message sort identifier values */
#endif
xmachine_message_tissue_damage_report_PBM * d_tissue_damage_report_partition_matrix;  /**< Pointer to PCB matrix */
glm::vec3 h_message_tissue_damage_report_min_bounds;           /**< min bounds (x,y,z) of partitioning environment */
glm::vec3 h_message_tissue_damage_report_max_bounds;           /**< max bounds (x,y,z) of partitioning environment */
glm::ivec3 h_message_tissue_damage_report_partitionDim;           /**< partition dimensions (x,y,z) of partitioning environment */
float h_message_tissue_damage_report_radius;                 /**< partition radius (used to determin the size of the partitions) */
/* Texture offset values for host */
int h_tex_xmachine_message_tissue_damage_report_id_offset;
int h_tex_xmachine_message_tissue_damage_report_x_offset;
int h_tex_xmachine_message_tissue_damage_report_y_offset;
int h_tex_xmachine_message_tissue_damage_report_z_offset;
int h_tex_xmachine_message_tissue_damage_report_damage_offset;
int h_tex_xmachine_message_tissue_damage_report_pbm_start_offset;
int h_tex_xmachine_message_tissue_damage_report_pbm_end_or_count_offset;

/* fibroblast_report Message variables */
xmachine_message_fibroblast_report_list* h_fibroblast_reports;         /**< Pointer to message list on host*/
xmachine_message_fibroblast_report_list* d_fibroblast_reports;         /**< Pointer to message list on device*/
xmachine_message_fibroblast_report_list* d_fibroblast_reports_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Non partitioned and spatial partitioned message variables  */
int h_message_fibroblast_report_count;         /**< message list counter*/
int h_message_fibroblast_report_output_type;   /**< message output type (single or optional)*/
/* Spatial Partitioning Variables*/
#ifdef FAST_ATOMIC_SORTING
	uint * d_xmachine_message_fibroblast_report_local_bin_index;	  /**< index offset within the assigned bin */
	uint * d_xmachine_message_fibroblast_report_unsorted_index;		/**< unsorted index (hash) value for message */
    // Values for CUB exclusive scan of spatially partitioned variables
    void * d_temp_scan_storage_xmachine_message_fibroblast_report;
    size_t temp_scan_bytes_xmachine_message_fibroblast_report;
#else
	uint * d_xmachine_message_fibroblast_report_keys;	  /**< message sort identifier keys*/
	uint * d_xmachine_message_fibroblast_report_values;  /**< message sort identifier values */
#endif
xmachine_message_fibroblast_report_PBM * d_fibroblast_report_partition_matrix;  /**< Pointer to PCB matrix */
glm::vec3 h_message_fibroblast_report_min_bounds;           /**< min bounds (x,y,z) of partitioning environment */
glm::vec3 h_message_fibroblast_report_max_bounds;           /**< max bounds (x,y,z) of partitioning environment */
glm::ivec3 h_message_fibroblast_report_partitionDim;           /**< partition dimensions (x,y,z) of partitioning environment */
float h_message_fibroblast_report_radius;                 /**< partition radius (used to determin the size of the partitions) */
/* Texture offset values for host */
int h_tex_xmachine_message_fibroblast_report_id_offset;
int h_tex_xmachine_message_fibroblast_report_x_offset;
int h_tex_xmachine_message_fibroblast_report_y_offset;
int h_tex_xmachine_message_fibroblast_report_z_offset;
int h_tex_xmachine_message_fibroblast_report_current_state_offset;
int h_tex_xmachine_message_fibroblast_report_go_to_state_offset;
int h_tex_xmachine_message_fibroblast_report_pbm_start_offset;
int h_tex_xmachine_message_fibroblast_report_pbm_end_or_count_offset;

  
/* CUDA Streams for function layers */
cudaStream_t stream1;

/* Device memory and sizes for CUB values */

void * d_temp_scan_storage_TissueBlock;
size_t temp_scan_storage_bytes_TissueBlock;

void * d_temp_scan_storage_Fibroblast;
size_t temp_scan_storage_bytes_Fibroblast;


/*Global condition counts*/

/* RNG rand48 */
RNG_rand48* h_rand48;    /**< Pointer to RNG_rand48 seed list on host*/
RNG_rand48* d_rand48;    /**< Pointer to RNG_rand48 seed list on device*/

/* Early simulation exit*/
bool g_exit_early;

/* Cuda Event Timers for Instrumentation */
#if defined(INSTRUMENT_ITERATIONS) && INSTRUMENT_ITERATIONS
	cudaEvent_t instrument_iteration_start, instrument_iteration_stop;
	float instrument_iteration_milliseconds = 0.0f;
#endif
#if (defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS) || (defined(INSTRUMENT_INIT_FUNCTIONS) && INSTRUMENT_INIT_FUNCTIONS) || (defined(INSTRUMENT_STEP_FUNCTIONS) && INSTRUMENT_STEP_FUNCTIONS) || (defined(INSTRUMENT_EXIT_FUNCTIONS) && INSTRUMENT_EXIT_FUNCTIONS)
	cudaEvent_t instrument_start, instrument_stop;
	float instrument_milliseconds = 0.0f;
#endif

/* CUDA Parallel Primatives variables */
int scan_last_sum;           /**< Indicates if the position (in message list) of last message*/
int scan_last_included;      /**< Indicates if last sum value is included in the total sum count*/

/* Agent function prototypes */

/** TissueBlock_TissueTakesDamage
 * Agent function prototype for TissueTakesDamage function of TissueBlock agent
 */
void TissueBlock_TissueTakesDamage(cudaStream_t &stream);

/** TissueBlock_RepairDamage
 * Agent function prototype for RepairDamage function of TissueBlock agent
 */
void TissueBlock_RepairDamage(cudaStream_t &stream);

/** Fibroblast_QuiescentMigration
 * Agent function prototype for QuiescentMigration function of Fibroblast agent
 */
void Fibroblast_QuiescentMigration(cudaStream_t &stream);

/** Fibroblast_TransitionToRepair
 * Agent function prototype for TransitionToRepair function of Fibroblast agent
 */
void Fibroblast_TransitionToRepair(cudaStream_t &stream);

/** Fibroblast_TransitionToQuiescent
 * Agent function prototype for TransitionToQuiescent function of Fibroblast agent
 */
void Fibroblast_TransitionToQuiescent(cudaStream_t &stream);

  
void setPaddingAndOffset()
{
    PROFILE_SCOPED_RANGE("setPaddingAndOffset");
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	int x64_sys = 0;

	// This function call returns 9999 for both major & minor fields, if no CUDA capable devices are present
	if (deviceProp.major == 9999 && deviceProp.minor == 9999){
		printf("Error: There is no device supporting CUDA.\n");
		exit(EXIT_FAILURE);
	}
    
    //check if double is used and supported
#ifdef _DOUBLE_SUPPORT_REQUIRED_
	printf("Simulation requires full precision double values\n");
	if ((deviceProp.major < 2)&&(deviceProp.minor < 3)){
		printf("Error: Hardware does not support full precision double values!\n");
		exit(EXIT_FAILURE);
	}
    
#endif

	//check 32 or 64bit
	x64_sys = (sizeof(void*)==8);
	if (x64_sys)
	{
		printf("64Bit System Detected\n");
	}
	else
	{
		printf("32Bit System Detected\n");
	}

	SM_START = 0;
	PADDING = 0;
  
	//copy padding and offset to GPU
	gpuErrchk(cudaMemcpyToSymbol( d_SM_START, &SM_START, sizeof(int)));
	gpuErrchk(cudaMemcpyToSymbol( d_PADDING, &PADDING, sizeof(int)));     
}

int is_sqr_pow2(int x){
	int r = (int)pow(4, ceil(log(x)/log(4)));
	return (r == x);
}

int lowest_sqr_pow2(int x){
	int l;
	
	//escape early if x is square power of 2
	if (is_sqr_pow2(x))
		return x;
	
	//lower bound		
	l = (int)pow(4, floor(log(x)/log(4)));
	
	return l;
}

/* Unary function required for cudaOccupancyMaxPotentialBlockSizeVariableSMem to avoid warnings */
int no_sm(int b){
	return 0;
}

/* Unary function to return shared memory size for reorder message kernels */
int reorder_messages_sm_size(int blockSize)
{
	return sizeof(unsigned int)*(blockSize+1);
}


/** getIterationNumber
 *  Get the iteration number (host)
 *  @return a 1 indexed value for the iteration number, which is incremented at the start of each simulation step.
 *      I.e. it is 0 on up until the first call to singleIteration()
 */
extern unsigned int getIterationNumber(){
    return g_iterationNumber;
}

void initialise(char * inputfile){
    PROFILE_SCOPED_RANGE("initialise");

	//set the padding and offset values depending on architecture and OS
	setPaddingAndOffset();
  
		// Initialise some global variables
		g_iterationNumber = 0;
		g_exit_early = false;

    // Initialise variables for tracking which iterations' data is accessible on the host.
    h_TissueBlocks_default_variable_id_data_iteration = 0;
    h_TissueBlocks_default_variable_x_data_iteration = 0;
    h_TissueBlocks_default_variable_y_data_iteration = 0;
    h_TissueBlocks_default_variable_z_data_iteration = 0;
    h_TissueBlocks_default_variable_damage_data_iteration = 0;
    h_Fibroblasts_Quiescent_variable_id_data_iteration = 0;
    h_Fibroblasts_Quiescent_variable_x_data_iteration = 0;
    h_Fibroblasts_Quiescent_variable_y_data_iteration = 0;
    h_Fibroblasts_Quiescent_variable_z_data_iteration = 0;
    h_Fibroblasts_Quiescent_variable_damage_data_iteration = 0;
    h_Fibroblasts_Quiescent_variable_current_state_data_iteration = 0;
    h_Fibroblasts_Quiescent_variable_go_to_state_data_iteration = 0;
    h_Fibroblasts_Repair_variable_id_data_iteration = 0;
    h_Fibroblasts_Repair_variable_x_data_iteration = 0;
    h_Fibroblasts_Repair_variable_y_data_iteration = 0;
    h_Fibroblasts_Repair_variable_z_data_iteration = 0;
    h_Fibroblasts_Repair_variable_damage_data_iteration = 0;
    h_Fibroblasts_Repair_variable_current_state_data_iteration = 0;
    h_Fibroblasts_Repair_variable_go_to_state_data_iteration = 0;
    



	printf("Allocating Host and Device memory\n");
    PROFILE_PUSH_RANGE("allocate host");
	/* Agent memory allocation (CPU) */
	int xmachine_TissueBlock_SoA_size = sizeof(xmachine_memory_TissueBlock_list);
	h_TissueBlocks_default = (xmachine_memory_TissueBlock_list*)malloc(xmachine_TissueBlock_SoA_size);
	int xmachine_Fibroblast_SoA_size = sizeof(xmachine_memory_Fibroblast_list);
	h_Fibroblasts_Quiescent = (xmachine_memory_Fibroblast_list*)malloc(xmachine_Fibroblast_SoA_size);
	h_Fibroblasts_Repair = (xmachine_memory_Fibroblast_list*)malloc(xmachine_Fibroblast_SoA_size);

	/* Message memory allocation (CPU) */
	int message_tissue_damage_report_SoA_size = sizeof(xmachine_message_tissue_damage_report_list);
	h_tissue_damage_reports = (xmachine_message_tissue_damage_report_list*)malloc(message_tissue_damage_report_SoA_size);
	int message_fibroblast_report_SoA_size = sizeof(xmachine_message_fibroblast_report_list);
	h_fibroblast_reports = (xmachine_message_fibroblast_report_list*)malloc(message_fibroblast_report_SoA_size);

	//Exit if agent or message buffer sizes are to small for function outputs

  /* Graph memory allocation (CPU) */
  

    PROFILE_POP_RANGE(); //"allocate host"
	
			
	/* Set spatial partitioning tissue_damage_report message variables (min_bounds, max_bounds)*/
	h_message_tissue_damage_report_radius = (float)0.1;
	gpuErrchk(cudaMemcpyToSymbol( d_message_tissue_damage_report_radius, &h_message_tissue_damage_report_radius, sizeof(float)));	
	    h_message_tissue_damage_report_min_bounds = glm::vec3((float)-1.0, (float)-1.0, (float)-1.0);
	gpuErrchk(cudaMemcpyToSymbol( d_message_tissue_damage_report_min_bounds, &h_message_tissue_damage_report_min_bounds, sizeof(glm::vec3)));	
	h_message_tissue_damage_report_max_bounds = glm::vec3((float)1.0, (float)1.0, (float)1.0);
	gpuErrchk(cudaMemcpyToSymbol( d_message_tissue_damage_report_max_bounds, &h_message_tissue_damage_report_max_bounds, sizeof(glm::vec3)));	
	h_message_tissue_damage_report_partitionDim.x = (int)ceil((h_message_tissue_damage_report_max_bounds.x - h_message_tissue_damage_report_min_bounds.x)/h_message_tissue_damage_report_radius);
	h_message_tissue_damage_report_partitionDim.y = (int)ceil((h_message_tissue_damage_report_max_bounds.y - h_message_tissue_damage_report_min_bounds.y)/h_message_tissue_damage_report_radius);
	h_message_tissue_damage_report_partitionDim.z = (int)ceil((h_message_tissue_damage_report_max_bounds.z - h_message_tissue_damage_report_min_bounds.z)/h_message_tissue_damage_report_radius);
	gpuErrchk(cudaMemcpyToSymbol( d_message_tissue_damage_report_partitionDim, &h_message_tissue_damage_report_partitionDim, sizeof(glm::ivec3)));	
	
			
	/* Set spatial partitioning fibroblast_report message variables (min_bounds, max_bounds)*/
	h_message_fibroblast_report_radius = (float)0.1;
	gpuErrchk(cudaMemcpyToSymbol( d_message_fibroblast_report_radius, &h_message_fibroblast_report_radius, sizeof(float)));	
	    h_message_fibroblast_report_min_bounds = glm::vec3((float)-1.0, (float)-1.0, (float)-1.0);
	gpuErrchk(cudaMemcpyToSymbol( d_message_fibroblast_report_min_bounds, &h_message_fibroblast_report_min_bounds, sizeof(glm::vec3)));	
	h_message_fibroblast_report_max_bounds = glm::vec3((float)1.0, (float)1.0, (float)1.0);
	gpuErrchk(cudaMemcpyToSymbol( d_message_fibroblast_report_max_bounds, &h_message_fibroblast_report_max_bounds, sizeof(glm::vec3)));	
	h_message_fibroblast_report_partitionDim.x = (int)ceil((h_message_fibroblast_report_max_bounds.x - h_message_fibroblast_report_min_bounds.x)/h_message_fibroblast_report_radius);
	h_message_fibroblast_report_partitionDim.y = (int)ceil((h_message_fibroblast_report_max_bounds.y - h_message_fibroblast_report_min_bounds.y)/h_message_fibroblast_report_radius);
	h_message_fibroblast_report_partitionDim.z = (int)ceil((h_message_fibroblast_report_max_bounds.z - h_message_fibroblast_report_min_bounds.z)/h_message_fibroblast_report_radius);
	gpuErrchk(cudaMemcpyToSymbol( d_message_fibroblast_report_partitionDim, &h_message_fibroblast_report_partitionDim, sizeof(glm::ivec3)));	
	

	//read initial states
	readInitialStates(inputfile, h_TissueBlocks_default, &h_xmachine_memory_TissueBlock_default_count, h_Fibroblasts_Quiescent, &h_xmachine_memory_Fibroblast_Quiescent_count);

  // Read graphs from disk
  

  PROFILE_PUSH_RANGE("allocate device");
	
	/* TissueBlock Agent memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_TissueBlocks, xmachine_TissueBlock_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_TissueBlocks_swap, xmachine_TissueBlock_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_TissueBlocks_new, xmachine_TissueBlock_SoA_size));
    //continuous agent sort identifiers
  gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_TissueBlock_keys, xmachine_memory_TissueBlock_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_TissueBlock_values, xmachine_memory_TissueBlock_MAX* sizeof(uint)));
	/* default memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_TissueBlocks_default, xmachine_TissueBlock_SoA_size));
	gpuErrchk( cudaMemcpy( d_TissueBlocks_default, h_TissueBlocks_default, xmachine_TissueBlock_SoA_size, cudaMemcpyHostToDevice));
    
	/* Fibroblast Agent memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_Fibroblasts, xmachine_Fibroblast_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_Fibroblasts_swap, xmachine_Fibroblast_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_Fibroblasts_new, xmachine_Fibroblast_SoA_size));
    //continuous agent sort identifiers
  gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_Fibroblast_keys, xmachine_memory_Fibroblast_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_Fibroblast_values, xmachine_memory_Fibroblast_MAX* sizeof(uint)));
	/* Quiescent memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_Fibroblasts_Quiescent, xmachine_Fibroblast_SoA_size));
	gpuErrchk( cudaMemcpy( d_Fibroblasts_Quiescent, h_Fibroblasts_Quiescent, xmachine_Fibroblast_SoA_size, cudaMemcpyHostToDevice));
    
	/* Repair memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_Fibroblasts_Repair, xmachine_Fibroblast_SoA_size));
	gpuErrchk( cudaMemcpy( d_Fibroblasts_Repair, h_Fibroblasts_Repair, xmachine_Fibroblast_SoA_size, cudaMemcpyHostToDevice));
    
	/* tissue_damage_report Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_tissue_damage_reports, message_tissue_damage_report_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_tissue_damage_reports_swap, message_tissue_damage_report_SoA_size));
	gpuErrchk( cudaMemcpy( d_tissue_damage_reports, h_tissue_damage_reports, message_tissue_damage_report_SoA_size, cudaMemcpyHostToDevice));
	gpuErrchk( cudaMalloc( (void**) &d_tissue_damage_report_partition_matrix, sizeof(xmachine_message_tissue_damage_report_PBM)));
#ifdef FAST_ATOMIC_SORTING
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_message_tissue_damage_report_local_bin_index, xmachine_message_tissue_damage_report_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_message_tissue_damage_report_unsorted_index, xmachine_message_tissue_damage_report_MAX* sizeof(uint)));
    /* Calculate and allocate CUB temporary memory for exclusive scans */
    d_temp_scan_storage_xmachine_message_tissue_damage_report = nullptr;
    temp_scan_bytes_xmachine_message_tissue_damage_report = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_xmachine_message_tissue_damage_report, 
        temp_scan_bytes_xmachine_message_tissue_damage_report, 
        (int*) nullptr, 
        (int*) nullptr, 
        xmachine_message_tissue_damage_report_grid_size
    );
    gpuErrchk(cudaMalloc(&d_temp_scan_storage_xmachine_message_tissue_damage_report, temp_scan_bytes_xmachine_message_tissue_damage_report));
#else
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_message_tissue_damage_report_keys, xmachine_message_tissue_damage_report_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_message_tissue_damage_report_values, xmachine_message_tissue_damage_report_MAX* sizeof(uint)));
#endif
	
	/* fibroblast_report Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_fibroblast_reports, message_fibroblast_report_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_fibroblast_reports_swap, message_fibroblast_report_SoA_size));
	gpuErrchk( cudaMemcpy( d_fibroblast_reports, h_fibroblast_reports, message_fibroblast_report_SoA_size, cudaMemcpyHostToDevice));
	gpuErrchk( cudaMalloc( (void**) &d_fibroblast_report_partition_matrix, sizeof(xmachine_message_fibroblast_report_PBM)));
#ifdef FAST_ATOMIC_SORTING
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_message_fibroblast_report_local_bin_index, xmachine_message_fibroblast_report_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_message_fibroblast_report_unsorted_index, xmachine_message_fibroblast_report_MAX* sizeof(uint)));
    /* Calculate and allocate CUB temporary memory for exclusive scans */
    d_temp_scan_storage_xmachine_message_fibroblast_report = nullptr;
    temp_scan_bytes_xmachine_message_fibroblast_report = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_xmachine_message_fibroblast_report, 
        temp_scan_bytes_xmachine_message_fibroblast_report, 
        (int*) nullptr, 
        (int*) nullptr, 
        xmachine_message_fibroblast_report_grid_size
    );
    gpuErrchk(cudaMalloc(&d_temp_scan_storage_xmachine_message_fibroblast_report, temp_scan_bytes_xmachine_message_fibroblast_report));
#else
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_message_fibroblast_report_keys, xmachine_message_fibroblast_report_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_message_fibroblast_report_values, xmachine_message_fibroblast_report_MAX* sizeof(uint)));
#endif
		


  /* Allocate device memory for graphs */
  

    PROFILE_POP_RANGE(); // "allocate device"

    /* Calculate and allocate CUB temporary memory for exclusive scans */
    
    d_temp_scan_storage_TissueBlock = nullptr;
    temp_scan_storage_bytes_TissueBlock = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_TissueBlock, 
        temp_scan_storage_bytes_TissueBlock, 
        (int*) nullptr, 
        (int*) nullptr, 
        xmachine_memory_TissueBlock_MAX
    );
    gpuErrchk(cudaMalloc(&d_temp_scan_storage_TissueBlock, temp_scan_storage_bytes_TissueBlock));
    
    d_temp_scan_storage_Fibroblast = nullptr;
    temp_scan_storage_bytes_Fibroblast = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_Fibroblast, 
        temp_scan_storage_bytes_Fibroblast, 
        (int*) nullptr, 
        (int*) nullptr, 
        xmachine_memory_Fibroblast_MAX
    );
    gpuErrchk(cudaMalloc(&d_temp_scan_storage_Fibroblast, temp_scan_storage_bytes_Fibroblast));
    

	/*Set global condition counts*/

	/* RNG rand48 */
    PROFILE_PUSH_RANGE("Initialse RNG_rand48");
	int h_rand48_SoA_size = sizeof(RNG_rand48);
	h_rand48 = (RNG_rand48*)malloc(h_rand48_SoA_size);
	//allocate on GPU
	gpuErrchk( cudaMalloc( (void**) &d_rand48, h_rand48_SoA_size));
	// calculate strided iteration constants
	static const unsigned long long a = 0x5DEECE66DLL, c = 0xB;
	int seed = 123;
	unsigned long long A, C;
	A = 1LL; C = 0LL;
	for (unsigned int i = 0; i < buffer_size_MAX; ++i) {
		C += A*c;
		A *= a;
	}
	h_rand48->A.x = A & 0xFFFFFFLL;
	h_rand48->A.y = (A >> 24) & 0xFFFFFFLL;
	h_rand48->C.x = C & 0xFFFFFFLL;
	h_rand48->C.y = (C >> 24) & 0xFFFFFFLL;
	// prepare first nThreads random numbers from seed
	unsigned long long x = (((unsigned long long)seed) << 16) | 0x330E;
	for (unsigned int i = 0; i < buffer_size_MAX; ++i) {
		x = a*x + c;
		h_rand48->seeds[i].x = x & 0xFFFFFFLL;
		h_rand48->seeds[i].y = (x >> 24) & 0xFFFFFFLL;
	}
	//copy to device
	gpuErrchk( cudaMemcpy( d_rand48, h_rand48, h_rand48_SoA_size, cudaMemcpyHostToDevice));

    PROFILE_POP_RANGE();

	/* Call all init functions */
	/* Prepare cuda event timers for instrumentation */
#if defined(INSTRUMENT_ITERATIONS) && INSTRUMENT_ITERATIONS
	cudaEventCreate(&instrument_iteration_start);
	cudaEventCreate(&instrument_iteration_stop);
#endif
#if (defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS) || (defined(INSTRUMENT_INIT_FUNCTIONS) && INSTRUMENT_INIT_FUNCTIONS) || (defined(INSTRUMENT_STEP_FUNCTIONS) && INSTRUMENT_STEP_FUNCTIONS) || (defined(INSTRUMENT_EXIT_FUNCTIONS) && INSTRUMENT_EXIT_FUNCTIONS)
	cudaEventCreate(&instrument_start);
	cudaEventCreate(&instrument_stop);
#endif

	
  
  /* Init CUDA Streams for function layers */
  
  gpuErrchk(cudaStreamCreate(&stream1));

#if defined(OUTPUT_POPULATION_PER_ITERATION) && OUTPUT_POPULATION_PER_ITERATION
	// Print the agent population size of all agents in all states
	
		printf("Init agent_TissueBlock_default_count: %u\n",get_agent_TissueBlock_default_count());
	
		printf("Init agent_Fibroblast_Quiescent_count: %u\n",get_agent_Fibroblast_Quiescent_count());
	
		printf("Init agent_Fibroblast_Repair_count: %u\n",get_agent_Fibroblast_Repair_count());
	
#endif
} 


void sort_TissueBlocks_default(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_TissueBlock_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_TissueBlock_default_count); 
	gridSize = (h_xmachine_memory_TissueBlock_default_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_TissueBlock_keys, d_xmachine_memory_TissueBlock_values, d_TissueBlocks_default);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_TissueBlock_keys),  thrust::device_pointer_cast(d_xmachine_memory_TissueBlock_keys) + h_xmachine_memory_TissueBlock_default_count,  thrust::device_pointer_cast(d_xmachine_memory_TissueBlock_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_TissueBlock_agents, no_sm, h_xmachine_memory_TissueBlock_default_count); 
	gridSize = (h_xmachine_memory_TissueBlock_default_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_TissueBlock_agents<<<gridSize, blockSize>>>(d_xmachine_memory_TissueBlock_values, d_TissueBlocks_default, d_TissueBlocks_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_TissueBlock_list* d_TissueBlocks_temp = d_TissueBlocks_default;
	d_TissueBlocks_default = d_TissueBlocks_swap;
	d_TissueBlocks_swap = d_TissueBlocks_temp;	
}

void sort_Fibroblasts_Quiescent(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_Fibroblast_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_Fibroblast_Quiescent_count); 
	gridSize = (h_xmachine_memory_Fibroblast_Quiescent_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_Fibroblast_keys, d_xmachine_memory_Fibroblast_values, d_Fibroblasts_Quiescent);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_Fibroblast_keys),  thrust::device_pointer_cast(d_xmachine_memory_Fibroblast_keys) + h_xmachine_memory_Fibroblast_Quiescent_count,  thrust::device_pointer_cast(d_xmachine_memory_Fibroblast_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_Fibroblast_agents, no_sm, h_xmachine_memory_Fibroblast_Quiescent_count); 
	gridSize = (h_xmachine_memory_Fibroblast_Quiescent_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_Fibroblast_agents<<<gridSize, blockSize>>>(d_xmachine_memory_Fibroblast_values, d_Fibroblasts_Quiescent, d_Fibroblasts_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_Fibroblast_list* d_Fibroblasts_temp = d_Fibroblasts_Quiescent;
	d_Fibroblasts_Quiescent = d_Fibroblasts_swap;
	d_Fibroblasts_swap = d_Fibroblasts_temp;	
}

void sort_Fibroblasts_Repair(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_Fibroblast_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_Fibroblast_Repair_count); 
	gridSize = (h_xmachine_memory_Fibroblast_Repair_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_Fibroblast_keys, d_xmachine_memory_Fibroblast_values, d_Fibroblasts_Repair);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_Fibroblast_keys),  thrust::device_pointer_cast(d_xmachine_memory_Fibroblast_keys) + h_xmachine_memory_Fibroblast_Repair_count,  thrust::device_pointer_cast(d_xmachine_memory_Fibroblast_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_Fibroblast_agents, no_sm, h_xmachine_memory_Fibroblast_Repair_count); 
	gridSize = (h_xmachine_memory_Fibroblast_Repair_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_Fibroblast_agents<<<gridSize, blockSize>>>(d_xmachine_memory_Fibroblast_values, d_Fibroblasts_Repair, d_Fibroblasts_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_Fibroblast_list* d_Fibroblasts_temp = d_Fibroblasts_Repair;
	d_Fibroblasts_Repair = d_Fibroblasts_swap;
	d_Fibroblasts_swap = d_Fibroblasts_temp;	
}


void cleanup(){
    PROFILE_SCOPED_RANGE("cleanup");

    /* Call all exit functions */
	

	/* Agent data free*/
	
	/* TissueBlock Agent variables */
	gpuErrchk(cudaFree(d_TissueBlocks));
	gpuErrchk(cudaFree(d_TissueBlocks_swap));
	gpuErrchk(cudaFree(d_TissueBlocks_new));
	
	free( h_TissueBlocks_default);
	gpuErrchk(cudaFree(d_TissueBlocks_default));
	
	/* Fibroblast Agent variables */
	gpuErrchk(cudaFree(d_Fibroblasts));
	gpuErrchk(cudaFree(d_Fibroblasts_swap));
	gpuErrchk(cudaFree(d_Fibroblasts_new));
	
	free( h_Fibroblasts_Quiescent);
	gpuErrchk(cudaFree(d_Fibroblasts_Quiescent));
	
	free( h_Fibroblasts_Repair);
	gpuErrchk(cudaFree(d_Fibroblasts_Repair));
	

	/* Message data free */
	
	/* tissue_damage_report Message variables */
	free( h_tissue_damage_reports);
	gpuErrchk(cudaFree(d_tissue_damage_reports));
	gpuErrchk(cudaFree(d_tissue_damage_reports_swap));
	gpuErrchk(cudaFree(d_tissue_damage_report_partition_matrix));
#ifdef FAST_ATOMIC_SORTING
	gpuErrchk(cudaFree(d_xmachine_message_tissue_damage_report_local_bin_index));
	gpuErrchk(cudaFree(d_xmachine_message_tissue_damage_report_unsorted_index));
  gpuErrchk(cudaFree(d_temp_scan_storage_xmachine_message_tissue_damage_report));
  d_temp_scan_storage_xmachine_message_tissue_damage_report = nullptr;
  temp_scan_bytes_xmachine_message_tissue_damage_report = 0;
#else
	gpuErrchk(cudaFree(d_xmachine_message_tissue_damage_report_keys));
	gpuErrchk(cudaFree(d_xmachine_message_tissue_damage_report_values));
#endif
	
	/* fibroblast_report Message variables */
	free( h_fibroblast_reports);
	gpuErrchk(cudaFree(d_fibroblast_reports));
	gpuErrchk(cudaFree(d_fibroblast_reports_swap));
	gpuErrchk(cudaFree(d_fibroblast_report_partition_matrix));
#ifdef FAST_ATOMIC_SORTING
	gpuErrchk(cudaFree(d_xmachine_message_fibroblast_report_local_bin_index));
	gpuErrchk(cudaFree(d_xmachine_message_fibroblast_report_unsorted_index));
  gpuErrchk(cudaFree(d_temp_scan_storage_xmachine_message_fibroblast_report));
  d_temp_scan_storage_xmachine_message_fibroblast_report = nullptr;
  temp_scan_bytes_xmachine_message_fibroblast_report = 0;
#else
	gpuErrchk(cudaFree(d_xmachine_message_fibroblast_report_keys));
	gpuErrchk(cudaFree(d_xmachine_message_fibroblast_report_values));
#endif
	

    /* Free temporary CUB memory if required. */
    
    if(d_temp_scan_storage_TissueBlock != nullptr){
      gpuErrchk(cudaFree(d_temp_scan_storage_TissueBlock));
      d_temp_scan_storage_TissueBlock = nullptr;
      temp_scan_storage_bytes_TissueBlock = 0;
    }
    
    if(d_temp_scan_storage_Fibroblast != nullptr){
      gpuErrchk(cudaFree(d_temp_scan_storage_Fibroblast));
      d_temp_scan_storage_Fibroblast = nullptr;
      temp_scan_storage_bytes_Fibroblast = 0;
    }
    

  /* Graph data free */
  
  
  /* CUDA Streams for function layers */
  
  gpuErrchk(cudaStreamDestroy(stream1));

  /* CUDA Event Timers for Instrumentation */
#if defined(INSTRUMENT_ITERATIONS) && INSTRUMENT_ITERATIONS
	cudaEventDestroy(instrument_iteration_start);
	cudaEventDestroy(instrument_iteration_stop);
#endif
#if (defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS) || (defined(INSTRUMENT_INIT_FUNCTIONS) && INSTRUMENT_INIT_FUNCTIONS) || (defined(INSTRUMENT_STEP_FUNCTIONS) && INSTRUMENT_STEP_FUNCTIONS) || (defined(INSTRUMENT_EXIT_FUNCTIONS) && INSTRUMENT_EXIT_FUNCTIONS)
	cudaEventDestroy(instrument_start);
	cudaEventDestroy(instrument_stop);
#endif
}

void singleIteration(){
PROFILE_SCOPED_RANGE("singleIteration");

#if defined(INSTRUMENT_ITERATIONS) && INSTRUMENT_ITERATIONS
	cudaEventRecord(instrument_iteration_start);
#endif

    // Increment the iteration number.
    g_iterationNumber++;

  /* set all non partitioned, spatial partitioned and On-Graph Partitioned message counts to 0*/
	h_message_tissue_damage_report_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_tissue_damage_report_count, &h_message_tissue_damage_report_count, sizeof(int)));
	
	h_message_fibroblast_report_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_fibroblast_report_count, &h_message_fibroblast_report_count, sizeof(int)));
	

	/* Call agent functions in order iterating through the layer functions */
	
	/* Layer 1*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("Fibroblast_TransitionToRepair");
	Fibroblast_TransitionToRepair(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: Fibroblast_TransitionToRepair = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 2*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("Fibroblast_TransitionToQuiescent");
	Fibroblast_TransitionToQuiescent(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: Fibroblast_TransitionToQuiescent = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 3*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("TissueBlock_TissueTakesDamage");
	TissueBlock_TissueTakesDamage(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: TissueBlock_TissueTakesDamage = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 4*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("TissueBlock_RepairDamage");
	TissueBlock_RepairDamage(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: TissueBlock_RepairDamage = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 5*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("Fibroblast_QuiescentMigration");
	Fibroblast_QuiescentMigration(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: Fibroblast_QuiescentMigration = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
    
    /* Call all step functions */
	
#if defined(INSTRUMENT_STEP_FUNCTIONS) && INSTRUMENT_STEP_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
    PROFILE_PUSH_RANGE("Tissuelogs");
	Tissuelogs();
	
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_STEP_FUNCTIONS) && INSTRUMENT_STEP_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: Tissuelogs = %f (ms)\n", instrument_milliseconds);
#endif
#if defined(INSTRUMENT_STEP_FUNCTIONS) && INSTRUMENT_STEP_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
    PROFILE_PUSH_RANGE("FibroblastQuiescentlogs");
	FibroblastQuiescentlogs();
	
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_STEP_FUNCTIONS) && INSTRUMENT_STEP_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: FibroblastQuiescentlogs = %f (ms)\n", instrument_milliseconds);
#endif
#if defined(INSTRUMENT_STEP_FUNCTIONS) && INSTRUMENT_STEP_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
    PROFILE_PUSH_RANGE("FibroblastRepairlogs");
	FibroblastRepairlogs();
	
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_STEP_FUNCTIONS) && INSTRUMENT_STEP_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: FibroblastRepairlogs = %f (ms)\n", instrument_milliseconds);
#endif

#if defined(OUTPUT_POPULATION_PER_ITERATION) && OUTPUT_POPULATION_PER_ITERATION
	// Print the agent population size of all agents in all states
	
		printf("agent_TissueBlock_default_count: %u\n",get_agent_TissueBlock_default_count());
	
		printf("agent_Fibroblast_Quiescent_count: %u\n",get_agent_Fibroblast_Quiescent_count());
	
		printf("agent_Fibroblast_Repair_count: %u\n",get_agent_Fibroblast_Repair_count());
	
#endif

#if defined(INSTRUMENT_ITERATIONS) && INSTRUMENT_ITERATIONS
	cudaEventRecord(instrument_iteration_stop);
	cudaEventSynchronize(instrument_iteration_stop);
	cudaEventElapsedTime(&instrument_iteration_milliseconds, instrument_iteration_start, instrument_iteration_stop);
	printf("Instrumentation: Iteration Time = %f (ms)\n", instrument_iteration_milliseconds);
#endif
}

/* finish whole simulation after this step */
void set_exit_early() {
	g_exit_early = true;
}

bool get_exit_early() {
	return g_exit_early;
}

/* Environment functions */

//host constant declaration
float h_env_TISSUE_DAMAGE_PROB;
float h_env_QUIESCENT_MIGRATION_SCALE;
float h_env_REPAIR_RANGE;
float h_env_DAMAGE_DETECTION_RANGE;
int h_env_REPAIR_RATE;


//constant setter
void set_TISSUE_DAMAGE_PROB(float* h_TISSUE_DAMAGE_PROB){
    gpuErrchk(cudaMemcpyToSymbol(TISSUE_DAMAGE_PROB, h_TISSUE_DAMAGE_PROB, sizeof(float)));
    memcpy(&h_env_TISSUE_DAMAGE_PROB, h_TISSUE_DAMAGE_PROB,sizeof(float));
}

//constant getter
const float* get_TISSUE_DAMAGE_PROB(){
    return &h_env_TISSUE_DAMAGE_PROB;
}



//constant setter
void set_QUIESCENT_MIGRATION_SCALE(float* h_QUIESCENT_MIGRATION_SCALE){
    gpuErrchk(cudaMemcpyToSymbol(QUIESCENT_MIGRATION_SCALE, h_QUIESCENT_MIGRATION_SCALE, sizeof(float)));
    memcpy(&h_env_QUIESCENT_MIGRATION_SCALE, h_QUIESCENT_MIGRATION_SCALE,sizeof(float));
}

//constant getter
const float* get_QUIESCENT_MIGRATION_SCALE(){
    return &h_env_QUIESCENT_MIGRATION_SCALE;
}



//constant setter
void set_REPAIR_RANGE(float* h_REPAIR_RANGE){
    gpuErrchk(cudaMemcpyToSymbol(REPAIR_RANGE, h_REPAIR_RANGE, sizeof(float)));
    memcpy(&h_env_REPAIR_RANGE, h_REPAIR_RANGE,sizeof(float));
}

//constant getter
const float* get_REPAIR_RANGE(){
    return &h_env_REPAIR_RANGE;
}



//constant setter
void set_DAMAGE_DETECTION_RANGE(float* h_DAMAGE_DETECTION_RANGE){
    gpuErrchk(cudaMemcpyToSymbol(DAMAGE_DETECTION_RANGE, h_DAMAGE_DETECTION_RANGE, sizeof(float)));
    memcpy(&h_env_DAMAGE_DETECTION_RANGE, h_DAMAGE_DETECTION_RANGE,sizeof(float));
}

//constant getter
const float* get_DAMAGE_DETECTION_RANGE(){
    return &h_env_DAMAGE_DETECTION_RANGE;
}



//constant setter
void set_REPAIR_RATE(int* h_REPAIR_RATE){
    gpuErrchk(cudaMemcpyToSymbol(REPAIR_RATE, h_REPAIR_RATE, sizeof(int)));
    memcpy(&h_env_REPAIR_RATE, h_REPAIR_RATE,sizeof(int));
}

//constant getter
const int* get_REPAIR_RATE(){
    return &h_env_REPAIR_RATE;
}




/* Agent data access functions*/

    
int get_agent_TissueBlock_MAX_count(){
    return xmachine_memory_TissueBlock_MAX;
}


int get_agent_TissueBlock_default_count(){
	//continuous agent
	return h_xmachine_memory_TissueBlock_default_count;
	
}

xmachine_memory_TissueBlock_list* get_device_TissueBlock_default_agents(){
	return d_TissueBlocks_default;
}

xmachine_memory_TissueBlock_list* get_host_TissueBlock_default_agents(){
	return h_TissueBlocks_default;
}

    
int get_agent_Fibroblast_MAX_count(){
    return xmachine_memory_Fibroblast_MAX;
}


int get_agent_Fibroblast_Quiescent_count(){
	//continuous agent
	return h_xmachine_memory_Fibroblast_Quiescent_count;
	
}

xmachine_memory_Fibroblast_list* get_device_Fibroblast_Quiescent_agents(){
	return d_Fibroblasts_Quiescent;
}

xmachine_memory_Fibroblast_list* get_host_Fibroblast_Quiescent_agents(){
	return h_Fibroblasts_Quiescent;
}

int get_agent_Fibroblast_Repair_count(){
	//continuous agent
	return h_xmachine_memory_Fibroblast_Repair_count;
	
}

xmachine_memory_Fibroblast_list* get_device_Fibroblast_Repair_agents(){
	return d_Fibroblasts_Repair;
}

xmachine_memory_Fibroblast_list* get_host_Fibroblast_Repair_agents(){
	return h_Fibroblasts_Repair;
}



/* Host based access of agent variables*/

/** int get_TissueBlock_default_variable_id(unsigned int index)
 * Gets the value of the id variable of an TissueBlock agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable id
 */
__host__ int get_TissueBlock_default_variable_id(unsigned int index){
    unsigned int count = get_agent_TissueBlock_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_TissueBlocks_default_variable_id_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_TissueBlocks_default->id,
                    d_TissueBlocks_default->id,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_TissueBlocks_default_variable_id_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_TissueBlocks_default->id[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access id for the %u th member of TissueBlock_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_TissueBlock_default_variable_x(unsigned int index)
 * Gets the value of the x variable of an TissueBlock agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable x
 */
__host__ float get_TissueBlock_default_variable_x(unsigned int index){
    unsigned int count = get_agent_TissueBlock_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_TissueBlocks_default_variable_x_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_TissueBlocks_default->x,
                    d_TissueBlocks_default->x,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_TissueBlocks_default_variable_x_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_TissueBlocks_default->x[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access x for the %u th member of TissueBlock_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_TissueBlock_default_variable_y(unsigned int index)
 * Gets the value of the y variable of an TissueBlock agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable y
 */
__host__ float get_TissueBlock_default_variable_y(unsigned int index){
    unsigned int count = get_agent_TissueBlock_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_TissueBlocks_default_variable_y_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_TissueBlocks_default->y,
                    d_TissueBlocks_default->y,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_TissueBlocks_default_variable_y_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_TissueBlocks_default->y[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access y for the %u th member of TissueBlock_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_TissueBlock_default_variable_z(unsigned int index)
 * Gets the value of the z variable of an TissueBlock agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable z
 */
__host__ float get_TissueBlock_default_variable_z(unsigned int index){
    unsigned int count = get_agent_TissueBlock_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_TissueBlocks_default_variable_z_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_TissueBlocks_default->z,
                    d_TissueBlocks_default->z,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_TissueBlocks_default_variable_z_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_TissueBlocks_default->z[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access z for the %u th member of TissueBlock_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_TissueBlock_default_variable_damage(unsigned int index)
 * Gets the value of the damage variable of an TissueBlock agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable damage
 */
__host__ int get_TissueBlock_default_variable_damage(unsigned int index){
    unsigned int count = get_agent_TissueBlock_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_TissueBlocks_default_variable_damage_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_TissueBlocks_default->damage,
                    d_TissueBlocks_default->damage,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_TissueBlocks_default_variable_damage_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_TissueBlocks_default->damage[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access damage for the %u th member of TissueBlock_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_Fibroblast_Quiescent_variable_id(unsigned int index)
 * Gets the value of the id variable of an Fibroblast agent in the Quiescent state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable id
 */
__host__ int get_Fibroblast_Quiescent_variable_id(unsigned int index){
    unsigned int count = get_agent_Fibroblast_Quiescent_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Fibroblasts_Quiescent_variable_id_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Fibroblasts_Quiescent->id,
                    d_Fibroblasts_Quiescent->id,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Fibroblasts_Quiescent_variable_id_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Fibroblasts_Quiescent->id[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access id for the %u th member of Fibroblast_Quiescent. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_Fibroblast_Quiescent_variable_x(unsigned int index)
 * Gets the value of the x variable of an Fibroblast agent in the Quiescent state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable x
 */
__host__ float get_Fibroblast_Quiescent_variable_x(unsigned int index){
    unsigned int count = get_agent_Fibroblast_Quiescent_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Fibroblasts_Quiescent_variable_x_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Fibroblasts_Quiescent->x,
                    d_Fibroblasts_Quiescent->x,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Fibroblasts_Quiescent_variable_x_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Fibroblasts_Quiescent->x[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access x for the %u th member of Fibroblast_Quiescent. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_Fibroblast_Quiescent_variable_y(unsigned int index)
 * Gets the value of the y variable of an Fibroblast agent in the Quiescent state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable y
 */
__host__ float get_Fibroblast_Quiescent_variable_y(unsigned int index){
    unsigned int count = get_agent_Fibroblast_Quiescent_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Fibroblasts_Quiescent_variable_y_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Fibroblasts_Quiescent->y,
                    d_Fibroblasts_Quiescent->y,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Fibroblasts_Quiescent_variable_y_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Fibroblasts_Quiescent->y[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access y for the %u th member of Fibroblast_Quiescent. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_Fibroblast_Quiescent_variable_z(unsigned int index)
 * Gets the value of the z variable of an Fibroblast agent in the Quiescent state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable z
 */
__host__ float get_Fibroblast_Quiescent_variable_z(unsigned int index){
    unsigned int count = get_agent_Fibroblast_Quiescent_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Fibroblasts_Quiescent_variable_z_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Fibroblasts_Quiescent->z,
                    d_Fibroblasts_Quiescent->z,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Fibroblasts_Quiescent_variable_z_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Fibroblasts_Quiescent->z[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access z for the %u th member of Fibroblast_Quiescent. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_Fibroblast_Quiescent_variable_damage(unsigned int index)
 * Gets the value of the damage variable of an Fibroblast agent in the Quiescent state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable damage
 */
__host__ int get_Fibroblast_Quiescent_variable_damage(unsigned int index){
    unsigned int count = get_agent_Fibroblast_Quiescent_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Fibroblasts_Quiescent_variable_damage_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Fibroblasts_Quiescent->damage,
                    d_Fibroblasts_Quiescent->damage,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Fibroblasts_Quiescent_variable_damage_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Fibroblasts_Quiescent->damage[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access damage for the %u th member of Fibroblast_Quiescent. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_Fibroblast_Quiescent_variable_current_state(unsigned int index)
 * Gets the value of the current_state variable of an Fibroblast agent in the Quiescent state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable current_state
 */
__host__ int get_Fibroblast_Quiescent_variable_current_state(unsigned int index){
    unsigned int count = get_agent_Fibroblast_Quiescent_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Fibroblasts_Quiescent_variable_current_state_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Fibroblasts_Quiescent->current_state,
                    d_Fibroblasts_Quiescent->current_state,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Fibroblasts_Quiescent_variable_current_state_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Fibroblasts_Quiescent->current_state[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access current_state for the %u th member of Fibroblast_Quiescent. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_Fibroblast_Quiescent_variable_go_to_state(unsigned int index)
 * Gets the value of the go_to_state variable of an Fibroblast agent in the Quiescent state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable go_to_state
 */
__host__ int get_Fibroblast_Quiescent_variable_go_to_state(unsigned int index){
    unsigned int count = get_agent_Fibroblast_Quiescent_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Fibroblasts_Quiescent_variable_go_to_state_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Fibroblasts_Quiescent->go_to_state,
                    d_Fibroblasts_Quiescent->go_to_state,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Fibroblasts_Quiescent_variable_go_to_state_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Fibroblasts_Quiescent->go_to_state[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access go_to_state for the %u th member of Fibroblast_Quiescent. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_Fibroblast_Repair_variable_id(unsigned int index)
 * Gets the value of the id variable of an Fibroblast agent in the Repair state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable id
 */
__host__ int get_Fibroblast_Repair_variable_id(unsigned int index){
    unsigned int count = get_agent_Fibroblast_Repair_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Fibroblasts_Repair_variable_id_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Fibroblasts_Repair->id,
                    d_Fibroblasts_Repair->id,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Fibroblasts_Repair_variable_id_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Fibroblasts_Repair->id[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access id for the %u th member of Fibroblast_Repair. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_Fibroblast_Repair_variable_x(unsigned int index)
 * Gets the value of the x variable of an Fibroblast agent in the Repair state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable x
 */
__host__ float get_Fibroblast_Repair_variable_x(unsigned int index){
    unsigned int count = get_agent_Fibroblast_Repair_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Fibroblasts_Repair_variable_x_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Fibroblasts_Repair->x,
                    d_Fibroblasts_Repair->x,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Fibroblasts_Repair_variable_x_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Fibroblasts_Repair->x[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access x for the %u th member of Fibroblast_Repair. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_Fibroblast_Repair_variable_y(unsigned int index)
 * Gets the value of the y variable of an Fibroblast agent in the Repair state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable y
 */
__host__ float get_Fibroblast_Repair_variable_y(unsigned int index){
    unsigned int count = get_agent_Fibroblast_Repair_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Fibroblasts_Repair_variable_y_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Fibroblasts_Repair->y,
                    d_Fibroblasts_Repair->y,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Fibroblasts_Repair_variable_y_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Fibroblasts_Repair->y[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access y for the %u th member of Fibroblast_Repair. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_Fibroblast_Repair_variable_z(unsigned int index)
 * Gets the value of the z variable of an Fibroblast agent in the Repair state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable z
 */
__host__ float get_Fibroblast_Repair_variable_z(unsigned int index){
    unsigned int count = get_agent_Fibroblast_Repair_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Fibroblasts_Repair_variable_z_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Fibroblasts_Repair->z,
                    d_Fibroblasts_Repair->z,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Fibroblasts_Repair_variable_z_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Fibroblasts_Repair->z[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access z for the %u th member of Fibroblast_Repair. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_Fibroblast_Repair_variable_damage(unsigned int index)
 * Gets the value of the damage variable of an Fibroblast agent in the Repair state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable damage
 */
__host__ int get_Fibroblast_Repair_variable_damage(unsigned int index){
    unsigned int count = get_agent_Fibroblast_Repair_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Fibroblasts_Repair_variable_damage_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Fibroblasts_Repair->damage,
                    d_Fibroblasts_Repair->damage,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Fibroblasts_Repair_variable_damage_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Fibroblasts_Repair->damage[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access damage for the %u th member of Fibroblast_Repair. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_Fibroblast_Repair_variable_current_state(unsigned int index)
 * Gets the value of the current_state variable of an Fibroblast agent in the Repair state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable current_state
 */
__host__ int get_Fibroblast_Repair_variable_current_state(unsigned int index){
    unsigned int count = get_agent_Fibroblast_Repair_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Fibroblasts_Repair_variable_current_state_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Fibroblasts_Repair->current_state,
                    d_Fibroblasts_Repair->current_state,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Fibroblasts_Repair_variable_current_state_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Fibroblasts_Repair->current_state[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access current_state for the %u th member of Fibroblast_Repair. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_Fibroblast_Repair_variable_go_to_state(unsigned int index)
 * Gets the value of the go_to_state variable of an Fibroblast agent in the Repair state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable go_to_state
 */
__host__ int get_Fibroblast_Repair_variable_go_to_state(unsigned int index){
    unsigned int count = get_agent_Fibroblast_Repair_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Fibroblasts_Repair_variable_go_to_state_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Fibroblasts_Repair->go_to_state,
                    d_Fibroblasts_Repair->go_to_state,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Fibroblasts_Repair_variable_go_to_state_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Fibroblasts_Repair->go_to_state[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access go_to_state for the %u th member of Fibroblast_Repair. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}



/* Host based agent creation functions */
// These are only available for continuous agents.



/* copy_single_xmachine_memory_TissueBlock_hostToDevice
 * Private function to copy a host agent struct into a device SoA agent list.
 * @param d_dst destination agent state list
 * @param h_agent agent struct
 */
void copy_single_xmachine_memory_TissueBlock_hostToDevice(xmachine_memory_TissueBlock_list * d_dst, xmachine_memory_TissueBlock * h_agent){
 
		gpuErrchk(cudaMemcpy(d_dst->id, &h_agent->id, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->x, &h_agent->x, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->y, &h_agent->y, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->z, &h_agent->z, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->damage, &h_agent->damage, sizeof(int), cudaMemcpyHostToDevice));

}
/*
 * Private function to copy some elements from a host based struct of arrays to a device based struct of arrays for a single agent state.
 * Individual copies of `count` elements are performed for each agent variable or each component of agent array variables, to avoid wasted data transfer.
 * There will be a point at which a single cudaMemcpy will outperform many smaller memcpys, however host based agent creation should typically only populate a fraction of the maximum buffer size, so this should be more efficient.
 * @optimisation - experimentally find the proportion at which transferring the whole SoA would be better and incorporate this. The same will apply to agent variable arrays.
 * 
 * @param d_dst device destination SoA
 * @oaram h_src host source SoA
 * @param count the number of agents to transfer data for
 */
void copy_partial_xmachine_memory_TissueBlock_hostToDevice(xmachine_memory_TissueBlock_list * d_dst, xmachine_memory_TissueBlock_list * h_src, unsigned int count){
    // Only copy elements if there is data to move.
    if (count > 0){
	 
		gpuErrchk(cudaMemcpy(d_dst->id, h_src->id, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->x, h_src->x, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->y, h_src->y, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->z, h_src->z, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->damage, h_src->damage, count * sizeof(int), cudaMemcpyHostToDevice));

    }
}


/* copy_single_xmachine_memory_Fibroblast_hostToDevice
 * Private function to copy a host agent struct into a device SoA agent list.
 * @param d_dst destination agent state list
 * @param h_agent agent struct
 */
void copy_single_xmachine_memory_Fibroblast_hostToDevice(xmachine_memory_Fibroblast_list * d_dst, xmachine_memory_Fibroblast * h_agent){
 
		gpuErrchk(cudaMemcpy(d_dst->id, &h_agent->id, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->x, &h_agent->x, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->y, &h_agent->y, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->z, &h_agent->z, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->damage, &h_agent->damage, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->current_state, &h_agent->current_state, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->go_to_state, &h_agent->go_to_state, sizeof(int), cudaMemcpyHostToDevice));

}
/*
 * Private function to copy some elements from a host based struct of arrays to a device based struct of arrays for a single agent state.
 * Individual copies of `count` elements are performed for each agent variable or each component of agent array variables, to avoid wasted data transfer.
 * There will be a point at which a single cudaMemcpy will outperform many smaller memcpys, however host based agent creation should typically only populate a fraction of the maximum buffer size, so this should be more efficient.
 * @optimisation - experimentally find the proportion at which transferring the whole SoA would be better and incorporate this. The same will apply to agent variable arrays.
 * 
 * @param d_dst device destination SoA
 * @oaram h_src host source SoA
 * @param count the number of agents to transfer data for
 */
void copy_partial_xmachine_memory_Fibroblast_hostToDevice(xmachine_memory_Fibroblast_list * d_dst, xmachine_memory_Fibroblast_list * h_src, unsigned int count){
    // Only copy elements if there is data to move.
    if (count > 0){
	 
		gpuErrchk(cudaMemcpy(d_dst->id, h_src->id, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->x, h_src->x, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->y, h_src->y, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->z, h_src->z, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->damage, h_src->damage, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->current_state, h_src->current_state, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->go_to_state, h_src->go_to_state, count * sizeof(int), cudaMemcpyHostToDevice));

    }
}

xmachine_memory_TissueBlock* h_allocate_agent_TissueBlock(){
	xmachine_memory_TissueBlock* agent = (xmachine_memory_TissueBlock*)malloc(sizeof(xmachine_memory_TissueBlock));
	// Memset the whole agent strcuture
    memset(agent, 0, sizeof(xmachine_memory_TissueBlock));

	return agent;
}
void h_free_agent_TissueBlock(xmachine_memory_TissueBlock** agent){
 
	free((*agent));
	(*agent) = NULL;
}
xmachine_memory_TissueBlock** h_allocate_agent_TissueBlock_array(unsigned int count){
	xmachine_memory_TissueBlock ** agents = (xmachine_memory_TissueBlock**)malloc(count * sizeof(xmachine_memory_TissueBlock*));
	for (unsigned int i = 0; i < count; i++) {
		agents[i] = h_allocate_agent_TissueBlock();
	}
	return agents;
}
void h_free_agent_TissueBlock_array(xmachine_memory_TissueBlock*** agents, unsigned int count){
	for (unsigned int i = 0; i < count; i++) {
		h_free_agent_TissueBlock(&((*agents)[i]));
	}
	free((*agents));
	(*agents) = NULL;
}

void h_unpack_agents_TissueBlock_AoS_to_SoA(xmachine_memory_TissueBlock_list * dst, xmachine_memory_TissueBlock** src, unsigned int count){
	if(count > 0){
		for(unsigned int i = 0; i < count; i++){
			 
			dst->id[i] = src[i]->id;
			 
			dst->x[i] = src[i]->x;
			 
			dst->y[i] = src[i]->y;
			 
			dst->z[i] = src[i]->z;
			 
			dst->damage[i] = src[i]->damage;
			
		}
	}
}


void h_add_agent_TissueBlock_default(xmachine_memory_TissueBlock* agent){
	if (h_xmachine_memory_TissueBlock_count + 1 > xmachine_memory_TissueBlock_MAX){
		printf("Error: Buffer size of TissueBlock agents in state default will be exceeded by h_add_agent_TissueBlock_default\n");
		exit(EXIT_FAILURE);
	}	

	int blockSize;
	int minGridSize;
	int gridSize;
	unsigned int count = 1;
	
	// Copy data from host struct to device SoA for target state
	copy_single_xmachine_memory_TissueBlock_hostToDevice(d_TissueBlocks_new, agent);

	// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_TissueBlock_Agents, no_sm, count);
	gridSize = (count + blockSize - 1) / blockSize;
	append_TissueBlock_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_TissueBlocks_default, d_TissueBlocks_new, h_xmachine_memory_TissueBlock_default_count, count);
	gpuErrchkLaunch();
	// Update the number of agents in this state.
	h_xmachine_memory_TissueBlock_default_count += count;
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_TissueBlock_default_count, &h_xmachine_memory_TissueBlock_default_count, sizeof(int)));
	cudaDeviceSynchronize();

    // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
    h_TissueBlocks_default_variable_id_data_iteration = 0;
    h_TissueBlocks_default_variable_x_data_iteration = 0;
    h_TissueBlocks_default_variable_y_data_iteration = 0;
    h_TissueBlocks_default_variable_z_data_iteration = 0;
    h_TissueBlocks_default_variable_damage_data_iteration = 0;
    

}
void h_add_agents_TissueBlock_default(xmachine_memory_TissueBlock** agents, unsigned int count){
	if(count > 0){
		int blockSize;
		int minGridSize;
		int gridSize;

		if (h_xmachine_memory_TissueBlock_count + count > xmachine_memory_TissueBlock_MAX){
			printf("Error: Buffer size of TissueBlock agents in state default will be exceeded by h_add_agents_TissueBlock_default\n");
			exit(EXIT_FAILURE);
		}

		// Unpack data from AoS into the pre-existing SoA
		h_unpack_agents_TissueBlock_AoS_to_SoA(h_TissueBlocks_default, agents, count);

		// Copy data from the host SoA to the device SoA for the target state
		copy_partial_xmachine_memory_TissueBlock_hostToDevice(d_TissueBlocks_new, h_TissueBlocks_default, count);

		// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
		cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_TissueBlock_Agents, no_sm, count);
		gridSize = (count + blockSize - 1) / blockSize;
		append_TissueBlock_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_TissueBlocks_default, d_TissueBlocks_new, h_xmachine_memory_TissueBlock_default_count, count);
		gpuErrchkLaunch();
		// Update the number of agents in this state.
		h_xmachine_memory_TissueBlock_default_count += count;
		gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_TissueBlock_default_count, &h_xmachine_memory_TissueBlock_default_count, sizeof(int)));
		cudaDeviceSynchronize();

        // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
        h_TissueBlocks_default_variable_id_data_iteration = 0;
        h_TissueBlocks_default_variable_x_data_iteration = 0;
        h_TissueBlocks_default_variable_y_data_iteration = 0;
        h_TissueBlocks_default_variable_z_data_iteration = 0;
        h_TissueBlocks_default_variable_damage_data_iteration = 0;
        

	}
}

xmachine_memory_Fibroblast* h_allocate_agent_Fibroblast(){
	xmachine_memory_Fibroblast* agent = (xmachine_memory_Fibroblast*)malloc(sizeof(xmachine_memory_Fibroblast));
	// Memset the whole agent strcuture
    memset(agent, 0, sizeof(xmachine_memory_Fibroblast));

	return agent;
}
void h_free_agent_Fibroblast(xmachine_memory_Fibroblast** agent){
 
	free((*agent));
	(*agent) = NULL;
}
xmachine_memory_Fibroblast** h_allocate_agent_Fibroblast_array(unsigned int count){
	xmachine_memory_Fibroblast ** agents = (xmachine_memory_Fibroblast**)malloc(count * sizeof(xmachine_memory_Fibroblast*));
	for (unsigned int i = 0; i < count; i++) {
		agents[i] = h_allocate_agent_Fibroblast();
	}
	return agents;
}
void h_free_agent_Fibroblast_array(xmachine_memory_Fibroblast*** agents, unsigned int count){
	for (unsigned int i = 0; i < count; i++) {
		h_free_agent_Fibroblast(&((*agents)[i]));
	}
	free((*agents));
	(*agents) = NULL;
}

void h_unpack_agents_Fibroblast_AoS_to_SoA(xmachine_memory_Fibroblast_list * dst, xmachine_memory_Fibroblast** src, unsigned int count){
	if(count > 0){
		for(unsigned int i = 0; i < count; i++){
			 
			dst->id[i] = src[i]->id;
			 
			dst->x[i] = src[i]->x;
			 
			dst->y[i] = src[i]->y;
			 
			dst->z[i] = src[i]->z;
			 
			dst->damage[i] = src[i]->damage;
			 
			dst->current_state[i] = src[i]->current_state;
			 
			dst->go_to_state[i] = src[i]->go_to_state;
			
		}
	}
}


void h_add_agent_Fibroblast_Quiescent(xmachine_memory_Fibroblast* agent){
	if (h_xmachine_memory_Fibroblast_count + 1 > xmachine_memory_Fibroblast_MAX){
		printf("Error: Buffer size of Fibroblast agents in state Quiescent will be exceeded by h_add_agent_Fibroblast_Quiescent\n");
		exit(EXIT_FAILURE);
	}	

	int blockSize;
	int minGridSize;
	int gridSize;
	unsigned int count = 1;
	
	// Copy data from host struct to device SoA for target state
	copy_single_xmachine_memory_Fibroblast_hostToDevice(d_Fibroblasts_new, agent);

	// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_Fibroblast_Agents, no_sm, count);
	gridSize = (count + blockSize - 1) / blockSize;
	append_Fibroblast_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_Fibroblasts_Quiescent, d_Fibroblasts_new, h_xmachine_memory_Fibroblast_Quiescent_count, count);
	gpuErrchkLaunch();
	// Update the number of agents in this state.
	h_xmachine_memory_Fibroblast_Quiescent_count += count;
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_Fibroblast_Quiescent_count, &h_xmachine_memory_Fibroblast_Quiescent_count, sizeof(int)));
	cudaDeviceSynchronize();

    // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
    h_Fibroblasts_Quiescent_variable_id_data_iteration = 0;
    h_Fibroblasts_Quiescent_variable_x_data_iteration = 0;
    h_Fibroblasts_Quiescent_variable_y_data_iteration = 0;
    h_Fibroblasts_Quiescent_variable_z_data_iteration = 0;
    h_Fibroblasts_Quiescent_variable_damage_data_iteration = 0;
    h_Fibroblasts_Quiescent_variable_current_state_data_iteration = 0;
    h_Fibroblasts_Quiescent_variable_go_to_state_data_iteration = 0;
    

}
void h_add_agents_Fibroblast_Quiescent(xmachine_memory_Fibroblast** agents, unsigned int count){
	if(count > 0){
		int blockSize;
		int minGridSize;
		int gridSize;

		if (h_xmachine_memory_Fibroblast_count + count > xmachine_memory_Fibroblast_MAX){
			printf("Error: Buffer size of Fibroblast agents in state Quiescent will be exceeded by h_add_agents_Fibroblast_Quiescent\n");
			exit(EXIT_FAILURE);
		}

		// Unpack data from AoS into the pre-existing SoA
		h_unpack_agents_Fibroblast_AoS_to_SoA(h_Fibroblasts_Quiescent, agents, count);

		// Copy data from the host SoA to the device SoA for the target state
		copy_partial_xmachine_memory_Fibroblast_hostToDevice(d_Fibroblasts_new, h_Fibroblasts_Quiescent, count);

		// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
		cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_Fibroblast_Agents, no_sm, count);
		gridSize = (count + blockSize - 1) / blockSize;
		append_Fibroblast_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_Fibroblasts_Quiescent, d_Fibroblasts_new, h_xmachine_memory_Fibroblast_Quiescent_count, count);
		gpuErrchkLaunch();
		// Update the number of agents in this state.
		h_xmachine_memory_Fibroblast_Quiescent_count += count;
		gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_Fibroblast_Quiescent_count, &h_xmachine_memory_Fibroblast_Quiescent_count, sizeof(int)));
		cudaDeviceSynchronize();

        // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
        h_Fibroblasts_Quiescent_variable_id_data_iteration = 0;
        h_Fibroblasts_Quiescent_variable_x_data_iteration = 0;
        h_Fibroblasts_Quiescent_variable_y_data_iteration = 0;
        h_Fibroblasts_Quiescent_variable_z_data_iteration = 0;
        h_Fibroblasts_Quiescent_variable_damage_data_iteration = 0;
        h_Fibroblasts_Quiescent_variable_current_state_data_iteration = 0;
        h_Fibroblasts_Quiescent_variable_go_to_state_data_iteration = 0;
        

	}
}


void h_add_agent_Fibroblast_Repair(xmachine_memory_Fibroblast* agent){
	if (h_xmachine_memory_Fibroblast_count + 1 > xmachine_memory_Fibroblast_MAX){
		printf("Error: Buffer size of Fibroblast agents in state Repair will be exceeded by h_add_agent_Fibroblast_Repair\n");
		exit(EXIT_FAILURE);
	}	

	int blockSize;
	int minGridSize;
	int gridSize;
	unsigned int count = 1;
	
	// Copy data from host struct to device SoA for target state
	copy_single_xmachine_memory_Fibroblast_hostToDevice(d_Fibroblasts_new, agent);

	// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_Fibroblast_Agents, no_sm, count);
	gridSize = (count + blockSize - 1) / blockSize;
	append_Fibroblast_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_Fibroblasts_Repair, d_Fibroblasts_new, h_xmachine_memory_Fibroblast_Repair_count, count);
	gpuErrchkLaunch();
	// Update the number of agents in this state.
	h_xmachine_memory_Fibroblast_Repair_count += count;
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_Fibroblast_Repair_count, &h_xmachine_memory_Fibroblast_Repair_count, sizeof(int)));
	cudaDeviceSynchronize();

    // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
    h_Fibroblasts_Repair_variable_id_data_iteration = 0;
    h_Fibroblasts_Repair_variable_x_data_iteration = 0;
    h_Fibroblasts_Repair_variable_y_data_iteration = 0;
    h_Fibroblasts_Repair_variable_z_data_iteration = 0;
    h_Fibroblasts_Repair_variable_damage_data_iteration = 0;
    h_Fibroblasts_Repair_variable_current_state_data_iteration = 0;
    h_Fibroblasts_Repair_variable_go_to_state_data_iteration = 0;
    

}
void h_add_agents_Fibroblast_Repair(xmachine_memory_Fibroblast** agents, unsigned int count){
	if(count > 0){
		int blockSize;
		int minGridSize;
		int gridSize;

		if (h_xmachine_memory_Fibroblast_count + count > xmachine_memory_Fibroblast_MAX){
			printf("Error: Buffer size of Fibroblast agents in state Repair will be exceeded by h_add_agents_Fibroblast_Repair\n");
			exit(EXIT_FAILURE);
		}

		// Unpack data from AoS into the pre-existing SoA
		h_unpack_agents_Fibroblast_AoS_to_SoA(h_Fibroblasts_Repair, agents, count);

		// Copy data from the host SoA to the device SoA for the target state
		copy_partial_xmachine_memory_Fibroblast_hostToDevice(d_Fibroblasts_new, h_Fibroblasts_Repair, count);

		// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
		cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_Fibroblast_Agents, no_sm, count);
		gridSize = (count + blockSize - 1) / blockSize;
		append_Fibroblast_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_Fibroblasts_Repair, d_Fibroblasts_new, h_xmachine_memory_Fibroblast_Repair_count, count);
		gpuErrchkLaunch();
		// Update the number of agents in this state.
		h_xmachine_memory_Fibroblast_Repair_count += count;
		gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_Fibroblast_Repair_count, &h_xmachine_memory_Fibroblast_Repair_count, sizeof(int)));
		cudaDeviceSynchronize();

        // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
        h_Fibroblasts_Repair_variable_id_data_iteration = 0;
        h_Fibroblasts_Repair_variable_x_data_iteration = 0;
        h_Fibroblasts_Repair_variable_y_data_iteration = 0;
        h_Fibroblasts_Repair_variable_z_data_iteration = 0;
        h_Fibroblasts_Repair_variable_damage_data_iteration = 0;
        h_Fibroblasts_Repair_variable_current_state_data_iteration = 0;
        h_Fibroblasts_Repair_variable_go_to_state_data_iteration = 0;
        

	}
}


/*  Analytics Functions */

int reduce_TissueBlock_default_id_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_TissueBlocks_default->id),  thrust::device_pointer_cast(d_TissueBlocks_default->id) + h_xmachine_memory_TissueBlock_default_count);
}

int count_TissueBlock_default_id_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_TissueBlocks_default->id),  thrust::device_pointer_cast(d_TissueBlocks_default->id) + h_xmachine_memory_TissueBlock_default_count, count_value);
}
int min_TissueBlock_default_id_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_TissueBlocks_default->id);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_TissueBlock_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_TissueBlock_default_id_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_TissueBlocks_default->id);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_TissueBlock_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_TissueBlock_default_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_TissueBlocks_default->x),  thrust::device_pointer_cast(d_TissueBlocks_default->x) + h_xmachine_memory_TissueBlock_default_count);
}

float min_TissueBlock_default_x_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_TissueBlocks_default->x);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_TissueBlock_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_TissueBlock_default_x_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_TissueBlocks_default->x);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_TissueBlock_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_TissueBlock_default_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_TissueBlocks_default->y),  thrust::device_pointer_cast(d_TissueBlocks_default->y) + h_xmachine_memory_TissueBlock_default_count);
}

float min_TissueBlock_default_y_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_TissueBlocks_default->y);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_TissueBlock_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_TissueBlock_default_y_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_TissueBlocks_default->y);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_TissueBlock_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_TissueBlock_default_z_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_TissueBlocks_default->z),  thrust::device_pointer_cast(d_TissueBlocks_default->z) + h_xmachine_memory_TissueBlock_default_count);
}

float min_TissueBlock_default_z_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_TissueBlocks_default->z);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_TissueBlock_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_TissueBlock_default_z_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_TissueBlocks_default->z);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_TissueBlock_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_TissueBlock_default_damage_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_TissueBlocks_default->damage),  thrust::device_pointer_cast(d_TissueBlocks_default->damage) + h_xmachine_memory_TissueBlock_default_count);
}

int count_TissueBlock_default_damage_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_TissueBlocks_default->damage),  thrust::device_pointer_cast(d_TissueBlocks_default->damage) + h_xmachine_memory_TissueBlock_default_count, count_value);
}
int min_TissueBlock_default_damage_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_TissueBlocks_default->damage);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_TissueBlock_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_TissueBlock_default_damage_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_TissueBlocks_default->damage);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_TissueBlock_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_Fibroblast_Quiescent_id_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Fibroblasts_Quiescent->id),  thrust::device_pointer_cast(d_Fibroblasts_Quiescent->id) + h_xmachine_memory_Fibroblast_Quiescent_count);
}

int count_Fibroblast_Quiescent_id_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Fibroblasts_Quiescent->id),  thrust::device_pointer_cast(d_Fibroblasts_Quiescent->id) + h_xmachine_memory_Fibroblast_Quiescent_count, count_value);
}
int min_Fibroblast_Quiescent_id_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Quiescent->id);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_Quiescent_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_Fibroblast_Quiescent_id_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Quiescent->id);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_Quiescent_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_Fibroblast_Quiescent_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Fibroblasts_Quiescent->x),  thrust::device_pointer_cast(d_Fibroblasts_Quiescent->x) + h_xmachine_memory_Fibroblast_Quiescent_count);
}

float min_Fibroblast_Quiescent_x_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Quiescent->x);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_Quiescent_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_Fibroblast_Quiescent_x_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Quiescent->x);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_Quiescent_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_Fibroblast_Quiescent_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Fibroblasts_Quiescent->y),  thrust::device_pointer_cast(d_Fibroblasts_Quiescent->y) + h_xmachine_memory_Fibroblast_Quiescent_count);
}

float min_Fibroblast_Quiescent_y_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Quiescent->y);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_Quiescent_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_Fibroblast_Quiescent_y_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Quiescent->y);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_Quiescent_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_Fibroblast_Quiescent_z_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Fibroblasts_Quiescent->z),  thrust::device_pointer_cast(d_Fibroblasts_Quiescent->z) + h_xmachine_memory_Fibroblast_Quiescent_count);
}

float min_Fibroblast_Quiescent_z_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Quiescent->z);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_Quiescent_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_Fibroblast_Quiescent_z_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Quiescent->z);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_Quiescent_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_Fibroblast_Quiescent_damage_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Fibroblasts_Quiescent->damage),  thrust::device_pointer_cast(d_Fibroblasts_Quiescent->damage) + h_xmachine_memory_Fibroblast_Quiescent_count);
}

int count_Fibroblast_Quiescent_damage_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Fibroblasts_Quiescent->damage),  thrust::device_pointer_cast(d_Fibroblasts_Quiescent->damage) + h_xmachine_memory_Fibroblast_Quiescent_count, count_value);
}
int min_Fibroblast_Quiescent_damage_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Quiescent->damage);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_Quiescent_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_Fibroblast_Quiescent_damage_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Quiescent->damage);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_Quiescent_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_Fibroblast_Quiescent_current_state_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Fibroblasts_Quiescent->current_state),  thrust::device_pointer_cast(d_Fibroblasts_Quiescent->current_state) + h_xmachine_memory_Fibroblast_Quiescent_count);
}

int count_Fibroblast_Quiescent_current_state_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Fibroblasts_Quiescent->current_state),  thrust::device_pointer_cast(d_Fibroblasts_Quiescent->current_state) + h_xmachine_memory_Fibroblast_Quiescent_count, count_value);
}
int min_Fibroblast_Quiescent_current_state_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Quiescent->current_state);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_Quiescent_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_Fibroblast_Quiescent_current_state_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Quiescent->current_state);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_Quiescent_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_Fibroblast_Quiescent_go_to_state_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Fibroblasts_Quiescent->go_to_state),  thrust::device_pointer_cast(d_Fibroblasts_Quiescent->go_to_state) + h_xmachine_memory_Fibroblast_Quiescent_count);
}

int count_Fibroblast_Quiescent_go_to_state_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Fibroblasts_Quiescent->go_to_state),  thrust::device_pointer_cast(d_Fibroblasts_Quiescent->go_to_state) + h_xmachine_memory_Fibroblast_Quiescent_count, count_value);
}
int min_Fibroblast_Quiescent_go_to_state_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Quiescent->go_to_state);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_Quiescent_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_Fibroblast_Quiescent_go_to_state_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Quiescent->go_to_state);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_Quiescent_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_Fibroblast_Repair_id_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Fibroblasts_Repair->id),  thrust::device_pointer_cast(d_Fibroblasts_Repair->id) + h_xmachine_memory_Fibroblast_Repair_count);
}

int count_Fibroblast_Repair_id_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Fibroblasts_Repair->id),  thrust::device_pointer_cast(d_Fibroblasts_Repair->id) + h_xmachine_memory_Fibroblast_Repair_count, count_value);
}
int min_Fibroblast_Repair_id_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Repair->id);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_Repair_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_Fibroblast_Repair_id_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Repair->id);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_Repair_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_Fibroblast_Repair_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Fibroblasts_Repair->x),  thrust::device_pointer_cast(d_Fibroblasts_Repair->x) + h_xmachine_memory_Fibroblast_Repair_count);
}

float min_Fibroblast_Repair_x_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Repair->x);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_Repair_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_Fibroblast_Repair_x_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Repair->x);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_Repair_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_Fibroblast_Repair_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Fibroblasts_Repair->y),  thrust::device_pointer_cast(d_Fibroblasts_Repair->y) + h_xmachine_memory_Fibroblast_Repair_count);
}

float min_Fibroblast_Repair_y_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Repair->y);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_Repair_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_Fibroblast_Repair_y_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Repair->y);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_Repair_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_Fibroblast_Repair_z_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Fibroblasts_Repair->z),  thrust::device_pointer_cast(d_Fibroblasts_Repair->z) + h_xmachine_memory_Fibroblast_Repair_count);
}

float min_Fibroblast_Repair_z_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Repair->z);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_Repair_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_Fibroblast_Repair_z_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Repair->z);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_Repair_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_Fibroblast_Repair_damage_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Fibroblasts_Repair->damage),  thrust::device_pointer_cast(d_Fibroblasts_Repair->damage) + h_xmachine_memory_Fibroblast_Repair_count);
}

int count_Fibroblast_Repair_damage_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Fibroblasts_Repair->damage),  thrust::device_pointer_cast(d_Fibroblasts_Repair->damage) + h_xmachine_memory_Fibroblast_Repair_count, count_value);
}
int min_Fibroblast_Repair_damage_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Repair->damage);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_Repair_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_Fibroblast_Repair_damage_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Repair->damage);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_Repair_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_Fibroblast_Repair_current_state_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Fibroblasts_Repair->current_state),  thrust::device_pointer_cast(d_Fibroblasts_Repair->current_state) + h_xmachine_memory_Fibroblast_Repair_count);
}

int count_Fibroblast_Repair_current_state_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Fibroblasts_Repair->current_state),  thrust::device_pointer_cast(d_Fibroblasts_Repair->current_state) + h_xmachine_memory_Fibroblast_Repair_count, count_value);
}
int min_Fibroblast_Repair_current_state_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Repair->current_state);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_Repair_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_Fibroblast_Repair_current_state_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Repair->current_state);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_Repair_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_Fibroblast_Repair_go_to_state_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Fibroblasts_Repair->go_to_state),  thrust::device_pointer_cast(d_Fibroblasts_Repair->go_to_state) + h_xmachine_memory_Fibroblast_Repair_count);
}

int count_Fibroblast_Repair_go_to_state_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Fibroblasts_Repair->go_to_state),  thrust::device_pointer_cast(d_Fibroblasts_Repair->go_to_state) + h_xmachine_memory_Fibroblast_Repair_count, count_value);
}
int min_Fibroblast_Repair_go_to_state_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Repair->go_to_state);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_Repair_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_Fibroblast_Repair_go_to_state_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Repair->go_to_state);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_Repair_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}



/* Agent functions */


	
/* Shared memory size calculator for agent function */
int TissueBlock_TissueTakesDamage_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** TissueBlock_TissueTakesDamage
 * Agent function prototype for TissueTakesDamage function of TissueBlock agent
 */
void TissueBlock_TissueTakesDamage(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_TissueBlock_default_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_TissueBlock_default_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_TissueBlock_list* TissueBlocks_default_temp = d_TissueBlocks;
	d_TissueBlocks = d_TissueBlocks_default;
	d_TissueBlocks_default = TissueBlocks_default_temp;
	//set working count to current state count
	h_xmachine_memory_TissueBlock_count = h_xmachine_memory_TissueBlock_default_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_TissueBlock_count, &h_xmachine_memory_TissueBlock_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_TissueBlock_default_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_TissueBlock_default_count, &h_xmachine_memory_TissueBlock_default_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	//CONTINUOUS AGENT CHECK FUNCTION OUTPUT BUFFERS FOR OUT OF BOUNDS
	if (h_message_tissue_damage_report_count + h_xmachine_memory_TissueBlock_count > xmachine_message_tissue_damage_report_MAX){
		printf("Error: Buffer size of tissue_damage_report message will be exceeded in function TissueTakesDamage\n");
		exit(EXIT_FAILURE);
	}
	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_TissueTakesDamage, TissueBlock_TissueTakesDamage_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = TissueBlock_TissueTakesDamage_sm_size(blockSize);
	
	
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	//Set the message_type for non partitioned, spatially partitioned and On-Graph Partitioned message outputs
	h_message_tissue_damage_report_output_type = optional_message;
	gpuErrchk( cudaMemcpyToSymbol( d_message_tissue_damage_report_output_type, &h_message_tissue_damage_report_output_type, sizeof(int)));
	//message is optional so reset the swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_tissue_damage_report_swaps, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_tissue_damage_report_swaps<<<gridSize, blockSize, 0, stream>>>(d_tissue_damage_reports); 
	gpuErrchkLaunch();
	
	
	//MAIN XMACHINE FUNCTION CALL (TissueTakesDamage)
	//Reallocate   : false
	//Input        : 
	//Output       : tissue_damage_report
	//Agent Output : 
	GPUFLAME_TissueTakesDamage<<<g, b, sm_size, stream>>>(d_TissueBlocks, d_tissue_damage_reports, d_rand48);
	gpuErrchkLaunch();
	
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	//tissue_damage_report Message Type Prefix Sum
	
	//swap output
	xmachine_message_tissue_damage_report_list* d_tissue_damage_reports_scanswap_temp = d_tissue_damage_reports;
	d_tissue_damage_reports = d_tissue_damage_reports_swap;
	d_tissue_damage_reports_swap = d_tissue_damage_reports_scanswap_temp;
	
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_TissueBlock, 
        temp_scan_storage_bytes_TissueBlock, 
        d_tissue_damage_reports_swap->_scan_input,
        d_tissue_damage_reports_swap->_position,
        h_xmachine_memory_TissueBlock_count, 
        stream
    );

	//Scatter
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_optional_tissue_damage_report_messages, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	scatter_optional_tissue_damage_report_messages<<<gridSize, blockSize, 0, stream>>>(d_tissue_damage_reports, d_tissue_damage_reports_swap);
	gpuErrchkLaunch();
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_tissue_damage_reports_swap->_position[h_xmachine_memory_TissueBlock_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_tissue_damage_reports_swap->_scan_input[h_xmachine_memory_TissueBlock_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	//If last item in prefix sum was 1 then increase its index to get the count
	if (scan_last_included == 1){
		h_message_tissue_damage_report_count += scan_last_sum+1;
	}else{
		h_message_tissue_damage_report_count += scan_last_sum;
	}
    //Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_message_tissue_damage_report_count, &h_message_tissue_damage_report_count, sizeof(int)));	
	
	//reset partition matrix
	gpuErrchk( cudaMemset( (void*) d_tissue_damage_report_partition_matrix, 0, sizeof(xmachine_message_tissue_damage_report_PBM)));
    //PR Bug fix: Second fix. This should prevent future problems when multiple agents write the same message as now the message structure is completely rebuilt after an output.
    if (h_message_tissue_damage_report_count > 0){
#ifdef FAST_ATOMIC_SORTING
      //USE ATOMICS TO BUILD PARTITION BOUNDARY
	  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, hist_tissue_damage_report_messages, no_sm, h_message_tissue_damage_report_count); 
	  gridSize = (h_message_tissue_damage_report_count + blockSize - 1) / blockSize;
	  hist_tissue_damage_report_messages<<<gridSize, blockSize, 0, stream>>>(d_xmachine_message_tissue_damage_report_local_bin_index, d_xmachine_message_tissue_damage_report_unsorted_index, d_tissue_damage_report_partition_matrix->end_or_count, d_tissue_damage_reports, h_message_tissue_damage_report_count);
	  gpuErrchkLaunch();
	
      // Scan
      cub::DeviceScan::ExclusiveSum(
          d_temp_scan_storage_xmachine_message_tissue_damage_report, 
          temp_scan_bytes_xmachine_message_tissue_damage_report, 
          d_tissue_damage_report_partition_matrix->end_or_count,
          d_tissue_damage_report_partition_matrix->start,
          xmachine_message_tissue_damage_report_grid_size, 
          stream
      );
	
	  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_tissue_damage_report_messages, no_sm, h_message_tissue_damage_report_count); 
	  gridSize = (h_message_tissue_damage_report_count + blockSize - 1) / blockSize; 	// Round up according to array size 
	  reorder_tissue_damage_report_messages <<<gridSize, blockSize, 0, stream>>>(d_xmachine_message_tissue_damage_report_local_bin_index, d_xmachine_message_tissue_damage_report_unsorted_index, d_tissue_damage_report_partition_matrix->start, d_tissue_damage_reports, d_tissue_damage_reports_swap, h_message_tissue_damage_report_count);
	  gpuErrchkLaunch();
#else
	  //HASH, SORT, REORDER AND BUILD PMB FOR SPATIAL PARTITIONING MESSAGE OUTPUTS
	  //Get message hash values for sorting
	  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, hash_tissue_damage_report_messages, no_sm, h_message_tissue_damage_report_count); 
	  gridSize = (h_message_tissue_damage_report_count + blockSize - 1) / blockSize;
	  hash_tissue_damage_report_messages<<<gridSize, blockSize, 0, stream>>>(d_xmachine_message_tissue_damage_report_keys, d_xmachine_message_tissue_damage_report_values, d_tissue_damage_reports);
	  gpuErrchkLaunch();
	  //Sort
	  thrust::sort_by_key(thrust::cuda::par.on(stream), thrust::device_pointer_cast(d_xmachine_message_tissue_damage_report_keys),  thrust::device_pointer_cast(d_xmachine_message_tissue_damage_report_keys) + h_message_tissue_damage_report_count,  thrust::device_pointer_cast(d_xmachine_message_tissue_damage_report_values));
	  gpuErrchkLaunch();
	  //reorder and build pcb
	  gpuErrchk(cudaMemset(d_tissue_damage_report_partition_matrix->start, 0xffffffff, xmachine_message_tissue_damage_report_grid_size* sizeof(int)));
	  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_tissue_damage_report_messages, reorder_messages_sm_size, h_message_tissue_damage_report_count); 
	  gridSize = (h_message_tissue_damage_report_count + blockSize - 1) / blockSize;
	  int reorder_sm_size = reorder_messages_sm_size(blockSize);
	  reorder_tissue_damage_report_messages<<<gridSize, blockSize, reorder_sm_size, stream>>>(d_xmachine_message_tissue_damage_report_keys, d_xmachine_message_tissue_damage_report_values, d_tissue_damage_report_partition_matrix, d_tissue_damage_reports, d_tissue_damage_reports_swap);
	  gpuErrchkLaunch();
#endif
  }
	//swap ordered list
	xmachine_message_tissue_damage_report_list* d_tissue_damage_reports_temp = d_tissue_damage_reports;
	d_tissue_damage_reports = d_tissue_damage_reports_swap;
	d_tissue_damage_reports_swap = d_tissue_damage_reports_temp;
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_TissueBlock_default_count+h_xmachine_memory_TissueBlock_count > xmachine_memory_TissueBlock_MAX){
		printf("Error: Buffer size of TissueTakesDamage agents in state default will be exceeded moving working agents to next state in function TissueTakesDamage\n");
      exit(EXIT_FAILURE);
      }
      
  //pointer swap the updated data
  TissueBlocks_default_temp = d_TissueBlocks;
  d_TissueBlocks = d_TissueBlocks_default;
  d_TissueBlocks_default = TissueBlocks_default_temp;
        
	//update new state agent size
	h_xmachine_memory_TissueBlock_default_count += h_xmachine_memory_TissueBlock_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_TissueBlock_default_count, &h_xmachine_memory_TissueBlock_default_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int TissueBlock_RepairDamage_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input is spatially partitioned
	sm_size += (blockSize * sizeof(xmachine_message_fibroblast_report));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** TissueBlock_RepairDamage
 * Agent function prototype for RepairDamage function of TissueBlock agent
 */
void TissueBlock_RepairDamage(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_TissueBlock_default_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_TissueBlock_default_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_TissueBlock_list* TissueBlocks_default_temp = d_TissueBlocks;
	d_TissueBlocks = d_TissueBlocks_default;
	d_TissueBlocks_default = TissueBlocks_default_temp;
	//set working count to current state count
	h_xmachine_memory_TissueBlock_count = h_xmachine_memory_TissueBlock_default_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_TissueBlock_count, &h_xmachine_memory_TissueBlock_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_TissueBlock_default_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_TissueBlock_default_count, &h_xmachine_memory_TissueBlock_default_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_RepairDamage, TissueBlock_RepairDamage_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = TissueBlock_RepairDamage_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	//any agent with discrete or partitioned message input uses texture caching
	size_t tex_xmachine_message_fibroblast_report_id_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_fibroblast_report_id_byte_offset, tex_xmachine_message_fibroblast_report_id, d_fibroblast_reports->id, sizeof(int)*xmachine_message_fibroblast_report_MAX));
	h_tex_xmachine_message_fibroblast_report_id_offset = (int)tex_xmachine_message_fibroblast_report_id_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_fibroblast_report_id_offset, &h_tex_xmachine_message_fibroblast_report_id_offset, sizeof(int)));
	size_t tex_xmachine_message_fibroblast_report_x_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_fibroblast_report_x_byte_offset, tex_xmachine_message_fibroblast_report_x, d_fibroblast_reports->x, sizeof(float)*xmachine_message_fibroblast_report_MAX));
	h_tex_xmachine_message_fibroblast_report_x_offset = (int)tex_xmachine_message_fibroblast_report_x_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_fibroblast_report_x_offset, &h_tex_xmachine_message_fibroblast_report_x_offset, sizeof(int)));
	size_t tex_xmachine_message_fibroblast_report_y_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_fibroblast_report_y_byte_offset, tex_xmachine_message_fibroblast_report_y, d_fibroblast_reports->y, sizeof(float)*xmachine_message_fibroblast_report_MAX));
	h_tex_xmachine_message_fibroblast_report_y_offset = (int)tex_xmachine_message_fibroblast_report_y_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_fibroblast_report_y_offset, &h_tex_xmachine_message_fibroblast_report_y_offset, sizeof(int)));
	size_t tex_xmachine_message_fibroblast_report_z_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_fibroblast_report_z_byte_offset, tex_xmachine_message_fibroblast_report_z, d_fibroblast_reports->z, sizeof(float)*xmachine_message_fibroblast_report_MAX));
	h_tex_xmachine_message_fibroblast_report_z_offset = (int)tex_xmachine_message_fibroblast_report_z_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_fibroblast_report_z_offset, &h_tex_xmachine_message_fibroblast_report_z_offset, sizeof(int)));
	size_t tex_xmachine_message_fibroblast_report_current_state_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_fibroblast_report_current_state_byte_offset, tex_xmachine_message_fibroblast_report_current_state, d_fibroblast_reports->current_state, sizeof(int)*xmachine_message_fibroblast_report_MAX));
	h_tex_xmachine_message_fibroblast_report_current_state_offset = (int)tex_xmachine_message_fibroblast_report_current_state_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_fibroblast_report_current_state_offset, &h_tex_xmachine_message_fibroblast_report_current_state_offset, sizeof(int)));
	size_t tex_xmachine_message_fibroblast_report_go_to_state_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_fibroblast_report_go_to_state_byte_offset, tex_xmachine_message_fibroblast_report_go_to_state, d_fibroblast_reports->go_to_state, sizeof(int)*xmachine_message_fibroblast_report_MAX));
	h_tex_xmachine_message_fibroblast_report_go_to_state_offset = (int)tex_xmachine_message_fibroblast_report_go_to_state_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_fibroblast_report_go_to_state_offset, &h_tex_xmachine_message_fibroblast_report_go_to_state_offset, sizeof(int)));
	//bind pbm start and end indices to textures
	size_t tex_xmachine_message_fibroblast_report_pbm_start_byte_offset;
	size_t tex_xmachine_message_fibroblast_report_pbm_end_or_count_byte_offset;
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_fibroblast_report_pbm_start_byte_offset, tex_xmachine_message_fibroblast_report_pbm_start, d_fibroblast_report_partition_matrix->start, sizeof(int)*xmachine_message_fibroblast_report_grid_size));
	h_tex_xmachine_message_fibroblast_report_pbm_start_offset = (int)tex_xmachine_message_fibroblast_report_pbm_start_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_fibroblast_report_pbm_start_offset, &h_tex_xmachine_message_fibroblast_report_pbm_start_offset, sizeof(int)));
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_fibroblast_report_pbm_end_or_count_byte_offset, tex_xmachine_message_fibroblast_report_pbm_end_or_count, d_fibroblast_report_partition_matrix->end_or_count, sizeof(int)*xmachine_message_fibroblast_report_grid_size));
  h_tex_xmachine_message_fibroblast_report_pbm_end_or_count_offset = (int)tex_xmachine_message_fibroblast_report_pbm_end_or_count_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_fibroblast_report_pbm_end_or_count_offset, &h_tex_xmachine_message_fibroblast_report_pbm_end_or_count_offset, sizeof(int)));

	
	
	//MAIN XMACHINE FUNCTION CALL (RepairDamage)
	//Reallocate   : false
	//Input        : fibroblast_report
	//Output       : 
	//Agent Output : 
	GPUFLAME_RepairDamage<<<g, b, sm_size, stream>>>(d_TissueBlocks, d_fibroblast_reports, d_fibroblast_report_partition_matrix);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	//any agent with discrete or partitioned message input uses texture caching
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_fibroblast_report_id));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_fibroblast_report_x));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_fibroblast_report_y));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_fibroblast_report_z));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_fibroblast_report_current_state));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_fibroblast_report_go_to_state));
	//unbind pbm indices
    gpuErrchk( cudaUnbindTexture(tex_xmachine_message_fibroblast_report_pbm_start));
    gpuErrchk( cudaUnbindTexture(tex_xmachine_message_fibroblast_report_pbm_end_or_count));
    
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_TissueBlock_default_count+h_xmachine_memory_TissueBlock_count > xmachine_memory_TissueBlock_MAX){
		printf("Error: Buffer size of RepairDamage agents in state default will be exceeded moving working agents to next state in function RepairDamage\n");
      exit(EXIT_FAILURE);
      }
      
  //pointer swap the updated data
  TissueBlocks_default_temp = d_TissueBlocks;
  d_TissueBlocks = d_TissueBlocks_default;
  d_TissueBlocks_default = TissueBlocks_default_temp;
        
	//update new state agent size
	h_xmachine_memory_TissueBlock_default_count += h_xmachine_memory_TissueBlock_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_TissueBlock_default_count, &h_xmachine_memory_TissueBlock_default_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int Fibroblast_QuiescentMigration_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input is spatially partitioned
	sm_size += (blockSize * sizeof(xmachine_message_tissue_damage_report));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** Fibroblast_QuiescentMigration
 * Agent function prototype for QuiescentMigration function of Fibroblast agent
 */
void Fibroblast_QuiescentMigration(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_Fibroblast_Quiescent_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_Fibroblast_Quiescent_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_Fibroblast_list* Fibroblasts_Quiescent_temp = d_Fibroblasts;
	d_Fibroblasts = d_Fibroblasts_Quiescent;
	d_Fibroblasts_Quiescent = Fibroblasts_Quiescent_temp;
	//set working count to current state count
	h_xmachine_memory_Fibroblast_count = h_xmachine_memory_Fibroblast_Quiescent_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Fibroblast_count, &h_xmachine_memory_Fibroblast_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_Fibroblast_Quiescent_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Fibroblast_Quiescent_count, &h_xmachine_memory_Fibroblast_Quiescent_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	//CONTINUOUS AGENT CHECK FUNCTION OUTPUT BUFFERS FOR OUT OF BOUNDS
	if (h_message_fibroblast_report_count + h_xmachine_memory_Fibroblast_count > xmachine_message_fibroblast_report_MAX){
		printf("Error: Buffer size of fibroblast_report message will be exceeded in function QuiescentMigration\n");
		exit(EXIT_FAILURE);
	}
	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_QuiescentMigration, Fibroblast_QuiescentMigration_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = Fibroblast_QuiescentMigration_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	//any agent with discrete or partitioned message input uses texture caching
	size_t tex_xmachine_message_tissue_damage_report_id_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_tissue_damage_report_id_byte_offset, tex_xmachine_message_tissue_damage_report_id, d_tissue_damage_reports->id, sizeof(int)*xmachine_message_tissue_damage_report_MAX));
	h_tex_xmachine_message_tissue_damage_report_id_offset = (int)tex_xmachine_message_tissue_damage_report_id_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_tissue_damage_report_id_offset, &h_tex_xmachine_message_tissue_damage_report_id_offset, sizeof(int)));
	size_t tex_xmachine_message_tissue_damage_report_x_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_tissue_damage_report_x_byte_offset, tex_xmachine_message_tissue_damage_report_x, d_tissue_damage_reports->x, sizeof(float)*xmachine_message_tissue_damage_report_MAX));
	h_tex_xmachine_message_tissue_damage_report_x_offset = (int)tex_xmachine_message_tissue_damage_report_x_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_tissue_damage_report_x_offset, &h_tex_xmachine_message_tissue_damage_report_x_offset, sizeof(int)));
	size_t tex_xmachine_message_tissue_damage_report_y_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_tissue_damage_report_y_byte_offset, tex_xmachine_message_tissue_damage_report_y, d_tissue_damage_reports->y, sizeof(float)*xmachine_message_tissue_damage_report_MAX));
	h_tex_xmachine_message_tissue_damage_report_y_offset = (int)tex_xmachine_message_tissue_damage_report_y_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_tissue_damage_report_y_offset, &h_tex_xmachine_message_tissue_damage_report_y_offset, sizeof(int)));
	size_t tex_xmachine_message_tissue_damage_report_z_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_tissue_damage_report_z_byte_offset, tex_xmachine_message_tissue_damage_report_z, d_tissue_damage_reports->z, sizeof(float)*xmachine_message_tissue_damage_report_MAX));
	h_tex_xmachine_message_tissue_damage_report_z_offset = (int)tex_xmachine_message_tissue_damage_report_z_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_tissue_damage_report_z_offset, &h_tex_xmachine_message_tissue_damage_report_z_offset, sizeof(int)));
	size_t tex_xmachine_message_tissue_damage_report_damage_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_tissue_damage_report_damage_byte_offset, tex_xmachine_message_tissue_damage_report_damage, d_tissue_damage_reports->damage, sizeof(int)*xmachine_message_tissue_damage_report_MAX));
	h_tex_xmachine_message_tissue_damage_report_damage_offset = (int)tex_xmachine_message_tissue_damage_report_damage_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_tissue_damage_report_damage_offset, &h_tex_xmachine_message_tissue_damage_report_damage_offset, sizeof(int)));
	//bind pbm start and end indices to textures
	size_t tex_xmachine_message_tissue_damage_report_pbm_start_byte_offset;
	size_t tex_xmachine_message_tissue_damage_report_pbm_end_or_count_byte_offset;
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_tissue_damage_report_pbm_start_byte_offset, tex_xmachine_message_tissue_damage_report_pbm_start, d_tissue_damage_report_partition_matrix->start, sizeof(int)*xmachine_message_tissue_damage_report_grid_size));
	h_tex_xmachine_message_tissue_damage_report_pbm_start_offset = (int)tex_xmachine_message_tissue_damage_report_pbm_start_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_tissue_damage_report_pbm_start_offset, &h_tex_xmachine_message_tissue_damage_report_pbm_start_offset, sizeof(int)));
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_tissue_damage_report_pbm_end_or_count_byte_offset, tex_xmachine_message_tissue_damage_report_pbm_end_or_count, d_tissue_damage_report_partition_matrix->end_or_count, sizeof(int)*xmachine_message_tissue_damage_report_grid_size));
  h_tex_xmachine_message_tissue_damage_report_pbm_end_or_count_offset = (int)tex_xmachine_message_tissue_damage_report_pbm_end_or_count_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_tissue_damage_report_pbm_end_or_count_offset, &h_tex_xmachine_message_tissue_damage_report_pbm_end_or_count_offset, sizeof(int)));

	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	//Set the message_type for non partitioned, spatially partitioned and On-Graph Partitioned message outputs
	h_message_fibroblast_report_output_type = single_message;
	gpuErrchk( cudaMemcpyToSymbol( d_message_fibroblast_report_output_type, &h_message_fibroblast_report_output_type, sizeof(int)));
	
	
	//MAIN XMACHINE FUNCTION CALL (QuiescentMigration)
	//Reallocate   : false
	//Input        : tissue_damage_report
	//Output       : fibroblast_report
	//Agent Output : 
	GPUFLAME_QuiescentMigration<<<g, b, sm_size, stream>>>(d_Fibroblasts, d_tissue_damage_reports, d_tissue_damage_report_partition_matrix, d_fibroblast_reports);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	//any agent with discrete or partitioned message input uses texture caching
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_tissue_damage_report_id));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_tissue_damage_report_x));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_tissue_damage_report_y));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_tissue_damage_report_z));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_tissue_damage_report_damage));
	//unbind pbm indices
    gpuErrchk( cudaUnbindTexture(tex_xmachine_message_tissue_damage_report_pbm_start));
    gpuErrchk( cudaUnbindTexture(tex_xmachine_message_tissue_damage_report_pbm_end_or_count));
    
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	h_message_fibroblast_report_count += h_xmachine_memory_Fibroblast_count;
	//Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_message_fibroblast_report_count, &h_message_fibroblast_report_count, sizeof(int)));	
	
	//reset partition matrix
	gpuErrchk( cudaMemset( (void*) d_fibroblast_report_partition_matrix, 0, sizeof(xmachine_message_fibroblast_report_PBM)));
    //PR Bug fix: Second fix. This should prevent future problems when multiple agents write the same message as now the message structure is completely rebuilt after an output.
    if (h_message_fibroblast_report_count > 0){
#ifdef FAST_ATOMIC_SORTING
      //USE ATOMICS TO BUILD PARTITION BOUNDARY
	  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, hist_fibroblast_report_messages, no_sm, h_message_fibroblast_report_count); 
	  gridSize = (h_message_fibroblast_report_count + blockSize - 1) / blockSize;
	  hist_fibroblast_report_messages<<<gridSize, blockSize, 0, stream>>>(d_xmachine_message_fibroblast_report_local_bin_index, d_xmachine_message_fibroblast_report_unsorted_index, d_fibroblast_report_partition_matrix->end_or_count, d_fibroblast_reports, h_message_fibroblast_report_count);
	  gpuErrchkLaunch();
	
      // Scan
      cub::DeviceScan::ExclusiveSum(
          d_temp_scan_storage_xmachine_message_fibroblast_report, 
          temp_scan_bytes_xmachine_message_fibroblast_report, 
          d_fibroblast_report_partition_matrix->end_or_count,
          d_fibroblast_report_partition_matrix->start,
          xmachine_message_fibroblast_report_grid_size, 
          stream
      );
	
	  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_fibroblast_report_messages, no_sm, h_message_fibroblast_report_count); 
	  gridSize = (h_message_fibroblast_report_count + blockSize - 1) / blockSize; 	// Round up according to array size 
	  reorder_fibroblast_report_messages <<<gridSize, blockSize, 0, stream>>>(d_xmachine_message_fibroblast_report_local_bin_index, d_xmachine_message_fibroblast_report_unsorted_index, d_fibroblast_report_partition_matrix->start, d_fibroblast_reports, d_fibroblast_reports_swap, h_message_fibroblast_report_count);
	  gpuErrchkLaunch();
#else
	  //HASH, SORT, REORDER AND BUILD PMB FOR SPATIAL PARTITIONING MESSAGE OUTPUTS
	  //Get message hash values for sorting
	  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, hash_fibroblast_report_messages, no_sm, h_message_fibroblast_report_count); 
	  gridSize = (h_message_fibroblast_report_count + blockSize - 1) / blockSize;
	  hash_fibroblast_report_messages<<<gridSize, blockSize, 0, stream>>>(d_xmachine_message_fibroblast_report_keys, d_xmachine_message_fibroblast_report_values, d_fibroblast_reports);
	  gpuErrchkLaunch();
	  //Sort
	  thrust::sort_by_key(thrust::cuda::par.on(stream), thrust::device_pointer_cast(d_xmachine_message_fibroblast_report_keys),  thrust::device_pointer_cast(d_xmachine_message_fibroblast_report_keys) + h_message_fibroblast_report_count,  thrust::device_pointer_cast(d_xmachine_message_fibroblast_report_values));
	  gpuErrchkLaunch();
	  //reorder and build pcb
	  gpuErrchk(cudaMemset(d_fibroblast_report_partition_matrix->start, 0xffffffff, xmachine_message_fibroblast_report_grid_size* sizeof(int)));
	  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_fibroblast_report_messages, reorder_messages_sm_size, h_message_fibroblast_report_count); 
	  gridSize = (h_message_fibroblast_report_count + blockSize - 1) / blockSize;
	  int reorder_sm_size = reorder_messages_sm_size(blockSize);
	  reorder_fibroblast_report_messages<<<gridSize, blockSize, reorder_sm_size, stream>>>(d_xmachine_message_fibroblast_report_keys, d_xmachine_message_fibroblast_report_values, d_fibroblast_report_partition_matrix, d_fibroblast_reports, d_fibroblast_reports_swap);
	  gpuErrchkLaunch();
#endif
  }
	//swap ordered list
	xmachine_message_fibroblast_report_list* d_fibroblast_reports_temp = d_fibroblast_reports;
	d_fibroblast_reports = d_fibroblast_reports_swap;
	d_fibroblast_reports_swap = d_fibroblast_reports_temp;
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_Fibroblast_Quiescent_count+h_xmachine_memory_Fibroblast_count > xmachine_memory_Fibroblast_MAX){
		printf("Error: Buffer size of QuiescentMigration agents in state Quiescent will be exceeded moving working agents to next state in function QuiescentMigration\n");
      exit(EXIT_FAILURE);
      }
      
  //pointer swap the updated data
  Fibroblasts_Quiescent_temp = d_Fibroblasts;
  d_Fibroblasts = d_Fibroblasts_Quiescent;
  d_Fibroblasts_Quiescent = Fibroblasts_Quiescent_temp;
        
	//update new state agent size
	h_xmachine_memory_Fibroblast_Quiescent_count += h_xmachine_memory_Fibroblast_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Fibroblast_Quiescent_count, &h_xmachine_memory_Fibroblast_Quiescent_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int Fibroblast_TransitionToRepair_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** Fibroblast_TransitionToRepair
 * Agent function prototype for TransitionToRepair function of Fibroblast agent
 */
void Fibroblast_TransitionToRepair(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_Fibroblast_Quiescent_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_Fibroblast_Quiescent_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_Fibroblast_list* Fibroblasts_Quiescent_temp = d_Fibroblasts;
	d_Fibroblasts = d_Fibroblasts_Quiescent;
	d_Fibroblasts_Quiescent = Fibroblasts_Quiescent_temp;
	//set working count to current state count
	h_xmachine_memory_Fibroblast_count = h_xmachine_memory_Fibroblast_Quiescent_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Fibroblast_count, &h_xmachine_memory_Fibroblast_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_Fibroblast_Quiescent_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Fibroblast_Quiescent_count, &h_xmachine_memory_Fibroblast_Quiescent_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_TransitionToRepair, Fibroblast_TransitionToRepair_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = Fibroblast_TransitionToRepair_sm_size(blockSize);
	
	
	
	
	//MAIN XMACHINE FUNCTION CALL (TransitionToRepair)
	//Reallocate   : false
	//Input        : 
	//Output       : 
	//Agent Output : 
	GPUFLAME_TransitionToRepair<<<g, b, sm_size, stream>>>(d_Fibroblasts);
	gpuErrchkLaunch();
	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_Fibroblast_Repair_count+h_xmachine_memory_Fibroblast_count > xmachine_memory_Fibroblast_MAX){
		printf("Error: Buffer size of TransitionToRepair agents in state Repair will be exceeded moving working agents to next state in function TransitionToRepair\n");
      exit(EXIT_FAILURE);
      }
      
  //append agents to next state list
  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_Fibroblast_Agents, no_sm, state_list_size);
  gridSize = (state_list_size + blockSize - 1) / blockSize;
  append_Fibroblast_Agents<<<gridSize, blockSize, 0, stream>>>(d_Fibroblasts_Repair, d_Fibroblasts, h_xmachine_memory_Fibroblast_Repair_count, h_xmachine_memory_Fibroblast_count);
  gpuErrchkLaunch();
        
	//update new state agent size
	h_xmachine_memory_Fibroblast_Repair_count += h_xmachine_memory_Fibroblast_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Fibroblast_Repair_count, &h_xmachine_memory_Fibroblast_Repair_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int Fibroblast_TransitionToQuiescent_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** Fibroblast_TransitionToQuiescent
 * Agent function prototype for TransitionToQuiescent function of Fibroblast agent
 */
void Fibroblast_TransitionToQuiescent(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_Fibroblast_Repair_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_Fibroblast_Repair_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_Fibroblast_list* Fibroblasts_Repair_temp = d_Fibroblasts;
	d_Fibroblasts = d_Fibroblasts_Repair;
	d_Fibroblasts_Repair = Fibroblasts_Repair_temp;
	//set working count to current state count
	h_xmachine_memory_Fibroblast_count = h_xmachine_memory_Fibroblast_Repair_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Fibroblast_count, &h_xmachine_memory_Fibroblast_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_Fibroblast_Repair_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Fibroblast_Repair_count, &h_xmachine_memory_Fibroblast_Repair_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_TransitionToQuiescent, Fibroblast_TransitionToQuiescent_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = Fibroblast_TransitionToQuiescent_sm_size(blockSize);
	
	
	
	
	//MAIN XMACHINE FUNCTION CALL (TransitionToQuiescent)
	//Reallocate   : false
	//Input        : 
	//Output       : 
	//Agent Output : 
	GPUFLAME_TransitionToQuiescent<<<g, b, sm_size, stream>>>(d_Fibroblasts);
	gpuErrchkLaunch();
	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_Fibroblast_Quiescent_count+h_xmachine_memory_Fibroblast_count > xmachine_memory_Fibroblast_MAX){
		printf("Error: Buffer size of TransitionToQuiescent agents in state Quiescent will be exceeded moving working agents to next state in function TransitionToQuiescent\n");
      exit(EXIT_FAILURE);
      }
      
  //append agents to next state list
  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_Fibroblast_Agents, no_sm, state_list_size);
  gridSize = (state_list_size + blockSize - 1) / blockSize;
  append_Fibroblast_Agents<<<gridSize, blockSize, 0, stream>>>(d_Fibroblasts_Quiescent, d_Fibroblasts, h_xmachine_memory_Fibroblast_Quiescent_count, h_xmachine_memory_Fibroblast_count);
  gpuErrchkLaunch();
        
	//update new state agent size
	h_xmachine_memory_Fibroblast_Quiescent_count += h_xmachine_memory_Fibroblast_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Fibroblast_Quiescent_count, &h_xmachine_memory_Fibroblast_Quiescent_count, sizeof(int)));	
	
	
}


 
extern void reset_TissueBlock_default_count()
{
    h_xmachine_memory_TissueBlock_default_count = 0;
}
 
extern void reset_Fibroblast_Quiescent_count()
{
    h_xmachine_memory_Fibroblast_Quiescent_count = 0;
}
 
extern void reset_Fibroblast_Repair_count()
{
    h_xmachine_memory_Fibroblast_Repair_count = 0;
}
