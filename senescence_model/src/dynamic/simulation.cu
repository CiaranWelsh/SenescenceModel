
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
xmachine_memory_Fibroblast_list* h_Fibroblasts_EarlySenescent;      /**< Pointer to agent list (population) on host*/
xmachine_memory_Fibroblast_list* d_Fibroblasts_EarlySenescent;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_Fibroblast_EarlySenescent_count;   /**< Agent population size counter */ 

/* Fibroblast state variables */
xmachine_memory_Fibroblast_list* h_Fibroblasts_Senescent;      /**< Pointer to agent list (population) on host*/
xmachine_memory_Fibroblast_list* d_Fibroblasts_Senescent;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_Fibroblast_Senescent_count;   /**< Agent population size counter */ 

/* Fibroblast state variables */
xmachine_memory_Fibroblast_list* h_Fibroblasts_Proliferating;      /**< Pointer to agent list (population) on host*/
xmachine_memory_Fibroblast_list* d_Fibroblasts_Proliferating;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_Fibroblast_Proliferating_count;   /**< Agent population size counter */ 

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
unsigned int h_Fibroblasts_Quiescent_variable_doublings_data_iteration;
unsigned int h_Fibroblasts_Quiescent_variable_damage_data_iteration;
unsigned int h_Fibroblasts_Quiescent_variable_early_sen_time_counter_data_iteration;
unsigned int h_Fibroblasts_Quiescent_variable_current_state_data_iteration;
unsigned int h_Fibroblasts_EarlySenescent_variable_id_data_iteration;
unsigned int h_Fibroblasts_EarlySenescent_variable_x_data_iteration;
unsigned int h_Fibroblasts_EarlySenescent_variable_y_data_iteration;
unsigned int h_Fibroblasts_EarlySenescent_variable_z_data_iteration;
unsigned int h_Fibroblasts_EarlySenescent_variable_doublings_data_iteration;
unsigned int h_Fibroblasts_EarlySenescent_variable_damage_data_iteration;
unsigned int h_Fibroblasts_EarlySenescent_variable_early_sen_time_counter_data_iteration;
unsigned int h_Fibroblasts_EarlySenescent_variable_current_state_data_iteration;
unsigned int h_Fibroblasts_Senescent_variable_id_data_iteration;
unsigned int h_Fibroblasts_Senescent_variable_x_data_iteration;
unsigned int h_Fibroblasts_Senescent_variable_y_data_iteration;
unsigned int h_Fibroblasts_Senescent_variable_z_data_iteration;
unsigned int h_Fibroblasts_Senescent_variable_doublings_data_iteration;
unsigned int h_Fibroblasts_Senescent_variable_damage_data_iteration;
unsigned int h_Fibroblasts_Senescent_variable_early_sen_time_counter_data_iteration;
unsigned int h_Fibroblasts_Senescent_variable_current_state_data_iteration;
unsigned int h_Fibroblasts_Proliferating_variable_id_data_iteration;
unsigned int h_Fibroblasts_Proliferating_variable_x_data_iteration;
unsigned int h_Fibroblasts_Proliferating_variable_y_data_iteration;
unsigned int h_Fibroblasts_Proliferating_variable_z_data_iteration;
unsigned int h_Fibroblasts_Proliferating_variable_doublings_data_iteration;
unsigned int h_Fibroblasts_Proliferating_variable_damage_data_iteration;
unsigned int h_Fibroblasts_Proliferating_variable_early_sen_time_counter_data_iteration;
unsigned int h_Fibroblasts_Proliferating_variable_current_state_data_iteration;
unsigned int h_Fibroblasts_Repair_variable_id_data_iteration;
unsigned int h_Fibroblasts_Repair_variable_x_data_iteration;
unsigned int h_Fibroblasts_Repair_variable_y_data_iteration;
unsigned int h_Fibroblasts_Repair_variable_z_data_iteration;
unsigned int h_Fibroblasts_Repair_variable_doublings_data_iteration;
unsigned int h_Fibroblasts_Repair_variable_damage_data_iteration;
unsigned int h_Fibroblasts_Repair_variable_early_sen_time_counter_data_iteration;
unsigned int h_Fibroblasts_Repair_variable_current_state_data_iteration;


/* Message Memory */

/* fibroblast_damage_report Message variables */
xmachine_message_fibroblast_damage_report_list* h_fibroblast_damage_reports;         /**< Pointer to message list on host*/
xmachine_message_fibroblast_damage_report_list* d_fibroblast_damage_reports;         /**< Pointer to message list on device*/
xmachine_message_fibroblast_damage_report_list* d_fibroblast_damage_reports_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Non partitioned and spatial partitioned message variables  */
int h_message_fibroblast_damage_report_count;         /**< message list counter*/
int h_message_fibroblast_damage_report_output_type;   /**< message output type (single or optional)*/
/* Spatial Partitioning Variables*/
#ifdef FAST_ATOMIC_SORTING
	uint * d_xmachine_message_fibroblast_damage_report_local_bin_index;	  /**< index offset within the assigned bin */
	uint * d_xmachine_message_fibroblast_damage_report_unsorted_index;		/**< unsorted index (hash) value for message */
    // Values for CUB exclusive scan of spatially partitioned variables
    void * d_temp_scan_storage_xmachine_message_fibroblast_damage_report;
    size_t temp_scan_bytes_xmachine_message_fibroblast_damage_report;
#else
	uint * d_xmachine_message_fibroblast_damage_report_keys;	  /**< message sort identifier keys*/
	uint * d_xmachine_message_fibroblast_damage_report_values;  /**< message sort identifier values */
#endif
xmachine_message_fibroblast_damage_report_PBM * d_fibroblast_damage_report_partition_matrix;  /**< Pointer to PCB matrix */
glm::vec3 h_message_fibroblast_damage_report_min_bounds;           /**< min bounds (x,y,z) of partitioning environment */
glm::vec3 h_message_fibroblast_damage_report_max_bounds;           /**< max bounds (x,y,z) of partitioning environment */
glm::ivec3 h_message_fibroblast_damage_report_partitionDim;           /**< partition dimensions (x,y,z) of partitioning environment */
float h_message_fibroblast_damage_report_radius;                 /**< partition radius (used to determin the size of the partitions) */
/* Texture offset values for host */
int h_tex_xmachine_message_fibroblast_damage_report_id_offset;
int h_tex_xmachine_message_fibroblast_damage_report_x_offset;
int h_tex_xmachine_message_fibroblast_damage_report_y_offset;
int h_tex_xmachine_message_fibroblast_damage_report_z_offset;
int h_tex_xmachine_message_fibroblast_damage_report_damage_offset;
int h_tex_xmachine_message_fibroblast_damage_report_pbm_start_offset;
int h_tex_xmachine_message_fibroblast_damage_report_pbm_end_or_count_offset;

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

/* doublings Message variables */
xmachine_message_doublings_list* h_doublingss;         /**< Pointer to message list on host*/
xmachine_message_doublings_list* d_doublingss;         /**< Pointer to message list on device*/
xmachine_message_doublings_list* d_doublingss_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Non partitioned and spatial partitioned message variables  */
int h_message_doublings_count;         /**< message list counter*/
int h_message_doublings_output_type;   /**< message output type (single or optional)*/

/* count Message variables */
xmachine_message_count_list* h_counts;         /**< Pointer to message list on host*/
xmachine_message_count_list* d_counts;         /**< Pointer to message list on device*/
xmachine_message_count_list* d_counts_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Non partitioned and spatial partitioned message variables  */
int h_message_count_count;         /**< message list counter*/
int h_message_count_output_type;   /**< message output type (single or optional)*/

/* fibroblast_location_report Message variables */
xmachine_message_fibroblast_location_report_list* h_fibroblast_location_reports;         /**< Pointer to message list on host*/
xmachine_message_fibroblast_location_report_list* d_fibroblast_location_reports;         /**< Pointer to message list on device*/
xmachine_message_fibroblast_location_report_list* d_fibroblast_location_reports_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Non partitioned and spatial partitioned message variables  */
int h_message_fibroblast_location_report_count;         /**< message list counter*/
int h_message_fibroblast_location_report_output_type;   /**< message output type (single or optional)*/
/* Spatial Partitioning Variables*/
#ifdef FAST_ATOMIC_SORTING
	uint * d_xmachine_message_fibroblast_location_report_local_bin_index;	  /**< index offset within the assigned bin */
	uint * d_xmachine_message_fibroblast_location_report_unsorted_index;		/**< unsorted index (hash) value for message */
    // Values for CUB exclusive scan of spatially partitioned variables
    void * d_temp_scan_storage_xmachine_message_fibroblast_location_report;
    size_t temp_scan_bytes_xmachine_message_fibroblast_location_report;
#else
	uint * d_xmachine_message_fibroblast_location_report_keys;	  /**< message sort identifier keys*/
	uint * d_xmachine_message_fibroblast_location_report_values;  /**< message sort identifier values */
#endif
xmachine_message_fibroblast_location_report_PBM * d_fibroblast_location_report_partition_matrix;  /**< Pointer to PCB matrix */
glm::vec3 h_message_fibroblast_location_report_min_bounds;           /**< min bounds (x,y,z) of partitioning environment */
glm::vec3 h_message_fibroblast_location_report_max_bounds;           /**< max bounds (x,y,z) of partitioning environment */
glm::ivec3 h_message_fibroblast_location_report_partitionDim;           /**< partition dimensions (x,y,z) of partitioning environment */
float h_message_fibroblast_location_report_radius;                 /**< partition radius (used to determin the size of the partitions) */
/* Texture offset values for host */
int h_tex_xmachine_message_fibroblast_location_report_id_offset;
int h_tex_xmachine_message_fibroblast_location_report_x_offset;
int h_tex_xmachine_message_fibroblast_location_report_y_offset;
int h_tex_xmachine_message_fibroblast_location_report_z_offset;
int h_tex_xmachine_message_fibroblast_location_report_current_state_offset;
int h_tex_xmachine_message_fibroblast_location_report_pbm_start_offset;
int h_tex_xmachine_message_fibroblast_location_report_pbm_end_or_count_offset;

  
/* CUDA Streams for function layers */
cudaStream_t stream1;
cudaStream_t stream2;
cudaStream_t stream3;

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

/** TissueBlock_TissueSendDamageReport
 * Agent function prototype for TissueSendDamageReport function of TissueBlock agent
 */
void TissueBlock_TissueSendDamageReport(cudaStream_t &stream);

/** TissueBlock_ReapirDamage
 * Agent function prototype for ReapirDamage function of TissueBlock agent
 */
void TissueBlock_ReapirDamage(cudaStream_t &stream);

/** Fibroblast_QuiescentMigration
 * Agent function prototype for QuiescentMigration function of Fibroblast agent
 */
void Fibroblast_QuiescentMigration(cudaStream_t &stream);

/** Fibroblast_SenescentMigration
 * Agent function prototype for SenescentMigration function of Fibroblast agent
 */
void Fibroblast_SenescentMigration(cudaStream_t &stream);

/** Fibroblast_EarlySenescentMigration
 * Agent function prototype for EarlySenescentMigration function of Fibroblast agent
 */
void Fibroblast_EarlySenescentMigration(cudaStream_t &stream);

/** Fibroblast_QuiescentTakesDamage
 * Agent function prototype for QuiescentTakesDamage function of Fibroblast agent
 */
void Fibroblast_QuiescentTakesDamage(cudaStream_t &stream);

/** Fibroblast_QuiescentSendDamageReport
 * Agent function prototype for QuiescentSendDamageReport function of Fibroblast agent
 */
void Fibroblast_QuiescentSendDamageReport(cudaStream_t &stream);

/** Fibroblast_TransitionToProliferating
 * Agent function prototype for TransitionToProliferating function of Fibroblast agent
 */
void Fibroblast_TransitionToProliferating(cudaStream_t &stream);

/** Fibroblast_Proliferation
 * Agent function prototype for Proliferation function of Fibroblast agent
 */
void Fibroblast_Proliferation(cudaStream_t &stream);

/** Fibroblast_BystanderEffect
 * Agent function prototype for BystanderEffect function of Fibroblast agent
 */
void Fibroblast_BystanderEffect(cudaStream_t &stream);

/** Fibroblast_ExcessiveDamage
 * Agent function prototype for ExcessiveDamage function of Fibroblast agent
 */
void Fibroblast_ExcessiveDamage(cudaStream_t &stream);

/** Fibroblast_ReplicativeSenescence
 * Agent function prototype for ReplicativeSenescence function of Fibroblast agent
 */
void Fibroblast_ReplicativeSenescence(cudaStream_t &stream);

/** Fibroblast_EarlySenCountTime
 * Agent function prototype for EarlySenCountTime function of Fibroblast agent
 */
void Fibroblast_EarlySenCountTime(cudaStream_t &stream);

/** Fibroblast_TransitionToFullSenescence
 * Agent function prototype for TransitionToFullSenescence function of Fibroblast agent
 */
void Fibroblast_TransitionToFullSenescence(cudaStream_t &stream);

/** Fibroblast_ClearanceOfEarlySenescent
 * Agent function prototype for ClearanceOfEarlySenescent function of Fibroblast agent
 */
void Fibroblast_ClearanceOfEarlySenescent(cudaStream_t &stream);

/** Fibroblast_ClearanceOfSenescent
 * Agent function prototype for ClearanceOfSenescent function of Fibroblast agent
 */
void Fibroblast_ClearanceOfSenescent(cudaStream_t &stream);

/** Fibroblast_DetectDamage
 * Agent function prototype for DetectDamage function of Fibroblast agent
 */
void Fibroblast_DetectDamage(cudaStream_t &stream);

  
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
    h_Fibroblasts_Quiescent_variable_doublings_data_iteration = 0;
    h_Fibroblasts_Quiescent_variable_damage_data_iteration = 0;
    h_Fibroblasts_Quiescent_variable_early_sen_time_counter_data_iteration = 0;
    h_Fibroblasts_Quiescent_variable_current_state_data_iteration = 0;
    h_Fibroblasts_EarlySenescent_variable_id_data_iteration = 0;
    h_Fibroblasts_EarlySenescent_variable_x_data_iteration = 0;
    h_Fibroblasts_EarlySenescent_variable_y_data_iteration = 0;
    h_Fibroblasts_EarlySenescent_variable_z_data_iteration = 0;
    h_Fibroblasts_EarlySenescent_variable_doublings_data_iteration = 0;
    h_Fibroblasts_EarlySenescent_variable_damage_data_iteration = 0;
    h_Fibroblasts_EarlySenescent_variable_early_sen_time_counter_data_iteration = 0;
    h_Fibroblasts_EarlySenescent_variable_current_state_data_iteration = 0;
    h_Fibroblasts_Senescent_variable_id_data_iteration = 0;
    h_Fibroblasts_Senescent_variable_x_data_iteration = 0;
    h_Fibroblasts_Senescent_variable_y_data_iteration = 0;
    h_Fibroblasts_Senescent_variable_z_data_iteration = 0;
    h_Fibroblasts_Senescent_variable_doublings_data_iteration = 0;
    h_Fibroblasts_Senescent_variable_damage_data_iteration = 0;
    h_Fibroblasts_Senescent_variable_early_sen_time_counter_data_iteration = 0;
    h_Fibroblasts_Senescent_variable_current_state_data_iteration = 0;
    h_Fibroblasts_Proliferating_variable_id_data_iteration = 0;
    h_Fibroblasts_Proliferating_variable_x_data_iteration = 0;
    h_Fibroblasts_Proliferating_variable_y_data_iteration = 0;
    h_Fibroblasts_Proliferating_variable_z_data_iteration = 0;
    h_Fibroblasts_Proliferating_variable_doublings_data_iteration = 0;
    h_Fibroblasts_Proliferating_variable_damage_data_iteration = 0;
    h_Fibroblasts_Proliferating_variable_early_sen_time_counter_data_iteration = 0;
    h_Fibroblasts_Proliferating_variable_current_state_data_iteration = 0;
    h_Fibroblasts_Repair_variable_id_data_iteration = 0;
    h_Fibroblasts_Repair_variable_x_data_iteration = 0;
    h_Fibroblasts_Repair_variable_y_data_iteration = 0;
    h_Fibroblasts_Repair_variable_z_data_iteration = 0;
    h_Fibroblasts_Repair_variable_doublings_data_iteration = 0;
    h_Fibroblasts_Repair_variable_damage_data_iteration = 0;
    h_Fibroblasts_Repair_variable_early_sen_time_counter_data_iteration = 0;
    h_Fibroblasts_Repair_variable_current_state_data_iteration = 0;
    



	printf("Allocating Host and Device memory\n");
    PROFILE_PUSH_RANGE("allocate host");
	/* Agent memory allocation (CPU) */
	int xmachine_TissueBlock_SoA_size = sizeof(xmachine_memory_TissueBlock_list);
	h_TissueBlocks_default = (xmachine_memory_TissueBlock_list*)malloc(xmachine_TissueBlock_SoA_size);
	int xmachine_Fibroblast_SoA_size = sizeof(xmachine_memory_Fibroblast_list);
	h_Fibroblasts_Quiescent = (xmachine_memory_Fibroblast_list*)malloc(xmachine_Fibroblast_SoA_size);
	h_Fibroblasts_EarlySenescent = (xmachine_memory_Fibroblast_list*)malloc(xmachine_Fibroblast_SoA_size);
	h_Fibroblasts_Senescent = (xmachine_memory_Fibroblast_list*)malloc(xmachine_Fibroblast_SoA_size);
	h_Fibroblasts_Proliferating = (xmachine_memory_Fibroblast_list*)malloc(xmachine_Fibroblast_SoA_size);
	h_Fibroblasts_Repair = (xmachine_memory_Fibroblast_list*)malloc(xmachine_Fibroblast_SoA_size);

	/* Message memory allocation (CPU) */
	int message_fibroblast_damage_report_SoA_size = sizeof(xmachine_message_fibroblast_damage_report_list);
	h_fibroblast_damage_reports = (xmachine_message_fibroblast_damage_report_list*)malloc(message_fibroblast_damage_report_SoA_size);
	int message_tissue_damage_report_SoA_size = sizeof(xmachine_message_tissue_damage_report_list);
	h_tissue_damage_reports = (xmachine_message_tissue_damage_report_list*)malloc(message_tissue_damage_report_SoA_size);
	int message_doublings_SoA_size = sizeof(xmachine_message_doublings_list);
	h_doublingss = (xmachine_message_doublings_list*)malloc(message_doublings_SoA_size);
	int message_count_SoA_size = sizeof(xmachine_message_count_list);
	h_counts = (xmachine_message_count_list*)malloc(message_count_SoA_size);
	int message_fibroblast_location_report_SoA_size = sizeof(xmachine_message_fibroblast_location_report_list);
	h_fibroblast_location_reports = (xmachine_message_fibroblast_location_report_list*)malloc(message_fibroblast_location_report_SoA_size);

	//Exit if agent or message buffer sizes are to small for function outputs

  /* Graph memory allocation (CPU) */
  

    PROFILE_POP_RANGE(); //"allocate host"
	
			
	/* Set spatial partitioning fibroblast_damage_report message variables (min_bounds, max_bounds)*/
	h_message_fibroblast_damage_report_radius = (float)1;
	gpuErrchk(cudaMemcpyToSymbol( d_message_fibroblast_damage_report_radius, &h_message_fibroblast_damage_report_radius, sizeof(float)));	
	    h_message_fibroblast_damage_report_min_bounds = glm::vec3((float)0.0, (float)0.0, (float)0.0);
	gpuErrchk(cudaMemcpyToSymbol( d_message_fibroblast_damage_report_min_bounds, &h_message_fibroblast_damage_report_min_bounds, sizeof(glm::vec3)));	
	h_message_fibroblast_damage_report_max_bounds = glm::vec3((float)10, (float)10, (float)10);
	gpuErrchk(cudaMemcpyToSymbol( d_message_fibroblast_damage_report_max_bounds, &h_message_fibroblast_damage_report_max_bounds, sizeof(glm::vec3)));	
	h_message_fibroblast_damage_report_partitionDim.x = (int)ceil((h_message_fibroblast_damage_report_max_bounds.x - h_message_fibroblast_damage_report_min_bounds.x)/h_message_fibroblast_damage_report_radius);
	h_message_fibroblast_damage_report_partitionDim.y = (int)ceil((h_message_fibroblast_damage_report_max_bounds.y - h_message_fibroblast_damage_report_min_bounds.y)/h_message_fibroblast_damage_report_radius);
	h_message_fibroblast_damage_report_partitionDim.z = (int)ceil((h_message_fibroblast_damage_report_max_bounds.z - h_message_fibroblast_damage_report_min_bounds.z)/h_message_fibroblast_damage_report_radius);
	gpuErrchk(cudaMemcpyToSymbol( d_message_fibroblast_damage_report_partitionDim, &h_message_fibroblast_damage_report_partitionDim, sizeof(glm::ivec3)));	
	
			
	/* Set spatial partitioning tissue_damage_report message variables (min_bounds, max_bounds)*/
	h_message_tissue_damage_report_radius = (float)1;
	gpuErrchk(cudaMemcpyToSymbol( d_message_tissue_damage_report_radius, &h_message_tissue_damage_report_radius, sizeof(float)));	
	    h_message_tissue_damage_report_min_bounds = glm::vec3((float)0.0, (float)0.0, (float)0.0);
	gpuErrchk(cudaMemcpyToSymbol( d_message_tissue_damage_report_min_bounds, &h_message_tissue_damage_report_min_bounds, sizeof(glm::vec3)));	
	h_message_tissue_damage_report_max_bounds = glm::vec3((float)10, (float)10, (float)10);
	gpuErrchk(cudaMemcpyToSymbol( d_message_tissue_damage_report_max_bounds, &h_message_tissue_damage_report_max_bounds, sizeof(glm::vec3)));	
	h_message_tissue_damage_report_partitionDim.x = (int)ceil((h_message_tissue_damage_report_max_bounds.x - h_message_tissue_damage_report_min_bounds.x)/h_message_tissue_damage_report_radius);
	h_message_tissue_damage_report_partitionDim.y = (int)ceil((h_message_tissue_damage_report_max_bounds.y - h_message_tissue_damage_report_min_bounds.y)/h_message_tissue_damage_report_radius);
	h_message_tissue_damage_report_partitionDim.z = (int)ceil((h_message_tissue_damage_report_max_bounds.z - h_message_tissue_damage_report_min_bounds.z)/h_message_tissue_damage_report_radius);
	gpuErrchk(cudaMemcpyToSymbol( d_message_tissue_damage_report_partitionDim, &h_message_tissue_damage_report_partitionDim, sizeof(glm::ivec3)));	
	
			
	/* Set spatial partitioning fibroblast_location_report message variables (min_bounds, max_bounds)*/
	h_message_fibroblast_location_report_radius = (float)1;
	gpuErrchk(cudaMemcpyToSymbol( d_message_fibroblast_location_report_radius, &h_message_fibroblast_location_report_radius, sizeof(float)));	
	    h_message_fibroblast_location_report_min_bounds = glm::vec3((float)0.0, (float)0.0, (float)0.0);
	gpuErrchk(cudaMemcpyToSymbol( d_message_fibroblast_location_report_min_bounds, &h_message_fibroblast_location_report_min_bounds, sizeof(glm::vec3)));	
	h_message_fibroblast_location_report_max_bounds = glm::vec3((float)10, (float)10, (float)10);
	gpuErrchk(cudaMemcpyToSymbol( d_message_fibroblast_location_report_max_bounds, &h_message_fibroblast_location_report_max_bounds, sizeof(glm::vec3)));	
	h_message_fibroblast_location_report_partitionDim.x = (int)ceil((h_message_fibroblast_location_report_max_bounds.x - h_message_fibroblast_location_report_min_bounds.x)/h_message_fibroblast_location_report_radius);
	h_message_fibroblast_location_report_partitionDim.y = (int)ceil((h_message_fibroblast_location_report_max_bounds.y - h_message_fibroblast_location_report_min_bounds.y)/h_message_fibroblast_location_report_radius);
	h_message_fibroblast_location_report_partitionDim.z = (int)ceil((h_message_fibroblast_location_report_max_bounds.z - h_message_fibroblast_location_report_min_bounds.z)/h_message_fibroblast_location_report_radius);
	gpuErrchk(cudaMemcpyToSymbol( d_message_fibroblast_location_report_partitionDim, &h_message_fibroblast_location_report_partitionDim, sizeof(glm::ivec3)));	
	

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
    
	/* EarlySenescent memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_Fibroblasts_EarlySenescent, xmachine_Fibroblast_SoA_size));
	gpuErrchk( cudaMemcpy( d_Fibroblasts_EarlySenescent, h_Fibroblasts_EarlySenescent, xmachine_Fibroblast_SoA_size, cudaMemcpyHostToDevice));
    
	/* Senescent memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_Fibroblasts_Senescent, xmachine_Fibroblast_SoA_size));
	gpuErrchk( cudaMemcpy( d_Fibroblasts_Senescent, h_Fibroblasts_Senescent, xmachine_Fibroblast_SoA_size, cudaMemcpyHostToDevice));
    
	/* Proliferating memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_Fibroblasts_Proliferating, xmachine_Fibroblast_SoA_size));
	gpuErrchk( cudaMemcpy( d_Fibroblasts_Proliferating, h_Fibroblasts_Proliferating, xmachine_Fibroblast_SoA_size, cudaMemcpyHostToDevice));
    
	/* Repair memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_Fibroblasts_Repair, xmachine_Fibroblast_SoA_size));
	gpuErrchk( cudaMemcpy( d_Fibroblasts_Repair, h_Fibroblasts_Repair, xmachine_Fibroblast_SoA_size, cudaMemcpyHostToDevice));
    
	/* fibroblast_damage_report Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_fibroblast_damage_reports, message_fibroblast_damage_report_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_fibroblast_damage_reports_swap, message_fibroblast_damage_report_SoA_size));
	gpuErrchk( cudaMemcpy( d_fibroblast_damage_reports, h_fibroblast_damage_reports, message_fibroblast_damage_report_SoA_size, cudaMemcpyHostToDevice));
	gpuErrchk( cudaMalloc( (void**) &d_fibroblast_damage_report_partition_matrix, sizeof(xmachine_message_fibroblast_damage_report_PBM)));
#ifdef FAST_ATOMIC_SORTING
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_message_fibroblast_damage_report_local_bin_index, xmachine_message_fibroblast_damage_report_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_message_fibroblast_damage_report_unsorted_index, xmachine_message_fibroblast_damage_report_MAX* sizeof(uint)));
    /* Calculate and allocate CUB temporary memory for exclusive scans */
    d_temp_scan_storage_xmachine_message_fibroblast_damage_report = nullptr;
    temp_scan_bytes_xmachine_message_fibroblast_damage_report = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_xmachine_message_fibroblast_damage_report, 
        temp_scan_bytes_xmachine_message_fibroblast_damage_report, 
        (int*) nullptr, 
        (int*) nullptr, 
        xmachine_message_fibroblast_damage_report_grid_size
    );
    gpuErrchk(cudaMalloc(&d_temp_scan_storage_xmachine_message_fibroblast_damage_report, temp_scan_bytes_xmachine_message_fibroblast_damage_report));
#else
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_message_fibroblast_damage_report_keys, xmachine_message_fibroblast_damage_report_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_message_fibroblast_damage_report_values, xmachine_message_fibroblast_damage_report_MAX* sizeof(uint)));
#endif
	
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
	
	/* doublings Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_doublingss, message_doublings_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_doublingss_swap, message_doublings_SoA_size));
	gpuErrchk( cudaMemcpy( d_doublingss, h_doublingss, message_doublings_SoA_size, cudaMemcpyHostToDevice));
	
	/* count Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_counts, message_count_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_counts_swap, message_count_SoA_size));
	gpuErrchk( cudaMemcpy( d_counts, h_counts, message_count_SoA_size, cudaMemcpyHostToDevice));
	
	/* fibroblast_location_report Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_fibroblast_location_reports, message_fibroblast_location_report_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_fibroblast_location_reports_swap, message_fibroblast_location_report_SoA_size));
	gpuErrchk( cudaMemcpy( d_fibroblast_location_reports, h_fibroblast_location_reports, message_fibroblast_location_report_SoA_size, cudaMemcpyHostToDevice));
	gpuErrchk( cudaMalloc( (void**) &d_fibroblast_location_report_partition_matrix, sizeof(xmachine_message_fibroblast_location_report_PBM)));
#ifdef FAST_ATOMIC_SORTING
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_message_fibroblast_location_report_local_bin_index, xmachine_message_fibroblast_location_report_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_message_fibroblast_location_report_unsorted_index, xmachine_message_fibroblast_location_report_MAX* sizeof(uint)));
    /* Calculate and allocate CUB temporary memory for exclusive scans */
    d_temp_scan_storage_xmachine_message_fibroblast_location_report = nullptr;
    temp_scan_bytes_xmachine_message_fibroblast_location_report = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_xmachine_message_fibroblast_location_report, 
        temp_scan_bytes_xmachine_message_fibroblast_location_report, 
        (int*) nullptr, 
        (int*) nullptr, 
        xmachine_message_fibroblast_location_report_grid_size
    );
    gpuErrchk(cudaMalloc(&d_temp_scan_storage_xmachine_message_fibroblast_location_report, temp_scan_bytes_xmachine_message_fibroblast_location_report));
#else
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_message_fibroblast_location_report_keys, xmachine_message_fibroblast_location_report_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_message_fibroblast_location_report_values, xmachine_message_fibroblast_location_report_MAX* sizeof(uint)));
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

	
#if defined(INSTRUMENT_INIT_FUNCTIONS) && INSTRUMENT_INIT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
    setConstants();
    PROFILE_PUSH_RANGE("setConstants");
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_INIT_FUNCTIONS) && INSTRUMENT_INIT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: setConstants = %f (ms)\n", instrument_milliseconds);
#endif
	
  
  /* Init CUDA Streams for function layers */
  
  gpuErrchk(cudaStreamCreate(&stream1));
  gpuErrchk(cudaStreamCreate(&stream2));
  gpuErrchk(cudaStreamCreate(&stream3));

#if defined(OUTPUT_POPULATION_PER_ITERATION) && OUTPUT_POPULATION_PER_ITERATION
	// Print the agent population size of all agents in all states
	
		printf("Init agent_TissueBlock_default_count: %u\n",get_agent_TissueBlock_default_count());
	
		printf("Init agent_Fibroblast_Quiescent_count: %u\n",get_agent_Fibroblast_Quiescent_count());
	
		printf("Init agent_Fibroblast_EarlySenescent_count: %u\n",get_agent_Fibroblast_EarlySenescent_count());
	
		printf("Init agent_Fibroblast_Senescent_count: %u\n",get_agent_Fibroblast_Senescent_count());
	
		printf("Init agent_Fibroblast_Proliferating_count: %u\n",get_agent_Fibroblast_Proliferating_count());
	
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

void sort_Fibroblasts_EarlySenescent(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_Fibroblast_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_Fibroblast_EarlySenescent_count); 
	gridSize = (h_xmachine_memory_Fibroblast_EarlySenescent_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_Fibroblast_keys, d_xmachine_memory_Fibroblast_values, d_Fibroblasts_EarlySenescent);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_Fibroblast_keys),  thrust::device_pointer_cast(d_xmachine_memory_Fibroblast_keys) + h_xmachine_memory_Fibroblast_EarlySenescent_count,  thrust::device_pointer_cast(d_xmachine_memory_Fibroblast_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_Fibroblast_agents, no_sm, h_xmachine_memory_Fibroblast_EarlySenescent_count); 
	gridSize = (h_xmachine_memory_Fibroblast_EarlySenescent_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_Fibroblast_agents<<<gridSize, blockSize>>>(d_xmachine_memory_Fibroblast_values, d_Fibroblasts_EarlySenescent, d_Fibroblasts_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_Fibroblast_list* d_Fibroblasts_temp = d_Fibroblasts_EarlySenescent;
	d_Fibroblasts_EarlySenescent = d_Fibroblasts_swap;
	d_Fibroblasts_swap = d_Fibroblasts_temp;	
}

void sort_Fibroblasts_Senescent(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_Fibroblast_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_Fibroblast_Senescent_count); 
	gridSize = (h_xmachine_memory_Fibroblast_Senescent_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_Fibroblast_keys, d_xmachine_memory_Fibroblast_values, d_Fibroblasts_Senescent);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_Fibroblast_keys),  thrust::device_pointer_cast(d_xmachine_memory_Fibroblast_keys) + h_xmachine_memory_Fibroblast_Senescent_count,  thrust::device_pointer_cast(d_xmachine_memory_Fibroblast_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_Fibroblast_agents, no_sm, h_xmachine_memory_Fibroblast_Senescent_count); 
	gridSize = (h_xmachine_memory_Fibroblast_Senescent_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_Fibroblast_agents<<<gridSize, blockSize>>>(d_xmachine_memory_Fibroblast_values, d_Fibroblasts_Senescent, d_Fibroblasts_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_Fibroblast_list* d_Fibroblasts_temp = d_Fibroblasts_Senescent;
	d_Fibroblasts_Senescent = d_Fibroblasts_swap;
	d_Fibroblasts_swap = d_Fibroblasts_temp;	
}

void sort_Fibroblasts_Proliferating(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_Fibroblast_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_Fibroblast_Proliferating_count); 
	gridSize = (h_xmachine_memory_Fibroblast_Proliferating_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_Fibroblast_keys, d_xmachine_memory_Fibroblast_values, d_Fibroblasts_Proliferating);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_Fibroblast_keys),  thrust::device_pointer_cast(d_xmachine_memory_Fibroblast_keys) + h_xmachine_memory_Fibroblast_Proliferating_count,  thrust::device_pointer_cast(d_xmachine_memory_Fibroblast_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_Fibroblast_agents, no_sm, h_xmachine_memory_Fibroblast_Proliferating_count); 
	gridSize = (h_xmachine_memory_Fibroblast_Proliferating_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_Fibroblast_agents<<<gridSize, blockSize>>>(d_xmachine_memory_Fibroblast_values, d_Fibroblasts_Proliferating, d_Fibroblasts_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_Fibroblast_list* d_Fibroblasts_temp = d_Fibroblasts_Proliferating;
	d_Fibroblasts_Proliferating = d_Fibroblasts_swap;
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
	
	free( h_Fibroblasts_EarlySenescent);
	gpuErrchk(cudaFree(d_Fibroblasts_EarlySenescent));
	
	free( h_Fibroblasts_Senescent);
	gpuErrchk(cudaFree(d_Fibroblasts_Senescent));
	
	free( h_Fibroblasts_Proliferating);
	gpuErrchk(cudaFree(d_Fibroblasts_Proliferating));
	
	free( h_Fibroblasts_Repair);
	gpuErrchk(cudaFree(d_Fibroblasts_Repair));
	

	/* Message data free */
	
	/* fibroblast_damage_report Message variables */
	free( h_fibroblast_damage_reports);
	gpuErrchk(cudaFree(d_fibroblast_damage_reports));
	gpuErrchk(cudaFree(d_fibroblast_damage_reports_swap));
	gpuErrchk(cudaFree(d_fibroblast_damage_report_partition_matrix));
#ifdef FAST_ATOMIC_SORTING
	gpuErrchk(cudaFree(d_xmachine_message_fibroblast_damage_report_local_bin_index));
	gpuErrchk(cudaFree(d_xmachine_message_fibroblast_damage_report_unsorted_index));
  gpuErrchk(cudaFree(d_temp_scan_storage_xmachine_message_fibroblast_damage_report));
  d_temp_scan_storage_xmachine_message_fibroblast_damage_report = nullptr;
  temp_scan_bytes_xmachine_message_fibroblast_damage_report = 0;
#else
	gpuErrchk(cudaFree(d_xmachine_message_fibroblast_damage_report_keys));
	gpuErrchk(cudaFree(d_xmachine_message_fibroblast_damage_report_values));
#endif
	
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
	
	/* doublings Message variables */
	free( h_doublingss);
	gpuErrchk(cudaFree(d_doublingss));
	gpuErrchk(cudaFree(d_doublingss_swap));
	
	/* count Message variables */
	free( h_counts);
	gpuErrchk(cudaFree(d_counts));
	gpuErrchk(cudaFree(d_counts_swap));
	
	/* fibroblast_location_report Message variables */
	free( h_fibroblast_location_reports);
	gpuErrchk(cudaFree(d_fibroblast_location_reports));
	gpuErrchk(cudaFree(d_fibroblast_location_reports_swap));
	gpuErrchk(cudaFree(d_fibroblast_location_report_partition_matrix));
#ifdef FAST_ATOMIC_SORTING
	gpuErrchk(cudaFree(d_xmachine_message_fibroblast_location_report_local_bin_index));
	gpuErrchk(cudaFree(d_xmachine_message_fibroblast_location_report_unsorted_index));
  gpuErrchk(cudaFree(d_temp_scan_storage_xmachine_message_fibroblast_location_report));
  d_temp_scan_storage_xmachine_message_fibroblast_location_report = nullptr;
  temp_scan_bytes_xmachine_message_fibroblast_location_report = 0;
#else
	gpuErrchk(cudaFree(d_xmachine_message_fibroblast_location_report_keys));
	gpuErrchk(cudaFree(d_xmachine_message_fibroblast_location_report_values));
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
  gpuErrchk(cudaStreamDestroy(stream2));
  gpuErrchk(cudaStreamDestroy(stream3));

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
	h_message_fibroblast_damage_report_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_fibroblast_damage_report_count, &h_message_fibroblast_damage_report_count, sizeof(int)));
	
	h_message_tissue_damage_report_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_tissue_damage_report_count, &h_message_tissue_damage_report_count, sizeof(int)));
	
	h_message_doublings_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_doublings_count, &h_message_doublings_count, sizeof(int)));
	
	h_message_count_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_count_count, &h_message_count_count, sizeof(int)));
	
	h_message_fibroblast_location_report_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_fibroblast_location_report_count, &h_message_fibroblast_location_report_count, sizeof(int)));
	

	/* Call agent functions in order iterating through the layer functions */
	
	/* Layer 1*/
	
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
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("Fibroblast_QuiescentTakesDamage");
	Fibroblast_QuiescentTakesDamage(stream2);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: Fibroblast_QuiescentTakesDamage = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 2*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("TissueBlock_TissueSendDamageReport");
	TissueBlock_TissueSendDamageReport(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: TissueBlock_TissueSendDamageReport = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 3*/
	
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
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("Fibroblast_EarlySenescentMigration");
	Fibroblast_EarlySenescentMigration(stream2);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: Fibroblast_EarlySenescentMigration = %f (ms)\n", instrument_milliseconds);
#endif
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("Fibroblast_SenescentMigration");
	Fibroblast_SenescentMigration(stream3);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: Fibroblast_SenescentMigration = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 4*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("TissueBlock_TissueSendDamageReport");
	TissueBlock_TissueSendDamageReport(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: TissueBlock_TissueSendDamageReport = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 5*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("Fibroblast_BystanderEffect");
	Fibroblast_BystanderEffect(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: Fibroblast_BystanderEffect = %f (ms)\n", instrument_milliseconds);
#endif
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("Fibroblast_ExcessiveDamage");
	Fibroblast_ExcessiveDamage(stream2);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: Fibroblast_ExcessiveDamage = %f (ms)\n", instrument_milliseconds);
#endif
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("Fibroblast_ReplicativeSenescence");
	Fibroblast_ReplicativeSenescence(stream3);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: Fibroblast_ReplicativeSenescence = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 6*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("Fibroblast_TransitionToProliferating");
	Fibroblast_TransitionToProliferating(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: Fibroblast_TransitionToProliferating = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 7*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("Fibroblast_EarlySenCountTime");
	Fibroblast_EarlySenCountTime(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: Fibroblast_EarlySenCountTime = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 8*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("Fibroblast_TransitionToFullSenescence");
	Fibroblast_TransitionToFullSenescence(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: Fibroblast_TransitionToFullSenescence = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 9*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("Fibroblast_ClearanceOfEarlySenescent");
	Fibroblast_ClearanceOfEarlySenescent(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: Fibroblast_ClearanceOfEarlySenescent = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 10*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("Fibroblast_ClearanceOfSenescent");
	Fibroblast_ClearanceOfSenescent(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: Fibroblast_ClearanceOfSenescent = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 11*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("Fibroblast_DetectDamage");
	Fibroblast_DetectDamage(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: Fibroblast_DetectDamage = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
    
    /* Call all step functions */
	

#if defined(OUTPUT_POPULATION_PER_ITERATION) && OUTPUT_POPULATION_PER_ITERATION
	// Print the agent population size of all agents in all states
	
		printf("agent_TissueBlock_default_count: %u\n",get_agent_TissueBlock_default_count());
	
		printf("agent_Fibroblast_Quiescent_count: %u\n",get_agent_Fibroblast_Quiescent_count());
	
		printf("agent_Fibroblast_EarlySenescent_count: %u\n",get_agent_Fibroblast_EarlySenescent_count());
	
		printf("agent_Fibroblast_Senescent_count: %u\n",get_agent_Fibroblast_Senescent_count());
	
		printf("agent_Fibroblast_Proliferating_count: %u\n",get_agent_Fibroblast_Proliferating_count());
	
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
float h_env_EARLY_SENESCENT_MIGRATION_SCALE;
float h_env_SENESCENT_MIGRATION_SCALE;
float h_env_QUIESCENT_MIGRATION_SCALE;
float h_env_PROLIFERATION_PROB;
float h_env_BYSTANDER_DISTANCE;
float h_env_BYSTANDER_PROB;
int h_env_EXCESSIVE_DAMAGE_AMOUNT;
float h_env_EXCESSIVE_DAMAGE_PROB;
int h_env_REPLICATIVE_SEN_AGE;
float h_env_REPLICATIVE_SEN_PROB;
int h_env_EARLY_SENESCENT_MATURATION_TIME;
float h_env_TRANSITION_TO_FULL_SENESCENCE_PROB;
float h_env_CLEARANCE_EARLY_SEN_PROB;
float h_env_CLEARANCE_SEN_PROB;
float h_env_REPAIR_RADIUS;


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
void set_EARLY_SENESCENT_MIGRATION_SCALE(float* h_EARLY_SENESCENT_MIGRATION_SCALE){
    gpuErrchk(cudaMemcpyToSymbol(EARLY_SENESCENT_MIGRATION_SCALE, h_EARLY_SENESCENT_MIGRATION_SCALE, sizeof(float)));
    memcpy(&h_env_EARLY_SENESCENT_MIGRATION_SCALE, h_EARLY_SENESCENT_MIGRATION_SCALE,sizeof(float));
}

//constant getter
const float* get_EARLY_SENESCENT_MIGRATION_SCALE(){
    return &h_env_EARLY_SENESCENT_MIGRATION_SCALE;
}



//constant setter
void set_SENESCENT_MIGRATION_SCALE(float* h_SENESCENT_MIGRATION_SCALE){
    gpuErrchk(cudaMemcpyToSymbol(SENESCENT_MIGRATION_SCALE, h_SENESCENT_MIGRATION_SCALE, sizeof(float)));
    memcpy(&h_env_SENESCENT_MIGRATION_SCALE, h_SENESCENT_MIGRATION_SCALE,sizeof(float));
}

//constant getter
const float* get_SENESCENT_MIGRATION_SCALE(){
    return &h_env_SENESCENT_MIGRATION_SCALE;
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
void set_PROLIFERATION_PROB(float* h_PROLIFERATION_PROB){
    gpuErrchk(cudaMemcpyToSymbol(PROLIFERATION_PROB, h_PROLIFERATION_PROB, sizeof(float)));
    memcpy(&h_env_PROLIFERATION_PROB, h_PROLIFERATION_PROB,sizeof(float));
}

//constant getter
const float* get_PROLIFERATION_PROB(){
    return &h_env_PROLIFERATION_PROB;
}



//constant setter
void set_BYSTANDER_DISTANCE(float* h_BYSTANDER_DISTANCE){
    gpuErrchk(cudaMemcpyToSymbol(BYSTANDER_DISTANCE, h_BYSTANDER_DISTANCE, sizeof(float)));
    memcpy(&h_env_BYSTANDER_DISTANCE, h_BYSTANDER_DISTANCE,sizeof(float));
}

//constant getter
const float* get_BYSTANDER_DISTANCE(){
    return &h_env_BYSTANDER_DISTANCE;
}



//constant setter
void set_BYSTANDER_PROB(float* h_BYSTANDER_PROB){
    gpuErrchk(cudaMemcpyToSymbol(BYSTANDER_PROB, h_BYSTANDER_PROB, sizeof(float)));
    memcpy(&h_env_BYSTANDER_PROB, h_BYSTANDER_PROB,sizeof(float));
}

//constant getter
const float* get_BYSTANDER_PROB(){
    return &h_env_BYSTANDER_PROB;
}



//constant setter
void set_EXCESSIVE_DAMAGE_AMOUNT(int* h_EXCESSIVE_DAMAGE_AMOUNT){
    gpuErrchk(cudaMemcpyToSymbol(EXCESSIVE_DAMAGE_AMOUNT, h_EXCESSIVE_DAMAGE_AMOUNT, sizeof(int)));
    memcpy(&h_env_EXCESSIVE_DAMAGE_AMOUNT, h_EXCESSIVE_DAMAGE_AMOUNT,sizeof(int));
}

//constant getter
const int* get_EXCESSIVE_DAMAGE_AMOUNT(){
    return &h_env_EXCESSIVE_DAMAGE_AMOUNT;
}



//constant setter
void set_EXCESSIVE_DAMAGE_PROB(float* h_EXCESSIVE_DAMAGE_PROB){
    gpuErrchk(cudaMemcpyToSymbol(EXCESSIVE_DAMAGE_PROB, h_EXCESSIVE_DAMAGE_PROB, sizeof(float)));
    memcpy(&h_env_EXCESSIVE_DAMAGE_PROB, h_EXCESSIVE_DAMAGE_PROB,sizeof(float));
}

//constant getter
const float* get_EXCESSIVE_DAMAGE_PROB(){
    return &h_env_EXCESSIVE_DAMAGE_PROB;
}



//constant setter
void set_REPLICATIVE_SEN_AGE(int* h_REPLICATIVE_SEN_AGE){
    gpuErrchk(cudaMemcpyToSymbol(REPLICATIVE_SEN_AGE, h_REPLICATIVE_SEN_AGE, sizeof(int)));
    memcpy(&h_env_REPLICATIVE_SEN_AGE, h_REPLICATIVE_SEN_AGE,sizeof(int));
}

//constant getter
const int* get_REPLICATIVE_SEN_AGE(){
    return &h_env_REPLICATIVE_SEN_AGE;
}



//constant setter
void set_REPLICATIVE_SEN_PROB(float* h_REPLICATIVE_SEN_PROB){
    gpuErrchk(cudaMemcpyToSymbol(REPLICATIVE_SEN_PROB, h_REPLICATIVE_SEN_PROB, sizeof(float)));
    memcpy(&h_env_REPLICATIVE_SEN_PROB, h_REPLICATIVE_SEN_PROB,sizeof(float));
}

//constant getter
const float* get_REPLICATIVE_SEN_PROB(){
    return &h_env_REPLICATIVE_SEN_PROB;
}



//constant setter
void set_EARLY_SENESCENT_MATURATION_TIME(int* h_EARLY_SENESCENT_MATURATION_TIME){
    gpuErrchk(cudaMemcpyToSymbol(EARLY_SENESCENT_MATURATION_TIME, h_EARLY_SENESCENT_MATURATION_TIME, sizeof(int)));
    memcpy(&h_env_EARLY_SENESCENT_MATURATION_TIME, h_EARLY_SENESCENT_MATURATION_TIME,sizeof(int));
}

//constant getter
const int* get_EARLY_SENESCENT_MATURATION_TIME(){
    return &h_env_EARLY_SENESCENT_MATURATION_TIME;
}



//constant setter
void set_TRANSITION_TO_FULL_SENESCENCE_PROB(float* h_TRANSITION_TO_FULL_SENESCENCE_PROB){
    gpuErrchk(cudaMemcpyToSymbol(TRANSITION_TO_FULL_SENESCENCE_PROB, h_TRANSITION_TO_FULL_SENESCENCE_PROB, sizeof(float)));
    memcpy(&h_env_TRANSITION_TO_FULL_SENESCENCE_PROB, h_TRANSITION_TO_FULL_SENESCENCE_PROB,sizeof(float));
}

//constant getter
const float* get_TRANSITION_TO_FULL_SENESCENCE_PROB(){
    return &h_env_TRANSITION_TO_FULL_SENESCENCE_PROB;
}



//constant setter
void set_CLEARANCE_EARLY_SEN_PROB(float* h_CLEARANCE_EARLY_SEN_PROB){
    gpuErrchk(cudaMemcpyToSymbol(CLEARANCE_EARLY_SEN_PROB, h_CLEARANCE_EARLY_SEN_PROB, sizeof(float)));
    memcpy(&h_env_CLEARANCE_EARLY_SEN_PROB, h_CLEARANCE_EARLY_SEN_PROB,sizeof(float));
}

//constant getter
const float* get_CLEARANCE_EARLY_SEN_PROB(){
    return &h_env_CLEARANCE_EARLY_SEN_PROB;
}



//constant setter
void set_CLEARANCE_SEN_PROB(float* h_CLEARANCE_SEN_PROB){
    gpuErrchk(cudaMemcpyToSymbol(CLEARANCE_SEN_PROB, h_CLEARANCE_SEN_PROB, sizeof(float)));
    memcpy(&h_env_CLEARANCE_SEN_PROB, h_CLEARANCE_SEN_PROB,sizeof(float));
}

//constant getter
const float* get_CLEARANCE_SEN_PROB(){
    return &h_env_CLEARANCE_SEN_PROB;
}



//constant setter
void set_REPAIR_RADIUS(float* h_REPAIR_RADIUS){
    gpuErrchk(cudaMemcpyToSymbol(REPAIR_RADIUS, h_REPAIR_RADIUS, sizeof(float)));
    memcpy(&h_env_REPAIR_RADIUS, h_REPAIR_RADIUS,sizeof(float));
}

//constant getter
const float* get_REPAIR_RADIUS(){
    return &h_env_REPAIR_RADIUS;
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

int get_agent_Fibroblast_EarlySenescent_count(){
	//continuous agent
	return h_xmachine_memory_Fibroblast_EarlySenescent_count;
	
}

xmachine_memory_Fibroblast_list* get_device_Fibroblast_EarlySenescent_agents(){
	return d_Fibroblasts_EarlySenescent;
}

xmachine_memory_Fibroblast_list* get_host_Fibroblast_EarlySenescent_agents(){
	return h_Fibroblasts_EarlySenescent;
}

int get_agent_Fibroblast_Senescent_count(){
	//continuous agent
	return h_xmachine_memory_Fibroblast_Senescent_count;
	
}

xmachine_memory_Fibroblast_list* get_device_Fibroblast_Senescent_agents(){
	return d_Fibroblasts_Senescent;
}

xmachine_memory_Fibroblast_list* get_host_Fibroblast_Senescent_agents(){
	return h_Fibroblasts_Senescent;
}

int get_agent_Fibroblast_Proliferating_count(){
	//continuous agent
	return h_xmachine_memory_Fibroblast_Proliferating_count;
	
}

xmachine_memory_Fibroblast_list* get_device_Fibroblast_Proliferating_agents(){
	return d_Fibroblasts_Proliferating;
}

xmachine_memory_Fibroblast_list* get_host_Fibroblast_Proliferating_agents(){
	return h_Fibroblasts_Proliferating;
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

/** float get_Fibroblast_Quiescent_variable_doublings(unsigned int index)
 * Gets the value of the doublings variable of an Fibroblast agent in the Quiescent state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable doublings
 */
__host__ float get_Fibroblast_Quiescent_variable_doublings(unsigned int index){
    unsigned int count = get_agent_Fibroblast_Quiescent_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Fibroblasts_Quiescent_variable_doublings_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Fibroblasts_Quiescent->doublings,
                    d_Fibroblasts_Quiescent->doublings,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Fibroblasts_Quiescent_variable_doublings_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Fibroblasts_Quiescent->doublings[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access doublings for the %u th member of Fibroblast_Quiescent. count is %u at iteration %u\n", index, count, currentIteration);
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

/** int get_Fibroblast_Quiescent_variable_early_sen_time_counter(unsigned int index)
 * Gets the value of the early_sen_time_counter variable of an Fibroblast agent in the Quiescent state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable early_sen_time_counter
 */
__host__ int get_Fibroblast_Quiescent_variable_early_sen_time_counter(unsigned int index){
    unsigned int count = get_agent_Fibroblast_Quiescent_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Fibroblasts_Quiescent_variable_early_sen_time_counter_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Fibroblasts_Quiescent->early_sen_time_counter,
                    d_Fibroblasts_Quiescent->early_sen_time_counter,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Fibroblasts_Quiescent_variable_early_sen_time_counter_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Fibroblasts_Quiescent->early_sen_time_counter[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access early_sen_time_counter for the %u th member of Fibroblast_Quiescent. count is %u at iteration %u\n", index, count, currentIteration);
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

/** int get_Fibroblast_EarlySenescent_variable_id(unsigned int index)
 * Gets the value of the id variable of an Fibroblast agent in the EarlySenescent state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable id
 */
__host__ int get_Fibroblast_EarlySenescent_variable_id(unsigned int index){
    unsigned int count = get_agent_Fibroblast_EarlySenescent_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Fibroblasts_EarlySenescent_variable_id_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Fibroblasts_EarlySenescent->id,
                    d_Fibroblasts_EarlySenescent->id,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Fibroblasts_EarlySenescent_variable_id_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Fibroblasts_EarlySenescent->id[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access id for the %u th member of Fibroblast_EarlySenescent. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_Fibroblast_EarlySenescent_variable_x(unsigned int index)
 * Gets the value of the x variable of an Fibroblast agent in the EarlySenescent state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable x
 */
__host__ float get_Fibroblast_EarlySenescent_variable_x(unsigned int index){
    unsigned int count = get_agent_Fibroblast_EarlySenescent_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Fibroblasts_EarlySenescent_variable_x_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Fibroblasts_EarlySenescent->x,
                    d_Fibroblasts_EarlySenescent->x,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Fibroblasts_EarlySenescent_variable_x_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Fibroblasts_EarlySenescent->x[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access x for the %u th member of Fibroblast_EarlySenescent. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_Fibroblast_EarlySenescent_variable_y(unsigned int index)
 * Gets the value of the y variable of an Fibroblast agent in the EarlySenescent state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable y
 */
__host__ float get_Fibroblast_EarlySenescent_variable_y(unsigned int index){
    unsigned int count = get_agent_Fibroblast_EarlySenescent_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Fibroblasts_EarlySenescent_variable_y_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Fibroblasts_EarlySenescent->y,
                    d_Fibroblasts_EarlySenescent->y,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Fibroblasts_EarlySenescent_variable_y_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Fibroblasts_EarlySenescent->y[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access y for the %u th member of Fibroblast_EarlySenescent. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_Fibroblast_EarlySenescent_variable_z(unsigned int index)
 * Gets the value of the z variable of an Fibroblast agent in the EarlySenescent state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable z
 */
__host__ float get_Fibroblast_EarlySenescent_variable_z(unsigned int index){
    unsigned int count = get_agent_Fibroblast_EarlySenescent_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Fibroblasts_EarlySenescent_variable_z_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Fibroblasts_EarlySenescent->z,
                    d_Fibroblasts_EarlySenescent->z,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Fibroblasts_EarlySenescent_variable_z_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Fibroblasts_EarlySenescent->z[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access z for the %u th member of Fibroblast_EarlySenescent. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_Fibroblast_EarlySenescent_variable_doublings(unsigned int index)
 * Gets the value of the doublings variable of an Fibroblast agent in the EarlySenescent state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable doublings
 */
__host__ float get_Fibroblast_EarlySenescent_variable_doublings(unsigned int index){
    unsigned int count = get_agent_Fibroblast_EarlySenescent_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Fibroblasts_EarlySenescent_variable_doublings_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Fibroblasts_EarlySenescent->doublings,
                    d_Fibroblasts_EarlySenescent->doublings,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Fibroblasts_EarlySenescent_variable_doublings_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Fibroblasts_EarlySenescent->doublings[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access doublings for the %u th member of Fibroblast_EarlySenescent. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_Fibroblast_EarlySenescent_variable_damage(unsigned int index)
 * Gets the value of the damage variable of an Fibroblast agent in the EarlySenescent state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable damage
 */
__host__ int get_Fibroblast_EarlySenescent_variable_damage(unsigned int index){
    unsigned int count = get_agent_Fibroblast_EarlySenescent_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Fibroblasts_EarlySenescent_variable_damage_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Fibroblasts_EarlySenescent->damage,
                    d_Fibroblasts_EarlySenescent->damage,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Fibroblasts_EarlySenescent_variable_damage_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Fibroblasts_EarlySenescent->damage[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access damage for the %u th member of Fibroblast_EarlySenescent. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_Fibroblast_EarlySenescent_variable_early_sen_time_counter(unsigned int index)
 * Gets the value of the early_sen_time_counter variable of an Fibroblast agent in the EarlySenescent state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable early_sen_time_counter
 */
__host__ int get_Fibroblast_EarlySenescent_variable_early_sen_time_counter(unsigned int index){
    unsigned int count = get_agent_Fibroblast_EarlySenescent_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Fibroblasts_EarlySenescent_variable_early_sen_time_counter_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Fibroblasts_EarlySenescent->early_sen_time_counter,
                    d_Fibroblasts_EarlySenescent->early_sen_time_counter,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Fibroblasts_EarlySenescent_variable_early_sen_time_counter_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Fibroblasts_EarlySenescent->early_sen_time_counter[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access early_sen_time_counter for the %u th member of Fibroblast_EarlySenescent. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_Fibroblast_EarlySenescent_variable_current_state(unsigned int index)
 * Gets the value of the current_state variable of an Fibroblast agent in the EarlySenescent state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable current_state
 */
__host__ int get_Fibroblast_EarlySenescent_variable_current_state(unsigned int index){
    unsigned int count = get_agent_Fibroblast_EarlySenescent_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Fibroblasts_EarlySenescent_variable_current_state_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Fibroblasts_EarlySenescent->current_state,
                    d_Fibroblasts_EarlySenescent->current_state,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Fibroblasts_EarlySenescent_variable_current_state_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Fibroblasts_EarlySenescent->current_state[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access current_state for the %u th member of Fibroblast_EarlySenescent. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_Fibroblast_Senescent_variable_id(unsigned int index)
 * Gets the value of the id variable of an Fibroblast agent in the Senescent state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable id
 */
__host__ int get_Fibroblast_Senescent_variable_id(unsigned int index){
    unsigned int count = get_agent_Fibroblast_Senescent_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Fibroblasts_Senescent_variable_id_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Fibroblasts_Senescent->id,
                    d_Fibroblasts_Senescent->id,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Fibroblasts_Senescent_variable_id_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Fibroblasts_Senescent->id[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access id for the %u th member of Fibroblast_Senescent. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_Fibroblast_Senescent_variable_x(unsigned int index)
 * Gets the value of the x variable of an Fibroblast agent in the Senescent state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable x
 */
__host__ float get_Fibroblast_Senescent_variable_x(unsigned int index){
    unsigned int count = get_agent_Fibroblast_Senescent_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Fibroblasts_Senescent_variable_x_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Fibroblasts_Senescent->x,
                    d_Fibroblasts_Senescent->x,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Fibroblasts_Senescent_variable_x_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Fibroblasts_Senescent->x[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access x for the %u th member of Fibroblast_Senescent. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_Fibroblast_Senescent_variable_y(unsigned int index)
 * Gets the value of the y variable of an Fibroblast agent in the Senescent state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable y
 */
__host__ float get_Fibroblast_Senescent_variable_y(unsigned int index){
    unsigned int count = get_agent_Fibroblast_Senescent_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Fibroblasts_Senescent_variable_y_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Fibroblasts_Senescent->y,
                    d_Fibroblasts_Senescent->y,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Fibroblasts_Senescent_variable_y_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Fibroblasts_Senescent->y[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access y for the %u th member of Fibroblast_Senescent. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_Fibroblast_Senescent_variable_z(unsigned int index)
 * Gets the value of the z variable of an Fibroblast agent in the Senescent state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable z
 */
__host__ float get_Fibroblast_Senescent_variable_z(unsigned int index){
    unsigned int count = get_agent_Fibroblast_Senescent_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Fibroblasts_Senescent_variable_z_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Fibroblasts_Senescent->z,
                    d_Fibroblasts_Senescent->z,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Fibroblasts_Senescent_variable_z_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Fibroblasts_Senescent->z[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access z for the %u th member of Fibroblast_Senescent. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_Fibroblast_Senescent_variable_doublings(unsigned int index)
 * Gets the value of the doublings variable of an Fibroblast agent in the Senescent state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable doublings
 */
__host__ float get_Fibroblast_Senescent_variable_doublings(unsigned int index){
    unsigned int count = get_agent_Fibroblast_Senescent_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Fibroblasts_Senescent_variable_doublings_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Fibroblasts_Senescent->doublings,
                    d_Fibroblasts_Senescent->doublings,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Fibroblasts_Senescent_variable_doublings_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Fibroblasts_Senescent->doublings[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access doublings for the %u th member of Fibroblast_Senescent. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_Fibroblast_Senescent_variable_damage(unsigned int index)
 * Gets the value of the damage variable of an Fibroblast agent in the Senescent state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable damage
 */
__host__ int get_Fibroblast_Senescent_variable_damage(unsigned int index){
    unsigned int count = get_agent_Fibroblast_Senescent_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Fibroblasts_Senescent_variable_damage_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Fibroblasts_Senescent->damage,
                    d_Fibroblasts_Senescent->damage,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Fibroblasts_Senescent_variable_damage_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Fibroblasts_Senescent->damage[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access damage for the %u th member of Fibroblast_Senescent. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_Fibroblast_Senescent_variable_early_sen_time_counter(unsigned int index)
 * Gets the value of the early_sen_time_counter variable of an Fibroblast agent in the Senescent state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable early_sen_time_counter
 */
__host__ int get_Fibroblast_Senescent_variable_early_sen_time_counter(unsigned int index){
    unsigned int count = get_agent_Fibroblast_Senescent_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Fibroblasts_Senescent_variable_early_sen_time_counter_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Fibroblasts_Senescent->early_sen_time_counter,
                    d_Fibroblasts_Senescent->early_sen_time_counter,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Fibroblasts_Senescent_variable_early_sen_time_counter_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Fibroblasts_Senescent->early_sen_time_counter[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access early_sen_time_counter for the %u th member of Fibroblast_Senescent. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_Fibroblast_Senescent_variable_current_state(unsigned int index)
 * Gets the value of the current_state variable of an Fibroblast agent in the Senescent state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable current_state
 */
__host__ int get_Fibroblast_Senescent_variable_current_state(unsigned int index){
    unsigned int count = get_agent_Fibroblast_Senescent_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Fibroblasts_Senescent_variable_current_state_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Fibroblasts_Senescent->current_state,
                    d_Fibroblasts_Senescent->current_state,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Fibroblasts_Senescent_variable_current_state_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Fibroblasts_Senescent->current_state[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access current_state for the %u th member of Fibroblast_Senescent. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_Fibroblast_Proliferating_variable_id(unsigned int index)
 * Gets the value of the id variable of an Fibroblast agent in the Proliferating state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable id
 */
__host__ int get_Fibroblast_Proliferating_variable_id(unsigned int index){
    unsigned int count = get_agent_Fibroblast_Proliferating_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Fibroblasts_Proliferating_variable_id_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Fibroblasts_Proliferating->id,
                    d_Fibroblasts_Proliferating->id,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Fibroblasts_Proliferating_variable_id_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Fibroblasts_Proliferating->id[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access id for the %u th member of Fibroblast_Proliferating. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_Fibroblast_Proliferating_variable_x(unsigned int index)
 * Gets the value of the x variable of an Fibroblast agent in the Proliferating state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable x
 */
__host__ float get_Fibroblast_Proliferating_variable_x(unsigned int index){
    unsigned int count = get_agent_Fibroblast_Proliferating_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Fibroblasts_Proliferating_variable_x_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Fibroblasts_Proliferating->x,
                    d_Fibroblasts_Proliferating->x,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Fibroblasts_Proliferating_variable_x_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Fibroblasts_Proliferating->x[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access x for the %u th member of Fibroblast_Proliferating. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_Fibroblast_Proliferating_variable_y(unsigned int index)
 * Gets the value of the y variable of an Fibroblast agent in the Proliferating state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable y
 */
__host__ float get_Fibroblast_Proliferating_variable_y(unsigned int index){
    unsigned int count = get_agent_Fibroblast_Proliferating_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Fibroblasts_Proliferating_variable_y_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Fibroblasts_Proliferating->y,
                    d_Fibroblasts_Proliferating->y,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Fibroblasts_Proliferating_variable_y_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Fibroblasts_Proliferating->y[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access y for the %u th member of Fibroblast_Proliferating. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_Fibroblast_Proliferating_variable_z(unsigned int index)
 * Gets the value of the z variable of an Fibroblast agent in the Proliferating state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable z
 */
__host__ float get_Fibroblast_Proliferating_variable_z(unsigned int index){
    unsigned int count = get_agent_Fibroblast_Proliferating_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Fibroblasts_Proliferating_variable_z_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Fibroblasts_Proliferating->z,
                    d_Fibroblasts_Proliferating->z,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Fibroblasts_Proliferating_variable_z_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Fibroblasts_Proliferating->z[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access z for the %u th member of Fibroblast_Proliferating. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_Fibroblast_Proliferating_variable_doublings(unsigned int index)
 * Gets the value of the doublings variable of an Fibroblast agent in the Proliferating state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable doublings
 */
__host__ float get_Fibroblast_Proliferating_variable_doublings(unsigned int index){
    unsigned int count = get_agent_Fibroblast_Proliferating_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Fibroblasts_Proliferating_variable_doublings_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Fibroblasts_Proliferating->doublings,
                    d_Fibroblasts_Proliferating->doublings,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Fibroblasts_Proliferating_variable_doublings_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Fibroblasts_Proliferating->doublings[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access doublings for the %u th member of Fibroblast_Proliferating. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_Fibroblast_Proliferating_variable_damage(unsigned int index)
 * Gets the value of the damage variable of an Fibroblast agent in the Proliferating state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable damage
 */
__host__ int get_Fibroblast_Proliferating_variable_damage(unsigned int index){
    unsigned int count = get_agent_Fibroblast_Proliferating_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Fibroblasts_Proliferating_variable_damage_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Fibroblasts_Proliferating->damage,
                    d_Fibroblasts_Proliferating->damage,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Fibroblasts_Proliferating_variable_damage_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Fibroblasts_Proliferating->damage[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access damage for the %u th member of Fibroblast_Proliferating. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_Fibroblast_Proliferating_variable_early_sen_time_counter(unsigned int index)
 * Gets the value of the early_sen_time_counter variable of an Fibroblast agent in the Proliferating state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable early_sen_time_counter
 */
__host__ int get_Fibroblast_Proliferating_variable_early_sen_time_counter(unsigned int index){
    unsigned int count = get_agent_Fibroblast_Proliferating_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Fibroblasts_Proliferating_variable_early_sen_time_counter_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Fibroblasts_Proliferating->early_sen_time_counter,
                    d_Fibroblasts_Proliferating->early_sen_time_counter,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Fibroblasts_Proliferating_variable_early_sen_time_counter_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Fibroblasts_Proliferating->early_sen_time_counter[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access early_sen_time_counter for the %u th member of Fibroblast_Proliferating. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_Fibroblast_Proliferating_variable_current_state(unsigned int index)
 * Gets the value of the current_state variable of an Fibroblast agent in the Proliferating state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable current_state
 */
__host__ int get_Fibroblast_Proliferating_variable_current_state(unsigned int index){
    unsigned int count = get_agent_Fibroblast_Proliferating_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Fibroblasts_Proliferating_variable_current_state_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Fibroblasts_Proliferating->current_state,
                    d_Fibroblasts_Proliferating->current_state,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Fibroblasts_Proliferating_variable_current_state_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Fibroblasts_Proliferating->current_state[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access current_state for the %u th member of Fibroblast_Proliferating. count is %u at iteration %u\n", index, count, currentIteration);
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

/** float get_Fibroblast_Repair_variable_doublings(unsigned int index)
 * Gets the value of the doublings variable of an Fibroblast agent in the Repair state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable doublings
 */
__host__ float get_Fibroblast_Repair_variable_doublings(unsigned int index){
    unsigned int count = get_agent_Fibroblast_Repair_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Fibroblasts_Repair_variable_doublings_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Fibroblasts_Repair->doublings,
                    d_Fibroblasts_Repair->doublings,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Fibroblasts_Repair_variable_doublings_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Fibroblasts_Repair->doublings[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access doublings for the %u th member of Fibroblast_Repair. count is %u at iteration %u\n", index, count, currentIteration);
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

/** int get_Fibroblast_Repair_variable_early_sen_time_counter(unsigned int index)
 * Gets the value of the early_sen_time_counter variable of an Fibroblast agent in the Repair state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable early_sen_time_counter
 */
__host__ int get_Fibroblast_Repair_variable_early_sen_time_counter(unsigned int index){
    unsigned int count = get_agent_Fibroblast_Repair_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Fibroblasts_Repair_variable_early_sen_time_counter_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Fibroblasts_Repair->early_sen_time_counter,
                    d_Fibroblasts_Repair->early_sen_time_counter,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Fibroblasts_Repair_variable_early_sen_time_counter_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Fibroblasts_Repair->early_sen_time_counter[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access early_sen_time_counter for the %u th member of Fibroblast_Repair. count is %u at iteration %u\n", index, count, currentIteration);
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
 
		gpuErrchk(cudaMemcpy(d_dst->doublings, &h_agent->doublings, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->damage, &h_agent->damage, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->early_sen_time_counter, &h_agent->early_sen_time_counter, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->current_state, &h_agent->current_state, sizeof(int), cudaMemcpyHostToDevice));

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
 
		gpuErrchk(cudaMemcpy(d_dst->doublings, h_src->doublings, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->damage, h_src->damage, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->early_sen_time_counter, h_src->early_sen_time_counter, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->current_state, h_src->current_state, count * sizeof(int), cudaMemcpyHostToDevice));

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
			 
			dst->doublings[i] = src[i]->doublings;
			 
			dst->damage[i] = src[i]->damage;
			 
			dst->early_sen_time_counter[i] = src[i]->early_sen_time_counter;
			 
			dst->current_state[i] = src[i]->current_state;
			
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
    h_Fibroblasts_Quiescent_variable_doublings_data_iteration = 0;
    h_Fibroblasts_Quiescent_variable_damage_data_iteration = 0;
    h_Fibroblasts_Quiescent_variable_early_sen_time_counter_data_iteration = 0;
    h_Fibroblasts_Quiescent_variable_current_state_data_iteration = 0;
    

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
        h_Fibroblasts_Quiescent_variable_doublings_data_iteration = 0;
        h_Fibroblasts_Quiescent_variable_damage_data_iteration = 0;
        h_Fibroblasts_Quiescent_variable_early_sen_time_counter_data_iteration = 0;
        h_Fibroblasts_Quiescent_variable_current_state_data_iteration = 0;
        

	}
}


void h_add_agent_Fibroblast_EarlySenescent(xmachine_memory_Fibroblast* agent){
	if (h_xmachine_memory_Fibroblast_count + 1 > xmachine_memory_Fibroblast_MAX){
		printf("Error: Buffer size of Fibroblast agents in state EarlySenescent will be exceeded by h_add_agent_Fibroblast_EarlySenescent\n");
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
	append_Fibroblast_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_Fibroblasts_EarlySenescent, d_Fibroblasts_new, h_xmachine_memory_Fibroblast_EarlySenescent_count, count);
	gpuErrchkLaunch();
	// Update the number of agents in this state.
	h_xmachine_memory_Fibroblast_EarlySenescent_count += count;
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_Fibroblast_EarlySenescent_count, &h_xmachine_memory_Fibroblast_EarlySenescent_count, sizeof(int)));
	cudaDeviceSynchronize();

    // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
    h_Fibroblasts_EarlySenescent_variable_id_data_iteration = 0;
    h_Fibroblasts_EarlySenescent_variable_x_data_iteration = 0;
    h_Fibroblasts_EarlySenescent_variable_y_data_iteration = 0;
    h_Fibroblasts_EarlySenescent_variable_z_data_iteration = 0;
    h_Fibroblasts_EarlySenescent_variable_doublings_data_iteration = 0;
    h_Fibroblasts_EarlySenescent_variable_damage_data_iteration = 0;
    h_Fibroblasts_EarlySenescent_variable_early_sen_time_counter_data_iteration = 0;
    h_Fibroblasts_EarlySenescent_variable_current_state_data_iteration = 0;
    

}
void h_add_agents_Fibroblast_EarlySenescent(xmachine_memory_Fibroblast** agents, unsigned int count){
	if(count > 0){
		int blockSize;
		int minGridSize;
		int gridSize;

		if (h_xmachine_memory_Fibroblast_count + count > xmachine_memory_Fibroblast_MAX){
			printf("Error: Buffer size of Fibroblast agents in state EarlySenescent will be exceeded by h_add_agents_Fibroblast_EarlySenescent\n");
			exit(EXIT_FAILURE);
		}

		// Unpack data from AoS into the pre-existing SoA
		h_unpack_agents_Fibroblast_AoS_to_SoA(h_Fibroblasts_EarlySenescent, agents, count);

		// Copy data from the host SoA to the device SoA for the target state
		copy_partial_xmachine_memory_Fibroblast_hostToDevice(d_Fibroblasts_new, h_Fibroblasts_EarlySenescent, count);

		// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
		cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_Fibroblast_Agents, no_sm, count);
		gridSize = (count + blockSize - 1) / blockSize;
		append_Fibroblast_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_Fibroblasts_EarlySenescent, d_Fibroblasts_new, h_xmachine_memory_Fibroblast_EarlySenescent_count, count);
		gpuErrchkLaunch();
		// Update the number of agents in this state.
		h_xmachine_memory_Fibroblast_EarlySenescent_count += count;
		gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_Fibroblast_EarlySenescent_count, &h_xmachine_memory_Fibroblast_EarlySenescent_count, sizeof(int)));
		cudaDeviceSynchronize();

        // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
        h_Fibroblasts_EarlySenescent_variable_id_data_iteration = 0;
        h_Fibroblasts_EarlySenescent_variable_x_data_iteration = 0;
        h_Fibroblasts_EarlySenescent_variable_y_data_iteration = 0;
        h_Fibroblasts_EarlySenescent_variable_z_data_iteration = 0;
        h_Fibroblasts_EarlySenescent_variable_doublings_data_iteration = 0;
        h_Fibroblasts_EarlySenescent_variable_damage_data_iteration = 0;
        h_Fibroblasts_EarlySenescent_variable_early_sen_time_counter_data_iteration = 0;
        h_Fibroblasts_EarlySenescent_variable_current_state_data_iteration = 0;
        

	}
}


void h_add_agent_Fibroblast_Senescent(xmachine_memory_Fibroblast* agent){
	if (h_xmachine_memory_Fibroblast_count + 1 > xmachine_memory_Fibroblast_MAX){
		printf("Error: Buffer size of Fibroblast agents in state Senescent will be exceeded by h_add_agent_Fibroblast_Senescent\n");
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
	append_Fibroblast_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_Fibroblasts_Senescent, d_Fibroblasts_new, h_xmachine_memory_Fibroblast_Senescent_count, count);
	gpuErrchkLaunch();
	// Update the number of agents in this state.
	h_xmachine_memory_Fibroblast_Senescent_count += count;
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_Fibroblast_Senescent_count, &h_xmachine_memory_Fibroblast_Senescent_count, sizeof(int)));
	cudaDeviceSynchronize();

    // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
    h_Fibroblasts_Senescent_variable_id_data_iteration = 0;
    h_Fibroblasts_Senescent_variable_x_data_iteration = 0;
    h_Fibroblasts_Senescent_variable_y_data_iteration = 0;
    h_Fibroblasts_Senescent_variable_z_data_iteration = 0;
    h_Fibroblasts_Senescent_variable_doublings_data_iteration = 0;
    h_Fibroblasts_Senescent_variable_damage_data_iteration = 0;
    h_Fibroblasts_Senescent_variable_early_sen_time_counter_data_iteration = 0;
    h_Fibroblasts_Senescent_variable_current_state_data_iteration = 0;
    

}
void h_add_agents_Fibroblast_Senescent(xmachine_memory_Fibroblast** agents, unsigned int count){
	if(count > 0){
		int blockSize;
		int minGridSize;
		int gridSize;

		if (h_xmachine_memory_Fibroblast_count + count > xmachine_memory_Fibroblast_MAX){
			printf("Error: Buffer size of Fibroblast agents in state Senescent will be exceeded by h_add_agents_Fibroblast_Senescent\n");
			exit(EXIT_FAILURE);
		}

		// Unpack data from AoS into the pre-existing SoA
		h_unpack_agents_Fibroblast_AoS_to_SoA(h_Fibroblasts_Senescent, agents, count);

		// Copy data from the host SoA to the device SoA for the target state
		copy_partial_xmachine_memory_Fibroblast_hostToDevice(d_Fibroblasts_new, h_Fibroblasts_Senescent, count);

		// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
		cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_Fibroblast_Agents, no_sm, count);
		gridSize = (count + blockSize - 1) / blockSize;
		append_Fibroblast_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_Fibroblasts_Senescent, d_Fibroblasts_new, h_xmachine_memory_Fibroblast_Senescent_count, count);
		gpuErrchkLaunch();
		// Update the number of agents in this state.
		h_xmachine_memory_Fibroblast_Senescent_count += count;
		gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_Fibroblast_Senescent_count, &h_xmachine_memory_Fibroblast_Senescent_count, sizeof(int)));
		cudaDeviceSynchronize();

        // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
        h_Fibroblasts_Senescent_variable_id_data_iteration = 0;
        h_Fibroblasts_Senescent_variable_x_data_iteration = 0;
        h_Fibroblasts_Senescent_variable_y_data_iteration = 0;
        h_Fibroblasts_Senescent_variable_z_data_iteration = 0;
        h_Fibroblasts_Senescent_variable_doublings_data_iteration = 0;
        h_Fibroblasts_Senescent_variable_damage_data_iteration = 0;
        h_Fibroblasts_Senescent_variable_early_sen_time_counter_data_iteration = 0;
        h_Fibroblasts_Senescent_variable_current_state_data_iteration = 0;
        

	}
}


void h_add_agent_Fibroblast_Proliferating(xmachine_memory_Fibroblast* agent){
	if (h_xmachine_memory_Fibroblast_count + 1 > xmachine_memory_Fibroblast_MAX){
		printf("Error: Buffer size of Fibroblast agents in state Proliferating will be exceeded by h_add_agent_Fibroblast_Proliferating\n");
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
	append_Fibroblast_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_Fibroblasts_Proliferating, d_Fibroblasts_new, h_xmachine_memory_Fibroblast_Proliferating_count, count);
	gpuErrchkLaunch();
	// Update the number of agents in this state.
	h_xmachine_memory_Fibroblast_Proliferating_count += count;
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_Fibroblast_Proliferating_count, &h_xmachine_memory_Fibroblast_Proliferating_count, sizeof(int)));
	cudaDeviceSynchronize();

    // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
    h_Fibroblasts_Proliferating_variable_id_data_iteration = 0;
    h_Fibroblasts_Proliferating_variable_x_data_iteration = 0;
    h_Fibroblasts_Proliferating_variable_y_data_iteration = 0;
    h_Fibroblasts_Proliferating_variable_z_data_iteration = 0;
    h_Fibroblasts_Proliferating_variable_doublings_data_iteration = 0;
    h_Fibroblasts_Proliferating_variable_damage_data_iteration = 0;
    h_Fibroblasts_Proliferating_variable_early_sen_time_counter_data_iteration = 0;
    h_Fibroblasts_Proliferating_variable_current_state_data_iteration = 0;
    

}
void h_add_agents_Fibroblast_Proliferating(xmachine_memory_Fibroblast** agents, unsigned int count){
	if(count > 0){
		int blockSize;
		int minGridSize;
		int gridSize;

		if (h_xmachine_memory_Fibroblast_count + count > xmachine_memory_Fibroblast_MAX){
			printf("Error: Buffer size of Fibroblast agents in state Proliferating will be exceeded by h_add_agents_Fibroblast_Proliferating\n");
			exit(EXIT_FAILURE);
		}

		// Unpack data from AoS into the pre-existing SoA
		h_unpack_agents_Fibroblast_AoS_to_SoA(h_Fibroblasts_Proliferating, agents, count);

		// Copy data from the host SoA to the device SoA for the target state
		copy_partial_xmachine_memory_Fibroblast_hostToDevice(d_Fibroblasts_new, h_Fibroblasts_Proliferating, count);

		// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
		cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_Fibroblast_Agents, no_sm, count);
		gridSize = (count + blockSize - 1) / blockSize;
		append_Fibroblast_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_Fibroblasts_Proliferating, d_Fibroblasts_new, h_xmachine_memory_Fibroblast_Proliferating_count, count);
		gpuErrchkLaunch();
		// Update the number of agents in this state.
		h_xmachine_memory_Fibroblast_Proliferating_count += count;
		gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_Fibroblast_Proliferating_count, &h_xmachine_memory_Fibroblast_Proliferating_count, sizeof(int)));
		cudaDeviceSynchronize();

        // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
        h_Fibroblasts_Proliferating_variable_id_data_iteration = 0;
        h_Fibroblasts_Proliferating_variable_x_data_iteration = 0;
        h_Fibroblasts_Proliferating_variable_y_data_iteration = 0;
        h_Fibroblasts_Proliferating_variable_z_data_iteration = 0;
        h_Fibroblasts_Proliferating_variable_doublings_data_iteration = 0;
        h_Fibroblasts_Proliferating_variable_damage_data_iteration = 0;
        h_Fibroblasts_Proliferating_variable_early_sen_time_counter_data_iteration = 0;
        h_Fibroblasts_Proliferating_variable_current_state_data_iteration = 0;
        

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
    h_Fibroblasts_Repair_variable_doublings_data_iteration = 0;
    h_Fibroblasts_Repair_variable_damage_data_iteration = 0;
    h_Fibroblasts_Repair_variable_early_sen_time_counter_data_iteration = 0;
    h_Fibroblasts_Repair_variable_current_state_data_iteration = 0;
    

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
        h_Fibroblasts_Repair_variable_doublings_data_iteration = 0;
        h_Fibroblasts_Repair_variable_damage_data_iteration = 0;
        h_Fibroblasts_Repair_variable_early_sen_time_counter_data_iteration = 0;
        h_Fibroblasts_Repair_variable_current_state_data_iteration = 0;
        

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
float reduce_Fibroblast_Quiescent_doublings_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Fibroblasts_Quiescent->doublings),  thrust::device_pointer_cast(d_Fibroblasts_Quiescent->doublings) + h_xmachine_memory_Fibroblast_Quiescent_count);
}

float min_Fibroblast_Quiescent_doublings_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Quiescent->doublings);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_Quiescent_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_Fibroblast_Quiescent_doublings_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Quiescent->doublings);
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
int reduce_Fibroblast_Quiescent_early_sen_time_counter_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Fibroblasts_Quiescent->early_sen_time_counter),  thrust::device_pointer_cast(d_Fibroblasts_Quiescent->early_sen_time_counter) + h_xmachine_memory_Fibroblast_Quiescent_count);
}

int count_Fibroblast_Quiescent_early_sen_time_counter_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Fibroblasts_Quiescent->early_sen_time_counter),  thrust::device_pointer_cast(d_Fibroblasts_Quiescent->early_sen_time_counter) + h_xmachine_memory_Fibroblast_Quiescent_count, count_value);
}
int min_Fibroblast_Quiescent_early_sen_time_counter_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Quiescent->early_sen_time_counter);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_Quiescent_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_Fibroblast_Quiescent_early_sen_time_counter_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Quiescent->early_sen_time_counter);
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
int reduce_Fibroblast_EarlySenescent_id_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Fibroblasts_EarlySenescent->id),  thrust::device_pointer_cast(d_Fibroblasts_EarlySenescent->id) + h_xmachine_memory_Fibroblast_EarlySenescent_count);
}

int count_Fibroblast_EarlySenescent_id_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Fibroblasts_EarlySenescent->id),  thrust::device_pointer_cast(d_Fibroblasts_EarlySenescent->id) + h_xmachine_memory_Fibroblast_EarlySenescent_count, count_value);
}
int min_Fibroblast_EarlySenescent_id_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_EarlySenescent->id);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_EarlySenescent_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_Fibroblast_EarlySenescent_id_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_EarlySenescent->id);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_EarlySenescent_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_Fibroblast_EarlySenescent_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Fibroblasts_EarlySenescent->x),  thrust::device_pointer_cast(d_Fibroblasts_EarlySenescent->x) + h_xmachine_memory_Fibroblast_EarlySenescent_count);
}

float min_Fibroblast_EarlySenescent_x_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_EarlySenescent->x);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_EarlySenescent_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_Fibroblast_EarlySenescent_x_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_EarlySenescent->x);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_EarlySenescent_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_Fibroblast_EarlySenescent_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Fibroblasts_EarlySenescent->y),  thrust::device_pointer_cast(d_Fibroblasts_EarlySenescent->y) + h_xmachine_memory_Fibroblast_EarlySenescent_count);
}

float min_Fibroblast_EarlySenescent_y_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_EarlySenescent->y);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_EarlySenescent_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_Fibroblast_EarlySenescent_y_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_EarlySenescent->y);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_EarlySenescent_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_Fibroblast_EarlySenescent_z_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Fibroblasts_EarlySenescent->z),  thrust::device_pointer_cast(d_Fibroblasts_EarlySenescent->z) + h_xmachine_memory_Fibroblast_EarlySenescent_count);
}

float min_Fibroblast_EarlySenescent_z_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_EarlySenescent->z);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_EarlySenescent_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_Fibroblast_EarlySenescent_z_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_EarlySenescent->z);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_EarlySenescent_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_Fibroblast_EarlySenescent_doublings_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Fibroblasts_EarlySenescent->doublings),  thrust::device_pointer_cast(d_Fibroblasts_EarlySenescent->doublings) + h_xmachine_memory_Fibroblast_EarlySenescent_count);
}

float min_Fibroblast_EarlySenescent_doublings_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_EarlySenescent->doublings);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_EarlySenescent_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_Fibroblast_EarlySenescent_doublings_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_EarlySenescent->doublings);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_EarlySenescent_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_Fibroblast_EarlySenescent_damage_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Fibroblasts_EarlySenescent->damage),  thrust::device_pointer_cast(d_Fibroblasts_EarlySenescent->damage) + h_xmachine_memory_Fibroblast_EarlySenescent_count);
}

int count_Fibroblast_EarlySenescent_damage_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Fibroblasts_EarlySenescent->damage),  thrust::device_pointer_cast(d_Fibroblasts_EarlySenescent->damage) + h_xmachine_memory_Fibroblast_EarlySenescent_count, count_value);
}
int min_Fibroblast_EarlySenescent_damage_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_EarlySenescent->damage);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_EarlySenescent_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_Fibroblast_EarlySenescent_damage_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_EarlySenescent->damage);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_EarlySenescent_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_Fibroblast_EarlySenescent_early_sen_time_counter_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Fibroblasts_EarlySenescent->early_sen_time_counter),  thrust::device_pointer_cast(d_Fibroblasts_EarlySenescent->early_sen_time_counter) + h_xmachine_memory_Fibroblast_EarlySenescent_count);
}

int count_Fibroblast_EarlySenescent_early_sen_time_counter_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Fibroblasts_EarlySenescent->early_sen_time_counter),  thrust::device_pointer_cast(d_Fibroblasts_EarlySenescent->early_sen_time_counter) + h_xmachine_memory_Fibroblast_EarlySenescent_count, count_value);
}
int min_Fibroblast_EarlySenescent_early_sen_time_counter_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_EarlySenescent->early_sen_time_counter);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_EarlySenescent_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_Fibroblast_EarlySenescent_early_sen_time_counter_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_EarlySenescent->early_sen_time_counter);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_EarlySenescent_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_Fibroblast_EarlySenescent_current_state_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Fibroblasts_EarlySenescent->current_state),  thrust::device_pointer_cast(d_Fibroblasts_EarlySenescent->current_state) + h_xmachine_memory_Fibroblast_EarlySenescent_count);
}

int count_Fibroblast_EarlySenescent_current_state_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Fibroblasts_EarlySenescent->current_state),  thrust::device_pointer_cast(d_Fibroblasts_EarlySenescent->current_state) + h_xmachine_memory_Fibroblast_EarlySenescent_count, count_value);
}
int min_Fibroblast_EarlySenescent_current_state_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_EarlySenescent->current_state);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_EarlySenescent_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_Fibroblast_EarlySenescent_current_state_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_EarlySenescent->current_state);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_EarlySenescent_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_Fibroblast_Senescent_id_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Fibroblasts_Senescent->id),  thrust::device_pointer_cast(d_Fibroblasts_Senescent->id) + h_xmachine_memory_Fibroblast_Senescent_count);
}

int count_Fibroblast_Senescent_id_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Fibroblasts_Senescent->id),  thrust::device_pointer_cast(d_Fibroblasts_Senescent->id) + h_xmachine_memory_Fibroblast_Senescent_count, count_value);
}
int min_Fibroblast_Senescent_id_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Senescent->id);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_Senescent_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_Fibroblast_Senescent_id_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Senescent->id);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_Senescent_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_Fibroblast_Senescent_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Fibroblasts_Senescent->x),  thrust::device_pointer_cast(d_Fibroblasts_Senescent->x) + h_xmachine_memory_Fibroblast_Senescent_count);
}

float min_Fibroblast_Senescent_x_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Senescent->x);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_Senescent_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_Fibroblast_Senescent_x_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Senescent->x);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_Senescent_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_Fibroblast_Senescent_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Fibroblasts_Senescent->y),  thrust::device_pointer_cast(d_Fibroblasts_Senescent->y) + h_xmachine_memory_Fibroblast_Senescent_count);
}

float min_Fibroblast_Senescent_y_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Senescent->y);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_Senescent_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_Fibroblast_Senescent_y_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Senescent->y);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_Senescent_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_Fibroblast_Senescent_z_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Fibroblasts_Senescent->z),  thrust::device_pointer_cast(d_Fibroblasts_Senescent->z) + h_xmachine_memory_Fibroblast_Senescent_count);
}

float min_Fibroblast_Senescent_z_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Senescent->z);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_Senescent_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_Fibroblast_Senescent_z_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Senescent->z);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_Senescent_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_Fibroblast_Senescent_doublings_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Fibroblasts_Senescent->doublings),  thrust::device_pointer_cast(d_Fibroblasts_Senescent->doublings) + h_xmachine_memory_Fibroblast_Senescent_count);
}

float min_Fibroblast_Senescent_doublings_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Senescent->doublings);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_Senescent_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_Fibroblast_Senescent_doublings_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Senescent->doublings);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_Senescent_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_Fibroblast_Senescent_damage_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Fibroblasts_Senescent->damage),  thrust::device_pointer_cast(d_Fibroblasts_Senescent->damage) + h_xmachine_memory_Fibroblast_Senescent_count);
}

int count_Fibroblast_Senescent_damage_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Fibroblasts_Senescent->damage),  thrust::device_pointer_cast(d_Fibroblasts_Senescent->damage) + h_xmachine_memory_Fibroblast_Senescent_count, count_value);
}
int min_Fibroblast_Senescent_damage_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Senescent->damage);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_Senescent_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_Fibroblast_Senescent_damage_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Senescent->damage);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_Senescent_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_Fibroblast_Senescent_early_sen_time_counter_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Fibroblasts_Senescent->early_sen_time_counter),  thrust::device_pointer_cast(d_Fibroblasts_Senescent->early_sen_time_counter) + h_xmachine_memory_Fibroblast_Senescent_count);
}

int count_Fibroblast_Senescent_early_sen_time_counter_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Fibroblasts_Senescent->early_sen_time_counter),  thrust::device_pointer_cast(d_Fibroblasts_Senescent->early_sen_time_counter) + h_xmachine_memory_Fibroblast_Senescent_count, count_value);
}
int min_Fibroblast_Senescent_early_sen_time_counter_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Senescent->early_sen_time_counter);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_Senescent_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_Fibroblast_Senescent_early_sen_time_counter_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Senescent->early_sen_time_counter);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_Senescent_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_Fibroblast_Senescent_current_state_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Fibroblasts_Senescent->current_state),  thrust::device_pointer_cast(d_Fibroblasts_Senescent->current_state) + h_xmachine_memory_Fibroblast_Senescent_count);
}

int count_Fibroblast_Senescent_current_state_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Fibroblasts_Senescent->current_state),  thrust::device_pointer_cast(d_Fibroblasts_Senescent->current_state) + h_xmachine_memory_Fibroblast_Senescent_count, count_value);
}
int min_Fibroblast_Senescent_current_state_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Senescent->current_state);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_Senescent_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_Fibroblast_Senescent_current_state_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Senescent->current_state);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_Senescent_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_Fibroblast_Proliferating_id_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Fibroblasts_Proliferating->id),  thrust::device_pointer_cast(d_Fibroblasts_Proliferating->id) + h_xmachine_memory_Fibroblast_Proliferating_count);
}

int count_Fibroblast_Proliferating_id_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Fibroblasts_Proliferating->id),  thrust::device_pointer_cast(d_Fibroblasts_Proliferating->id) + h_xmachine_memory_Fibroblast_Proliferating_count, count_value);
}
int min_Fibroblast_Proliferating_id_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Proliferating->id);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_Proliferating_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_Fibroblast_Proliferating_id_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Proliferating->id);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_Proliferating_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_Fibroblast_Proliferating_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Fibroblasts_Proliferating->x),  thrust::device_pointer_cast(d_Fibroblasts_Proliferating->x) + h_xmachine_memory_Fibroblast_Proliferating_count);
}

float min_Fibroblast_Proliferating_x_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Proliferating->x);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_Proliferating_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_Fibroblast_Proliferating_x_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Proliferating->x);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_Proliferating_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_Fibroblast_Proliferating_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Fibroblasts_Proliferating->y),  thrust::device_pointer_cast(d_Fibroblasts_Proliferating->y) + h_xmachine_memory_Fibroblast_Proliferating_count);
}

float min_Fibroblast_Proliferating_y_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Proliferating->y);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_Proliferating_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_Fibroblast_Proliferating_y_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Proliferating->y);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_Proliferating_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_Fibroblast_Proliferating_z_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Fibroblasts_Proliferating->z),  thrust::device_pointer_cast(d_Fibroblasts_Proliferating->z) + h_xmachine_memory_Fibroblast_Proliferating_count);
}

float min_Fibroblast_Proliferating_z_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Proliferating->z);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_Proliferating_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_Fibroblast_Proliferating_z_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Proliferating->z);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_Proliferating_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_Fibroblast_Proliferating_doublings_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Fibroblasts_Proliferating->doublings),  thrust::device_pointer_cast(d_Fibroblasts_Proliferating->doublings) + h_xmachine_memory_Fibroblast_Proliferating_count);
}

float min_Fibroblast_Proliferating_doublings_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Proliferating->doublings);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_Proliferating_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_Fibroblast_Proliferating_doublings_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Proliferating->doublings);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_Proliferating_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_Fibroblast_Proliferating_damage_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Fibroblasts_Proliferating->damage),  thrust::device_pointer_cast(d_Fibroblasts_Proliferating->damage) + h_xmachine_memory_Fibroblast_Proliferating_count);
}

int count_Fibroblast_Proliferating_damage_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Fibroblasts_Proliferating->damage),  thrust::device_pointer_cast(d_Fibroblasts_Proliferating->damage) + h_xmachine_memory_Fibroblast_Proliferating_count, count_value);
}
int min_Fibroblast_Proliferating_damage_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Proliferating->damage);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_Proliferating_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_Fibroblast_Proliferating_damage_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Proliferating->damage);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_Proliferating_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_Fibroblast_Proliferating_early_sen_time_counter_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Fibroblasts_Proliferating->early_sen_time_counter),  thrust::device_pointer_cast(d_Fibroblasts_Proliferating->early_sen_time_counter) + h_xmachine_memory_Fibroblast_Proliferating_count);
}

int count_Fibroblast_Proliferating_early_sen_time_counter_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Fibroblasts_Proliferating->early_sen_time_counter),  thrust::device_pointer_cast(d_Fibroblasts_Proliferating->early_sen_time_counter) + h_xmachine_memory_Fibroblast_Proliferating_count, count_value);
}
int min_Fibroblast_Proliferating_early_sen_time_counter_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Proliferating->early_sen_time_counter);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_Proliferating_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_Fibroblast_Proliferating_early_sen_time_counter_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Proliferating->early_sen_time_counter);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_Proliferating_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_Fibroblast_Proliferating_current_state_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Fibroblasts_Proliferating->current_state),  thrust::device_pointer_cast(d_Fibroblasts_Proliferating->current_state) + h_xmachine_memory_Fibroblast_Proliferating_count);
}

int count_Fibroblast_Proliferating_current_state_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Fibroblasts_Proliferating->current_state),  thrust::device_pointer_cast(d_Fibroblasts_Proliferating->current_state) + h_xmachine_memory_Fibroblast_Proliferating_count, count_value);
}
int min_Fibroblast_Proliferating_current_state_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Proliferating->current_state);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_Proliferating_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_Fibroblast_Proliferating_current_state_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Proliferating->current_state);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_Proliferating_count) - thrust_ptr;
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
float reduce_Fibroblast_Repair_doublings_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Fibroblasts_Repair->doublings),  thrust::device_pointer_cast(d_Fibroblasts_Repair->doublings) + h_xmachine_memory_Fibroblast_Repair_count);
}

float min_Fibroblast_Repair_doublings_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Repair->doublings);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_Repair_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_Fibroblast_Repair_doublings_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Repair->doublings);
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
int reduce_Fibroblast_Repair_early_sen_time_counter_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Fibroblasts_Repair->early_sen_time_counter),  thrust::device_pointer_cast(d_Fibroblasts_Repair->early_sen_time_counter) + h_xmachine_memory_Fibroblast_Repair_count);
}

int count_Fibroblast_Repair_early_sen_time_counter_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Fibroblasts_Repair->early_sen_time_counter),  thrust::device_pointer_cast(d_Fibroblasts_Repair->early_sen_time_counter) + h_xmachine_memory_Fibroblast_Repair_count, count_value);
}
int min_Fibroblast_Repair_early_sen_time_counter_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Repair->early_sen_time_counter);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Fibroblast_Repair_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_Fibroblast_Repair_early_sen_time_counter_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Fibroblasts_Repair->early_sen_time_counter);
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

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_TissueTakesDamage, TissueBlock_TissueTakesDamage_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = TissueBlock_TissueTakesDamage_sm_size(blockSize);
	
	
	
	
	//MAIN XMACHINE FUNCTION CALL (TissueTakesDamage)
	//Reallocate   : false
	//Input        : 
	//Output       : 
	//Agent Output : 
	GPUFLAME_TissueTakesDamage<<<g, b, sm_size, stream>>>(d_TissueBlocks, d_rand48);
	gpuErrchkLaunch();
	
	
	
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
int TissueBlock_TissueSendDamageReport_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** TissueBlock_TissueSendDamageReport
 * Agent function prototype for TissueSendDamageReport function of TissueBlock agent
 */
void TissueBlock_TissueSendDamageReport(cudaStream_t &stream){

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
		printf("Error: Buffer size of tissue_damage_report message will be exceeded in function TissueSendDamageReport\n");
		exit(EXIT_FAILURE);
	}
	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_TissueSendDamageReport, TissueBlock_TissueSendDamageReport_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = TissueBlock_TissueSendDamageReport_sm_size(blockSize);
	
	
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	//Set the message_type for non partitioned, spatially partitioned and On-Graph Partitioned message outputs
	h_message_tissue_damage_report_output_type = single_message;
	gpuErrchk( cudaMemcpyToSymbol( d_message_tissue_damage_report_output_type, &h_message_tissue_damage_report_output_type, sizeof(int)));
	
	
	//MAIN XMACHINE FUNCTION CALL (TissueSendDamageReport)
	//Reallocate   : false
	//Input        : 
	//Output       : tissue_damage_report
	//Agent Output : 
	GPUFLAME_TissueSendDamageReport<<<g, b, sm_size, stream>>>(d_TissueBlocks, d_tissue_damage_reports);
	gpuErrchkLaunch();
	
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	h_message_tissue_damage_report_count += h_xmachine_memory_TissueBlock_count;
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
		printf("Error: Buffer size of TissueSendDamageReport agents in state default will be exceeded moving working agents to next state in function TissueSendDamageReport\n");
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
int TissueBlock_ReapirDamage_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input is spatially partitioned
	sm_size += (blockSize * sizeof(xmachine_message_fibroblast_location_report));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** TissueBlock_ReapirDamage
 * Agent function prototype for ReapirDamage function of TissueBlock agent
 */
void TissueBlock_ReapirDamage(cudaStream_t &stream){

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
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_ReapirDamage, TissueBlock_ReapirDamage_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = TissueBlock_ReapirDamage_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	//any agent with discrete or partitioned message input uses texture caching
	size_t tex_xmachine_message_fibroblast_location_report_id_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_fibroblast_location_report_id_byte_offset, tex_xmachine_message_fibroblast_location_report_id, d_fibroblast_location_reports->id, sizeof(int)*xmachine_message_fibroblast_location_report_MAX));
	h_tex_xmachine_message_fibroblast_location_report_id_offset = (int)tex_xmachine_message_fibroblast_location_report_id_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_fibroblast_location_report_id_offset, &h_tex_xmachine_message_fibroblast_location_report_id_offset, sizeof(int)));
	size_t tex_xmachine_message_fibroblast_location_report_x_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_fibroblast_location_report_x_byte_offset, tex_xmachine_message_fibroblast_location_report_x, d_fibroblast_location_reports->x, sizeof(float)*xmachine_message_fibroblast_location_report_MAX));
	h_tex_xmachine_message_fibroblast_location_report_x_offset = (int)tex_xmachine_message_fibroblast_location_report_x_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_fibroblast_location_report_x_offset, &h_tex_xmachine_message_fibroblast_location_report_x_offset, sizeof(int)));
	size_t tex_xmachine_message_fibroblast_location_report_y_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_fibroblast_location_report_y_byte_offset, tex_xmachine_message_fibroblast_location_report_y, d_fibroblast_location_reports->y, sizeof(float)*xmachine_message_fibroblast_location_report_MAX));
	h_tex_xmachine_message_fibroblast_location_report_y_offset = (int)tex_xmachine_message_fibroblast_location_report_y_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_fibroblast_location_report_y_offset, &h_tex_xmachine_message_fibroblast_location_report_y_offset, sizeof(int)));
	size_t tex_xmachine_message_fibroblast_location_report_z_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_fibroblast_location_report_z_byte_offset, tex_xmachine_message_fibroblast_location_report_z, d_fibroblast_location_reports->z, sizeof(float)*xmachine_message_fibroblast_location_report_MAX));
	h_tex_xmachine_message_fibroblast_location_report_z_offset = (int)tex_xmachine_message_fibroblast_location_report_z_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_fibroblast_location_report_z_offset, &h_tex_xmachine_message_fibroblast_location_report_z_offset, sizeof(int)));
	size_t tex_xmachine_message_fibroblast_location_report_current_state_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_fibroblast_location_report_current_state_byte_offset, tex_xmachine_message_fibroblast_location_report_current_state, d_fibroblast_location_reports->current_state, sizeof(int)*xmachine_message_fibroblast_location_report_MAX));
	h_tex_xmachine_message_fibroblast_location_report_current_state_offset = (int)tex_xmachine_message_fibroblast_location_report_current_state_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_fibroblast_location_report_current_state_offset, &h_tex_xmachine_message_fibroblast_location_report_current_state_offset, sizeof(int)));
	//bind pbm start and end indices to textures
	size_t tex_xmachine_message_fibroblast_location_report_pbm_start_byte_offset;
	size_t tex_xmachine_message_fibroblast_location_report_pbm_end_or_count_byte_offset;
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_fibroblast_location_report_pbm_start_byte_offset, tex_xmachine_message_fibroblast_location_report_pbm_start, d_fibroblast_location_report_partition_matrix->start, sizeof(int)*xmachine_message_fibroblast_location_report_grid_size));
	h_tex_xmachine_message_fibroblast_location_report_pbm_start_offset = (int)tex_xmachine_message_fibroblast_location_report_pbm_start_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_fibroblast_location_report_pbm_start_offset, &h_tex_xmachine_message_fibroblast_location_report_pbm_start_offset, sizeof(int)));
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_fibroblast_location_report_pbm_end_or_count_byte_offset, tex_xmachine_message_fibroblast_location_report_pbm_end_or_count, d_fibroblast_location_report_partition_matrix->end_or_count, sizeof(int)*xmachine_message_fibroblast_location_report_grid_size));
  h_tex_xmachine_message_fibroblast_location_report_pbm_end_or_count_offset = (int)tex_xmachine_message_fibroblast_location_report_pbm_end_or_count_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_fibroblast_location_report_pbm_end_or_count_offset, &h_tex_xmachine_message_fibroblast_location_report_pbm_end_or_count_offset, sizeof(int)));

	
	
	//MAIN XMACHINE FUNCTION CALL (ReapirDamage)
	//Reallocate   : false
	//Input        : fibroblast_location_report
	//Output       : 
	//Agent Output : 
	GPUFLAME_ReapirDamage<<<g, b, sm_size, stream>>>(d_TissueBlocks, d_fibroblast_location_reports, d_fibroblast_location_report_partition_matrix);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	//any agent with discrete or partitioned message input uses texture caching
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_fibroblast_location_report_id));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_fibroblast_location_report_x));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_fibroblast_location_report_y));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_fibroblast_location_report_z));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_fibroblast_location_report_current_state));
	//unbind pbm indices
    gpuErrchk( cudaUnbindTexture(tex_xmachine_message_fibroblast_location_report_pbm_start));
    gpuErrchk( cudaUnbindTexture(tex_xmachine_message_fibroblast_location_report_pbm_end_or_count));
    
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_TissueBlock_default_count+h_xmachine_memory_TissueBlock_count > xmachine_memory_TissueBlock_MAX){
		printf("Error: Buffer size of ReapirDamage agents in state default will be exceeded moving working agents to next state in function ReapirDamage\n");
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

	
	
	//MAIN XMACHINE FUNCTION CALL (QuiescentMigration)
	//Reallocate   : false
	//Input        : tissue_damage_report
	//Output       : 
	//Agent Output : 
	GPUFLAME_QuiescentMigration<<<g, b, sm_size, stream>>>(d_Fibroblasts, d_tissue_damage_reports, d_tissue_damage_report_partition_matrix);
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
int Fibroblast_SenescentMigration_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input is spatially partitioned
	sm_size += (blockSize * sizeof(xmachine_message_tissue_damage_report));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** Fibroblast_SenescentMigration
 * Agent function prototype for SenescentMigration function of Fibroblast agent
 */
void Fibroblast_SenescentMigration(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_Fibroblast_Senescent_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_Fibroblast_Senescent_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_Fibroblast_list* Fibroblasts_Senescent_temp = d_Fibroblasts;
	d_Fibroblasts = d_Fibroblasts_Senescent;
	d_Fibroblasts_Senescent = Fibroblasts_Senescent_temp;
	//set working count to current state count
	h_xmachine_memory_Fibroblast_count = h_xmachine_memory_Fibroblast_Senescent_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Fibroblast_count, &h_xmachine_memory_Fibroblast_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_Fibroblast_Senescent_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Fibroblast_Senescent_count, &h_xmachine_memory_Fibroblast_Senescent_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_SenescentMigration, Fibroblast_SenescentMigration_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = Fibroblast_SenescentMigration_sm_size(blockSize);
	
	
	
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

	
	
	//MAIN XMACHINE FUNCTION CALL (SenescentMigration)
	//Reallocate   : false
	//Input        : tissue_damage_report
	//Output       : 
	//Agent Output : 
	GPUFLAME_SenescentMigration<<<g, b, sm_size, stream>>>(d_Fibroblasts, d_tissue_damage_reports, d_tissue_damage_report_partition_matrix);
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
    
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_Fibroblast_Senescent_count+h_xmachine_memory_Fibroblast_count > xmachine_memory_Fibroblast_MAX){
		printf("Error: Buffer size of SenescentMigration agents in state Senescent will be exceeded moving working agents to next state in function SenescentMigration\n");
      exit(EXIT_FAILURE);
      }
      
  //pointer swap the updated data
  Fibroblasts_Senescent_temp = d_Fibroblasts;
  d_Fibroblasts = d_Fibroblasts_Senescent;
  d_Fibroblasts_Senescent = Fibroblasts_Senescent_temp;
        
	//update new state agent size
	h_xmachine_memory_Fibroblast_Senescent_count += h_xmachine_memory_Fibroblast_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Fibroblast_Senescent_count, &h_xmachine_memory_Fibroblast_Senescent_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int Fibroblast_EarlySenescentMigration_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input is spatially partitioned
	sm_size += (blockSize * sizeof(xmachine_message_tissue_damage_report));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** Fibroblast_EarlySenescentMigration
 * Agent function prototype for EarlySenescentMigration function of Fibroblast agent
 */
void Fibroblast_EarlySenescentMigration(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_Fibroblast_EarlySenescent_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_Fibroblast_EarlySenescent_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_Fibroblast_list* Fibroblasts_EarlySenescent_temp = d_Fibroblasts;
	d_Fibroblasts = d_Fibroblasts_EarlySenescent;
	d_Fibroblasts_EarlySenescent = Fibroblasts_EarlySenescent_temp;
	//set working count to current state count
	h_xmachine_memory_Fibroblast_count = h_xmachine_memory_Fibroblast_EarlySenescent_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Fibroblast_count, &h_xmachine_memory_Fibroblast_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_Fibroblast_EarlySenescent_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Fibroblast_EarlySenescent_count, &h_xmachine_memory_Fibroblast_EarlySenescent_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_EarlySenescentMigration, Fibroblast_EarlySenescentMigration_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = Fibroblast_EarlySenescentMigration_sm_size(blockSize);
	
	
	
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

	
	
	//MAIN XMACHINE FUNCTION CALL (EarlySenescentMigration)
	//Reallocate   : false
	//Input        : tissue_damage_report
	//Output       : 
	//Agent Output : 
	GPUFLAME_EarlySenescentMigration<<<g, b, sm_size, stream>>>(d_Fibroblasts, d_tissue_damage_reports, d_tissue_damage_report_partition_matrix);
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
    
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_Fibroblast_EarlySenescent_count+h_xmachine_memory_Fibroblast_count > xmachine_memory_Fibroblast_MAX){
		printf("Error: Buffer size of EarlySenescentMigration agents in state EarlySenescent will be exceeded moving working agents to next state in function EarlySenescentMigration\n");
      exit(EXIT_FAILURE);
      }
      
  //pointer swap the updated data
  Fibroblasts_EarlySenescent_temp = d_Fibroblasts;
  d_Fibroblasts = d_Fibroblasts_EarlySenescent;
  d_Fibroblasts_EarlySenescent = Fibroblasts_EarlySenescent_temp;
        
	//update new state agent size
	h_xmachine_memory_Fibroblast_EarlySenescent_count += h_xmachine_memory_Fibroblast_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Fibroblast_EarlySenescent_count, &h_xmachine_memory_Fibroblast_EarlySenescent_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int Fibroblast_QuiescentTakesDamage_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input is spatially partitioned
	sm_size += (blockSize * sizeof(xmachine_message_fibroblast_damage_report));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** Fibroblast_QuiescentTakesDamage
 * Agent function prototype for QuiescentTakesDamage function of Fibroblast agent
 */
void Fibroblast_QuiescentTakesDamage(cudaStream_t &stream){

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
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_QuiescentTakesDamage, Fibroblast_QuiescentTakesDamage_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = Fibroblast_QuiescentTakesDamage_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	//any agent with discrete or partitioned message input uses texture caching
	size_t tex_xmachine_message_fibroblast_damage_report_id_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_fibroblast_damage_report_id_byte_offset, tex_xmachine_message_fibroblast_damage_report_id, d_fibroblast_damage_reports->id, sizeof(int)*xmachine_message_fibroblast_damage_report_MAX));
	h_tex_xmachine_message_fibroblast_damage_report_id_offset = (int)tex_xmachine_message_fibroblast_damage_report_id_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_fibroblast_damage_report_id_offset, &h_tex_xmachine_message_fibroblast_damage_report_id_offset, sizeof(int)));
	size_t tex_xmachine_message_fibroblast_damage_report_x_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_fibroblast_damage_report_x_byte_offset, tex_xmachine_message_fibroblast_damage_report_x, d_fibroblast_damage_reports->x, sizeof(float)*xmachine_message_fibroblast_damage_report_MAX));
	h_tex_xmachine_message_fibroblast_damage_report_x_offset = (int)tex_xmachine_message_fibroblast_damage_report_x_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_fibroblast_damage_report_x_offset, &h_tex_xmachine_message_fibroblast_damage_report_x_offset, sizeof(int)));
	size_t tex_xmachine_message_fibroblast_damage_report_y_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_fibroblast_damage_report_y_byte_offset, tex_xmachine_message_fibroblast_damage_report_y, d_fibroblast_damage_reports->y, sizeof(float)*xmachine_message_fibroblast_damage_report_MAX));
	h_tex_xmachine_message_fibroblast_damage_report_y_offset = (int)tex_xmachine_message_fibroblast_damage_report_y_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_fibroblast_damage_report_y_offset, &h_tex_xmachine_message_fibroblast_damage_report_y_offset, sizeof(int)));
	size_t tex_xmachine_message_fibroblast_damage_report_z_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_fibroblast_damage_report_z_byte_offset, tex_xmachine_message_fibroblast_damage_report_z, d_fibroblast_damage_reports->z, sizeof(float)*xmachine_message_fibroblast_damage_report_MAX));
	h_tex_xmachine_message_fibroblast_damage_report_z_offset = (int)tex_xmachine_message_fibroblast_damage_report_z_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_fibroblast_damage_report_z_offset, &h_tex_xmachine_message_fibroblast_damage_report_z_offset, sizeof(int)));
	size_t tex_xmachine_message_fibroblast_damage_report_damage_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_fibroblast_damage_report_damage_byte_offset, tex_xmachine_message_fibroblast_damage_report_damage, d_fibroblast_damage_reports->damage, sizeof(int)*xmachine_message_fibroblast_damage_report_MAX));
	h_tex_xmachine_message_fibroblast_damage_report_damage_offset = (int)tex_xmachine_message_fibroblast_damage_report_damage_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_fibroblast_damage_report_damage_offset, &h_tex_xmachine_message_fibroblast_damage_report_damage_offset, sizeof(int)));
	//bind pbm start and end indices to textures
	size_t tex_xmachine_message_fibroblast_damage_report_pbm_start_byte_offset;
	size_t tex_xmachine_message_fibroblast_damage_report_pbm_end_or_count_byte_offset;
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_fibroblast_damage_report_pbm_start_byte_offset, tex_xmachine_message_fibroblast_damage_report_pbm_start, d_fibroblast_damage_report_partition_matrix->start, sizeof(int)*xmachine_message_fibroblast_damage_report_grid_size));
	h_tex_xmachine_message_fibroblast_damage_report_pbm_start_offset = (int)tex_xmachine_message_fibroblast_damage_report_pbm_start_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_fibroblast_damage_report_pbm_start_offset, &h_tex_xmachine_message_fibroblast_damage_report_pbm_start_offset, sizeof(int)));
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_fibroblast_damage_report_pbm_end_or_count_byte_offset, tex_xmachine_message_fibroblast_damage_report_pbm_end_or_count, d_fibroblast_damage_report_partition_matrix->end_or_count, sizeof(int)*xmachine_message_fibroblast_damage_report_grid_size));
  h_tex_xmachine_message_fibroblast_damage_report_pbm_end_or_count_offset = (int)tex_xmachine_message_fibroblast_damage_report_pbm_end_or_count_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_fibroblast_damage_report_pbm_end_or_count_offset, &h_tex_xmachine_message_fibroblast_damage_report_pbm_end_or_count_offset, sizeof(int)));

	
	
	//MAIN XMACHINE FUNCTION CALL (QuiescentTakesDamage)
	//Reallocate   : false
	//Input        : fibroblast_damage_report
	//Output       : 
	//Agent Output : 
	GPUFLAME_QuiescentTakesDamage<<<g, b, sm_size, stream>>>(d_Fibroblasts, d_fibroblast_damage_reports, d_fibroblast_damage_report_partition_matrix, d_rand48);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	//any agent with discrete or partitioned message input uses texture caching
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_fibroblast_damage_report_id));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_fibroblast_damage_report_x));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_fibroblast_damage_report_y));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_fibroblast_damage_report_z));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_fibroblast_damage_report_damage));
	//unbind pbm indices
    gpuErrchk( cudaUnbindTexture(tex_xmachine_message_fibroblast_damage_report_pbm_start));
    gpuErrchk( cudaUnbindTexture(tex_xmachine_message_fibroblast_damage_report_pbm_end_or_count));
    
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_Fibroblast_Quiescent_count+h_xmachine_memory_Fibroblast_count > xmachine_memory_Fibroblast_MAX){
		printf("Error: Buffer size of QuiescentTakesDamage agents in state Quiescent will be exceeded moving working agents to next state in function QuiescentTakesDamage\n");
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
int Fibroblast_QuiescentSendDamageReport_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** Fibroblast_QuiescentSendDamageReport
 * Agent function prototype for QuiescentSendDamageReport function of Fibroblast agent
 */
void Fibroblast_QuiescentSendDamageReport(cudaStream_t &stream){

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
	if (h_message_fibroblast_damage_report_count + h_xmachine_memory_Fibroblast_count > xmachine_message_fibroblast_damage_report_MAX){
		printf("Error: Buffer size of fibroblast_damage_report message will be exceeded in function QuiescentSendDamageReport\n");
		exit(EXIT_FAILURE);
	}
	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_QuiescentSendDamageReport, Fibroblast_QuiescentSendDamageReport_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = Fibroblast_QuiescentSendDamageReport_sm_size(blockSize);
	
	
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	//Set the message_type for non partitioned, spatially partitioned and On-Graph Partitioned message outputs
	h_message_fibroblast_damage_report_output_type = single_message;
	gpuErrchk( cudaMemcpyToSymbol( d_message_fibroblast_damage_report_output_type, &h_message_fibroblast_damage_report_output_type, sizeof(int)));
	
	
	//MAIN XMACHINE FUNCTION CALL (QuiescentSendDamageReport)
	//Reallocate   : false
	//Input        : 
	//Output       : fibroblast_damage_report
	//Agent Output : 
	GPUFLAME_QuiescentSendDamageReport<<<g, b, sm_size, stream>>>(d_Fibroblasts, d_fibroblast_damage_reports, d_rand48);
	gpuErrchkLaunch();
	
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	h_message_fibroblast_damage_report_count += h_xmachine_memory_Fibroblast_count;
	//Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_message_fibroblast_damage_report_count, &h_message_fibroblast_damage_report_count, sizeof(int)));	
	
	//reset partition matrix
	gpuErrchk( cudaMemset( (void*) d_fibroblast_damage_report_partition_matrix, 0, sizeof(xmachine_message_fibroblast_damage_report_PBM)));
    //PR Bug fix: Second fix. This should prevent future problems when multiple agents write the same message as now the message structure is completely rebuilt after an output.
    if (h_message_fibroblast_damage_report_count > 0){
#ifdef FAST_ATOMIC_SORTING
      //USE ATOMICS TO BUILD PARTITION BOUNDARY
	  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, hist_fibroblast_damage_report_messages, no_sm, h_message_fibroblast_damage_report_count); 
	  gridSize = (h_message_fibroblast_damage_report_count + blockSize - 1) / blockSize;
	  hist_fibroblast_damage_report_messages<<<gridSize, blockSize, 0, stream>>>(d_xmachine_message_fibroblast_damage_report_local_bin_index, d_xmachine_message_fibroblast_damage_report_unsorted_index, d_fibroblast_damage_report_partition_matrix->end_or_count, d_fibroblast_damage_reports, h_message_fibroblast_damage_report_count);
	  gpuErrchkLaunch();
	
      // Scan
      cub::DeviceScan::ExclusiveSum(
          d_temp_scan_storage_xmachine_message_fibroblast_damage_report, 
          temp_scan_bytes_xmachine_message_fibroblast_damage_report, 
          d_fibroblast_damage_report_partition_matrix->end_or_count,
          d_fibroblast_damage_report_partition_matrix->start,
          xmachine_message_fibroblast_damage_report_grid_size, 
          stream
      );
	
	  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_fibroblast_damage_report_messages, no_sm, h_message_fibroblast_damage_report_count); 
	  gridSize = (h_message_fibroblast_damage_report_count + blockSize - 1) / blockSize; 	// Round up according to array size 
	  reorder_fibroblast_damage_report_messages <<<gridSize, blockSize, 0, stream>>>(d_xmachine_message_fibroblast_damage_report_local_bin_index, d_xmachine_message_fibroblast_damage_report_unsorted_index, d_fibroblast_damage_report_partition_matrix->start, d_fibroblast_damage_reports, d_fibroblast_damage_reports_swap, h_message_fibroblast_damage_report_count);
	  gpuErrchkLaunch();
#else
	  //HASH, SORT, REORDER AND BUILD PMB FOR SPATIAL PARTITIONING MESSAGE OUTPUTS
	  //Get message hash values for sorting
	  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, hash_fibroblast_damage_report_messages, no_sm, h_message_fibroblast_damage_report_count); 
	  gridSize = (h_message_fibroblast_damage_report_count + blockSize - 1) / blockSize;
	  hash_fibroblast_damage_report_messages<<<gridSize, blockSize, 0, stream>>>(d_xmachine_message_fibroblast_damage_report_keys, d_xmachine_message_fibroblast_damage_report_values, d_fibroblast_damage_reports);
	  gpuErrchkLaunch();
	  //Sort
	  thrust::sort_by_key(thrust::cuda::par.on(stream), thrust::device_pointer_cast(d_xmachine_message_fibroblast_damage_report_keys),  thrust::device_pointer_cast(d_xmachine_message_fibroblast_damage_report_keys) + h_message_fibroblast_damage_report_count,  thrust::device_pointer_cast(d_xmachine_message_fibroblast_damage_report_values));
	  gpuErrchkLaunch();
	  //reorder and build pcb
	  gpuErrchk(cudaMemset(d_fibroblast_damage_report_partition_matrix->start, 0xffffffff, xmachine_message_fibroblast_damage_report_grid_size* sizeof(int)));
	  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_fibroblast_damage_report_messages, reorder_messages_sm_size, h_message_fibroblast_damage_report_count); 
	  gridSize = (h_message_fibroblast_damage_report_count + blockSize - 1) / blockSize;
	  int reorder_sm_size = reorder_messages_sm_size(blockSize);
	  reorder_fibroblast_damage_report_messages<<<gridSize, blockSize, reorder_sm_size, stream>>>(d_xmachine_message_fibroblast_damage_report_keys, d_xmachine_message_fibroblast_damage_report_values, d_fibroblast_damage_report_partition_matrix, d_fibroblast_damage_reports, d_fibroblast_damage_reports_swap);
	  gpuErrchkLaunch();
#endif
  }
	//swap ordered list
	xmachine_message_fibroblast_damage_report_list* d_fibroblast_damage_reports_temp = d_fibroblast_damage_reports;
	d_fibroblast_damage_reports = d_fibroblast_damage_reports_swap;
	d_fibroblast_damage_reports_swap = d_fibroblast_damage_reports_temp;
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_Fibroblast_Quiescent_count+h_xmachine_memory_Fibroblast_count > xmachine_memory_Fibroblast_MAX){
		printf("Error: Buffer size of QuiescentSendDamageReport agents in state Quiescent will be exceeded moving working agents to next state in function QuiescentSendDamageReport\n");
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
int Fibroblast_TransitionToProliferating_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** Fibroblast_TransitionToProliferating
 * Agent function prototype for TransitionToProliferating function of Fibroblast agent
 */
void Fibroblast_TransitionToProliferating(cudaStream_t &stream){

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
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_TransitionToProliferating, Fibroblast_TransitionToProliferating_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = Fibroblast_TransitionToProliferating_sm_size(blockSize);
	
	
	
	
	//MAIN XMACHINE FUNCTION CALL (TransitionToProliferating)
	//Reallocate   : false
	//Input        : 
	//Output       : 
	//Agent Output : 
	GPUFLAME_TransitionToProliferating<<<g, b, sm_size, stream>>>(d_Fibroblasts, d_rand48);
	gpuErrchkLaunch();
	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_Fibroblast_Proliferating_count+h_xmachine_memory_Fibroblast_count > xmachine_memory_Fibroblast_MAX){
		printf("Error: Buffer size of TransitionToProliferating agents in state Proliferating will be exceeded moving working agents to next state in function TransitionToProliferating\n");
      exit(EXIT_FAILURE);
      }
      
  //append agents to next state list
  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_Fibroblast_Agents, no_sm, state_list_size);
  gridSize = (state_list_size + blockSize - 1) / blockSize;
  append_Fibroblast_Agents<<<gridSize, blockSize, 0, stream>>>(d_Fibroblasts_Proliferating, d_Fibroblasts, h_xmachine_memory_Fibroblast_Proliferating_count, h_xmachine_memory_Fibroblast_count);
  gpuErrchkLaunch();
        
	//update new state agent size
	h_xmachine_memory_Fibroblast_Proliferating_count += h_xmachine_memory_Fibroblast_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Fibroblast_Proliferating_count, &h_xmachine_memory_Fibroblast_Proliferating_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int Fibroblast_Proliferation_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** Fibroblast_Proliferation
 * Agent function prototype for Proliferation function of Fibroblast agent
 */
void Fibroblast_Proliferation(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_Fibroblast_Proliferating_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_Fibroblast_Proliferating_count;

	
	//FOR Fibroblast AGENT OUTPUT, RESET THE AGENT NEW LIST SCAN INPUT
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_Fibroblast_scan_input, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_Fibroblast_scan_input<<<gridSize, blockSize, 0, stream>>>(d_Fibroblasts_new);
	gpuErrchkLaunch();
	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_Fibroblast_list* Fibroblasts_Proliferating_temp = d_Fibroblasts;
	d_Fibroblasts = d_Fibroblasts_Proliferating;
	d_Fibroblasts_Proliferating = Fibroblasts_Proliferating_temp;
	//set working count to current state count
	h_xmachine_memory_Fibroblast_count = h_xmachine_memory_Fibroblast_Proliferating_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Fibroblast_count, &h_xmachine_memory_Fibroblast_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_Fibroblast_Proliferating_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Fibroblast_Proliferating_count, &h_xmachine_memory_Fibroblast_Proliferating_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_Proliferation, Fibroblast_Proliferation_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = Fibroblast_Proliferation_sm_size(blockSize);
	
	
	
	//IF CONTINUOUS AGENT CAN REALLOCATE (process dead agents) THEN RESET AGENT SWAPS	
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_Fibroblast_scan_input, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_Fibroblast_scan_input<<<gridSize, blockSize, 0, stream>>>(d_Fibroblasts);
	gpuErrchkLaunch();
	
	
	//MAIN XMACHINE FUNCTION CALL (Proliferation)
	//Reallocate   : true
	//Input        : 
	//Output       : 
	//Agent Output : Fibroblast
	GPUFLAME_Proliferation<<<g, b, sm_size, stream>>>(d_Fibroblasts, d_Fibroblasts_new);
	gpuErrchkLaunch();
	
	
    //COPY ANY AGENT COUNT BEFORE Fibroblast AGENTS ARE KILLED (needed for scatter)
	int Fibroblasts_pre_death_count = h_xmachine_memory_Fibroblast_count;
	
	//FOR CONTINUOUS AGENTS WITH REALLOCATION REMOVE POSSIBLE DEAD AGENTS	
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_Fibroblast, 
        temp_scan_storage_bytes_Fibroblast, 
        d_Fibroblasts->_scan_input,
        d_Fibroblasts->_position,
        h_xmachine_memory_Fibroblast_count, 
        stream
    );

	//Scatter into swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_Fibroblast_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	scatter_Fibroblast_Agents<<<gridSize, blockSize, 0, stream>>>(d_Fibroblasts_swap, d_Fibroblasts, 0, h_xmachine_memory_Fibroblast_count);
	gpuErrchkLaunch();
	//use a temp pointer to make swap default
	xmachine_memory_Fibroblast_list* Proliferation_Fibroblasts_temp = d_Fibroblasts;
	d_Fibroblasts = d_Fibroblasts_swap;
	d_Fibroblasts_swap = Proliferation_Fibroblasts_temp;
	//reset agent count
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_Fibroblasts_swap->_position[h_xmachine_memory_Fibroblast_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_Fibroblasts_swap->_scan_input[h_xmachine_memory_Fibroblast_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	if (scan_last_included == 1)
		h_xmachine_memory_Fibroblast_count = scan_last_sum+1;
	else
		h_xmachine_memory_Fibroblast_count = scan_last_sum;
	//Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Fibroblast_count, &h_xmachine_memory_Fibroblast_count, sizeof(int)));	
	
	//FOR Fibroblast AGENT OUTPUT SCATTER AGENTS 

    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_Fibroblast, 
        temp_scan_storage_bytes_Fibroblast, 
        d_Fibroblasts_new->_scan_input, 
        d_Fibroblasts_new->_position, 
        Fibroblasts_pre_death_count,
        stream
    );

	//reset agent count
	int Fibroblast_after_birth_count;
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_Fibroblasts_new->_position[Fibroblasts_pre_death_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_Fibroblasts_new->_scan_input[Fibroblasts_pre_death_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	if (scan_last_included == 1)
		Fibroblast_after_birth_count = h_xmachine_memory_Fibroblast_Quiescent_count + scan_last_sum+1;
	else
		Fibroblast_after_birth_count = h_xmachine_memory_Fibroblast_Quiescent_count + scan_last_sum;
	//check buffer is not exceeded
	if (Fibroblast_after_birth_count > xmachine_memory_Fibroblast_MAX){
		printf("Error: Buffer size of Fibroblast agents in state Quiescent will be exceeded writing new agents in function Proliferation\n");
		exit(EXIT_FAILURE);
	}
	//Scatter into swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_Fibroblast_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	scatter_Fibroblast_Agents<<<gridSize, blockSize, 0, stream>>>(d_Fibroblasts_Quiescent, d_Fibroblasts_new, h_xmachine_memory_Fibroblast_Quiescent_count, Fibroblasts_pre_death_count);
	gpuErrchkLaunch();
	//Copy count to device
	h_xmachine_memory_Fibroblast_Quiescent_count = Fibroblast_after_birth_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Fibroblast_Quiescent_count, &h_xmachine_memory_Fibroblast_Quiescent_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_Fibroblast_Quiescent_count+h_xmachine_memory_Fibroblast_count > xmachine_memory_Fibroblast_MAX){
		printf("Error: Buffer size of Proliferation agents in state Quiescent will be exceeded moving working agents to next state in function Proliferation\n");
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



	
/* Shared memory size calculator for agent function */
int Fibroblast_BystanderEffect_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input is spatially partitioned
	sm_size += (blockSize * sizeof(xmachine_message_fibroblast_location_report));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** Fibroblast_BystanderEffect
 * Agent function prototype for BystanderEffect function of Fibroblast agent
 */
void Fibroblast_BystanderEffect(cudaStream_t &stream){

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
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_BystanderEffect, Fibroblast_BystanderEffect_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = Fibroblast_BystanderEffect_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	//any agent with discrete or partitioned message input uses texture caching
	size_t tex_xmachine_message_fibroblast_location_report_id_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_fibroblast_location_report_id_byte_offset, tex_xmachine_message_fibroblast_location_report_id, d_fibroblast_location_reports->id, sizeof(int)*xmachine_message_fibroblast_location_report_MAX));
	h_tex_xmachine_message_fibroblast_location_report_id_offset = (int)tex_xmachine_message_fibroblast_location_report_id_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_fibroblast_location_report_id_offset, &h_tex_xmachine_message_fibroblast_location_report_id_offset, sizeof(int)));
	size_t tex_xmachine_message_fibroblast_location_report_x_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_fibroblast_location_report_x_byte_offset, tex_xmachine_message_fibroblast_location_report_x, d_fibroblast_location_reports->x, sizeof(float)*xmachine_message_fibroblast_location_report_MAX));
	h_tex_xmachine_message_fibroblast_location_report_x_offset = (int)tex_xmachine_message_fibroblast_location_report_x_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_fibroblast_location_report_x_offset, &h_tex_xmachine_message_fibroblast_location_report_x_offset, sizeof(int)));
	size_t tex_xmachine_message_fibroblast_location_report_y_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_fibroblast_location_report_y_byte_offset, tex_xmachine_message_fibroblast_location_report_y, d_fibroblast_location_reports->y, sizeof(float)*xmachine_message_fibroblast_location_report_MAX));
	h_tex_xmachine_message_fibroblast_location_report_y_offset = (int)tex_xmachine_message_fibroblast_location_report_y_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_fibroblast_location_report_y_offset, &h_tex_xmachine_message_fibroblast_location_report_y_offset, sizeof(int)));
	size_t tex_xmachine_message_fibroblast_location_report_z_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_fibroblast_location_report_z_byte_offset, tex_xmachine_message_fibroblast_location_report_z, d_fibroblast_location_reports->z, sizeof(float)*xmachine_message_fibroblast_location_report_MAX));
	h_tex_xmachine_message_fibroblast_location_report_z_offset = (int)tex_xmachine_message_fibroblast_location_report_z_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_fibroblast_location_report_z_offset, &h_tex_xmachine_message_fibroblast_location_report_z_offset, sizeof(int)));
	size_t tex_xmachine_message_fibroblast_location_report_current_state_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_fibroblast_location_report_current_state_byte_offset, tex_xmachine_message_fibroblast_location_report_current_state, d_fibroblast_location_reports->current_state, sizeof(int)*xmachine_message_fibroblast_location_report_MAX));
	h_tex_xmachine_message_fibroblast_location_report_current_state_offset = (int)tex_xmachine_message_fibroblast_location_report_current_state_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_fibroblast_location_report_current_state_offset, &h_tex_xmachine_message_fibroblast_location_report_current_state_offset, sizeof(int)));
	//bind pbm start and end indices to textures
	size_t tex_xmachine_message_fibroblast_location_report_pbm_start_byte_offset;
	size_t tex_xmachine_message_fibroblast_location_report_pbm_end_or_count_byte_offset;
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_fibroblast_location_report_pbm_start_byte_offset, tex_xmachine_message_fibroblast_location_report_pbm_start, d_fibroblast_location_report_partition_matrix->start, sizeof(int)*xmachine_message_fibroblast_location_report_grid_size));
	h_tex_xmachine_message_fibroblast_location_report_pbm_start_offset = (int)tex_xmachine_message_fibroblast_location_report_pbm_start_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_fibroblast_location_report_pbm_start_offset, &h_tex_xmachine_message_fibroblast_location_report_pbm_start_offset, sizeof(int)));
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_fibroblast_location_report_pbm_end_or_count_byte_offset, tex_xmachine_message_fibroblast_location_report_pbm_end_or_count, d_fibroblast_location_report_partition_matrix->end_or_count, sizeof(int)*xmachine_message_fibroblast_location_report_grid_size));
  h_tex_xmachine_message_fibroblast_location_report_pbm_end_or_count_offset = (int)tex_xmachine_message_fibroblast_location_report_pbm_end_or_count_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_fibroblast_location_report_pbm_end_or_count_offset, &h_tex_xmachine_message_fibroblast_location_report_pbm_end_or_count_offset, sizeof(int)));

	
	
	//MAIN XMACHINE FUNCTION CALL (BystanderEffect)
	//Reallocate   : false
	//Input        : fibroblast_location_report
	//Output       : 
	//Agent Output : 
	GPUFLAME_BystanderEffect<<<g, b, sm_size, stream>>>(d_Fibroblasts, d_fibroblast_location_reports, d_fibroblast_location_report_partition_matrix, d_rand48);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	//any agent with discrete or partitioned message input uses texture caching
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_fibroblast_location_report_id));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_fibroblast_location_report_x));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_fibroblast_location_report_y));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_fibroblast_location_report_z));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_fibroblast_location_report_current_state));
	//unbind pbm indices
    gpuErrchk( cudaUnbindTexture(tex_xmachine_message_fibroblast_location_report_pbm_start));
    gpuErrchk( cudaUnbindTexture(tex_xmachine_message_fibroblast_location_report_pbm_end_or_count));
    
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_Fibroblast_EarlySenescent_count+h_xmachine_memory_Fibroblast_count > xmachine_memory_Fibroblast_MAX){
		printf("Error: Buffer size of BystanderEffect agents in state EarlySenescent will be exceeded moving working agents to next state in function BystanderEffect\n");
      exit(EXIT_FAILURE);
      }
      
  //append agents to next state list
  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_Fibroblast_Agents, no_sm, state_list_size);
  gridSize = (state_list_size + blockSize - 1) / blockSize;
  append_Fibroblast_Agents<<<gridSize, blockSize, 0, stream>>>(d_Fibroblasts_EarlySenescent, d_Fibroblasts, h_xmachine_memory_Fibroblast_EarlySenescent_count, h_xmachine_memory_Fibroblast_count);
  gpuErrchkLaunch();
        
	//update new state agent size
	h_xmachine_memory_Fibroblast_EarlySenescent_count += h_xmachine_memory_Fibroblast_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Fibroblast_EarlySenescent_count, &h_xmachine_memory_Fibroblast_EarlySenescent_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int Fibroblast_ExcessiveDamage_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** Fibroblast_ExcessiveDamage
 * Agent function prototype for ExcessiveDamage function of Fibroblast agent
 */
void Fibroblast_ExcessiveDamage(cudaStream_t &stream){

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
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_ExcessiveDamage, Fibroblast_ExcessiveDamage_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = Fibroblast_ExcessiveDamage_sm_size(blockSize);
	
	
	
	
	//MAIN XMACHINE FUNCTION CALL (ExcessiveDamage)
	//Reallocate   : false
	//Input        : 
	//Output       : 
	//Agent Output : 
	GPUFLAME_ExcessiveDamage<<<g, b, sm_size, stream>>>(d_Fibroblasts, d_rand48);
	gpuErrchkLaunch();
	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_Fibroblast_EarlySenescent_count+h_xmachine_memory_Fibroblast_count > xmachine_memory_Fibroblast_MAX){
		printf("Error: Buffer size of ExcessiveDamage agents in state EarlySenescent will be exceeded moving working agents to next state in function ExcessiveDamage\n");
      exit(EXIT_FAILURE);
      }
      
  //append agents to next state list
  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_Fibroblast_Agents, no_sm, state_list_size);
  gridSize = (state_list_size + blockSize - 1) / blockSize;
  append_Fibroblast_Agents<<<gridSize, blockSize, 0, stream>>>(d_Fibroblasts_EarlySenescent, d_Fibroblasts, h_xmachine_memory_Fibroblast_EarlySenescent_count, h_xmachine_memory_Fibroblast_count);
  gpuErrchkLaunch();
        
	//update new state agent size
	h_xmachine_memory_Fibroblast_EarlySenescent_count += h_xmachine_memory_Fibroblast_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Fibroblast_EarlySenescent_count, &h_xmachine_memory_Fibroblast_EarlySenescent_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int Fibroblast_ReplicativeSenescence_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** Fibroblast_ReplicativeSenescence
 * Agent function prototype for ReplicativeSenescence function of Fibroblast agent
 */
void Fibroblast_ReplicativeSenescence(cudaStream_t &stream){

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
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_ReplicativeSenescence, Fibroblast_ReplicativeSenescence_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = Fibroblast_ReplicativeSenescence_sm_size(blockSize);
	
	
	
	
	//MAIN XMACHINE FUNCTION CALL (ReplicativeSenescence)
	//Reallocate   : false
	//Input        : 
	//Output       : 
	//Agent Output : 
	GPUFLAME_ReplicativeSenescence<<<g, b, sm_size, stream>>>(d_Fibroblasts, d_rand48);
	gpuErrchkLaunch();
	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_Fibroblast_EarlySenescent_count+h_xmachine_memory_Fibroblast_count > xmachine_memory_Fibroblast_MAX){
		printf("Error: Buffer size of ReplicativeSenescence agents in state EarlySenescent will be exceeded moving working agents to next state in function ReplicativeSenescence\n");
      exit(EXIT_FAILURE);
      }
      
  //append agents to next state list
  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_Fibroblast_Agents, no_sm, state_list_size);
  gridSize = (state_list_size + blockSize - 1) / blockSize;
  append_Fibroblast_Agents<<<gridSize, blockSize, 0, stream>>>(d_Fibroblasts_EarlySenescent, d_Fibroblasts, h_xmachine_memory_Fibroblast_EarlySenescent_count, h_xmachine_memory_Fibroblast_count);
  gpuErrchkLaunch();
        
	//update new state agent size
	h_xmachine_memory_Fibroblast_EarlySenescent_count += h_xmachine_memory_Fibroblast_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Fibroblast_EarlySenescent_count, &h_xmachine_memory_Fibroblast_EarlySenescent_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int Fibroblast_EarlySenCountTime_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** Fibroblast_EarlySenCountTime
 * Agent function prototype for EarlySenCountTime function of Fibroblast agent
 */
void Fibroblast_EarlySenCountTime(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_Fibroblast_EarlySenescent_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_Fibroblast_EarlySenescent_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_Fibroblast_list* Fibroblasts_EarlySenescent_temp = d_Fibroblasts;
	d_Fibroblasts = d_Fibroblasts_EarlySenescent;
	d_Fibroblasts_EarlySenescent = Fibroblasts_EarlySenescent_temp;
	//set working count to current state count
	h_xmachine_memory_Fibroblast_count = h_xmachine_memory_Fibroblast_EarlySenescent_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Fibroblast_count, &h_xmachine_memory_Fibroblast_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_Fibroblast_EarlySenescent_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Fibroblast_EarlySenescent_count, &h_xmachine_memory_Fibroblast_EarlySenescent_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_EarlySenCountTime, Fibroblast_EarlySenCountTime_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = Fibroblast_EarlySenCountTime_sm_size(blockSize);
	
	
	
	
	//MAIN XMACHINE FUNCTION CALL (EarlySenCountTime)
	//Reallocate   : false
	//Input        : 
	//Output       : 
	//Agent Output : 
	GPUFLAME_EarlySenCountTime<<<g, b, sm_size, stream>>>(d_Fibroblasts);
	gpuErrchkLaunch();
	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_Fibroblast_EarlySenescent_count+h_xmachine_memory_Fibroblast_count > xmachine_memory_Fibroblast_MAX){
		printf("Error: Buffer size of EarlySenCountTime agents in state EarlySenescent will be exceeded moving working agents to next state in function EarlySenCountTime\n");
      exit(EXIT_FAILURE);
      }
      
  //pointer swap the updated data
  Fibroblasts_EarlySenescent_temp = d_Fibroblasts;
  d_Fibroblasts = d_Fibroblasts_EarlySenescent;
  d_Fibroblasts_EarlySenescent = Fibroblasts_EarlySenescent_temp;
        
	//update new state agent size
	h_xmachine_memory_Fibroblast_EarlySenescent_count += h_xmachine_memory_Fibroblast_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Fibroblast_EarlySenescent_count, &h_xmachine_memory_Fibroblast_EarlySenescent_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int Fibroblast_TransitionToFullSenescence_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** Fibroblast_TransitionToFullSenescence
 * Agent function prototype for TransitionToFullSenescence function of Fibroblast agent
 */
void Fibroblast_TransitionToFullSenescence(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_Fibroblast_EarlySenescent_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_Fibroblast_EarlySenescent_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_Fibroblast_list* Fibroblasts_EarlySenescent_temp = d_Fibroblasts;
	d_Fibroblasts = d_Fibroblasts_EarlySenescent;
	d_Fibroblasts_EarlySenescent = Fibroblasts_EarlySenescent_temp;
	//set working count to current state count
	h_xmachine_memory_Fibroblast_count = h_xmachine_memory_Fibroblast_EarlySenescent_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Fibroblast_count, &h_xmachine_memory_Fibroblast_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_Fibroblast_EarlySenescent_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Fibroblast_EarlySenescent_count, &h_xmachine_memory_Fibroblast_EarlySenescent_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_TransitionToFullSenescence, Fibroblast_TransitionToFullSenescence_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = Fibroblast_TransitionToFullSenescence_sm_size(blockSize);
	
	
	
	
	//MAIN XMACHINE FUNCTION CALL (TransitionToFullSenescence)
	//Reallocate   : false
	//Input        : 
	//Output       : 
	//Agent Output : 
	GPUFLAME_TransitionToFullSenescence<<<g, b, sm_size, stream>>>(d_Fibroblasts, d_rand48);
	gpuErrchkLaunch();
	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_Fibroblast_Senescent_count+h_xmachine_memory_Fibroblast_count > xmachine_memory_Fibroblast_MAX){
		printf("Error: Buffer size of TransitionToFullSenescence agents in state Senescent will be exceeded moving working agents to next state in function TransitionToFullSenescence\n");
      exit(EXIT_FAILURE);
      }
      
  //append agents to next state list
  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_Fibroblast_Agents, no_sm, state_list_size);
  gridSize = (state_list_size + blockSize - 1) / blockSize;
  append_Fibroblast_Agents<<<gridSize, blockSize, 0, stream>>>(d_Fibroblasts_Senescent, d_Fibroblasts, h_xmachine_memory_Fibroblast_Senescent_count, h_xmachine_memory_Fibroblast_count);
  gpuErrchkLaunch();
        
	//update new state agent size
	h_xmachine_memory_Fibroblast_Senescent_count += h_xmachine_memory_Fibroblast_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Fibroblast_Senescent_count, &h_xmachine_memory_Fibroblast_Senescent_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int Fibroblast_ClearanceOfEarlySenescent_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** Fibroblast_ClearanceOfEarlySenescent
 * Agent function prototype for ClearanceOfEarlySenescent function of Fibroblast agent
 */
void Fibroblast_ClearanceOfEarlySenescent(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_Fibroblast_EarlySenescent_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_Fibroblast_EarlySenescent_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_Fibroblast_list* Fibroblasts_EarlySenescent_temp = d_Fibroblasts;
	d_Fibroblasts = d_Fibroblasts_EarlySenescent;
	d_Fibroblasts_EarlySenescent = Fibroblasts_EarlySenescent_temp;
	//set working count to current state count
	h_xmachine_memory_Fibroblast_count = h_xmachine_memory_Fibroblast_EarlySenescent_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Fibroblast_count, &h_xmachine_memory_Fibroblast_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_Fibroblast_EarlySenescent_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Fibroblast_EarlySenescent_count, &h_xmachine_memory_Fibroblast_EarlySenescent_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_ClearanceOfEarlySenescent, Fibroblast_ClearanceOfEarlySenescent_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = Fibroblast_ClearanceOfEarlySenescent_sm_size(blockSize);
	
	
	
	//IF CONTINUOUS AGENT CAN REALLOCATE (process dead agents) THEN RESET AGENT SWAPS	
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_Fibroblast_scan_input, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_Fibroblast_scan_input<<<gridSize, blockSize, 0, stream>>>(d_Fibroblasts);
	gpuErrchkLaunch();
	
	
	//MAIN XMACHINE FUNCTION CALL (ClearanceOfEarlySenescent)
	//Reallocate   : true
	//Input        : 
	//Output       : 
	//Agent Output : 
	GPUFLAME_ClearanceOfEarlySenescent<<<g, b, sm_size, stream>>>(d_Fibroblasts, d_rand48);
	gpuErrchkLaunch();
	
	
	//FOR CONTINUOUS AGENTS WITH REALLOCATION REMOVE POSSIBLE DEAD AGENTS	
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_Fibroblast, 
        temp_scan_storage_bytes_Fibroblast, 
        d_Fibroblasts->_scan_input,
        d_Fibroblasts->_position,
        h_xmachine_memory_Fibroblast_count, 
        stream
    );

	//Scatter into swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_Fibroblast_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	scatter_Fibroblast_Agents<<<gridSize, blockSize, 0, stream>>>(d_Fibroblasts_swap, d_Fibroblasts, 0, h_xmachine_memory_Fibroblast_count);
	gpuErrchkLaunch();
	//use a temp pointer to make swap default
	xmachine_memory_Fibroblast_list* ClearanceOfEarlySenescent_Fibroblasts_temp = d_Fibroblasts;
	d_Fibroblasts = d_Fibroblasts_swap;
	d_Fibroblasts_swap = ClearanceOfEarlySenescent_Fibroblasts_temp;
	//reset agent count
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_Fibroblasts_swap->_position[h_xmachine_memory_Fibroblast_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_Fibroblasts_swap->_scan_input[h_xmachine_memory_Fibroblast_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	if (scan_last_included == 1)
		h_xmachine_memory_Fibroblast_count = scan_last_sum+1;
	else
		h_xmachine_memory_Fibroblast_count = scan_last_sum;
	//Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Fibroblast_count, &h_xmachine_memory_Fibroblast_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_Fibroblast_EarlySenescent_count+h_xmachine_memory_Fibroblast_count > xmachine_memory_Fibroblast_MAX){
		printf("Error: Buffer size of ClearanceOfEarlySenescent agents in state EarlySenescent will be exceeded moving working agents to next state in function ClearanceOfEarlySenescent\n");
      exit(EXIT_FAILURE);
      }
      
  //append agents to next state list
  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_Fibroblast_Agents, no_sm, state_list_size);
  gridSize = (state_list_size + blockSize - 1) / blockSize;
  append_Fibroblast_Agents<<<gridSize, blockSize, 0, stream>>>(d_Fibroblasts_EarlySenescent, d_Fibroblasts, h_xmachine_memory_Fibroblast_EarlySenescent_count, h_xmachine_memory_Fibroblast_count);
  gpuErrchkLaunch();
        
	//update new state agent size
	h_xmachine_memory_Fibroblast_EarlySenescent_count += h_xmachine_memory_Fibroblast_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Fibroblast_EarlySenescent_count, &h_xmachine_memory_Fibroblast_EarlySenescent_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int Fibroblast_ClearanceOfSenescent_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** Fibroblast_ClearanceOfSenescent
 * Agent function prototype for ClearanceOfSenescent function of Fibroblast agent
 */
void Fibroblast_ClearanceOfSenescent(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_Fibroblast_Senescent_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_Fibroblast_Senescent_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_Fibroblast_list* Fibroblasts_Senescent_temp = d_Fibroblasts;
	d_Fibroblasts = d_Fibroblasts_Senescent;
	d_Fibroblasts_Senescent = Fibroblasts_Senescent_temp;
	//set working count to current state count
	h_xmachine_memory_Fibroblast_count = h_xmachine_memory_Fibroblast_Senescent_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Fibroblast_count, &h_xmachine_memory_Fibroblast_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_Fibroblast_Senescent_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Fibroblast_Senescent_count, &h_xmachine_memory_Fibroblast_Senescent_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_ClearanceOfSenescent, Fibroblast_ClearanceOfSenescent_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = Fibroblast_ClearanceOfSenescent_sm_size(blockSize);
	
	
	
	//IF CONTINUOUS AGENT CAN REALLOCATE (process dead agents) THEN RESET AGENT SWAPS	
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_Fibroblast_scan_input, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_Fibroblast_scan_input<<<gridSize, blockSize, 0, stream>>>(d_Fibroblasts);
	gpuErrchkLaunch();
	
	
	//MAIN XMACHINE FUNCTION CALL (ClearanceOfSenescent)
	//Reallocate   : true
	//Input        : 
	//Output       : 
	//Agent Output : 
	GPUFLAME_ClearanceOfSenescent<<<g, b, sm_size, stream>>>(d_Fibroblasts, d_rand48);
	gpuErrchkLaunch();
	
	
	//FOR CONTINUOUS AGENTS WITH REALLOCATION REMOVE POSSIBLE DEAD AGENTS	
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_Fibroblast, 
        temp_scan_storage_bytes_Fibroblast, 
        d_Fibroblasts->_scan_input,
        d_Fibroblasts->_position,
        h_xmachine_memory_Fibroblast_count, 
        stream
    );

	//Scatter into swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_Fibroblast_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	scatter_Fibroblast_Agents<<<gridSize, blockSize, 0, stream>>>(d_Fibroblasts_swap, d_Fibroblasts, 0, h_xmachine_memory_Fibroblast_count);
	gpuErrchkLaunch();
	//use a temp pointer to make swap default
	xmachine_memory_Fibroblast_list* ClearanceOfSenescent_Fibroblasts_temp = d_Fibroblasts;
	d_Fibroblasts = d_Fibroblasts_swap;
	d_Fibroblasts_swap = ClearanceOfSenescent_Fibroblasts_temp;
	//reset agent count
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_Fibroblasts_swap->_position[h_xmachine_memory_Fibroblast_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_Fibroblasts_swap->_scan_input[h_xmachine_memory_Fibroblast_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	if (scan_last_included == 1)
		h_xmachine_memory_Fibroblast_count = scan_last_sum+1;
	else
		h_xmachine_memory_Fibroblast_count = scan_last_sum;
	//Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Fibroblast_count, &h_xmachine_memory_Fibroblast_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_Fibroblast_Senescent_count+h_xmachine_memory_Fibroblast_count > xmachine_memory_Fibroblast_MAX){
		printf("Error: Buffer size of ClearanceOfSenescent agents in state Senescent will be exceeded moving working agents to next state in function ClearanceOfSenescent\n");
      exit(EXIT_FAILURE);
      }
      
  //append agents to next state list
  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_Fibroblast_Agents, no_sm, state_list_size);
  gridSize = (state_list_size + blockSize - 1) / blockSize;
  append_Fibroblast_Agents<<<gridSize, blockSize, 0, stream>>>(d_Fibroblasts_Senescent, d_Fibroblasts, h_xmachine_memory_Fibroblast_Senescent_count, h_xmachine_memory_Fibroblast_count);
  gpuErrchkLaunch();
        
	//update new state agent size
	h_xmachine_memory_Fibroblast_Senescent_count += h_xmachine_memory_Fibroblast_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Fibroblast_Senescent_count, &h_xmachine_memory_Fibroblast_Senescent_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int Fibroblast_DetectDamage_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input is spatially partitioned
	sm_size += (blockSize * sizeof(xmachine_message_tissue_damage_report));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** Fibroblast_DetectDamage
 * Agent function prototype for DetectDamage function of Fibroblast agent
 */
void Fibroblast_DetectDamage(cudaStream_t &stream){

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
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_DetectDamage, Fibroblast_DetectDamage_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = Fibroblast_DetectDamage_sm_size(blockSize);
	
	
	
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

	
	
	//MAIN XMACHINE FUNCTION CALL (DetectDamage)
	//Reallocate   : false
	//Input        : tissue_damage_report
	//Output       : 
	//Agent Output : 
	GPUFLAME_DetectDamage<<<g, b, sm_size, stream>>>(d_Fibroblasts, d_tissue_damage_reports, d_tissue_damage_report_partition_matrix);
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
    
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_Fibroblast_Repair_count+h_xmachine_memory_Fibroblast_count > xmachine_memory_Fibroblast_MAX){
		printf("Error: Buffer size of DetectDamage agents in state Repair will be exceeded moving working agents to next state in function DetectDamage\n");
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


 
extern void reset_TissueBlock_default_count()
{
    h_xmachine_memory_TissueBlock_default_count = 0;
}
 
extern void reset_Fibroblast_Quiescent_count()
{
    h_xmachine_memory_Fibroblast_Quiescent_count = 0;
}
 
extern void reset_Fibroblast_EarlySenescent_count()
{
    h_xmachine_memory_Fibroblast_EarlySenescent_count = 0;
}
 
extern void reset_Fibroblast_Senescent_count()
{
    h_xmachine_memory_Fibroblast_Senescent_count = 0;
}
 
extern void reset_Fibroblast_Proliferating_count()
{
    h_xmachine_memory_Fibroblast_Proliferating_count = 0;
}
 
extern void reset_Fibroblast_Repair_count()
{
    h_xmachine_memory_Fibroblast_Repair_count = 0;
}
