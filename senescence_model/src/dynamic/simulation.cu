
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

/* A Agent variables these lists are used in the agent function where as the other lists are used only outside the agent functions*/
xmachine_memory_A_list* d_As;      /**< Pointer to agent list (population) on the device*/
xmachine_memory_A_list* d_As_swap; /**< Pointer to agent list swap on the device (used when killing agents)*/
xmachine_memory_A_list* d_As_new;  /**< Pointer to new agent list on the device (used to hold new agents before they are appended to the population)*/
int h_xmachine_memory_A_count;   /**< Agent population size counter */ 
uint * d_xmachine_memory_A_keys;	  /**< Agent sort identifiers keys*/
uint * d_xmachine_memory_A_values;  /**< Agent sort identifiers value */

/* A state variables */
xmachine_memory_A_list* h_As_moving_A;      /**< Pointer to agent list (population) on host*/
xmachine_memory_A_list* d_As_moving_A;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_A_moving_A_count;   /**< Agent population size counter */ 

/* A state variables */
xmachine_memory_A_list* h_As_change_direction_A;      /**< Pointer to agent list (population) on host*/
xmachine_memory_A_list* d_As_change_direction_A;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_A_change_direction_A_count;   /**< Agent population size counter */ 

/* A state variables */
xmachine_memory_A_list* h_As_get_going_again_A;      /**< Pointer to agent list (population) on host*/
xmachine_memory_A_list* d_As_get_going_again_A;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_A_get_going_again_A_count;   /**< Agent population size counter */ 

/* B Agent variables these lists are used in the agent function where as the other lists are used only outside the agent functions*/
xmachine_memory_B_list* d_Bs;      /**< Pointer to agent list (population) on the device*/
xmachine_memory_B_list* d_Bs_swap; /**< Pointer to agent list swap on the device (used when killing agents)*/
xmachine_memory_B_list* d_Bs_new;  /**< Pointer to new agent list on the device (used to hold new agents before they are appended to the population)*/
int h_xmachine_memory_B_count;   /**< Agent population size counter */ 
uint * d_xmachine_memory_B_keys;	  /**< Agent sort identifiers keys*/
uint * d_xmachine_memory_B_values;  /**< Agent sort identifiers value */

/* B state variables */
xmachine_memory_B_list* h_Bs_moving_B;      /**< Pointer to agent list (population) on host*/
xmachine_memory_B_list* d_Bs_moving_B;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_B_moving_B_count;   /**< Agent population size counter */ 

/* B state variables */
xmachine_memory_B_list* h_Bs_change_direction_B;      /**< Pointer to agent list (population) on host*/
xmachine_memory_B_list* d_Bs_change_direction_B;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_B_change_direction_B_count;   /**< Agent population size counter */ 

/* B state variables */
xmachine_memory_B_list* h_Bs_get_going_again_B;      /**< Pointer to agent list (population) on host*/
xmachine_memory_B_list* d_Bs_get_going_again_B;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_B_get_going_again_B_count;   /**< Agent population size counter */ 


/* Variables to track the state of host copies of state lists, for the purposes of host agent data access.
 * @future - if the host data is current it may be possible to avoid duplicating memcpy in xml output.
 */
unsigned int h_As_moving_A_variable_id_data_iteration;
unsigned int h_As_moving_A_variable_x_data_iteration;
unsigned int h_As_moving_A_variable_y_data_iteration;
unsigned int h_As_moving_A_variable_z_data_iteration;
unsigned int h_As_moving_A_variable_fx_data_iteration;
unsigned int h_As_moving_A_variable_fy_data_iteration;
unsigned int h_As_moving_A_variable_fz_data_iteration;
unsigned int h_As_moving_A_variable_colour_data_iteration;
unsigned int h_As_change_direction_A_variable_id_data_iteration;
unsigned int h_As_change_direction_A_variable_x_data_iteration;
unsigned int h_As_change_direction_A_variable_y_data_iteration;
unsigned int h_As_change_direction_A_variable_z_data_iteration;
unsigned int h_As_change_direction_A_variable_fx_data_iteration;
unsigned int h_As_change_direction_A_variable_fy_data_iteration;
unsigned int h_As_change_direction_A_variable_fz_data_iteration;
unsigned int h_As_change_direction_A_variable_colour_data_iteration;
unsigned int h_As_get_going_again_A_variable_id_data_iteration;
unsigned int h_As_get_going_again_A_variable_x_data_iteration;
unsigned int h_As_get_going_again_A_variable_y_data_iteration;
unsigned int h_As_get_going_again_A_variable_z_data_iteration;
unsigned int h_As_get_going_again_A_variable_fx_data_iteration;
unsigned int h_As_get_going_again_A_variable_fy_data_iteration;
unsigned int h_As_get_going_again_A_variable_fz_data_iteration;
unsigned int h_As_get_going_again_A_variable_colour_data_iteration;
unsigned int h_Bs_moving_B_variable_id_data_iteration;
unsigned int h_Bs_moving_B_variable_x_data_iteration;
unsigned int h_Bs_moving_B_variable_y_data_iteration;
unsigned int h_Bs_moving_B_variable_z_data_iteration;
unsigned int h_Bs_moving_B_variable_fx_data_iteration;
unsigned int h_Bs_moving_B_variable_fy_data_iteration;
unsigned int h_Bs_moving_B_variable_fz_data_iteration;
unsigned int h_Bs_moving_B_variable_colour_data_iteration;
unsigned int h_Bs_change_direction_B_variable_id_data_iteration;
unsigned int h_Bs_change_direction_B_variable_x_data_iteration;
unsigned int h_Bs_change_direction_B_variable_y_data_iteration;
unsigned int h_Bs_change_direction_B_variable_z_data_iteration;
unsigned int h_Bs_change_direction_B_variable_fx_data_iteration;
unsigned int h_Bs_change_direction_B_variable_fy_data_iteration;
unsigned int h_Bs_change_direction_B_variable_fz_data_iteration;
unsigned int h_Bs_change_direction_B_variable_colour_data_iteration;
unsigned int h_Bs_get_going_again_B_variable_id_data_iteration;
unsigned int h_Bs_get_going_again_B_variable_x_data_iteration;
unsigned int h_Bs_get_going_again_B_variable_y_data_iteration;
unsigned int h_Bs_get_going_again_B_variable_z_data_iteration;
unsigned int h_Bs_get_going_again_B_variable_fx_data_iteration;
unsigned int h_Bs_get_going_again_B_variable_fy_data_iteration;
unsigned int h_Bs_get_going_again_B_variable_fz_data_iteration;
unsigned int h_Bs_get_going_again_B_variable_colour_data_iteration;


/* Message Memory */

/* location Message variables */
xmachine_message_location_list* h_locations;         /**< Pointer to message list on host*/
xmachine_message_location_list* d_locations;         /**< Pointer to message list on device*/
xmachine_message_location_list* d_locations_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Non partitioned and spatial partitioned message variables  */
int h_message_location_count;         /**< message list counter*/
int h_message_location_output_type;   /**< message output type (single or optional)*/

  
/* CUDA Streams for function layers */
cudaStream_t stream1;
cudaStream_t stream2;

/* Device memory and sizes for CUB values */

void * d_temp_scan_storage_A;
size_t temp_scan_storage_bytes_A;

void * d_temp_scan_storage_B;
size_t temp_scan_storage_bytes_B;


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

/** A_move_A
 * Agent function prototype for move_A function of A agent
 */
void A_move_A(cudaStream_t &stream);

/** A_reverse_direction_A
 * Agent function prototype for reverse_direction_A function of A agent
 */
void A_reverse_direction_A(cudaStream_t &stream);

/** A_resume_movement_A
 * Agent function prototype for resume_movement_A function of A agent
 */
void A_resume_movement_A(cudaStream_t &stream);

/** B_move_B
 * Agent function prototype for move_B function of B agent
 */
void B_move_B(cudaStream_t &stream);

/** B_reverse_direction_B
 * Agent function prototype for reverse_direction_B function of B agent
 */
void B_reverse_direction_B(cudaStream_t &stream);

/** B_resume_movement_B
 * Agent function prototype for resume_movement_B function of B agent
 */
void B_resume_movement_B(cudaStream_t &stream);

  
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
    h_As_moving_A_variable_id_data_iteration = 0;
    h_As_moving_A_variable_x_data_iteration = 0;
    h_As_moving_A_variable_y_data_iteration = 0;
    h_As_moving_A_variable_z_data_iteration = 0;
    h_As_moving_A_variable_fx_data_iteration = 0;
    h_As_moving_A_variable_fy_data_iteration = 0;
    h_As_moving_A_variable_fz_data_iteration = 0;
    h_As_moving_A_variable_colour_data_iteration = 0;
    h_As_change_direction_A_variable_id_data_iteration = 0;
    h_As_change_direction_A_variable_x_data_iteration = 0;
    h_As_change_direction_A_variable_y_data_iteration = 0;
    h_As_change_direction_A_variable_z_data_iteration = 0;
    h_As_change_direction_A_variable_fx_data_iteration = 0;
    h_As_change_direction_A_variable_fy_data_iteration = 0;
    h_As_change_direction_A_variable_fz_data_iteration = 0;
    h_As_change_direction_A_variable_colour_data_iteration = 0;
    h_As_get_going_again_A_variable_id_data_iteration = 0;
    h_As_get_going_again_A_variable_x_data_iteration = 0;
    h_As_get_going_again_A_variable_y_data_iteration = 0;
    h_As_get_going_again_A_variable_z_data_iteration = 0;
    h_As_get_going_again_A_variable_fx_data_iteration = 0;
    h_As_get_going_again_A_variable_fy_data_iteration = 0;
    h_As_get_going_again_A_variable_fz_data_iteration = 0;
    h_As_get_going_again_A_variable_colour_data_iteration = 0;
    h_Bs_moving_B_variable_id_data_iteration = 0;
    h_Bs_moving_B_variable_x_data_iteration = 0;
    h_Bs_moving_B_variable_y_data_iteration = 0;
    h_Bs_moving_B_variable_z_data_iteration = 0;
    h_Bs_moving_B_variable_fx_data_iteration = 0;
    h_Bs_moving_B_variable_fy_data_iteration = 0;
    h_Bs_moving_B_variable_fz_data_iteration = 0;
    h_Bs_moving_B_variable_colour_data_iteration = 0;
    h_Bs_change_direction_B_variable_id_data_iteration = 0;
    h_Bs_change_direction_B_variable_x_data_iteration = 0;
    h_Bs_change_direction_B_variable_y_data_iteration = 0;
    h_Bs_change_direction_B_variable_z_data_iteration = 0;
    h_Bs_change_direction_B_variable_fx_data_iteration = 0;
    h_Bs_change_direction_B_variable_fy_data_iteration = 0;
    h_Bs_change_direction_B_variable_fz_data_iteration = 0;
    h_Bs_change_direction_B_variable_colour_data_iteration = 0;
    h_Bs_get_going_again_B_variable_id_data_iteration = 0;
    h_Bs_get_going_again_B_variable_x_data_iteration = 0;
    h_Bs_get_going_again_B_variable_y_data_iteration = 0;
    h_Bs_get_going_again_B_variable_z_data_iteration = 0;
    h_Bs_get_going_again_B_variable_fx_data_iteration = 0;
    h_Bs_get_going_again_B_variable_fy_data_iteration = 0;
    h_Bs_get_going_again_B_variable_fz_data_iteration = 0;
    h_Bs_get_going_again_B_variable_colour_data_iteration = 0;
    



	printf("Allocating Host and Device memory\n");
    PROFILE_PUSH_RANGE("allocate host");
	/* Agent memory allocation (CPU) */
	int xmachine_A_SoA_size = sizeof(xmachine_memory_A_list);
	h_As_moving_A = (xmachine_memory_A_list*)malloc(xmachine_A_SoA_size);
	h_As_change_direction_A = (xmachine_memory_A_list*)malloc(xmachine_A_SoA_size);
	h_As_get_going_again_A = (xmachine_memory_A_list*)malloc(xmachine_A_SoA_size);
	int xmachine_B_SoA_size = sizeof(xmachine_memory_B_list);
	h_Bs_moving_B = (xmachine_memory_B_list*)malloc(xmachine_B_SoA_size);
	h_Bs_change_direction_B = (xmachine_memory_B_list*)malloc(xmachine_B_SoA_size);
	h_Bs_get_going_again_B = (xmachine_memory_B_list*)malloc(xmachine_B_SoA_size);

	/* Message memory allocation (CPU) */
	int message_location_SoA_size = sizeof(xmachine_message_location_list);
	h_locations = (xmachine_message_location_list*)malloc(message_location_SoA_size);

	//Exit if agent or message buffer sizes are to small for function outputs

  /* Graph memory allocation (CPU) */
  

    PROFILE_POP_RANGE(); //"allocate host"
	

	//read initial states
	readInitialStates(inputfile, h_As_moving_A, &h_xmachine_memory_A_moving_A_count, h_Bs_moving_B, &h_xmachine_memory_B_moving_B_count);

  // Read graphs from disk
  

  PROFILE_PUSH_RANGE("allocate device");
	
	/* A Agent memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_As, xmachine_A_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_As_swap, xmachine_A_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_As_new, xmachine_A_SoA_size));
    //continuous agent sort identifiers
  gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_A_keys, xmachine_memory_A_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_A_values, xmachine_memory_A_MAX* sizeof(uint)));
	/* moving_A memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_As_moving_A, xmachine_A_SoA_size));
	gpuErrchk( cudaMemcpy( d_As_moving_A, h_As_moving_A, xmachine_A_SoA_size, cudaMemcpyHostToDevice));
    
	/* change_direction_A memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_As_change_direction_A, xmachine_A_SoA_size));
	gpuErrchk( cudaMemcpy( d_As_change_direction_A, h_As_change_direction_A, xmachine_A_SoA_size, cudaMemcpyHostToDevice));
    
	/* get_going_again_A memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_As_get_going_again_A, xmachine_A_SoA_size));
	gpuErrchk( cudaMemcpy( d_As_get_going_again_A, h_As_get_going_again_A, xmachine_A_SoA_size, cudaMemcpyHostToDevice));
    
	/* B Agent memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_Bs, xmachine_B_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_Bs_swap, xmachine_B_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_Bs_new, xmachine_B_SoA_size));
    //continuous agent sort identifiers
  gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_B_keys, xmachine_memory_B_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_B_values, xmachine_memory_B_MAX* sizeof(uint)));
	/* moving_B memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_Bs_moving_B, xmachine_B_SoA_size));
	gpuErrchk( cudaMemcpy( d_Bs_moving_B, h_Bs_moving_B, xmachine_B_SoA_size, cudaMemcpyHostToDevice));
    
	/* change_direction_B memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_Bs_change_direction_B, xmachine_B_SoA_size));
	gpuErrchk( cudaMemcpy( d_Bs_change_direction_B, h_Bs_change_direction_B, xmachine_B_SoA_size, cudaMemcpyHostToDevice));
    
	/* get_going_again_B memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_Bs_get_going_again_B, xmachine_B_SoA_size));
	gpuErrchk( cudaMemcpy( d_Bs_get_going_again_B, h_Bs_get_going_again_B, xmachine_B_SoA_size, cudaMemcpyHostToDevice));
    
	/* location Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_locations, message_location_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_locations_swap, message_location_SoA_size));
	gpuErrchk( cudaMemcpy( d_locations, h_locations, message_location_SoA_size, cudaMemcpyHostToDevice));
		


  /* Allocate device memory for graphs */
  

    PROFILE_POP_RANGE(); // "allocate device"

    /* Calculate and allocate CUB temporary memory for exclusive scans */
    
    d_temp_scan_storage_A = nullptr;
    temp_scan_storage_bytes_A = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_A, 
        temp_scan_storage_bytes_A, 
        (int*) nullptr, 
        (int*) nullptr, 
        xmachine_memory_A_MAX
    );
    gpuErrchk(cudaMalloc(&d_temp_scan_storage_A, temp_scan_storage_bytes_A));
    
    d_temp_scan_storage_B = nullptr;
    temp_scan_storage_bytes_B = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_B, 
        temp_scan_storage_bytes_B, 
        (int*) nullptr, 
        (int*) nullptr, 
        xmachine_memory_B_MAX
    );
    gpuErrchk(cudaMalloc(&d_temp_scan_storage_B, temp_scan_storage_bytes_B));
    

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
  gpuErrchk(cudaStreamCreate(&stream2));

#if defined(OUTPUT_POPULATION_PER_ITERATION) && OUTPUT_POPULATION_PER_ITERATION
	// Print the agent population size of all agents in all states
	
		printf("Init agent_A_moving_A_count: %u\n",get_agent_A_moving_A_count());
	
		printf("Init agent_A_change_direction_A_count: %u\n",get_agent_A_change_direction_A_count());
	
		printf("Init agent_A_get_going_again_A_count: %u\n",get_agent_A_get_going_again_A_count());
	
		printf("Init agent_B_moving_B_count: %u\n",get_agent_B_moving_B_count());
	
		printf("Init agent_B_change_direction_B_count: %u\n",get_agent_B_change_direction_B_count());
	
		printf("Init agent_B_get_going_again_B_count: %u\n",get_agent_B_get_going_again_B_count());
	
#endif
} 


void sort_As_moving_A(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_A_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_A_moving_A_count); 
	gridSize = (h_xmachine_memory_A_moving_A_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_A_keys, d_xmachine_memory_A_values, d_As_moving_A);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_A_keys),  thrust::device_pointer_cast(d_xmachine_memory_A_keys) + h_xmachine_memory_A_moving_A_count,  thrust::device_pointer_cast(d_xmachine_memory_A_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_A_agents, no_sm, h_xmachine_memory_A_moving_A_count); 
	gridSize = (h_xmachine_memory_A_moving_A_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_A_agents<<<gridSize, blockSize>>>(d_xmachine_memory_A_values, d_As_moving_A, d_As_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_A_list* d_As_temp = d_As_moving_A;
	d_As_moving_A = d_As_swap;
	d_As_swap = d_As_temp;	
}

void sort_As_change_direction_A(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_A_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_A_change_direction_A_count); 
	gridSize = (h_xmachine_memory_A_change_direction_A_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_A_keys, d_xmachine_memory_A_values, d_As_change_direction_A);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_A_keys),  thrust::device_pointer_cast(d_xmachine_memory_A_keys) + h_xmachine_memory_A_change_direction_A_count,  thrust::device_pointer_cast(d_xmachine_memory_A_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_A_agents, no_sm, h_xmachine_memory_A_change_direction_A_count); 
	gridSize = (h_xmachine_memory_A_change_direction_A_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_A_agents<<<gridSize, blockSize>>>(d_xmachine_memory_A_values, d_As_change_direction_A, d_As_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_A_list* d_As_temp = d_As_change_direction_A;
	d_As_change_direction_A = d_As_swap;
	d_As_swap = d_As_temp;	
}

void sort_As_get_going_again_A(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_A_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_A_get_going_again_A_count); 
	gridSize = (h_xmachine_memory_A_get_going_again_A_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_A_keys, d_xmachine_memory_A_values, d_As_get_going_again_A);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_A_keys),  thrust::device_pointer_cast(d_xmachine_memory_A_keys) + h_xmachine_memory_A_get_going_again_A_count,  thrust::device_pointer_cast(d_xmachine_memory_A_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_A_agents, no_sm, h_xmachine_memory_A_get_going_again_A_count); 
	gridSize = (h_xmachine_memory_A_get_going_again_A_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_A_agents<<<gridSize, blockSize>>>(d_xmachine_memory_A_values, d_As_get_going_again_A, d_As_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_A_list* d_As_temp = d_As_get_going_again_A;
	d_As_get_going_again_A = d_As_swap;
	d_As_swap = d_As_temp;	
}

void sort_Bs_moving_B(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_B_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_B_moving_B_count); 
	gridSize = (h_xmachine_memory_B_moving_B_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_B_keys, d_xmachine_memory_B_values, d_Bs_moving_B);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_B_keys),  thrust::device_pointer_cast(d_xmachine_memory_B_keys) + h_xmachine_memory_B_moving_B_count,  thrust::device_pointer_cast(d_xmachine_memory_B_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_B_agents, no_sm, h_xmachine_memory_B_moving_B_count); 
	gridSize = (h_xmachine_memory_B_moving_B_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_B_agents<<<gridSize, blockSize>>>(d_xmachine_memory_B_values, d_Bs_moving_B, d_Bs_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_B_list* d_Bs_temp = d_Bs_moving_B;
	d_Bs_moving_B = d_Bs_swap;
	d_Bs_swap = d_Bs_temp;	
}

void sort_Bs_change_direction_B(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_B_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_B_change_direction_B_count); 
	gridSize = (h_xmachine_memory_B_change_direction_B_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_B_keys, d_xmachine_memory_B_values, d_Bs_change_direction_B);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_B_keys),  thrust::device_pointer_cast(d_xmachine_memory_B_keys) + h_xmachine_memory_B_change_direction_B_count,  thrust::device_pointer_cast(d_xmachine_memory_B_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_B_agents, no_sm, h_xmachine_memory_B_change_direction_B_count); 
	gridSize = (h_xmachine_memory_B_change_direction_B_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_B_agents<<<gridSize, blockSize>>>(d_xmachine_memory_B_values, d_Bs_change_direction_B, d_Bs_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_B_list* d_Bs_temp = d_Bs_change_direction_B;
	d_Bs_change_direction_B = d_Bs_swap;
	d_Bs_swap = d_Bs_temp;	
}

void sort_Bs_get_going_again_B(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_B_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_B_get_going_again_B_count); 
	gridSize = (h_xmachine_memory_B_get_going_again_B_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_B_keys, d_xmachine_memory_B_values, d_Bs_get_going_again_B);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_B_keys),  thrust::device_pointer_cast(d_xmachine_memory_B_keys) + h_xmachine_memory_B_get_going_again_B_count,  thrust::device_pointer_cast(d_xmachine_memory_B_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_B_agents, no_sm, h_xmachine_memory_B_get_going_again_B_count); 
	gridSize = (h_xmachine_memory_B_get_going_again_B_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_B_agents<<<gridSize, blockSize>>>(d_xmachine_memory_B_values, d_Bs_get_going_again_B, d_Bs_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_B_list* d_Bs_temp = d_Bs_get_going_again_B;
	d_Bs_get_going_again_B = d_Bs_swap;
	d_Bs_swap = d_Bs_temp;	
}


void cleanup(){
    PROFILE_SCOPED_RANGE("cleanup");

    /* Call all exit functions */
	

	/* Agent data free*/
	
	/* A Agent variables */
	gpuErrchk(cudaFree(d_As));
	gpuErrchk(cudaFree(d_As_swap));
	gpuErrchk(cudaFree(d_As_new));
	
	free( h_As_moving_A);
	gpuErrchk(cudaFree(d_As_moving_A));
	
	free( h_As_change_direction_A);
	gpuErrchk(cudaFree(d_As_change_direction_A));
	
	free( h_As_get_going_again_A);
	gpuErrchk(cudaFree(d_As_get_going_again_A));
	
	/* B Agent variables */
	gpuErrchk(cudaFree(d_Bs));
	gpuErrchk(cudaFree(d_Bs_swap));
	gpuErrchk(cudaFree(d_Bs_new));
	
	free( h_Bs_moving_B);
	gpuErrchk(cudaFree(d_Bs_moving_B));
	
	free( h_Bs_change_direction_B);
	gpuErrchk(cudaFree(d_Bs_change_direction_B));
	
	free( h_Bs_get_going_again_B);
	gpuErrchk(cudaFree(d_Bs_get_going_again_B));
	

	/* Message data free */
	
	/* location Message variables */
	free( h_locations);
	gpuErrchk(cudaFree(d_locations));
	gpuErrchk(cudaFree(d_locations_swap));
	

    /* Free temporary CUB memory if required. */
    
    if(d_temp_scan_storage_A != nullptr){
      gpuErrchk(cudaFree(d_temp_scan_storage_A));
      d_temp_scan_storage_A = nullptr;
      temp_scan_storage_bytes_A = 0;
    }
    
    if(d_temp_scan_storage_B != nullptr){
      gpuErrchk(cudaFree(d_temp_scan_storage_B));
      d_temp_scan_storage_B = nullptr;
      temp_scan_storage_bytes_B = 0;
    }
    

  /* Graph data free */
  
  
  /* CUDA Streams for function layers */
  
  gpuErrchk(cudaStreamDestroy(stream1));
  gpuErrchk(cudaStreamDestroy(stream2));

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
	h_message_location_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_location_count, &h_message_location_count, sizeof(int)));
	

	/* Call agent functions in order iterating through the layer functions */
	
	/* Layer 1*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("A_move_A");
	A_move_A(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: A_move_A = %f (ms)\n", instrument_milliseconds);
#endif
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("B_move_B");
	B_move_B(stream2);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: B_move_B = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 2*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("A_reverse_direction_A");
	A_reverse_direction_A(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: A_reverse_direction_A = %f (ms)\n", instrument_milliseconds);
#endif
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("B_reverse_direction_B");
	B_reverse_direction_B(stream2);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: B_reverse_direction_B = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 3*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("A_resume_movement_A");
	A_resume_movement_A(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: A_resume_movement_A = %f (ms)\n", instrument_milliseconds);
#endif
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("B_resume_movement_B");
	B_resume_movement_B(stream2);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: B_resume_movement_B = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
    
    /* Call all step functions */
	

#if defined(OUTPUT_POPULATION_PER_ITERATION) && OUTPUT_POPULATION_PER_ITERATION
	// Print the agent population size of all agents in all states
	
		printf("agent_A_moving_A_count: %u\n",get_agent_A_moving_A_count());
	
		printf("agent_A_change_direction_A_count: %u\n",get_agent_A_change_direction_A_count());
	
		printf("agent_A_get_going_again_A_count: %u\n",get_agent_A_get_going_again_A_count());
	
		printf("agent_B_moving_B_count: %u\n",get_agent_B_moving_B_count());
	
		printf("agent_B_change_direction_B_count: %u\n",get_agent_B_change_direction_B_count());
	
		printf("agent_B_get_going_again_B_count: %u\n",get_agent_B_get_going_again_B_count());
	
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



/* Agent data access functions*/

    
int get_agent_A_MAX_count(){
    return xmachine_memory_A_MAX;
}


int get_agent_A_moving_A_count(){
	//continuous agent
	return h_xmachine_memory_A_moving_A_count;
	
}

xmachine_memory_A_list* get_device_A_moving_A_agents(){
	return d_As_moving_A;
}

xmachine_memory_A_list* get_host_A_moving_A_agents(){
	return h_As_moving_A;
}

int get_agent_A_change_direction_A_count(){
	//continuous agent
	return h_xmachine_memory_A_change_direction_A_count;
	
}

xmachine_memory_A_list* get_device_A_change_direction_A_agents(){
	return d_As_change_direction_A;
}

xmachine_memory_A_list* get_host_A_change_direction_A_agents(){
	return h_As_change_direction_A;
}

int get_agent_A_get_going_again_A_count(){
	//continuous agent
	return h_xmachine_memory_A_get_going_again_A_count;
	
}

xmachine_memory_A_list* get_device_A_get_going_again_A_agents(){
	return d_As_get_going_again_A;
}

xmachine_memory_A_list* get_host_A_get_going_again_A_agents(){
	return h_As_get_going_again_A;
}

    
int get_agent_B_MAX_count(){
    return xmachine_memory_B_MAX;
}


int get_agent_B_moving_B_count(){
	//continuous agent
	return h_xmachine_memory_B_moving_B_count;
	
}

xmachine_memory_B_list* get_device_B_moving_B_agents(){
	return d_Bs_moving_B;
}

xmachine_memory_B_list* get_host_B_moving_B_agents(){
	return h_Bs_moving_B;
}

int get_agent_B_change_direction_B_count(){
	//continuous agent
	return h_xmachine_memory_B_change_direction_B_count;
	
}

xmachine_memory_B_list* get_device_B_change_direction_B_agents(){
	return d_Bs_change_direction_B;
}

xmachine_memory_B_list* get_host_B_change_direction_B_agents(){
	return h_Bs_change_direction_B;
}

int get_agent_B_get_going_again_B_count(){
	//continuous agent
	return h_xmachine_memory_B_get_going_again_B_count;
	
}

xmachine_memory_B_list* get_device_B_get_going_again_B_agents(){
	return d_Bs_get_going_again_B;
}

xmachine_memory_B_list* get_host_B_get_going_again_B_agents(){
	return h_Bs_get_going_again_B;
}



/* Host based access of agent variables*/

/** int get_A_moving_A_variable_id(unsigned int index)
 * Gets the value of the id variable of an A agent in the moving_A state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable id
 */
__host__ int get_A_moving_A_variable_id(unsigned int index){
    unsigned int count = get_agent_A_moving_A_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_As_moving_A_variable_id_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_As_moving_A->id,
                    d_As_moving_A->id,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_As_moving_A_variable_id_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_As_moving_A->id[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access id for the %u th member of A_moving_A. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_A_moving_A_variable_x(unsigned int index)
 * Gets the value of the x variable of an A agent in the moving_A state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable x
 */
__host__ float get_A_moving_A_variable_x(unsigned int index){
    unsigned int count = get_agent_A_moving_A_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_As_moving_A_variable_x_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_As_moving_A->x,
                    d_As_moving_A->x,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_As_moving_A_variable_x_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_As_moving_A->x[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access x for the %u th member of A_moving_A. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_A_moving_A_variable_y(unsigned int index)
 * Gets the value of the y variable of an A agent in the moving_A state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable y
 */
__host__ float get_A_moving_A_variable_y(unsigned int index){
    unsigned int count = get_agent_A_moving_A_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_As_moving_A_variable_y_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_As_moving_A->y,
                    d_As_moving_A->y,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_As_moving_A_variable_y_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_As_moving_A->y[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access y for the %u th member of A_moving_A. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_A_moving_A_variable_z(unsigned int index)
 * Gets the value of the z variable of an A agent in the moving_A state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable z
 */
__host__ float get_A_moving_A_variable_z(unsigned int index){
    unsigned int count = get_agent_A_moving_A_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_As_moving_A_variable_z_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_As_moving_A->z,
                    d_As_moving_A->z,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_As_moving_A_variable_z_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_As_moving_A->z[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access z for the %u th member of A_moving_A. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_A_moving_A_variable_fx(unsigned int index)
 * Gets the value of the fx variable of an A agent in the moving_A state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable fx
 */
__host__ float get_A_moving_A_variable_fx(unsigned int index){
    unsigned int count = get_agent_A_moving_A_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_As_moving_A_variable_fx_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_As_moving_A->fx,
                    d_As_moving_A->fx,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_As_moving_A_variable_fx_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_As_moving_A->fx[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access fx for the %u th member of A_moving_A. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_A_moving_A_variable_fy(unsigned int index)
 * Gets the value of the fy variable of an A agent in the moving_A state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable fy
 */
__host__ float get_A_moving_A_variable_fy(unsigned int index){
    unsigned int count = get_agent_A_moving_A_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_As_moving_A_variable_fy_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_As_moving_A->fy,
                    d_As_moving_A->fy,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_As_moving_A_variable_fy_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_As_moving_A->fy[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access fy for the %u th member of A_moving_A. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_A_moving_A_variable_fz(unsigned int index)
 * Gets the value of the fz variable of an A agent in the moving_A state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable fz
 */
__host__ float get_A_moving_A_variable_fz(unsigned int index){
    unsigned int count = get_agent_A_moving_A_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_As_moving_A_variable_fz_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_As_moving_A->fz,
                    d_As_moving_A->fz,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_As_moving_A_variable_fz_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_As_moving_A->fz[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access fz for the %u th member of A_moving_A. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_A_moving_A_variable_colour(unsigned int index)
 * Gets the value of the colour variable of an A agent in the moving_A state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable colour
 */
__host__ int get_A_moving_A_variable_colour(unsigned int index){
    unsigned int count = get_agent_A_moving_A_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_As_moving_A_variable_colour_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_As_moving_A->colour,
                    d_As_moving_A->colour,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_As_moving_A_variable_colour_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_As_moving_A->colour[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access colour for the %u th member of A_moving_A. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_A_change_direction_A_variable_id(unsigned int index)
 * Gets the value of the id variable of an A agent in the change_direction_A state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable id
 */
__host__ int get_A_change_direction_A_variable_id(unsigned int index){
    unsigned int count = get_agent_A_change_direction_A_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_As_change_direction_A_variable_id_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_As_change_direction_A->id,
                    d_As_change_direction_A->id,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_As_change_direction_A_variable_id_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_As_change_direction_A->id[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access id for the %u th member of A_change_direction_A. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_A_change_direction_A_variable_x(unsigned int index)
 * Gets the value of the x variable of an A agent in the change_direction_A state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable x
 */
__host__ float get_A_change_direction_A_variable_x(unsigned int index){
    unsigned int count = get_agent_A_change_direction_A_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_As_change_direction_A_variable_x_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_As_change_direction_A->x,
                    d_As_change_direction_A->x,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_As_change_direction_A_variable_x_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_As_change_direction_A->x[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access x for the %u th member of A_change_direction_A. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_A_change_direction_A_variable_y(unsigned int index)
 * Gets the value of the y variable of an A agent in the change_direction_A state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable y
 */
__host__ float get_A_change_direction_A_variable_y(unsigned int index){
    unsigned int count = get_agent_A_change_direction_A_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_As_change_direction_A_variable_y_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_As_change_direction_A->y,
                    d_As_change_direction_A->y,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_As_change_direction_A_variable_y_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_As_change_direction_A->y[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access y for the %u th member of A_change_direction_A. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_A_change_direction_A_variable_z(unsigned int index)
 * Gets the value of the z variable of an A agent in the change_direction_A state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable z
 */
__host__ float get_A_change_direction_A_variable_z(unsigned int index){
    unsigned int count = get_agent_A_change_direction_A_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_As_change_direction_A_variable_z_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_As_change_direction_A->z,
                    d_As_change_direction_A->z,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_As_change_direction_A_variable_z_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_As_change_direction_A->z[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access z for the %u th member of A_change_direction_A. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_A_change_direction_A_variable_fx(unsigned int index)
 * Gets the value of the fx variable of an A agent in the change_direction_A state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable fx
 */
__host__ float get_A_change_direction_A_variable_fx(unsigned int index){
    unsigned int count = get_agent_A_change_direction_A_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_As_change_direction_A_variable_fx_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_As_change_direction_A->fx,
                    d_As_change_direction_A->fx,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_As_change_direction_A_variable_fx_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_As_change_direction_A->fx[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access fx for the %u th member of A_change_direction_A. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_A_change_direction_A_variable_fy(unsigned int index)
 * Gets the value of the fy variable of an A agent in the change_direction_A state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable fy
 */
__host__ float get_A_change_direction_A_variable_fy(unsigned int index){
    unsigned int count = get_agent_A_change_direction_A_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_As_change_direction_A_variable_fy_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_As_change_direction_A->fy,
                    d_As_change_direction_A->fy,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_As_change_direction_A_variable_fy_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_As_change_direction_A->fy[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access fy for the %u th member of A_change_direction_A. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_A_change_direction_A_variable_fz(unsigned int index)
 * Gets the value of the fz variable of an A agent in the change_direction_A state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable fz
 */
__host__ float get_A_change_direction_A_variable_fz(unsigned int index){
    unsigned int count = get_agent_A_change_direction_A_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_As_change_direction_A_variable_fz_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_As_change_direction_A->fz,
                    d_As_change_direction_A->fz,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_As_change_direction_A_variable_fz_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_As_change_direction_A->fz[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access fz for the %u th member of A_change_direction_A. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_A_change_direction_A_variable_colour(unsigned int index)
 * Gets the value of the colour variable of an A agent in the change_direction_A state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable colour
 */
__host__ int get_A_change_direction_A_variable_colour(unsigned int index){
    unsigned int count = get_agent_A_change_direction_A_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_As_change_direction_A_variable_colour_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_As_change_direction_A->colour,
                    d_As_change_direction_A->colour,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_As_change_direction_A_variable_colour_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_As_change_direction_A->colour[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access colour for the %u th member of A_change_direction_A. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_A_get_going_again_A_variable_id(unsigned int index)
 * Gets the value of the id variable of an A agent in the get_going_again_A state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable id
 */
__host__ int get_A_get_going_again_A_variable_id(unsigned int index){
    unsigned int count = get_agent_A_get_going_again_A_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_As_get_going_again_A_variable_id_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_As_get_going_again_A->id,
                    d_As_get_going_again_A->id,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_As_get_going_again_A_variable_id_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_As_get_going_again_A->id[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access id for the %u th member of A_get_going_again_A. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_A_get_going_again_A_variable_x(unsigned int index)
 * Gets the value of the x variable of an A agent in the get_going_again_A state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable x
 */
__host__ float get_A_get_going_again_A_variable_x(unsigned int index){
    unsigned int count = get_agent_A_get_going_again_A_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_As_get_going_again_A_variable_x_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_As_get_going_again_A->x,
                    d_As_get_going_again_A->x,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_As_get_going_again_A_variable_x_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_As_get_going_again_A->x[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access x for the %u th member of A_get_going_again_A. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_A_get_going_again_A_variable_y(unsigned int index)
 * Gets the value of the y variable of an A agent in the get_going_again_A state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable y
 */
__host__ float get_A_get_going_again_A_variable_y(unsigned int index){
    unsigned int count = get_agent_A_get_going_again_A_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_As_get_going_again_A_variable_y_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_As_get_going_again_A->y,
                    d_As_get_going_again_A->y,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_As_get_going_again_A_variable_y_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_As_get_going_again_A->y[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access y for the %u th member of A_get_going_again_A. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_A_get_going_again_A_variable_z(unsigned int index)
 * Gets the value of the z variable of an A agent in the get_going_again_A state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable z
 */
__host__ float get_A_get_going_again_A_variable_z(unsigned int index){
    unsigned int count = get_agent_A_get_going_again_A_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_As_get_going_again_A_variable_z_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_As_get_going_again_A->z,
                    d_As_get_going_again_A->z,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_As_get_going_again_A_variable_z_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_As_get_going_again_A->z[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access z for the %u th member of A_get_going_again_A. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_A_get_going_again_A_variable_fx(unsigned int index)
 * Gets the value of the fx variable of an A agent in the get_going_again_A state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable fx
 */
__host__ float get_A_get_going_again_A_variable_fx(unsigned int index){
    unsigned int count = get_agent_A_get_going_again_A_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_As_get_going_again_A_variable_fx_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_As_get_going_again_A->fx,
                    d_As_get_going_again_A->fx,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_As_get_going_again_A_variable_fx_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_As_get_going_again_A->fx[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access fx for the %u th member of A_get_going_again_A. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_A_get_going_again_A_variable_fy(unsigned int index)
 * Gets the value of the fy variable of an A agent in the get_going_again_A state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable fy
 */
__host__ float get_A_get_going_again_A_variable_fy(unsigned int index){
    unsigned int count = get_agent_A_get_going_again_A_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_As_get_going_again_A_variable_fy_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_As_get_going_again_A->fy,
                    d_As_get_going_again_A->fy,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_As_get_going_again_A_variable_fy_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_As_get_going_again_A->fy[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access fy for the %u th member of A_get_going_again_A. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_A_get_going_again_A_variable_fz(unsigned int index)
 * Gets the value of the fz variable of an A agent in the get_going_again_A state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable fz
 */
__host__ float get_A_get_going_again_A_variable_fz(unsigned int index){
    unsigned int count = get_agent_A_get_going_again_A_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_As_get_going_again_A_variable_fz_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_As_get_going_again_A->fz,
                    d_As_get_going_again_A->fz,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_As_get_going_again_A_variable_fz_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_As_get_going_again_A->fz[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access fz for the %u th member of A_get_going_again_A. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_A_get_going_again_A_variable_colour(unsigned int index)
 * Gets the value of the colour variable of an A agent in the get_going_again_A state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable colour
 */
__host__ int get_A_get_going_again_A_variable_colour(unsigned int index){
    unsigned int count = get_agent_A_get_going_again_A_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_As_get_going_again_A_variable_colour_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_As_get_going_again_A->colour,
                    d_As_get_going_again_A->colour,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_As_get_going_again_A_variable_colour_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_As_get_going_again_A->colour[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access colour for the %u th member of A_get_going_again_A. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_B_moving_B_variable_id(unsigned int index)
 * Gets the value of the id variable of an B agent in the moving_B state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable id
 */
__host__ int get_B_moving_B_variable_id(unsigned int index){
    unsigned int count = get_agent_B_moving_B_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Bs_moving_B_variable_id_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Bs_moving_B->id,
                    d_Bs_moving_B->id,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Bs_moving_B_variable_id_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Bs_moving_B->id[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access id for the %u th member of B_moving_B. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_B_moving_B_variable_x(unsigned int index)
 * Gets the value of the x variable of an B agent in the moving_B state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable x
 */
__host__ float get_B_moving_B_variable_x(unsigned int index){
    unsigned int count = get_agent_B_moving_B_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Bs_moving_B_variable_x_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Bs_moving_B->x,
                    d_Bs_moving_B->x,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Bs_moving_B_variable_x_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Bs_moving_B->x[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access x for the %u th member of B_moving_B. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_B_moving_B_variable_y(unsigned int index)
 * Gets the value of the y variable of an B agent in the moving_B state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable y
 */
__host__ float get_B_moving_B_variable_y(unsigned int index){
    unsigned int count = get_agent_B_moving_B_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Bs_moving_B_variable_y_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Bs_moving_B->y,
                    d_Bs_moving_B->y,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Bs_moving_B_variable_y_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Bs_moving_B->y[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access y for the %u th member of B_moving_B. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_B_moving_B_variable_z(unsigned int index)
 * Gets the value of the z variable of an B agent in the moving_B state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable z
 */
__host__ float get_B_moving_B_variable_z(unsigned int index){
    unsigned int count = get_agent_B_moving_B_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Bs_moving_B_variable_z_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Bs_moving_B->z,
                    d_Bs_moving_B->z,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Bs_moving_B_variable_z_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Bs_moving_B->z[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access z for the %u th member of B_moving_B. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_B_moving_B_variable_fx(unsigned int index)
 * Gets the value of the fx variable of an B agent in the moving_B state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable fx
 */
__host__ float get_B_moving_B_variable_fx(unsigned int index){
    unsigned int count = get_agent_B_moving_B_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Bs_moving_B_variable_fx_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Bs_moving_B->fx,
                    d_Bs_moving_B->fx,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Bs_moving_B_variable_fx_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Bs_moving_B->fx[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access fx for the %u th member of B_moving_B. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_B_moving_B_variable_fy(unsigned int index)
 * Gets the value of the fy variable of an B agent in the moving_B state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable fy
 */
__host__ float get_B_moving_B_variable_fy(unsigned int index){
    unsigned int count = get_agent_B_moving_B_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Bs_moving_B_variable_fy_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Bs_moving_B->fy,
                    d_Bs_moving_B->fy,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Bs_moving_B_variable_fy_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Bs_moving_B->fy[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access fy for the %u th member of B_moving_B. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_B_moving_B_variable_fz(unsigned int index)
 * Gets the value of the fz variable of an B agent in the moving_B state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable fz
 */
__host__ float get_B_moving_B_variable_fz(unsigned int index){
    unsigned int count = get_agent_B_moving_B_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Bs_moving_B_variable_fz_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Bs_moving_B->fz,
                    d_Bs_moving_B->fz,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Bs_moving_B_variable_fz_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Bs_moving_B->fz[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access fz for the %u th member of B_moving_B. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_B_moving_B_variable_colour(unsigned int index)
 * Gets the value of the colour variable of an B agent in the moving_B state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable colour
 */
__host__ int get_B_moving_B_variable_colour(unsigned int index){
    unsigned int count = get_agent_B_moving_B_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Bs_moving_B_variable_colour_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Bs_moving_B->colour,
                    d_Bs_moving_B->colour,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Bs_moving_B_variable_colour_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Bs_moving_B->colour[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access colour for the %u th member of B_moving_B. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_B_change_direction_B_variable_id(unsigned int index)
 * Gets the value of the id variable of an B agent in the change_direction_B state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable id
 */
__host__ int get_B_change_direction_B_variable_id(unsigned int index){
    unsigned int count = get_agent_B_change_direction_B_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Bs_change_direction_B_variable_id_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Bs_change_direction_B->id,
                    d_Bs_change_direction_B->id,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Bs_change_direction_B_variable_id_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Bs_change_direction_B->id[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access id for the %u th member of B_change_direction_B. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_B_change_direction_B_variable_x(unsigned int index)
 * Gets the value of the x variable of an B agent in the change_direction_B state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable x
 */
__host__ float get_B_change_direction_B_variable_x(unsigned int index){
    unsigned int count = get_agent_B_change_direction_B_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Bs_change_direction_B_variable_x_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Bs_change_direction_B->x,
                    d_Bs_change_direction_B->x,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Bs_change_direction_B_variable_x_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Bs_change_direction_B->x[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access x for the %u th member of B_change_direction_B. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_B_change_direction_B_variable_y(unsigned int index)
 * Gets the value of the y variable of an B agent in the change_direction_B state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable y
 */
__host__ float get_B_change_direction_B_variable_y(unsigned int index){
    unsigned int count = get_agent_B_change_direction_B_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Bs_change_direction_B_variable_y_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Bs_change_direction_B->y,
                    d_Bs_change_direction_B->y,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Bs_change_direction_B_variable_y_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Bs_change_direction_B->y[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access y for the %u th member of B_change_direction_B. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_B_change_direction_B_variable_z(unsigned int index)
 * Gets the value of the z variable of an B agent in the change_direction_B state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable z
 */
__host__ float get_B_change_direction_B_variable_z(unsigned int index){
    unsigned int count = get_agent_B_change_direction_B_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Bs_change_direction_B_variable_z_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Bs_change_direction_B->z,
                    d_Bs_change_direction_B->z,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Bs_change_direction_B_variable_z_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Bs_change_direction_B->z[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access z for the %u th member of B_change_direction_B. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_B_change_direction_B_variable_fx(unsigned int index)
 * Gets the value of the fx variable of an B agent in the change_direction_B state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable fx
 */
__host__ float get_B_change_direction_B_variable_fx(unsigned int index){
    unsigned int count = get_agent_B_change_direction_B_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Bs_change_direction_B_variable_fx_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Bs_change_direction_B->fx,
                    d_Bs_change_direction_B->fx,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Bs_change_direction_B_variable_fx_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Bs_change_direction_B->fx[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access fx for the %u th member of B_change_direction_B. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_B_change_direction_B_variable_fy(unsigned int index)
 * Gets the value of the fy variable of an B agent in the change_direction_B state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable fy
 */
__host__ float get_B_change_direction_B_variable_fy(unsigned int index){
    unsigned int count = get_agent_B_change_direction_B_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Bs_change_direction_B_variable_fy_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Bs_change_direction_B->fy,
                    d_Bs_change_direction_B->fy,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Bs_change_direction_B_variable_fy_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Bs_change_direction_B->fy[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access fy for the %u th member of B_change_direction_B. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_B_change_direction_B_variable_fz(unsigned int index)
 * Gets the value of the fz variable of an B agent in the change_direction_B state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable fz
 */
__host__ float get_B_change_direction_B_variable_fz(unsigned int index){
    unsigned int count = get_agent_B_change_direction_B_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Bs_change_direction_B_variable_fz_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Bs_change_direction_B->fz,
                    d_Bs_change_direction_B->fz,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Bs_change_direction_B_variable_fz_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Bs_change_direction_B->fz[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access fz for the %u th member of B_change_direction_B. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_B_change_direction_B_variable_colour(unsigned int index)
 * Gets the value of the colour variable of an B agent in the change_direction_B state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable colour
 */
__host__ int get_B_change_direction_B_variable_colour(unsigned int index){
    unsigned int count = get_agent_B_change_direction_B_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Bs_change_direction_B_variable_colour_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Bs_change_direction_B->colour,
                    d_Bs_change_direction_B->colour,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Bs_change_direction_B_variable_colour_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Bs_change_direction_B->colour[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access colour for the %u th member of B_change_direction_B. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_B_get_going_again_B_variable_id(unsigned int index)
 * Gets the value of the id variable of an B agent in the get_going_again_B state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable id
 */
__host__ int get_B_get_going_again_B_variable_id(unsigned int index){
    unsigned int count = get_agent_B_get_going_again_B_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Bs_get_going_again_B_variable_id_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Bs_get_going_again_B->id,
                    d_Bs_get_going_again_B->id,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Bs_get_going_again_B_variable_id_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Bs_get_going_again_B->id[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access id for the %u th member of B_get_going_again_B. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_B_get_going_again_B_variable_x(unsigned int index)
 * Gets the value of the x variable of an B agent in the get_going_again_B state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable x
 */
__host__ float get_B_get_going_again_B_variable_x(unsigned int index){
    unsigned int count = get_agent_B_get_going_again_B_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Bs_get_going_again_B_variable_x_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Bs_get_going_again_B->x,
                    d_Bs_get_going_again_B->x,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Bs_get_going_again_B_variable_x_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Bs_get_going_again_B->x[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access x for the %u th member of B_get_going_again_B. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_B_get_going_again_B_variable_y(unsigned int index)
 * Gets the value of the y variable of an B agent in the get_going_again_B state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable y
 */
__host__ float get_B_get_going_again_B_variable_y(unsigned int index){
    unsigned int count = get_agent_B_get_going_again_B_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Bs_get_going_again_B_variable_y_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Bs_get_going_again_B->y,
                    d_Bs_get_going_again_B->y,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Bs_get_going_again_B_variable_y_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Bs_get_going_again_B->y[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access y for the %u th member of B_get_going_again_B. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_B_get_going_again_B_variable_z(unsigned int index)
 * Gets the value of the z variable of an B agent in the get_going_again_B state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable z
 */
__host__ float get_B_get_going_again_B_variable_z(unsigned int index){
    unsigned int count = get_agent_B_get_going_again_B_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Bs_get_going_again_B_variable_z_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Bs_get_going_again_B->z,
                    d_Bs_get_going_again_B->z,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Bs_get_going_again_B_variable_z_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Bs_get_going_again_B->z[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access z for the %u th member of B_get_going_again_B. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_B_get_going_again_B_variable_fx(unsigned int index)
 * Gets the value of the fx variable of an B agent in the get_going_again_B state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable fx
 */
__host__ float get_B_get_going_again_B_variable_fx(unsigned int index){
    unsigned int count = get_agent_B_get_going_again_B_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Bs_get_going_again_B_variable_fx_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Bs_get_going_again_B->fx,
                    d_Bs_get_going_again_B->fx,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Bs_get_going_again_B_variable_fx_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Bs_get_going_again_B->fx[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access fx for the %u th member of B_get_going_again_B. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_B_get_going_again_B_variable_fy(unsigned int index)
 * Gets the value of the fy variable of an B agent in the get_going_again_B state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable fy
 */
__host__ float get_B_get_going_again_B_variable_fy(unsigned int index){
    unsigned int count = get_agent_B_get_going_again_B_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Bs_get_going_again_B_variable_fy_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Bs_get_going_again_B->fy,
                    d_Bs_get_going_again_B->fy,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Bs_get_going_again_B_variable_fy_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Bs_get_going_again_B->fy[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access fy for the %u th member of B_get_going_again_B. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_B_get_going_again_B_variable_fz(unsigned int index)
 * Gets the value of the fz variable of an B agent in the get_going_again_B state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable fz
 */
__host__ float get_B_get_going_again_B_variable_fz(unsigned int index){
    unsigned int count = get_agent_B_get_going_again_B_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Bs_get_going_again_B_variable_fz_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Bs_get_going_again_B->fz,
                    d_Bs_get_going_again_B->fz,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Bs_get_going_again_B_variable_fz_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Bs_get_going_again_B->fz[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access fz for the %u th member of B_get_going_again_B. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_B_get_going_again_B_variable_colour(unsigned int index)
 * Gets the value of the colour variable of an B agent in the get_going_again_B state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable colour
 */
__host__ int get_B_get_going_again_B_variable_colour(unsigned int index){
    unsigned int count = get_agent_B_get_going_again_B_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Bs_get_going_again_B_variable_colour_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Bs_get_going_again_B->colour,
                    d_Bs_get_going_again_B->colour,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Bs_get_going_again_B_variable_colour_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Bs_get_going_again_B->colour[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access colour for the %u th member of B_get_going_again_B. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}



/* Host based agent creation functions */
// These are only available for continuous agents.



/* copy_single_xmachine_memory_A_hostToDevice
 * Private function to copy a host agent struct into a device SoA agent list.
 * @param d_dst destination agent state list
 * @param h_agent agent struct
 */
void copy_single_xmachine_memory_A_hostToDevice(xmachine_memory_A_list * d_dst, xmachine_memory_A * h_agent){
 
		gpuErrchk(cudaMemcpy(d_dst->id, &h_agent->id, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->x, &h_agent->x, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->y, &h_agent->y, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->z, &h_agent->z, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->fx, &h_agent->fx, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->fy, &h_agent->fy, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->fz, &h_agent->fz, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->colour, &h_agent->colour, sizeof(int), cudaMemcpyHostToDevice));

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
void copy_partial_xmachine_memory_A_hostToDevice(xmachine_memory_A_list * d_dst, xmachine_memory_A_list * h_src, unsigned int count){
    // Only copy elements if there is data to move.
    if (count > 0){
	 
		gpuErrchk(cudaMemcpy(d_dst->id, h_src->id, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->x, h_src->x, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->y, h_src->y, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->z, h_src->z, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->fx, h_src->fx, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->fy, h_src->fy, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->fz, h_src->fz, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->colour, h_src->colour, count * sizeof(int), cudaMemcpyHostToDevice));

    }
}


/* copy_single_xmachine_memory_B_hostToDevice
 * Private function to copy a host agent struct into a device SoA agent list.
 * @param d_dst destination agent state list
 * @param h_agent agent struct
 */
void copy_single_xmachine_memory_B_hostToDevice(xmachine_memory_B_list * d_dst, xmachine_memory_B * h_agent){
 
		gpuErrchk(cudaMemcpy(d_dst->id, &h_agent->id, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->x, &h_agent->x, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->y, &h_agent->y, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->z, &h_agent->z, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->fx, &h_agent->fx, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->fy, &h_agent->fy, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->fz, &h_agent->fz, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->colour, &h_agent->colour, sizeof(int), cudaMemcpyHostToDevice));

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
void copy_partial_xmachine_memory_B_hostToDevice(xmachine_memory_B_list * d_dst, xmachine_memory_B_list * h_src, unsigned int count){
    // Only copy elements if there is data to move.
    if (count > 0){
	 
		gpuErrchk(cudaMemcpy(d_dst->id, h_src->id, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->x, h_src->x, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->y, h_src->y, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->z, h_src->z, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->fx, h_src->fx, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->fy, h_src->fy, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->fz, h_src->fz, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->colour, h_src->colour, count * sizeof(int), cudaMemcpyHostToDevice));

    }
}

xmachine_memory_A* h_allocate_agent_A(){
	xmachine_memory_A* agent = (xmachine_memory_A*)malloc(sizeof(xmachine_memory_A));
	// Memset the whole agent strcuture
    memset(agent, 0, sizeof(xmachine_memory_A));

	return agent;
}
void h_free_agent_A(xmachine_memory_A** agent){
 
	free((*agent));
	(*agent) = NULL;
}
xmachine_memory_A** h_allocate_agent_A_array(unsigned int count){
	xmachine_memory_A ** agents = (xmachine_memory_A**)malloc(count * sizeof(xmachine_memory_A*));
	for (unsigned int i = 0; i < count; i++) {
		agents[i] = h_allocate_agent_A();
	}
	return agents;
}
void h_free_agent_A_array(xmachine_memory_A*** agents, unsigned int count){
	for (unsigned int i = 0; i < count; i++) {
		h_free_agent_A(&((*agents)[i]));
	}
	free((*agents));
	(*agents) = NULL;
}

void h_unpack_agents_A_AoS_to_SoA(xmachine_memory_A_list * dst, xmachine_memory_A** src, unsigned int count){
	if(count > 0){
		for(unsigned int i = 0; i < count; i++){
			 
			dst->id[i] = src[i]->id;
			 
			dst->x[i] = src[i]->x;
			 
			dst->y[i] = src[i]->y;
			 
			dst->z[i] = src[i]->z;
			 
			dst->fx[i] = src[i]->fx;
			 
			dst->fy[i] = src[i]->fy;
			 
			dst->fz[i] = src[i]->fz;
			 
			dst->colour[i] = src[i]->colour;
			
		}
	}
}


void h_add_agent_A_moving_A(xmachine_memory_A* agent){
	if (h_xmachine_memory_A_count + 1 > xmachine_memory_A_MAX){
		printf("Error: Buffer size of A agents in state moving_A will be exceeded by h_add_agent_A_moving_A\n");
		exit(EXIT_FAILURE);
	}	

	int blockSize;
	int minGridSize;
	int gridSize;
	unsigned int count = 1;
	
	// Copy data from host struct to device SoA for target state
	copy_single_xmachine_memory_A_hostToDevice(d_As_new, agent);

	// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_A_Agents, no_sm, count);
	gridSize = (count + blockSize - 1) / blockSize;
	append_A_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_As_moving_A, d_As_new, h_xmachine_memory_A_moving_A_count, count);
	gpuErrchkLaunch();
	// Update the number of agents in this state.
	h_xmachine_memory_A_moving_A_count += count;
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_A_moving_A_count, &h_xmachine_memory_A_moving_A_count, sizeof(int)));
	cudaDeviceSynchronize();

    // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
    h_As_moving_A_variable_id_data_iteration = 0;
    h_As_moving_A_variable_x_data_iteration = 0;
    h_As_moving_A_variable_y_data_iteration = 0;
    h_As_moving_A_variable_z_data_iteration = 0;
    h_As_moving_A_variable_fx_data_iteration = 0;
    h_As_moving_A_variable_fy_data_iteration = 0;
    h_As_moving_A_variable_fz_data_iteration = 0;
    h_As_moving_A_variable_colour_data_iteration = 0;
    

}
void h_add_agents_A_moving_A(xmachine_memory_A** agents, unsigned int count){
	if(count > 0){
		int blockSize;
		int minGridSize;
		int gridSize;

		if (h_xmachine_memory_A_count + count > xmachine_memory_A_MAX){
			printf("Error: Buffer size of A agents in state moving_A will be exceeded by h_add_agents_A_moving_A\n");
			exit(EXIT_FAILURE);
		}

		// Unpack data from AoS into the pre-existing SoA
		h_unpack_agents_A_AoS_to_SoA(h_As_moving_A, agents, count);

		// Copy data from the host SoA to the device SoA for the target state
		copy_partial_xmachine_memory_A_hostToDevice(d_As_new, h_As_moving_A, count);

		// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
		cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_A_Agents, no_sm, count);
		gridSize = (count + blockSize - 1) / blockSize;
		append_A_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_As_moving_A, d_As_new, h_xmachine_memory_A_moving_A_count, count);
		gpuErrchkLaunch();
		// Update the number of agents in this state.
		h_xmachine_memory_A_moving_A_count += count;
		gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_A_moving_A_count, &h_xmachine_memory_A_moving_A_count, sizeof(int)));
		cudaDeviceSynchronize();

        // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
        h_As_moving_A_variable_id_data_iteration = 0;
        h_As_moving_A_variable_x_data_iteration = 0;
        h_As_moving_A_variable_y_data_iteration = 0;
        h_As_moving_A_variable_z_data_iteration = 0;
        h_As_moving_A_variable_fx_data_iteration = 0;
        h_As_moving_A_variable_fy_data_iteration = 0;
        h_As_moving_A_variable_fz_data_iteration = 0;
        h_As_moving_A_variable_colour_data_iteration = 0;
        

	}
}


void h_add_agent_A_change_direction_A(xmachine_memory_A* agent){
	if (h_xmachine_memory_A_count + 1 > xmachine_memory_A_MAX){
		printf("Error: Buffer size of A agents in state change_direction_A will be exceeded by h_add_agent_A_change_direction_A\n");
		exit(EXIT_FAILURE);
	}	

	int blockSize;
	int minGridSize;
	int gridSize;
	unsigned int count = 1;
	
	// Copy data from host struct to device SoA for target state
	copy_single_xmachine_memory_A_hostToDevice(d_As_new, agent);

	// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_A_Agents, no_sm, count);
	gridSize = (count + blockSize - 1) / blockSize;
	append_A_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_As_change_direction_A, d_As_new, h_xmachine_memory_A_change_direction_A_count, count);
	gpuErrchkLaunch();
	// Update the number of agents in this state.
	h_xmachine_memory_A_change_direction_A_count += count;
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_A_change_direction_A_count, &h_xmachine_memory_A_change_direction_A_count, sizeof(int)));
	cudaDeviceSynchronize();

    // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
    h_As_change_direction_A_variable_id_data_iteration = 0;
    h_As_change_direction_A_variable_x_data_iteration = 0;
    h_As_change_direction_A_variable_y_data_iteration = 0;
    h_As_change_direction_A_variable_z_data_iteration = 0;
    h_As_change_direction_A_variable_fx_data_iteration = 0;
    h_As_change_direction_A_variable_fy_data_iteration = 0;
    h_As_change_direction_A_variable_fz_data_iteration = 0;
    h_As_change_direction_A_variable_colour_data_iteration = 0;
    

}
void h_add_agents_A_change_direction_A(xmachine_memory_A** agents, unsigned int count){
	if(count > 0){
		int blockSize;
		int minGridSize;
		int gridSize;

		if (h_xmachine_memory_A_count + count > xmachine_memory_A_MAX){
			printf("Error: Buffer size of A agents in state change_direction_A will be exceeded by h_add_agents_A_change_direction_A\n");
			exit(EXIT_FAILURE);
		}

		// Unpack data from AoS into the pre-existing SoA
		h_unpack_agents_A_AoS_to_SoA(h_As_change_direction_A, agents, count);

		// Copy data from the host SoA to the device SoA for the target state
		copy_partial_xmachine_memory_A_hostToDevice(d_As_new, h_As_change_direction_A, count);

		// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
		cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_A_Agents, no_sm, count);
		gridSize = (count + blockSize - 1) / blockSize;
		append_A_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_As_change_direction_A, d_As_new, h_xmachine_memory_A_change_direction_A_count, count);
		gpuErrchkLaunch();
		// Update the number of agents in this state.
		h_xmachine_memory_A_change_direction_A_count += count;
		gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_A_change_direction_A_count, &h_xmachine_memory_A_change_direction_A_count, sizeof(int)));
		cudaDeviceSynchronize();

        // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
        h_As_change_direction_A_variable_id_data_iteration = 0;
        h_As_change_direction_A_variable_x_data_iteration = 0;
        h_As_change_direction_A_variable_y_data_iteration = 0;
        h_As_change_direction_A_variable_z_data_iteration = 0;
        h_As_change_direction_A_variable_fx_data_iteration = 0;
        h_As_change_direction_A_variable_fy_data_iteration = 0;
        h_As_change_direction_A_variable_fz_data_iteration = 0;
        h_As_change_direction_A_variable_colour_data_iteration = 0;
        

	}
}


void h_add_agent_A_get_going_again_A(xmachine_memory_A* agent){
	if (h_xmachine_memory_A_count + 1 > xmachine_memory_A_MAX){
		printf("Error: Buffer size of A agents in state get_going_again_A will be exceeded by h_add_agent_A_get_going_again_A\n");
		exit(EXIT_FAILURE);
	}	

	int blockSize;
	int minGridSize;
	int gridSize;
	unsigned int count = 1;
	
	// Copy data from host struct to device SoA for target state
	copy_single_xmachine_memory_A_hostToDevice(d_As_new, agent);

	// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_A_Agents, no_sm, count);
	gridSize = (count + blockSize - 1) / blockSize;
	append_A_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_As_get_going_again_A, d_As_new, h_xmachine_memory_A_get_going_again_A_count, count);
	gpuErrchkLaunch();
	// Update the number of agents in this state.
	h_xmachine_memory_A_get_going_again_A_count += count;
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_A_get_going_again_A_count, &h_xmachine_memory_A_get_going_again_A_count, sizeof(int)));
	cudaDeviceSynchronize();

    // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
    h_As_get_going_again_A_variable_id_data_iteration = 0;
    h_As_get_going_again_A_variable_x_data_iteration = 0;
    h_As_get_going_again_A_variable_y_data_iteration = 0;
    h_As_get_going_again_A_variable_z_data_iteration = 0;
    h_As_get_going_again_A_variable_fx_data_iteration = 0;
    h_As_get_going_again_A_variable_fy_data_iteration = 0;
    h_As_get_going_again_A_variable_fz_data_iteration = 0;
    h_As_get_going_again_A_variable_colour_data_iteration = 0;
    

}
void h_add_agents_A_get_going_again_A(xmachine_memory_A** agents, unsigned int count){
	if(count > 0){
		int blockSize;
		int minGridSize;
		int gridSize;

		if (h_xmachine_memory_A_count + count > xmachine_memory_A_MAX){
			printf("Error: Buffer size of A agents in state get_going_again_A will be exceeded by h_add_agents_A_get_going_again_A\n");
			exit(EXIT_FAILURE);
		}

		// Unpack data from AoS into the pre-existing SoA
		h_unpack_agents_A_AoS_to_SoA(h_As_get_going_again_A, agents, count);

		// Copy data from the host SoA to the device SoA for the target state
		copy_partial_xmachine_memory_A_hostToDevice(d_As_new, h_As_get_going_again_A, count);

		// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
		cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_A_Agents, no_sm, count);
		gridSize = (count + blockSize - 1) / blockSize;
		append_A_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_As_get_going_again_A, d_As_new, h_xmachine_memory_A_get_going_again_A_count, count);
		gpuErrchkLaunch();
		// Update the number of agents in this state.
		h_xmachine_memory_A_get_going_again_A_count += count;
		gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_A_get_going_again_A_count, &h_xmachine_memory_A_get_going_again_A_count, sizeof(int)));
		cudaDeviceSynchronize();

        // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
        h_As_get_going_again_A_variable_id_data_iteration = 0;
        h_As_get_going_again_A_variable_x_data_iteration = 0;
        h_As_get_going_again_A_variable_y_data_iteration = 0;
        h_As_get_going_again_A_variable_z_data_iteration = 0;
        h_As_get_going_again_A_variable_fx_data_iteration = 0;
        h_As_get_going_again_A_variable_fy_data_iteration = 0;
        h_As_get_going_again_A_variable_fz_data_iteration = 0;
        h_As_get_going_again_A_variable_colour_data_iteration = 0;
        

	}
}

xmachine_memory_B* h_allocate_agent_B(){
	xmachine_memory_B* agent = (xmachine_memory_B*)malloc(sizeof(xmachine_memory_B));
	// Memset the whole agent strcuture
    memset(agent, 0, sizeof(xmachine_memory_B));

	return agent;
}
void h_free_agent_B(xmachine_memory_B** agent){
 
	free((*agent));
	(*agent) = NULL;
}
xmachine_memory_B** h_allocate_agent_B_array(unsigned int count){
	xmachine_memory_B ** agents = (xmachine_memory_B**)malloc(count * sizeof(xmachine_memory_B*));
	for (unsigned int i = 0; i < count; i++) {
		agents[i] = h_allocate_agent_B();
	}
	return agents;
}
void h_free_agent_B_array(xmachine_memory_B*** agents, unsigned int count){
	for (unsigned int i = 0; i < count; i++) {
		h_free_agent_B(&((*agents)[i]));
	}
	free((*agents));
	(*agents) = NULL;
}

void h_unpack_agents_B_AoS_to_SoA(xmachine_memory_B_list * dst, xmachine_memory_B** src, unsigned int count){
	if(count > 0){
		for(unsigned int i = 0; i < count; i++){
			 
			dst->id[i] = src[i]->id;
			 
			dst->x[i] = src[i]->x;
			 
			dst->y[i] = src[i]->y;
			 
			dst->z[i] = src[i]->z;
			 
			dst->fx[i] = src[i]->fx;
			 
			dst->fy[i] = src[i]->fy;
			 
			dst->fz[i] = src[i]->fz;
			 
			dst->colour[i] = src[i]->colour;
			
		}
	}
}


void h_add_agent_B_moving_B(xmachine_memory_B* agent){
	if (h_xmachine_memory_B_count + 1 > xmachine_memory_B_MAX){
		printf("Error: Buffer size of B agents in state moving_B will be exceeded by h_add_agent_B_moving_B\n");
		exit(EXIT_FAILURE);
	}	

	int blockSize;
	int minGridSize;
	int gridSize;
	unsigned int count = 1;
	
	// Copy data from host struct to device SoA for target state
	copy_single_xmachine_memory_B_hostToDevice(d_Bs_new, agent);

	// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_B_Agents, no_sm, count);
	gridSize = (count + blockSize - 1) / blockSize;
	append_B_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_Bs_moving_B, d_Bs_new, h_xmachine_memory_B_moving_B_count, count);
	gpuErrchkLaunch();
	// Update the number of agents in this state.
	h_xmachine_memory_B_moving_B_count += count;
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_B_moving_B_count, &h_xmachine_memory_B_moving_B_count, sizeof(int)));
	cudaDeviceSynchronize();

    // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
    h_Bs_moving_B_variable_id_data_iteration = 0;
    h_Bs_moving_B_variable_x_data_iteration = 0;
    h_Bs_moving_B_variable_y_data_iteration = 0;
    h_Bs_moving_B_variable_z_data_iteration = 0;
    h_Bs_moving_B_variable_fx_data_iteration = 0;
    h_Bs_moving_B_variable_fy_data_iteration = 0;
    h_Bs_moving_B_variable_fz_data_iteration = 0;
    h_Bs_moving_B_variable_colour_data_iteration = 0;
    

}
void h_add_agents_B_moving_B(xmachine_memory_B** agents, unsigned int count){
	if(count > 0){
		int blockSize;
		int minGridSize;
		int gridSize;

		if (h_xmachine_memory_B_count + count > xmachine_memory_B_MAX){
			printf("Error: Buffer size of B agents in state moving_B will be exceeded by h_add_agents_B_moving_B\n");
			exit(EXIT_FAILURE);
		}

		// Unpack data from AoS into the pre-existing SoA
		h_unpack_agents_B_AoS_to_SoA(h_Bs_moving_B, agents, count);

		// Copy data from the host SoA to the device SoA for the target state
		copy_partial_xmachine_memory_B_hostToDevice(d_Bs_new, h_Bs_moving_B, count);

		// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
		cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_B_Agents, no_sm, count);
		gridSize = (count + blockSize - 1) / blockSize;
		append_B_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_Bs_moving_B, d_Bs_new, h_xmachine_memory_B_moving_B_count, count);
		gpuErrchkLaunch();
		// Update the number of agents in this state.
		h_xmachine_memory_B_moving_B_count += count;
		gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_B_moving_B_count, &h_xmachine_memory_B_moving_B_count, sizeof(int)));
		cudaDeviceSynchronize();

        // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
        h_Bs_moving_B_variable_id_data_iteration = 0;
        h_Bs_moving_B_variable_x_data_iteration = 0;
        h_Bs_moving_B_variable_y_data_iteration = 0;
        h_Bs_moving_B_variable_z_data_iteration = 0;
        h_Bs_moving_B_variable_fx_data_iteration = 0;
        h_Bs_moving_B_variable_fy_data_iteration = 0;
        h_Bs_moving_B_variable_fz_data_iteration = 0;
        h_Bs_moving_B_variable_colour_data_iteration = 0;
        

	}
}


void h_add_agent_B_change_direction_B(xmachine_memory_B* agent){
	if (h_xmachine_memory_B_count + 1 > xmachine_memory_B_MAX){
		printf("Error: Buffer size of B agents in state change_direction_B will be exceeded by h_add_agent_B_change_direction_B\n");
		exit(EXIT_FAILURE);
	}	

	int blockSize;
	int minGridSize;
	int gridSize;
	unsigned int count = 1;
	
	// Copy data from host struct to device SoA for target state
	copy_single_xmachine_memory_B_hostToDevice(d_Bs_new, agent);

	// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_B_Agents, no_sm, count);
	gridSize = (count + blockSize - 1) / blockSize;
	append_B_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_Bs_change_direction_B, d_Bs_new, h_xmachine_memory_B_change_direction_B_count, count);
	gpuErrchkLaunch();
	// Update the number of agents in this state.
	h_xmachine_memory_B_change_direction_B_count += count;
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_B_change_direction_B_count, &h_xmachine_memory_B_change_direction_B_count, sizeof(int)));
	cudaDeviceSynchronize();

    // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
    h_Bs_change_direction_B_variable_id_data_iteration = 0;
    h_Bs_change_direction_B_variable_x_data_iteration = 0;
    h_Bs_change_direction_B_variable_y_data_iteration = 0;
    h_Bs_change_direction_B_variable_z_data_iteration = 0;
    h_Bs_change_direction_B_variable_fx_data_iteration = 0;
    h_Bs_change_direction_B_variable_fy_data_iteration = 0;
    h_Bs_change_direction_B_variable_fz_data_iteration = 0;
    h_Bs_change_direction_B_variable_colour_data_iteration = 0;
    

}
void h_add_agents_B_change_direction_B(xmachine_memory_B** agents, unsigned int count){
	if(count > 0){
		int blockSize;
		int minGridSize;
		int gridSize;

		if (h_xmachine_memory_B_count + count > xmachine_memory_B_MAX){
			printf("Error: Buffer size of B agents in state change_direction_B will be exceeded by h_add_agents_B_change_direction_B\n");
			exit(EXIT_FAILURE);
		}

		// Unpack data from AoS into the pre-existing SoA
		h_unpack_agents_B_AoS_to_SoA(h_Bs_change_direction_B, agents, count);

		// Copy data from the host SoA to the device SoA for the target state
		copy_partial_xmachine_memory_B_hostToDevice(d_Bs_new, h_Bs_change_direction_B, count);

		// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
		cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_B_Agents, no_sm, count);
		gridSize = (count + blockSize - 1) / blockSize;
		append_B_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_Bs_change_direction_B, d_Bs_new, h_xmachine_memory_B_change_direction_B_count, count);
		gpuErrchkLaunch();
		// Update the number of agents in this state.
		h_xmachine_memory_B_change_direction_B_count += count;
		gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_B_change_direction_B_count, &h_xmachine_memory_B_change_direction_B_count, sizeof(int)));
		cudaDeviceSynchronize();

        // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
        h_Bs_change_direction_B_variable_id_data_iteration = 0;
        h_Bs_change_direction_B_variable_x_data_iteration = 0;
        h_Bs_change_direction_B_variable_y_data_iteration = 0;
        h_Bs_change_direction_B_variable_z_data_iteration = 0;
        h_Bs_change_direction_B_variable_fx_data_iteration = 0;
        h_Bs_change_direction_B_variable_fy_data_iteration = 0;
        h_Bs_change_direction_B_variable_fz_data_iteration = 0;
        h_Bs_change_direction_B_variable_colour_data_iteration = 0;
        

	}
}


void h_add_agent_B_get_going_again_B(xmachine_memory_B* agent){
	if (h_xmachine_memory_B_count + 1 > xmachine_memory_B_MAX){
		printf("Error: Buffer size of B agents in state get_going_again_B will be exceeded by h_add_agent_B_get_going_again_B\n");
		exit(EXIT_FAILURE);
	}	

	int blockSize;
	int minGridSize;
	int gridSize;
	unsigned int count = 1;
	
	// Copy data from host struct to device SoA for target state
	copy_single_xmachine_memory_B_hostToDevice(d_Bs_new, agent);

	// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_B_Agents, no_sm, count);
	gridSize = (count + blockSize - 1) / blockSize;
	append_B_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_Bs_get_going_again_B, d_Bs_new, h_xmachine_memory_B_get_going_again_B_count, count);
	gpuErrchkLaunch();
	// Update the number of agents in this state.
	h_xmachine_memory_B_get_going_again_B_count += count;
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_B_get_going_again_B_count, &h_xmachine_memory_B_get_going_again_B_count, sizeof(int)));
	cudaDeviceSynchronize();

    // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
    h_Bs_get_going_again_B_variable_id_data_iteration = 0;
    h_Bs_get_going_again_B_variable_x_data_iteration = 0;
    h_Bs_get_going_again_B_variable_y_data_iteration = 0;
    h_Bs_get_going_again_B_variable_z_data_iteration = 0;
    h_Bs_get_going_again_B_variable_fx_data_iteration = 0;
    h_Bs_get_going_again_B_variable_fy_data_iteration = 0;
    h_Bs_get_going_again_B_variable_fz_data_iteration = 0;
    h_Bs_get_going_again_B_variable_colour_data_iteration = 0;
    

}
void h_add_agents_B_get_going_again_B(xmachine_memory_B** agents, unsigned int count){
	if(count > 0){
		int blockSize;
		int minGridSize;
		int gridSize;

		if (h_xmachine_memory_B_count + count > xmachine_memory_B_MAX){
			printf("Error: Buffer size of B agents in state get_going_again_B will be exceeded by h_add_agents_B_get_going_again_B\n");
			exit(EXIT_FAILURE);
		}

		// Unpack data from AoS into the pre-existing SoA
		h_unpack_agents_B_AoS_to_SoA(h_Bs_get_going_again_B, agents, count);

		// Copy data from the host SoA to the device SoA for the target state
		copy_partial_xmachine_memory_B_hostToDevice(d_Bs_new, h_Bs_get_going_again_B, count);

		// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
		cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_B_Agents, no_sm, count);
		gridSize = (count + blockSize - 1) / blockSize;
		append_B_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_Bs_get_going_again_B, d_Bs_new, h_xmachine_memory_B_get_going_again_B_count, count);
		gpuErrchkLaunch();
		// Update the number of agents in this state.
		h_xmachine_memory_B_get_going_again_B_count += count;
		gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_B_get_going_again_B_count, &h_xmachine_memory_B_get_going_again_B_count, sizeof(int)));
		cudaDeviceSynchronize();

        // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
        h_Bs_get_going_again_B_variable_id_data_iteration = 0;
        h_Bs_get_going_again_B_variable_x_data_iteration = 0;
        h_Bs_get_going_again_B_variable_y_data_iteration = 0;
        h_Bs_get_going_again_B_variable_z_data_iteration = 0;
        h_Bs_get_going_again_B_variable_fx_data_iteration = 0;
        h_Bs_get_going_again_B_variable_fy_data_iteration = 0;
        h_Bs_get_going_again_B_variable_fz_data_iteration = 0;
        h_Bs_get_going_again_B_variable_colour_data_iteration = 0;
        

	}
}


/*  Analytics Functions */

int reduce_A_moving_A_id_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_As_moving_A->id),  thrust::device_pointer_cast(d_As_moving_A->id) + h_xmachine_memory_A_moving_A_count);
}

int count_A_moving_A_id_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_As_moving_A->id),  thrust::device_pointer_cast(d_As_moving_A->id) + h_xmachine_memory_A_moving_A_count, count_value);
}
int min_A_moving_A_id_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_As_moving_A->id);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_A_moving_A_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_A_moving_A_id_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_As_moving_A->id);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_A_moving_A_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_A_moving_A_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_As_moving_A->x),  thrust::device_pointer_cast(d_As_moving_A->x) + h_xmachine_memory_A_moving_A_count);
}

float min_A_moving_A_x_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_As_moving_A->x);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_A_moving_A_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_A_moving_A_x_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_As_moving_A->x);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_A_moving_A_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_A_moving_A_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_As_moving_A->y),  thrust::device_pointer_cast(d_As_moving_A->y) + h_xmachine_memory_A_moving_A_count);
}

float min_A_moving_A_y_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_As_moving_A->y);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_A_moving_A_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_A_moving_A_y_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_As_moving_A->y);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_A_moving_A_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_A_moving_A_z_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_As_moving_A->z),  thrust::device_pointer_cast(d_As_moving_A->z) + h_xmachine_memory_A_moving_A_count);
}

float min_A_moving_A_z_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_As_moving_A->z);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_A_moving_A_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_A_moving_A_z_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_As_moving_A->z);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_A_moving_A_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_A_moving_A_fx_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_As_moving_A->fx),  thrust::device_pointer_cast(d_As_moving_A->fx) + h_xmachine_memory_A_moving_A_count);
}

float min_A_moving_A_fx_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_As_moving_A->fx);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_A_moving_A_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_A_moving_A_fx_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_As_moving_A->fx);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_A_moving_A_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_A_moving_A_fy_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_As_moving_A->fy),  thrust::device_pointer_cast(d_As_moving_A->fy) + h_xmachine_memory_A_moving_A_count);
}

float min_A_moving_A_fy_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_As_moving_A->fy);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_A_moving_A_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_A_moving_A_fy_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_As_moving_A->fy);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_A_moving_A_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_A_moving_A_fz_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_As_moving_A->fz),  thrust::device_pointer_cast(d_As_moving_A->fz) + h_xmachine_memory_A_moving_A_count);
}

float min_A_moving_A_fz_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_As_moving_A->fz);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_A_moving_A_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_A_moving_A_fz_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_As_moving_A->fz);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_A_moving_A_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_A_moving_A_colour_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_As_moving_A->colour),  thrust::device_pointer_cast(d_As_moving_A->colour) + h_xmachine_memory_A_moving_A_count);
}

int count_A_moving_A_colour_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_As_moving_A->colour),  thrust::device_pointer_cast(d_As_moving_A->colour) + h_xmachine_memory_A_moving_A_count, count_value);
}
int min_A_moving_A_colour_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_As_moving_A->colour);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_A_moving_A_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_A_moving_A_colour_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_As_moving_A->colour);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_A_moving_A_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_A_change_direction_A_id_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_As_change_direction_A->id),  thrust::device_pointer_cast(d_As_change_direction_A->id) + h_xmachine_memory_A_change_direction_A_count);
}

int count_A_change_direction_A_id_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_As_change_direction_A->id),  thrust::device_pointer_cast(d_As_change_direction_A->id) + h_xmachine_memory_A_change_direction_A_count, count_value);
}
int min_A_change_direction_A_id_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_As_change_direction_A->id);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_A_change_direction_A_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_A_change_direction_A_id_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_As_change_direction_A->id);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_A_change_direction_A_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_A_change_direction_A_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_As_change_direction_A->x),  thrust::device_pointer_cast(d_As_change_direction_A->x) + h_xmachine_memory_A_change_direction_A_count);
}

float min_A_change_direction_A_x_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_As_change_direction_A->x);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_A_change_direction_A_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_A_change_direction_A_x_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_As_change_direction_A->x);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_A_change_direction_A_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_A_change_direction_A_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_As_change_direction_A->y),  thrust::device_pointer_cast(d_As_change_direction_A->y) + h_xmachine_memory_A_change_direction_A_count);
}

float min_A_change_direction_A_y_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_As_change_direction_A->y);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_A_change_direction_A_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_A_change_direction_A_y_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_As_change_direction_A->y);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_A_change_direction_A_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_A_change_direction_A_z_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_As_change_direction_A->z),  thrust::device_pointer_cast(d_As_change_direction_A->z) + h_xmachine_memory_A_change_direction_A_count);
}

float min_A_change_direction_A_z_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_As_change_direction_A->z);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_A_change_direction_A_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_A_change_direction_A_z_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_As_change_direction_A->z);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_A_change_direction_A_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_A_change_direction_A_fx_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_As_change_direction_A->fx),  thrust::device_pointer_cast(d_As_change_direction_A->fx) + h_xmachine_memory_A_change_direction_A_count);
}

float min_A_change_direction_A_fx_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_As_change_direction_A->fx);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_A_change_direction_A_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_A_change_direction_A_fx_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_As_change_direction_A->fx);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_A_change_direction_A_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_A_change_direction_A_fy_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_As_change_direction_A->fy),  thrust::device_pointer_cast(d_As_change_direction_A->fy) + h_xmachine_memory_A_change_direction_A_count);
}

float min_A_change_direction_A_fy_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_As_change_direction_A->fy);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_A_change_direction_A_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_A_change_direction_A_fy_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_As_change_direction_A->fy);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_A_change_direction_A_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_A_change_direction_A_fz_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_As_change_direction_A->fz),  thrust::device_pointer_cast(d_As_change_direction_A->fz) + h_xmachine_memory_A_change_direction_A_count);
}

float min_A_change_direction_A_fz_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_As_change_direction_A->fz);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_A_change_direction_A_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_A_change_direction_A_fz_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_As_change_direction_A->fz);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_A_change_direction_A_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_A_change_direction_A_colour_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_As_change_direction_A->colour),  thrust::device_pointer_cast(d_As_change_direction_A->colour) + h_xmachine_memory_A_change_direction_A_count);
}

int count_A_change_direction_A_colour_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_As_change_direction_A->colour),  thrust::device_pointer_cast(d_As_change_direction_A->colour) + h_xmachine_memory_A_change_direction_A_count, count_value);
}
int min_A_change_direction_A_colour_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_As_change_direction_A->colour);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_A_change_direction_A_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_A_change_direction_A_colour_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_As_change_direction_A->colour);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_A_change_direction_A_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_A_get_going_again_A_id_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_As_get_going_again_A->id),  thrust::device_pointer_cast(d_As_get_going_again_A->id) + h_xmachine_memory_A_get_going_again_A_count);
}

int count_A_get_going_again_A_id_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_As_get_going_again_A->id),  thrust::device_pointer_cast(d_As_get_going_again_A->id) + h_xmachine_memory_A_get_going_again_A_count, count_value);
}
int min_A_get_going_again_A_id_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_As_get_going_again_A->id);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_A_get_going_again_A_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_A_get_going_again_A_id_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_As_get_going_again_A->id);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_A_get_going_again_A_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_A_get_going_again_A_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_As_get_going_again_A->x),  thrust::device_pointer_cast(d_As_get_going_again_A->x) + h_xmachine_memory_A_get_going_again_A_count);
}

float min_A_get_going_again_A_x_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_As_get_going_again_A->x);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_A_get_going_again_A_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_A_get_going_again_A_x_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_As_get_going_again_A->x);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_A_get_going_again_A_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_A_get_going_again_A_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_As_get_going_again_A->y),  thrust::device_pointer_cast(d_As_get_going_again_A->y) + h_xmachine_memory_A_get_going_again_A_count);
}

float min_A_get_going_again_A_y_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_As_get_going_again_A->y);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_A_get_going_again_A_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_A_get_going_again_A_y_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_As_get_going_again_A->y);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_A_get_going_again_A_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_A_get_going_again_A_z_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_As_get_going_again_A->z),  thrust::device_pointer_cast(d_As_get_going_again_A->z) + h_xmachine_memory_A_get_going_again_A_count);
}

float min_A_get_going_again_A_z_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_As_get_going_again_A->z);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_A_get_going_again_A_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_A_get_going_again_A_z_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_As_get_going_again_A->z);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_A_get_going_again_A_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_A_get_going_again_A_fx_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_As_get_going_again_A->fx),  thrust::device_pointer_cast(d_As_get_going_again_A->fx) + h_xmachine_memory_A_get_going_again_A_count);
}

float min_A_get_going_again_A_fx_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_As_get_going_again_A->fx);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_A_get_going_again_A_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_A_get_going_again_A_fx_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_As_get_going_again_A->fx);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_A_get_going_again_A_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_A_get_going_again_A_fy_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_As_get_going_again_A->fy),  thrust::device_pointer_cast(d_As_get_going_again_A->fy) + h_xmachine_memory_A_get_going_again_A_count);
}

float min_A_get_going_again_A_fy_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_As_get_going_again_A->fy);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_A_get_going_again_A_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_A_get_going_again_A_fy_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_As_get_going_again_A->fy);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_A_get_going_again_A_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_A_get_going_again_A_fz_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_As_get_going_again_A->fz),  thrust::device_pointer_cast(d_As_get_going_again_A->fz) + h_xmachine_memory_A_get_going_again_A_count);
}

float min_A_get_going_again_A_fz_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_As_get_going_again_A->fz);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_A_get_going_again_A_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_A_get_going_again_A_fz_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_As_get_going_again_A->fz);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_A_get_going_again_A_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_A_get_going_again_A_colour_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_As_get_going_again_A->colour),  thrust::device_pointer_cast(d_As_get_going_again_A->colour) + h_xmachine_memory_A_get_going_again_A_count);
}

int count_A_get_going_again_A_colour_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_As_get_going_again_A->colour),  thrust::device_pointer_cast(d_As_get_going_again_A->colour) + h_xmachine_memory_A_get_going_again_A_count, count_value);
}
int min_A_get_going_again_A_colour_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_As_get_going_again_A->colour);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_A_get_going_again_A_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_A_get_going_again_A_colour_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_As_get_going_again_A->colour);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_A_get_going_again_A_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_B_moving_B_id_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Bs_moving_B->id),  thrust::device_pointer_cast(d_Bs_moving_B->id) + h_xmachine_memory_B_moving_B_count);
}

int count_B_moving_B_id_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Bs_moving_B->id),  thrust::device_pointer_cast(d_Bs_moving_B->id) + h_xmachine_memory_B_moving_B_count, count_value);
}
int min_B_moving_B_id_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Bs_moving_B->id);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_B_moving_B_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_B_moving_B_id_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Bs_moving_B->id);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_B_moving_B_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_B_moving_B_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Bs_moving_B->x),  thrust::device_pointer_cast(d_Bs_moving_B->x) + h_xmachine_memory_B_moving_B_count);
}

float min_B_moving_B_x_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Bs_moving_B->x);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_B_moving_B_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_B_moving_B_x_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Bs_moving_B->x);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_B_moving_B_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_B_moving_B_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Bs_moving_B->y),  thrust::device_pointer_cast(d_Bs_moving_B->y) + h_xmachine_memory_B_moving_B_count);
}

float min_B_moving_B_y_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Bs_moving_B->y);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_B_moving_B_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_B_moving_B_y_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Bs_moving_B->y);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_B_moving_B_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_B_moving_B_z_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Bs_moving_B->z),  thrust::device_pointer_cast(d_Bs_moving_B->z) + h_xmachine_memory_B_moving_B_count);
}

float min_B_moving_B_z_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Bs_moving_B->z);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_B_moving_B_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_B_moving_B_z_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Bs_moving_B->z);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_B_moving_B_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_B_moving_B_fx_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Bs_moving_B->fx),  thrust::device_pointer_cast(d_Bs_moving_B->fx) + h_xmachine_memory_B_moving_B_count);
}

float min_B_moving_B_fx_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Bs_moving_B->fx);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_B_moving_B_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_B_moving_B_fx_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Bs_moving_B->fx);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_B_moving_B_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_B_moving_B_fy_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Bs_moving_B->fy),  thrust::device_pointer_cast(d_Bs_moving_B->fy) + h_xmachine_memory_B_moving_B_count);
}

float min_B_moving_B_fy_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Bs_moving_B->fy);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_B_moving_B_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_B_moving_B_fy_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Bs_moving_B->fy);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_B_moving_B_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_B_moving_B_fz_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Bs_moving_B->fz),  thrust::device_pointer_cast(d_Bs_moving_B->fz) + h_xmachine_memory_B_moving_B_count);
}

float min_B_moving_B_fz_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Bs_moving_B->fz);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_B_moving_B_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_B_moving_B_fz_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Bs_moving_B->fz);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_B_moving_B_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_B_moving_B_colour_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Bs_moving_B->colour),  thrust::device_pointer_cast(d_Bs_moving_B->colour) + h_xmachine_memory_B_moving_B_count);
}

int count_B_moving_B_colour_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Bs_moving_B->colour),  thrust::device_pointer_cast(d_Bs_moving_B->colour) + h_xmachine_memory_B_moving_B_count, count_value);
}
int min_B_moving_B_colour_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Bs_moving_B->colour);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_B_moving_B_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_B_moving_B_colour_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Bs_moving_B->colour);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_B_moving_B_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_B_change_direction_B_id_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Bs_change_direction_B->id),  thrust::device_pointer_cast(d_Bs_change_direction_B->id) + h_xmachine_memory_B_change_direction_B_count);
}

int count_B_change_direction_B_id_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Bs_change_direction_B->id),  thrust::device_pointer_cast(d_Bs_change_direction_B->id) + h_xmachine_memory_B_change_direction_B_count, count_value);
}
int min_B_change_direction_B_id_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Bs_change_direction_B->id);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_B_change_direction_B_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_B_change_direction_B_id_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Bs_change_direction_B->id);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_B_change_direction_B_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_B_change_direction_B_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Bs_change_direction_B->x),  thrust::device_pointer_cast(d_Bs_change_direction_B->x) + h_xmachine_memory_B_change_direction_B_count);
}

float min_B_change_direction_B_x_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Bs_change_direction_B->x);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_B_change_direction_B_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_B_change_direction_B_x_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Bs_change_direction_B->x);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_B_change_direction_B_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_B_change_direction_B_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Bs_change_direction_B->y),  thrust::device_pointer_cast(d_Bs_change_direction_B->y) + h_xmachine_memory_B_change_direction_B_count);
}

float min_B_change_direction_B_y_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Bs_change_direction_B->y);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_B_change_direction_B_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_B_change_direction_B_y_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Bs_change_direction_B->y);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_B_change_direction_B_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_B_change_direction_B_z_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Bs_change_direction_B->z),  thrust::device_pointer_cast(d_Bs_change_direction_B->z) + h_xmachine_memory_B_change_direction_B_count);
}

float min_B_change_direction_B_z_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Bs_change_direction_B->z);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_B_change_direction_B_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_B_change_direction_B_z_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Bs_change_direction_B->z);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_B_change_direction_B_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_B_change_direction_B_fx_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Bs_change_direction_B->fx),  thrust::device_pointer_cast(d_Bs_change_direction_B->fx) + h_xmachine_memory_B_change_direction_B_count);
}

float min_B_change_direction_B_fx_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Bs_change_direction_B->fx);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_B_change_direction_B_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_B_change_direction_B_fx_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Bs_change_direction_B->fx);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_B_change_direction_B_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_B_change_direction_B_fy_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Bs_change_direction_B->fy),  thrust::device_pointer_cast(d_Bs_change_direction_B->fy) + h_xmachine_memory_B_change_direction_B_count);
}

float min_B_change_direction_B_fy_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Bs_change_direction_B->fy);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_B_change_direction_B_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_B_change_direction_B_fy_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Bs_change_direction_B->fy);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_B_change_direction_B_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_B_change_direction_B_fz_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Bs_change_direction_B->fz),  thrust::device_pointer_cast(d_Bs_change_direction_B->fz) + h_xmachine_memory_B_change_direction_B_count);
}

float min_B_change_direction_B_fz_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Bs_change_direction_B->fz);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_B_change_direction_B_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_B_change_direction_B_fz_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Bs_change_direction_B->fz);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_B_change_direction_B_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_B_change_direction_B_colour_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Bs_change_direction_B->colour),  thrust::device_pointer_cast(d_Bs_change_direction_B->colour) + h_xmachine_memory_B_change_direction_B_count);
}

int count_B_change_direction_B_colour_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Bs_change_direction_B->colour),  thrust::device_pointer_cast(d_Bs_change_direction_B->colour) + h_xmachine_memory_B_change_direction_B_count, count_value);
}
int min_B_change_direction_B_colour_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Bs_change_direction_B->colour);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_B_change_direction_B_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_B_change_direction_B_colour_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Bs_change_direction_B->colour);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_B_change_direction_B_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_B_get_going_again_B_id_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Bs_get_going_again_B->id),  thrust::device_pointer_cast(d_Bs_get_going_again_B->id) + h_xmachine_memory_B_get_going_again_B_count);
}

int count_B_get_going_again_B_id_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Bs_get_going_again_B->id),  thrust::device_pointer_cast(d_Bs_get_going_again_B->id) + h_xmachine_memory_B_get_going_again_B_count, count_value);
}
int min_B_get_going_again_B_id_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Bs_get_going_again_B->id);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_B_get_going_again_B_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_B_get_going_again_B_id_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Bs_get_going_again_B->id);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_B_get_going_again_B_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_B_get_going_again_B_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Bs_get_going_again_B->x),  thrust::device_pointer_cast(d_Bs_get_going_again_B->x) + h_xmachine_memory_B_get_going_again_B_count);
}

float min_B_get_going_again_B_x_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Bs_get_going_again_B->x);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_B_get_going_again_B_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_B_get_going_again_B_x_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Bs_get_going_again_B->x);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_B_get_going_again_B_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_B_get_going_again_B_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Bs_get_going_again_B->y),  thrust::device_pointer_cast(d_Bs_get_going_again_B->y) + h_xmachine_memory_B_get_going_again_B_count);
}

float min_B_get_going_again_B_y_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Bs_get_going_again_B->y);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_B_get_going_again_B_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_B_get_going_again_B_y_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Bs_get_going_again_B->y);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_B_get_going_again_B_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_B_get_going_again_B_z_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Bs_get_going_again_B->z),  thrust::device_pointer_cast(d_Bs_get_going_again_B->z) + h_xmachine_memory_B_get_going_again_B_count);
}

float min_B_get_going_again_B_z_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Bs_get_going_again_B->z);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_B_get_going_again_B_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_B_get_going_again_B_z_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Bs_get_going_again_B->z);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_B_get_going_again_B_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_B_get_going_again_B_fx_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Bs_get_going_again_B->fx),  thrust::device_pointer_cast(d_Bs_get_going_again_B->fx) + h_xmachine_memory_B_get_going_again_B_count);
}

float min_B_get_going_again_B_fx_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Bs_get_going_again_B->fx);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_B_get_going_again_B_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_B_get_going_again_B_fx_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Bs_get_going_again_B->fx);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_B_get_going_again_B_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_B_get_going_again_B_fy_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Bs_get_going_again_B->fy),  thrust::device_pointer_cast(d_Bs_get_going_again_B->fy) + h_xmachine_memory_B_get_going_again_B_count);
}

float min_B_get_going_again_B_fy_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Bs_get_going_again_B->fy);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_B_get_going_again_B_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_B_get_going_again_B_fy_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Bs_get_going_again_B->fy);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_B_get_going_again_B_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_B_get_going_again_B_fz_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Bs_get_going_again_B->fz),  thrust::device_pointer_cast(d_Bs_get_going_again_B->fz) + h_xmachine_memory_B_get_going_again_B_count);
}

float min_B_get_going_again_B_fz_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Bs_get_going_again_B->fz);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_B_get_going_again_B_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_B_get_going_again_B_fz_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Bs_get_going_again_B->fz);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_B_get_going_again_B_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_B_get_going_again_B_colour_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Bs_get_going_again_B->colour),  thrust::device_pointer_cast(d_Bs_get_going_again_B->colour) + h_xmachine_memory_B_get_going_again_B_count);
}

int count_B_get_going_again_B_colour_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Bs_get_going_again_B->colour),  thrust::device_pointer_cast(d_Bs_get_going_again_B->colour) + h_xmachine_memory_B_get_going_again_B_count, count_value);
}
int min_B_get_going_again_B_colour_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Bs_get_going_again_B->colour);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_B_get_going_again_B_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_B_get_going_again_B_colour_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Bs_get_going_again_B->colour);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_B_get_going_again_B_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}



/* Agent functions */


	
/* Shared memory size calculator for agent function */
int A_move_A_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** A_move_A
 * Agent function prototype for move_A function of A agent
 */
void A_move_A(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_A_moving_A_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_A_moving_A_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_A_list* As_moving_A_temp = d_As;
	d_As = d_As_moving_A;
	d_As_moving_A = As_moving_A_temp;
	//set working count to current state count
	h_xmachine_memory_A_count = h_xmachine_memory_A_moving_A_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_A_count, &h_xmachine_memory_A_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_A_moving_A_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_A_moving_A_count, &h_xmachine_memory_A_moving_A_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_move_A, A_move_A_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = A_move_A_sm_size(blockSize);
	
	
	
	
	//MAIN XMACHINE FUNCTION CALL (move_A)
	//Reallocate   : false
	//Input        : 
	//Output       : 
	//Agent Output : 
	GPUFLAME_move_A<<<g, b, sm_size, stream>>>(d_As);
	gpuErrchkLaunch();
	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_A_moving_A_count+h_xmachine_memory_A_count > xmachine_memory_A_MAX){
		printf("Error: Buffer size of move_A agents in state moving_A will be exceeded moving working agents to next state in function move_A\n");
      exit(EXIT_FAILURE);
      }
      
  //pointer swap the updated data
  As_moving_A_temp = d_As;
  d_As = d_As_moving_A;
  d_As_moving_A = As_moving_A_temp;
        
	//update new state agent size
	h_xmachine_memory_A_moving_A_count += h_xmachine_memory_A_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_A_moving_A_count, &h_xmachine_memory_A_moving_A_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int A_reverse_direction_A_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** A_reverse_direction_A
 * Agent function prototype for reverse_direction_A function of A agent
 */
void A_reverse_direction_A(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_A_moving_A_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_A_moving_A_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_A_list* As_moving_A_temp = d_As;
	d_As = d_As_moving_A;
	d_As_moving_A = As_moving_A_temp;
	//set working count to current state count
	h_xmachine_memory_A_count = h_xmachine_memory_A_moving_A_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_A_count, &h_xmachine_memory_A_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_A_moving_A_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_A_moving_A_count, &h_xmachine_memory_A_moving_A_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_reverse_direction_A, A_reverse_direction_A_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = A_reverse_direction_A_sm_size(blockSize);
	
	
	
	
	//MAIN XMACHINE FUNCTION CALL (reverse_direction_A)
	//Reallocate   : false
	//Input        : 
	//Output       : 
	//Agent Output : 
	GPUFLAME_reverse_direction_A<<<g, b, sm_size, stream>>>(d_As);
	gpuErrchkLaunch();
	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_A_change_direction_A_count+h_xmachine_memory_A_count > xmachine_memory_A_MAX){
		printf("Error: Buffer size of reverse_direction_A agents in state change_direction_A will be exceeded moving working agents to next state in function reverse_direction_A\n");
      exit(EXIT_FAILURE);
      }
      
  //append agents to next state list
  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_A_Agents, no_sm, state_list_size);
  gridSize = (state_list_size + blockSize - 1) / blockSize;
  append_A_Agents<<<gridSize, blockSize, 0, stream>>>(d_As_change_direction_A, d_As, h_xmachine_memory_A_change_direction_A_count, h_xmachine_memory_A_count);
  gpuErrchkLaunch();
        
	//update new state agent size
	h_xmachine_memory_A_change_direction_A_count += h_xmachine_memory_A_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_A_change_direction_A_count, &h_xmachine_memory_A_change_direction_A_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int A_resume_movement_A_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** A_resume_movement_A
 * Agent function prototype for resume_movement_A function of A agent
 */
void A_resume_movement_A(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_A_change_direction_A_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_A_change_direction_A_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_A_list* As_change_direction_A_temp = d_As;
	d_As = d_As_change_direction_A;
	d_As_change_direction_A = As_change_direction_A_temp;
	//set working count to current state count
	h_xmachine_memory_A_count = h_xmachine_memory_A_change_direction_A_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_A_count, &h_xmachine_memory_A_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_A_change_direction_A_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_A_change_direction_A_count, &h_xmachine_memory_A_change_direction_A_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_resume_movement_A, A_resume_movement_A_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = A_resume_movement_A_sm_size(blockSize);
	
	
	
	
	//MAIN XMACHINE FUNCTION CALL (resume_movement_A)
	//Reallocate   : false
	//Input        : 
	//Output       : 
	//Agent Output : 
	GPUFLAME_resume_movement_A<<<g, b, sm_size, stream>>>(d_As);
	gpuErrchkLaunch();
	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_A_moving_A_count+h_xmachine_memory_A_count > xmachine_memory_A_MAX){
		printf("Error: Buffer size of resume_movement_A agents in state moving_A will be exceeded moving working agents to next state in function resume_movement_A\n");
      exit(EXIT_FAILURE);
      }
      
  //append agents to next state list
  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_A_Agents, no_sm, state_list_size);
  gridSize = (state_list_size + blockSize - 1) / blockSize;
  append_A_Agents<<<gridSize, blockSize, 0, stream>>>(d_As_moving_A, d_As, h_xmachine_memory_A_moving_A_count, h_xmachine_memory_A_count);
  gpuErrchkLaunch();
        
	//update new state agent size
	h_xmachine_memory_A_moving_A_count += h_xmachine_memory_A_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_A_moving_A_count, &h_xmachine_memory_A_moving_A_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int B_move_B_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** B_move_B
 * Agent function prototype for move_B function of B agent
 */
void B_move_B(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_B_moving_B_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_B_moving_B_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_B_list* Bs_moving_B_temp = d_Bs;
	d_Bs = d_Bs_moving_B;
	d_Bs_moving_B = Bs_moving_B_temp;
	//set working count to current state count
	h_xmachine_memory_B_count = h_xmachine_memory_B_moving_B_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_B_count, &h_xmachine_memory_B_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_B_moving_B_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_B_moving_B_count, &h_xmachine_memory_B_moving_B_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_move_B, B_move_B_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = B_move_B_sm_size(blockSize);
	
	
	
	
	//MAIN XMACHINE FUNCTION CALL (move_B)
	//Reallocate   : false
	//Input        : 
	//Output       : 
	//Agent Output : 
	GPUFLAME_move_B<<<g, b, sm_size, stream>>>(d_Bs);
	gpuErrchkLaunch();
	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_B_moving_B_count+h_xmachine_memory_B_count > xmachine_memory_B_MAX){
		printf("Error: Buffer size of move_B agents in state moving_B will be exceeded moving working agents to next state in function move_B\n");
      exit(EXIT_FAILURE);
      }
      
  //pointer swap the updated data
  Bs_moving_B_temp = d_Bs;
  d_Bs = d_Bs_moving_B;
  d_Bs_moving_B = Bs_moving_B_temp;
        
	//update new state agent size
	h_xmachine_memory_B_moving_B_count += h_xmachine_memory_B_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_B_moving_B_count, &h_xmachine_memory_B_moving_B_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int B_reverse_direction_B_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** B_reverse_direction_B
 * Agent function prototype for reverse_direction_B function of B agent
 */
void B_reverse_direction_B(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_B_moving_B_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_B_moving_B_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_B_list* Bs_moving_B_temp = d_Bs;
	d_Bs = d_Bs_moving_B;
	d_Bs_moving_B = Bs_moving_B_temp;
	//set working count to current state count
	h_xmachine_memory_B_count = h_xmachine_memory_B_moving_B_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_B_count, &h_xmachine_memory_B_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_B_moving_B_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_B_moving_B_count, &h_xmachine_memory_B_moving_B_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_reverse_direction_B, B_reverse_direction_B_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = B_reverse_direction_B_sm_size(blockSize);
	
	
	
	
	//MAIN XMACHINE FUNCTION CALL (reverse_direction_B)
	//Reallocate   : false
	//Input        : 
	//Output       : 
	//Agent Output : 
	GPUFLAME_reverse_direction_B<<<g, b, sm_size, stream>>>(d_Bs);
	gpuErrchkLaunch();
	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_B_change_direction_B_count+h_xmachine_memory_B_count > xmachine_memory_B_MAX){
		printf("Error: Buffer size of reverse_direction_B agents in state change_direction_B will be exceeded moving working agents to next state in function reverse_direction_B\n");
      exit(EXIT_FAILURE);
      }
      
  //append agents to next state list
  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_B_Agents, no_sm, state_list_size);
  gridSize = (state_list_size + blockSize - 1) / blockSize;
  append_B_Agents<<<gridSize, blockSize, 0, stream>>>(d_Bs_change_direction_B, d_Bs, h_xmachine_memory_B_change_direction_B_count, h_xmachine_memory_B_count);
  gpuErrchkLaunch();
        
	//update new state agent size
	h_xmachine_memory_B_change_direction_B_count += h_xmachine_memory_B_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_B_change_direction_B_count, &h_xmachine_memory_B_change_direction_B_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int B_resume_movement_B_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** B_resume_movement_B
 * Agent function prototype for resume_movement_B function of B agent
 */
void B_resume_movement_B(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_B_change_direction_B_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_B_change_direction_B_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_B_list* Bs_change_direction_B_temp = d_Bs;
	d_Bs = d_Bs_change_direction_B;
	d_Bs_change_direction_B = Bs_change_direction_B_temp;
	//set working count to current state count
	h_xmachine_memory_B_count = h_xmachine_memory_B_change_direction_B_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_B_count, &h_xmachine_memory_B_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_B_change_direction_B_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_B_change_direction_B_count, &h_xmachine_memory_B_change_direction_B_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_resume_movement_B, B_resume_movement_B_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = B_resume_movement_B_sm_size(blockSize);
	
	
	
	
	//MAIN XMACHINE FUNCTION CALL (resume_movement_B)
	//Reallocate   : false
	//Input        : 
	//Output       : 
	//Agent Output : 
	GPUFLAME_resume_movement_B<<<g, b, sm_size, stream>>>(d_Bs);
	gpuErrchkLaunch();
	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_B_moving_B_count+h_xmachine_memory_B_count > xmachine_memory_B_MAX){
		printf("Error: Buffer size of resume_movement_B agents in state moving_B will be exceeded moving working agents to next state in function resume_movement_B\n");
      exit(EXIT_FAILURE);
      }
      
  //append agents to next state list
  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_B_Agents, no_sm, state_list_size);
  gridSize = (state_list_size + blockSize - 1) / blockSize;
  append_B_Agents<<<gridSize, blockSize, 0, stream>>>(d_Bs_moving_B, d_Bs, h_xmachine_memory_B_moving_B_count, h_xmachine_memory_B_count);
  gpuErrchkLaunch();
        
	//update new state agent size
	h_xmachine_memory_B_moving_B_count += h_xmachine_memory_B_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_B_moving_B_count, &h_xmachine_memory_B_moving_B_count, sizeof(int)));	
	
	
}


 
extern void reset_A_moving_A_count()
{
    h_xmachine_memory_A_moving_A_count = 0;
}
 
extern void reset_A_change_direction_A_count()
{
    h_xmachine_memory_A_change_direction_A_count = 0;
}
 
extern void reset_A_get_going_again_A_count()
{
    h_xmachine_memory_A_get_going_again_A_count = 0;
}
 
extern void reset_B_moving_B_count()
{
    h_xmachine_memory_B_moving_B_count = 0;
}
 
extern void reset_B_change_direction_B_count()
{
    h_xmachine_memory_B_change_direction_B_count = 0;
}
 
extern void reset_B_get_going_again_B_count()
{
    h_xmachine_memory_B_get_going_again_B_count = 0;
}
