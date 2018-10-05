
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



#ifndef __HEADER
#define __HEADER

#if defined(__NVCC__) && defined(__CUDACC_VER_MAJOR__) && __CUDACC_VER_MAJOR__ >= 9
   // Disable annotation on defaulted function warnings (glm 0.9.9 and CUDA 9.0 introduced this warning)
   #pragma diag_suppress esa_on_defaulted_function_ignored 
#endif

#define GLM_FORCE_NO_CTOR_INIT
#include <glm/glm.hpp>

/* General standard definitions */
//Threads per block (agents per block)
#define THREADS_PER_TILE 64
//Definition for any agent function or helper function
#define __FLAME_GPU_FUNC__ __device__
//Definition for a function used to initialise environment variables
#define __FLAME_GPU_INIT_FUNC__
#define __FLAME_GPU_STEP_FUNC__
#define __FLAME_GPU_EXIT_FUNC__
#define __FLAME_GPU_HOST_FUNC__ __host__

#define USE_CUDA_STREAMS
#define FAST_ATOMIC_SORTING

// FLAME GPU Version Macros.
#define FLAME_GPU_MAJOR_VERSION 1
#define FLAME_GPU_MINOR_VERSION 5
#define FLAME_GPU_PATCH_VERSION 0

typedef unsigned int uint;

//FLAME GPU vector types float, (i)nteger, (u)nsigned integer, (d)ouble
typedef glm::vec2 fvec2;
typedef glm::vec3 fvec3;
typedef glm::vec4 fvec4;
typedef glm::ivec2 ivec2;
typedef glm::ivec3 ivec3;
typedef glm::ivec4 ivec4;
typedef glm::uvec2 uvec2;
typedef glm::uvec3 uvec3;
typedef glm::uvec4 uvec4;
typedef glm::dvec2 dvec2;
typedef glm::dvec3 dvec3;
typedef glm::dvec4 dvec4;

	

/* Agent population size definitions must be a multiple of THREADS_PER_TILE (default 64) */
//Maximum buffer size (largest agent buffer size)
#define buffer_size_MAX 1024

//Maximum population size of xmachine_memory_TissueBlock
#define xmachine_memory_TissueBlock_MAX 1024

//Maximum population size of xmachine_memory_Fibroblast
#define xmachine_memory_Fibroblast_MAX 1024


  
  
/* Message population size definitions */
//Maximum population size of xmachine_mmessage_tissue_damage_report
#define xmachine_message_tissue_damage_report_MAX 4096

//Maximum population size of xmachine_mmessage_fibroblast_report
#define xmachine_message_fibroblast_report_MAX 4096


/* Define preprocessor symbols for each message to specify the type, to simplify / improve portability */

#define xmachine_message_tissue_damage_report_partitioningSpatial
#define xmachine_message_fibroblast_report_partitioningSpatial

/* Spatial partitioning grid size definitions */
//xmachine_message_tissue_damage_report partition grid size (gridDim.X*gridDim.Y*gridDim.Z)
#define xmachine_message_tissue_damage_report_grid_size 1000
//xmachine_message_fibroblast_report partition grid size (gridDim.X*gridDim.Y*gridDim.Z)
#define xmachine_message_fibroblast_report_grid_size 64

/* Static Graph size definitions*/
  

/* Default visualisation Colour indices */
 
#define FLAME_GPU_VISUALISATION_COLOUR_BLACK 0
#define FLAME_GPU_VISUALISATION_COLOUR_RED 1
#define FLAME_GPU_VISUALISATION_COLOUR_GREEN 2
#define FLAME_GPU_VISUALISATION_COLOUR_BLUE 3
#define FLAME_GPU_VISUALISATION_COLOUR_YELLOW 4
#define FLAME_GPU_VISUALISATION_COLOUR_CYAN 5
#define FLAME_GPU_VISUALISATION_COLOUR_MAGENTA 6
#define FLAME_GPU_VISUALISATION_COLOUR_WHITE 7
#define FLAME_GPU_VISUALISATION_COLOUR_BROWN 8

/* enum types */

/**
 * MESSAGE_OUTPUT used for all continuous messaging
 */
enum MESSAGE_OUTPUT{
	single_message,
	optional_message,
};

/**
 * AGENT_TYPE used for templates device message functions
 */
enum AGENT_TYPE{
	CONTINUOUS,
	DISCRETE_2D
};


/* Agent structures */

/** struct xmachine_memory_TissueBlock
 * continuous valued agent
 * Holds all agent variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_memory_TissueBlock
{
    int id;    /**< X-machine memory variable id of type int.*/
    float x;    /**< X-machine memory variable x of type float.*/
    float y;    /**< X-machine memory variable y of type float.*/
    float z;    /**< X-machine memory variable z of type float.*/
    int damage;    /**< X-machine memory variable damage of type int.*/
};

/** struct xmachine_memory_Fibroblast
 * continuous valued agent
 * Holds all agent variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_memory_Fibroblast
{
    int id;    /**< X-machine memory variable id of type int.*/
    float x;    /**< X-machine memory variable x of type float.*/
    float y;    /**< X-machine memory variable y of type float.*/
    float z;    /**< X-machine memory variable z of type float.*/
    int damage;    /**< X-machine memory variable damage of type int.*/
    int current_state;    /**< X-machine memory variable current_state of type int.*/
    int go_to_state;    /**< X-machine memory variable go_to_state of type int.*/
};



/* Message structures */

/** struct xmachine_message_tissue_damage_report
 * Spatial Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_tissue_damage_report
{	
    /* Spatial Partitioning Variables */
    glm::ivec3 _relative_cell;    /**< Relative cell position from agent grid cell position range -1 to 1 */
    int _cell_index_max;    /**< Max boundary value of current cell */
    glm::ivec3 _agent_grid_cell;  /**< Agents partition cell position */
    int _cell_index;        /**< Index of position in current cell */  
      
    int id;        /**< Message variable id of type int.*/  
    float x;        /**< Message variable x of type float.*/  
    float y;        /**< Message variable y of type float.*/  
    float z;        /**< Message variable z of type float.*/  
    int damage;        /**< Message variable damage of type int.*/
};

/** struct xmachine_message_fibroblast_report
 * Spatial Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_fibroblast_report
{	
    /* Spatial Partitioning Variables */
    glm::ivec3 _relative_cell;    /**< Relative cell position from agent grid cell position range -1 to 1 */
    int _cell_index_max;    /**< Max boundary value of current cell */
    glm::ivec3 _agent_grid_cell;  /**< Agents partition cell position */
    int _cell_index;        /**< Index of position in current cell */  
      
    int id;        /**< Message variable id of type int.*/  
    float x;        /**< Message variable x of type float.*/  
    float y;        /**< Message variable y of type float.*/  
    float z;        /**< Message variable z of type float.*/  
    int current_state;        /**< Message variable current_state of type int.*/  
    int go_to_state;        /**< Message variable go_to_state of type int.*/
};



/* Agent lists. Structure of Array (SoA) for memory coalescing on GPU */

/** struct xmachine_memory_TissueBlock_list
 * continuous valued agent
 * Variables lists for all agent variables
 */
struct xmachine_memory_TissueBlock_list
{	
    /* Temp variables for agents. Used for parallel operations such as prefix sum */
    int _position [xmachine_memory_TissueBlock_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_memory_TissueBlock_MAX];  /**< Used during parallel prefix sum */
    
    int id [xmachine_memory_TissueBlock_MAX];    /**< X-machine memory variable list id of type int.*/
    float x [xmachine_memory_TissueBlock_MAX];    /**< X-machine memory variable list x of type float.*/
    float y [xmachine_memory_TissueBlock_MAX];    /**< X-machine memory variable list y of type float.*/
    float z [xmachine_memory_TissueBlock_MAX];    /**< X-machine memory variable list z of type float.*/
    int damage [xmachine_memory_TissueBlock_MAX];    /**< X-machine memory variable list damage of type int.*/
};

/** struct xmachine_memory_Fibroblast_list
 * continuous valued agent
 * Variables lists for all agent variables
 */
struct xmachine_memory_Fibroblast_list
{	
    /* Temp variables for agents. Used for parallel operations such as prefix sum */
    int _position [xmachine_memory_Fibroblast_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_memory_Fibroblast_MAX];  /**< Used during parallel prefix sum */
    
    int id [xmachine_memory_Fibroblast_MAX];    /**< X-machine memory variable list id of type int.*/
    float x [xmachine_memory_Fibroblast_MAX];    /**< X-machine memory variable list x of type float.*/
    float y [xmachine_memory_Fibroblast_MAX];    /**< X-machine memory variable list y of type float.*/
    float z [xmachine_memory_Fibroblast_MAX];    /**< X-machine memory variable list z of type float.*/
    int damage [xmachine_memory_Fibroblast_MAX];    /**< X-machine memory variable list damage of type int.*/
    int current_state [xmachine_memory_Fibroblast_MAX];    /**< X-machine memory variable list current_state of type int.*/
    int go_to_state [xmachine_memory_Fibroblast_MAX];    /**< X-machine memory variable list go_to_state of type int.*/
};



/* Message lists. Structure of Array (SoA) for memory coalescing on GPU */

/** struct xmachine_message_tissue_damage_report_list
 * Spatial Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_tissue_damage_report_list
{
    /* Non discrete messages have temp variables used for reductions with optional message outputs */
    int _position [xmachine_message_tissue_damage_report_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_message_tissue_damage_report_MAX];  /**< Used during parallel prefix sum */
    
    int id [xmachine_message_tissue_damage_report_MAX];    /**< Message memory variable list id of type int.*/
    float x [xmachine_message_tissue_damage_report_MAX];    /**< Message memory variable list x of type float.*/
    float y [xmachine_message_tissue_damage_report_MAX];    /**< Message memory variable list y of type float.*/
    float z [xmachine_message_tissue_damage_report_MAX];    /**< Message memory variable list z of type float.*/
    int damage [xmachine_message_tissue_damage_report_MAX];    /**< Message memory variable list damage of type int.*/
    
};

/** struct xmachine_message_fibroblast_report_list
 * Spatial Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_fibroblast_report_list
{
    /* Non discrete messages have temp variables used for reductions with optional message outputs */
    int _position [xmachine_message_fibroblast_report_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_message_fibroblast_report_MAX];  /**< Used during parallel prefix sum */
    
    int id [xmachine_message_fibroblast_report_MAX];    /**< Message memory variable list id of type int.*/
    float x [xmachine_message_fibroblast_report_MAX];    /**< Message memory variable list x of type float.*/
    float y [xmachine_message_fibroblast_report_MAX];    /**< Message memory variable list y of type float.*/
    float z [xmachine_message_fibroblast_report_MAX];    /**< Message memory variable list z of type float.*/
    int current_state [xmachine_message_fibroblast_report_MAX];    /**< Message memory variable list current_state of type int.*/
    int go_to_state [xmachine_message_fibroblast_report_MAX];    /**< Message memory variable list go_to_state of type int.*/
    
};



/* Spatially Partitioned Message boundary Matrices */

/** struct xmachine_message_tissue_damage_report_PBM
 * Partition Boundary Matrix (PBM) for xmachine_message_tissue_damage_report 
 */
struct xmachine_message_tissue_damage_report_PBM
{
	int start[xmachine_message_tissue_damage_report_grid_size];
	int end_or_count[xmachine_message_tissue_damage_report_grid_size];
};

/** struct xmachine_message_fibroblast_report_PBM
 * Partition Boundary Matrix (PBM) for xmachine_message_fibroblast_report 
 */
struct xmachine_message_fibroblast_report_PBM
{
	int start[xmachine_message_fibroblast_report_grid_size];
	int end_or_count[xmachine_message_fibroblast_report_grid_size];
};



/* Graph structures */


/* Graph Edge Partitioned message boundary structures */


/* Graph utility functions, usable in agent functions and implemented in FLAMEGPU_Kernels */


  /* Random */
  /** struct RNG_rand48
  *	structure used to hold list seeds
  */
  struct RNG_rand48
  {
  glm::uvec2 A, C;
  glm::uvec2 seeds[buffer_size_MAX];
  };


/** getOutputDir
* Gets the output directory of the simulation. This is the same as the 0.xml input directory.
* @return a const char pointer to string denoting the output directory
*/
const char* getOutputDir();

  /* Random Functions (usable in agent functions) implemented in FLAMEGPU_Kernels */

  /**
  * Templated random function using a DISCRETE_2D template calculates the agent index using a 2D block
  * which requires extra processing but will work for CONTINUOUS agents. Using a CONTINUOUS template will
  * not work for DISCRETE_2D agent.
  * @param	rand48	an RNG_rand48 struct which holds the seeds sued to generate a random number on the GPU
  * @return			returns a random float value
  */
  template <int AGENT_TYPE> __FLAME_GPU_FUNC__ float rnd(RNG_rand48* rand48);
/**
 * Non templated random function calls the templated version with DISCRETE_2D which will work in either case
 * @param	rand48	an RNG_rand48 struct which holds the seeds sued to generate a random number on the GPU
 * @return			returns a random float value
 */
__FLAME_GPU_FUNC__ float rnd(RNG_rand48* rand48);

/* Agent function prototypes */

/**
 * TissueTakesDamage FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_TissueBlock. This represents a single agent instance and can be modified directly.
 * @param tissue_damage_report_messages Pointer to output message list of type xmachine_message_tissue_damage_report_list. Must be passed as an argument to the add_tissue_damage_report_message function ??.* @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an argument to the rand48 function for generating random numbers on the GPU.
 */
__FLAME_GPU_FUNC__ int TissueTakesDamage(xmachine_memory_TissueBlock* agent, xmachine_message_tissue_damage_report_list* tissue_damage_report_messages, RNG_rand48* rand48);

/**
 * RepairDamage FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_TissueBlock. This represents a single agent instance and can be modified directly.
 * @param fibroblast_report_messages  fibroblast_report_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_fibroblast_report_message and get_next_fibroblast_report_message functions.* @param partition_matrix Pointer to the partition matrix of type xmachine_message_fibroblast_report_PBM. Used within the get_first__message and get_next__message functions for spatially partitioned message access.
 */
__FLAME_GPU_FUNC__ int RepairDamage(xmachine_memory_TissueBlock* agent, xmachine_message_fibroblast_report_list* fibroblast_report_messages, xmachine_message_fibroblast_report_PBM* partition_matrix);

/**
 * QuiescentMigration FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_Fibroblast. This represents a single agent instance and can be modified directly.
 * @param tissue_damage_report_messages  tissue_damage_report_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_tissue_damage_report_message and get_next_tissue_damage_report_message functions.* @param partition_matrix Pointer to the partition matrix of type xmachine_message_tissue_damage_report_PBM. Used within the get_first__message and get_next__message functions for spatially partitioned message access.* @param fibroblast_report_messages Pointer to output message list of type xmachine_message_fibroblast_report_list. Must be passed as an argument to the add_fibroblast_report_message function ??.
 */
__FLAME_GPU_FUNC__ int QuiescentMigration(xmachine_memory_Fibroblast* agent, xmachine_message_tissue_damage_report_list* tissue_damage_report_messages, xmachine_message_tissue_damage_report_PBM* partition_matrix, xmachine_message_fibroblast_report_list* fibroblast_report_messages);

/**
 * TransitionToRepair FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_Fibroblast. This represents a single agent instance and can be modified directly.
 
 */
__FLAME_GPU_FUNC__ int TransitionToRepair(xmachine_memory_Fibroblast* agent);

/**
 * TransitionToQuiescent FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_Fibroblast. This represents a single agent instance and can be modified directly.
 
 */
__FLAME_GPU_FUNC__ int TransitionToQuiescent(xmachine_memory_Fibroblast* agent);

  
/* Message Function Prototypes for Spatially Partitioned tissue_damage_report message implemented in FLAMEGPU_Kernels */

/** add_tissue_damage_report_message
 * Function for all types of message partitioning
 * Adds a new tissue_damage_report agent to the xmachine_memory_tissue_damage_report_list list using a linear mapping
 * @param agents	xmachine_memory_tissue_damage_report_list agent list
 * @param id	message variable of type int
 * @param x	message variable of type float
 * @param y	message variable of type float
 * @param z	message variable of type float
 * @param damage	message variable of type int
 */
 
 __FLAME_GPU_FUNC__ void add_tissue_damage_report_message(xmachine_message_tissue_damage_report_list* tissue_damage_report_messages, int id, float x, float y, float z, int damage);
 
/** get_first_tissue_damage_report_message
 * Get first message function for spatially partitioned messages
 * @param tissue_damage_report_messages message list
 * @param partition_matrix the boundary partition matrix for the spatially partitioned message list
 * @param agentx x position of the agent
 * @param agenty y position of the agent
 * @param agentz z position of the agent
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_tissue_damage_report * get_first_tissue_damage_report_message(xmachine_message_tissue_damage_report_list* tissue_damage_report_messages, xmachine_message_tissue_damage_report_PBM* partition_matrix, float x, float y, float z);

/** get_next_tissue_damage_report_message
 * Get first message function for discrete partitioned messages. Template function will call either shared memory or texture cache implementation depending on AGENT_TYPE
 * @param current the current message struct
 * @param tissue_damage_report_messages message list
 * @param partition_matrix the boundary partition matrix for the spatially partitioned message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_tissue_damage_report * get_next_tissue_damage_report_message(xmachine_message_tissue_damage_report* current, xmachine_message_tissue_damage_report_list* tissue_damage_report_messages, xmachine_message_tissue_damage_report_PBM* partition_matrix);

  
/* Message Function Prototypes for Spatially Partitioned fibroblast_report message implemented in FLAMEGPU_Kernels */

/** add_fibroblast_report_message
 * Function for all types of message partitioning
 * Adds a new fibroblast_report agent to the xmachine_memory_fibroblast_report_list list using a linear mapping
 * @param agents	xmachine_memory_fibroblast_report_list agent list
 * @param id	message variable of type int
 * @param x	message variable of type float
 * @param y	message variable of type float
 * @param z	message variable of type float
 * @param current_state	message variable of type int
 * @param go_to_state	message variable of type int
 */
 
 __FLAME_GPU_FUNC__ void add_fibroblast_report_message(xmachine_message_fibroblast_report_list* fibroblast_report_messages, int id, float x, float y, float z, int current_state, int go_to_state);
 
/** get_first_fibroblast_report_message
 * Get first message function for spatially partitioned messages
 * @param fibroblast_report_messages message list
 * @param partition_matrix the boundary partition matrix for the spatially partitioned message list
 * @param agentx x position of the agent
 * @param agenty y position of the agent
 * @param agentz z position of the agent
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_fibroblast_report * get_first_fibroblast_report_message(xmachine_message_fibroblast_report_list* fibroblast_report_messages, xmachine_message_fibroblast_report_PBM* partition_matrix, float x, float y, float z);

/** get_next_fibroblast_report_message
 * Get first message function for discrete partitioned messages. Template function will call either shared memory or texture cache implementation depending on AGENT_TYPE
 * @param current the current message struct
 * @param fibroblast_report_messages message list
 * @param partition_matrix the boundary partition matrix for the spatially partitioned message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_fibroblast_report * get_next_fibroblast_report_message(xmachine_message_fibroblast_report* current, xmachine_message_fibroblast_report_list* fibroblast_report_messages, xmachine_message_fibroblast_report_PBM* partition_matrix);

  
/* Agent Function Prototypes implemented in FLAMEGPU_Kernels */

/** add_TissueBlock_agent
 * Adds a new continuous valued TissueBlock agent to the xmachine_memory_TissueBlock_list list using a linear mapping. Note that any agent variables with an arrayLength are ommited and not support during the creation of new agents on the fly.
 * @param agents xmachine_memory_TissueBlock_list agent list
 * @param id	agent agent variable of type int
 * @param x	agent agent variable of type float
 * @param y	agent agent variable of type float
 * @param z	agent agent variable of type float
 * @param damage	agent agent variable of type int
 */
__FLAME_GPU_FUNC__ void add_TissueBlock_agent(xmachine_memory_TissueBlock_list* agents, int id, float x, float y, float z, int damage);

/** add_Fibroblast_agent
 * Adds a new continuous valued Fibroblast agent to the xmachine_memory_Fibroblast_list list using a linear mapping. Note that any agent variables with an arrayLength are ommited and not support during the creation of new agents on the fly.
 * @param agents xmachine_memory_Fibroblast_list agent list
 * @param id	agent agent variable of type int
 * @param x	agent agent variable of type float
 * @param y	agent agent variable of type float
 * @param z	agent agent variable of type float
 * @param damage	agent agent variable of type int
 * @param current_state	agent agent variable of type int
 * @param go_to_state	agent agent variable of type int
 */
__FLAME_GPU_FUNC__ void add_Fibroblast_agent(xmachine_memory_Fibroblast_list* agents, int id, float x, float y, float z, int damage, int current_state, int go_to_state);


/* Graph loading function prototypes implemented in io.cu */


  
/* Simulation function prototypes implemented in simulation.cu */
/** getIterationNumber
 *  Get the iteration number (host)
 */
extern unsigned int getIterationNumber();

/** initialise
 * Initialise the simulation. Allocated host and device memory. Reads the initial agent configuration from XML.
 * @param input        XML file path for agent initial configuration
 */
extern void initialise(char * input);

/** cleanup
 * Function cleans up any memory allocations on the host and device
 */
extern void cleanup();

/** singleIteration
 *	Performs a single iteration of the simulation. I.e. performs each agent function on each function layer in the correct order.
 */
extern void singleIteration();

/** saveIterationData
 * Reads the current agent data fromt he device and saves it to XML
 * @param	outputpath	file path to XML file used for output of agent data
 * @param	iteration_number
 * @param h_TissueBlocks Pointer to agent list on the host
 * @param d_TissueBlocks Pointer to agent list on the GPU device
 * @param h_xmachine_memory_TissueBlock_count Pointer to agent counter
 * @param h_Fibroblasts Pointer to agent list on the host
 * @param d_Fibroblasts Pointer to agent list on the GPU device
 * @param h_xmachine_memory_Fibroblast_count Pointer to agent counter
 */
extern void saveIterationData(char* outputpath, int iteration_number, xmachine_memory_TissueBlock_list* h_TissueBlocks_default, xmachine_memory_TissueBlock_list* d_TissueBlocks_default, int h_xmachine_memory_TissueBlock_default_count,xmachine_memory_Fibroblast_list* h_Fibroblasts_Quiescent, xmachine_memory_Fibroblast_list* d_Fibroblasts_Quiescent, int h_xmachine_memory_Fibroblast_Quiescent_count,xmachine_memory_Fibroblast_list* h_Fibroblasts_Repair, xmachine_memory_Fibroblast_list* d_Fibroblasts_Repair, int h_xmachine_memory_Fibroblast_Repair_count);


/** readInitialStates
 * Reads the current agent data from the device and saves it to XML
 * @param	inputpath	file path to XML file used for input of agent data
 * @param h_TissueBlocks Pointer to agent list on the host
 * @param h_xmachine_memory_TissueBlock_count Pointer to agent counter
 * @param h_Fibroblasts Pointer to agent list on the host
 * @param h_xmachine_memory_Fibroblast_count Pointer to agent counter
 */
extern void readInitialStates(char* inputpath, xmachine_memory_TissueBlock_list* h_TissueBlocks, int* h_xmachine_memory_TissueBlock_count,xmachine_memory_Fibroblast_list* h_Fibroblasts, int* h_xmachine_memory_Fibroblast_count);

/** set_exit_early
 * exits the simulation on the step this is called
 */
extern void set_exit_early();

/** get_exit_early
 * gets whether the simulation is ending this simulation step
 */
extern bool get_exit_early();




/* Return functions used by external code to get agent data from device */

    
/** get_agent_TissueBlock_MAX_count
 * Gets the max agent count for the TissueBlock agent type 
 * @return		the maximum TissueBlock agent count
 */
extern int get_agent_TissueBlock_MAX_count();



/** get_agent_TissueBlock_default_count
 * Gets the agent count for the TissueBlock agent type in state default
 * @return		the current TissueBlock agent count in state default
 */
extern int get_agent_TissueBlock_default_count();

/** reset_default_count
 * Resets the agent count of the TissueBlock in state default to 0. This is useful for interacting with some visualisations.
 */
extern void reset_TissueBlock_default_count();

/** get_device_TissueBlock_default_agents
 * Gets a pointer to xmachine_memory_TissueBlock_list on the GPU device
 * @return		a xmachine_memory_TissueBlock_list on the GPU device
 */
extern xmachine_memory_TissueBlock_list* get_device_TissueBlock_default_agents();

/** get_host_TissueBlock_default_agents
 * Gets a pointer to xmachine_memory_TissueBlock_list on the CPU host
 * @return		a xmachine_memory_TissueBlock_list on the CPU host
 */
extern xmachine_memory_TissueBlock_list* get_host_TissueBlock_default_agents();


/** sort_TissueBlocks_default
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_TissueBlocks_default(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_TissueBlock_list* agents));


    
/** get_agent_Fibroblast_MAX_count
 * Gets the max agent count for the Fibroblast agent type 
 * @return		the maximum Fibroblast agent count
 */
extern int get_agent_Fibroblast_MAX_count();



/** get_agent_Fibroblast_Quiescent_count
 * Gets the agent count for the Fibroblast agent type in state Quiescent
 * @return		the current Fibroblast agent count in state Quiescent
 */
extern int get_agent_Fibroblast_Quiescent_count();

/** reset_Quiescent_count
 * Resets the agent count of the Fibroblast in state Quiescent to 0. This is useful for interacting with some visualisations.
 */
extern void reset_Fibroblast_Quiescent_count();

/** get_device_Fibroblast_Quiescent_agents
 * Gets a pointer to xmachine_memory_Fibroblast_list on the GPU device
 * @return		a xmachine_memory_Fibroblast_list on the GPU device
 */
extern xmachine_memory_Fibroblast_list* get_device_Fibroblast_Quiescent_agents();

/** get_host_Fibroblast_Quiescent_agents
 * Gets a pointer to xmachine_memory_Fibroblast_list on the CPU host
 * @return		a xmachine_memory_Fibroblast_list on the CPU host
 */
extern xmachine_memory_Fibroblast_list* get_host_Fibroblast_Quiescent_agents();


/** sort_Fibroblasts_Quiescent
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_Fibroblasts_Quiescent(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_Fibroblast_list* agents));


/** get_agent_Fibroblast_Repair_count
 * Gets the agent count for the Fibroblast agent type in state Repair
 * @return		the current Fibroblast agent count in state Repair
 */
extern int get_agent_Fibroblast_Repair_count();

/** reset_Repair_count
 * Resets the agent count of the Fibroblast in state Repair to 0. This is useful for interacting with some visualisations.
 */
extern void reset_Fibroblast_Repair_count();

/** get_device_Fibroblast_Repair_agents
 * Gets a pointer to xmachine_memory_Fibroblast_list on the GPU device
 * @return		a xmachine_memory_Fibroblast_list on the GPU device
 */
extern xmachine_memory_Fibroblast_list* get_device_Fibroblast_Repair_agents();

/** get_host_Fibroblast_Repair_agents
 * Gets a pointer to xmachine_memory_Fibroblast_list on the CPU host
 * @return		a xmachine_memory_Fibroblast_list on the CPU host
 */
extern xmachine_memory_Fibroblast_list* get_host_Fibroblast_Repair_agents();


/** sort_Fibroblasts_Repair
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_Fibroblasts_Repair(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_Fibroblast_list* agents));



/* Host based access of agent variables*/

/** int get_TissueBlock_default_variable_id(unsigned int index)
 * Gets the value of the id variable of an TissueBlock agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable id
 */
__host__ int get_TissueBlock_default_variable_id(unsigned int index);

/** float get_TissueBlock_default_variable_x(unsigned int index)
 * Gets the value of the x variable of an TissueBlock agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable x
 */
__host__ float get_TissueBlock_default_variable_x(unsigned int index);

/** float get_TissueBlock_default_variable_y(unsigned int index)
 * Gets the value of the y variable of an TissueBlock agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable y
 */
__host__ float get_TissueBlock_default_variable_y(unsigned int index);

/** float get_TissueBlock_default_variable_z(unsigned int index)
 * Gets the value of the z variable of an TissueBlock agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable z
 */
__host__ float get_TissueBlock_default_variable_z(unsigned int index);

/** int get_TissueBlock_default_variable_damage(unsigned int index)
 * Gets the value of the damage variable of an TissueBlock agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable damage
 */
__host__ int get_TissueBlock_default_variable_damage(unsigned int index);

/** int get_Fibroblast_Quiescent_variable_id(unsigned int index)
 * Gets the value of the id variable of an Fibroblast agent in the Quiescent state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable id
 */
__host__ int get_Fibroblast_Quiescent_variable_id(unsigned int index);

/** float get_Fibroblast_Quiescent_variable_x(unsigned int index)
 * Gets the value of the x variable of an Fibroblast agent in the Quiescent state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable x
 */
__host__ float get_Fibroblast_Quiescent_variable_x(unsigned int index);

/** float get_Fibroblast_Quiescent_variable_y(unsigned int index)
 * Gets the value of the y variable of an Fibroblast agent in the Quiescent state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable y
 */
__host__ float get_Fibroblast_Quiescent_variable_y(unsigned int index);

/** float get_Fibroblast_Quiescent_variable_z(unsigned int index)
 * Gets the value of the z variable of an Fibroblast agent in the Quiescent state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable z
 */
__host__ float get_Fibroblast_Quiescent_variable_z(unsigned int index);

/** int get_Fibroblast_Quiescent_variable_damage(unsigned int index)
 * Gets the value of the damage variable of an Fibroblast agent in the Quiescent state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable damage
 */
__host__ int get_Fibroblast_Quiescent_variable_damage(unsigned int index);

/** int get_Fibroblast_Quiescent_variable_current_state(unsigned int index)
 * Gets the value of the current_state variable of an Fibroblast agent in the Quiescent state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable current_state
 */
__host__ int get_Fibroblast_Quiescent_variable_current_state(unsigned int index);

/** int get_Fibroblast_Quiescent_variable_go_to_state(unsigned int index)
 * Gets the value of the go_to_state variable of an Fibroblast agent in the Quiescent state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable go_to_state
 */
__host__ int get_Fibroblast_Quiescent_variable_go_to_state(unsigned int index);

/** int get_Fibroblast_Repair_variable_id(unsigned int index)
 * Gets the value of the id variable of an Fibroblast agent in the Repair state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable id
 */
__host__ int get_Fibroblast_Repair_variable_id(unsigned int index);

/** float get_Fibroblast_Repair_variable_x(unsigned int index)
 * Gets the value of the x variable of an Fibroblast agent in the Repair state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable x
 */
__host__ float get_Fibroblast_Repair_variable_x(unsigned int index);

/** float get_Fibroblast_Repair_variable_y(unsigned int index)
 * Gets the value of the y variable of an Fibroblast agent in the Repair state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable y
 */
__host__ float get_Fibroblast_Repair_variable_y(unsigned int index);

/** float get_Fibroblast_Repair_variable_z(unsigned int index)
 * Gets the value of the z variable of an Fibroblast agent in the Repair state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable z
 */
__host__ float get_Fibroblast_Repair_variable_z(unsigned int index);

/** int get_Fibroblast_Repair_variable_damage(unsigned int index)
 * Gets the value of the damage variable of an Fibroblast agent in the Repair state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable damage
 */
__host__ int get_Fibroblast_Repair_variable_damage(unsigned int index);

/** int get_Fibroblast_Repair_variable_current_state(unsigned int index)
 * Gets the value of the current_state variable of an Fibroblast agent in the Repair state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable current_state
 */
__host__ int get_Fibroblast_Repair_variable_current_state(unsigned int index);

/** int get_Fibroblast_Repair_variable_go_to_state(unsigned int index)
 * Gets the value of the go_to_state variable of an Fibroblast agent in the Repair state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable go_to_state
 */
__host__ int get_Fibroblast_Repair_variable_go_to_state(unsigned int index);




/* Host based agent creation functions */

/** h_allocate_agent_TissueBlock
 * Utility function to allocate and initialise an agent struct on the host.
 * @return address of a host-allocated TissueBlock struct.
 */
xmachine_memory_TissueBlock* h_allocate_agent_TissueBlock();
/** h_free_agent_TissueBlock
 * Utility function to free a host-allocated agent struct.
 * This also deallocates any agent variable arrays, and sets the pointer to null
 * @param agent address of pointer to the host allocated struct
 */
void h_free_agent_TissueBlock(xmachine_memory_TissueBlock** agent);
/** h_allocate_agent_TissueBlock_array
 * Utility function to allocate an array of structs for  TissueBlock agents.
 * @param count the number of structs to allocate memory for.
 * @return pointer to the allocated array of structs
 */
xmachine_memory_TissueBlock** h_allocate_agent_TissueBlock_array(unsigned int count);
/** h_free_agent_TissueBlock_array(
 * Utility function to deallocate a host array of agent structs, including agent variables, and set pointer values to NULL.
 * @param agents the address of the pointer to the host array of structs.
 * @param count the number of elements in the AoS, to deallocate individual elements.
 */
void h_free_agent_TissueBlock_array(xmachine_memory_TissueBlock*** agents, unsigned int count);


/** h_add_agent_TissueBlock_default
 * Host function to add a single agent of type TissueBlock to the default state on the device.
 * This invokes many cudaMempcy, and an append kernel launch. 
 * If multiple agents are to be created in a single iteration, consider h_add_agent_TissueBlock_default instead.
 * @param agent pointer to agent struct on the host. Agent member arrays are supported.
 */
void h_add_agent_TissueBlock_default(xmachine_memory_TissueBlock* agent);

/** h_add_agents_TissueBlock_default(
 * Host function to add multiple agents of type TissueBlock to the default state on the device if possible.
 * This includes the transparent conversion from AoS to SoA, many calls to cudaMemcpy and an append kernel.
 * @param agents pointer to host struct of arrays of TissueBlock agents
 * @param count the number of agents to copy from the host to the device.
 */
void h_add_agents_TissueBlock_default(xmachine_memory_TissueBlock** agents, unsigned int count);

/** h_allocate_agent_Fibroblast
 * Utility function to allocate and initialise an agent struct on the host.
 * @return address of a host-allocated Fibroblast struct.
 */
xmachine_memory_Fibroblast* h_allocate_agent_Fibroblast();
/** h_free_agent_Fibroblast
 * Utility function to free a host-allocated agent struct.
 * This also deallocates any agent variable arrays, and sets the pointer to null
 * @param agent address of pointer to the host allocated struct
 */
void h_free_agent_Fibroblast(xmachine_memory_Fibroblast** agent);
/** h_allocate_agent_Fibroblast_array
 * Utility function to allocate an array of structs for  Fibroblast agents.
 * @param count the number of structs to allocate memory for.
 * @return pointer to the allocated array of structs
 */
xmachine_memory_Fibroblast** h_allocate_agent_Fibroblast_array(unsigned int count);
/** h_free_agent_Fibroblast_array(
 * Utility function to deallocate a host array of agent structs, including agent variables, and set pointer values to NULL.
 * @param agents the address of the pointer to the host array of structs.
 * @param count the number of elements in the AoS, to deallocate individual elements.
 */
void h_free_agent_Fibroblast_array(xmachine_memory_Fibroblast*** agents, unsigned int count);


/** h_add_agent_Fibroblast_Quiescent
 * Host function to add a single agent of type Fibroblast to the Quiescent state on the device.
 * This invokes many cudaMempcy, and an append kernel launch. 
 * If multiple agents are to be created in a single iteration, consider h_add_agent_Fibroblast_Quiescent instead.
 * @param agent pointer to agent struct on the host. Agent member arrays are supported.
 */
void h_add_agent_Fibroblast_Quiescent(xmachine_memory_Fibroblast* agent);

/** h_add_agents_Fibroblast_Quiescent(
 * Host function to add multiple agents of type Fibroblast to the Quiescent state on the device if possible.
 * This includes the transparent conversion from AoS to SoA, many calls to cudaMemcpy and an append kernel.
 * @param agents pointer to host struct of arrays of Fibroblast agents
 * @param count the number of agents to copy from the host to the device.
 */
void h_add_agents_Fibroblast_Quiescent(xmachine_memory_Fibroblast** agents, unsigned int count);


/** h_add_agent_Fibroblast_Repair
 * Host function to add a single agent of type Fibroblast to the Repair state on the device.
 * This invokes many cudaMempcy, and an append kernel launch. 
 * If multiple agents are to be created in a single iteration, consider h_add_agent_Fibroblast_Repair instead.
 * @param agent pointer to agent struct on the host. Agent member arrays are supported.
 */
void h_add_agent_Fibroblast_Repair(xmachine_memory_Fibroblast* agent);

/** h_add_agents_Fibroblast_Repair(
 * Host function to add multiple agents of type Fibroblast to the Repair state on the device if possible.
 * This includes the transparent conversion from AoS to SoA, many calls to cudaMemcpy and an append kernel.
 * @param agents pointer to host struct of arrays of Fibroblast agents
 * @param count the number of agents to copy from the host to the device.
 */
void h_add_agents_Fibroblast_Repair(xmachine_memory_Fibroblast** agents, unsigned int count);

  
  
/* Analytics functions for each varible in each state*/
typedef enum {
  REDUCTION_MAX,
  REDUCTION_MIN,
  REDUCTION_SUM
}reduction_operator;


/** int reduce_TissueBlock_default_id_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_TissueBlock_default_id_variable();



/** int count_TissueBlock_default_id_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_TissueBlock_default_id_variable(int count_value);

/** int min_TissueBlock_default_id_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_TissueBlock_default_id_variable();
/** int max_TissueBlock_default_id_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_TissueBlock_default_id_variable();

/** float reduce_TissueBlock_default_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_TissueBlock_default_x_variable();



/** float min_TissueBlock_default_x_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_TissueBlock_default_x_variable();
/** float max_TissueBlock_default_x_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_TissueBlock_default_x_variable();

/** float reduce_TissueBlock_default_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_TissueBlock_default_y_variable();



/** float min_TissueBlock_default_y_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_TissueBlock_default_y_variable();
/** float max_TissueBlock_default_y_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_TissueBlock_default_y_variable();

/** float reduce_TissueBlock_default_z_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_TissueBlock_default_z_variable();



/** float min_TissueBlock_default_z_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_TissueBlock_default_z_variable();
/** float max_TissueBlock_default_z_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_TissueBlock_default_z_variable();

/** int reduce_TissueBlock_default_damage_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_TissueBlock_default_damage_variable();



/** int count_TissueBlock_default_damage_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_TissueBlock_default_damage_variable(int count_value);

/** int min_TissueBlock_default_damage_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_TissueBlock_default_damage_variable();
/** int max_TissueBlock_default_damage_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_TissueBlock_default_damage_variable();

/** int reduce_Fibroblast_Quiescent_id_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_Fibroblast_Quiescent_id_variable();



/** int count_Fibroblast_Quiescent_id_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_Fibroblast_Quiescent_id_variable(int count_value);

/** int min_Fibroblast_Quiescent_id_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_Fibroblast_Quiescent_id_variable();
/** int max_Fibroblast_Quiescent_id_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_Fibroblast_Quiescent_id_variable();

/** float reduce_Fibroblast_Quiescent_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_Fibroblast_Quiescent_x_variable();



/** float min_Fibroblast_Quiescent_x_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_Fibroblast_Quiescent_x_variable();
/** float max_Fibroblast_Quiescent_x_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_Fibroblast_Quiescent_x_variable();

/** float reduce_Fibroblast_Quiescent_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_Fibroblast_Quiescent_y_variable();



/** float min_Fibroblast_Quiescent_y_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_Fibroblast_Quiescent_y_variable();
/** float max_Fibroblast_Quiescent_y_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_Fibroblast_Quiescent_y_variable();

/** float reduce_Fibroblast_Quiescent_z_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_Fibroblast_Quiescent_z_variable();



/** float min_Fibroblast_Quiescent_z_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_Fibroblast_Quiescent_z_variable();
/** float max_Fibroblast_Quiescent_z_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_Fibroblast_Quiescent_z_variable();

/** int reduce_Fibroblast_Quiescent_damage_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_Fibroblast_Quiescent_damage_variable();



/** int count_Fibroblast_Quiescent_damage_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_Fibroblast_Quiescent_damage_variable(int count_value);

/** int min_Fibroblast_Quiescent_damage_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_Fibroblast_Quiescent_damage_variable();
/** int max_Fibroblast_Quiescent_damage_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_Fibroblast_Quiescent_damage_variable();

/** int reduce_Fibroblast_Quiescent_current_state_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_Fibroblast_Quiescent_current_state_variable();



/** int count_Fibroblast_Quiescent_current_state_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_Fibroblast_Quiescent_current_state_variable(int count_value);

/** int min_Fibroblast_Quiescent_current_state_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_Fibroblast_Quiescent_current_state_variable();
/** int max_Fibroblast_Quiescent_current_state_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_Fibroblast_Quiescent_current_state_variable();

/** int reduce_Fibroblast_Quiescent_go_to_state_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_Fibroblast_Quiescent_go_to_state_variable();



/** int count_Fibroblast_Quiescent_go_to_state_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_Fibroblast_Quiescent_go_to_state_variable(int count_value);

/** int min_Fibroblast_Quiescent_go_to_state_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_Fibroblast_Quiescent_go_to_state_variable();
/** int max_Fibroblast_Quiescent_go_to_state_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_Fibroblast_Quiescent_go_to_state_variable();

/** int reduce_Fibroblast_Repair_id_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_Fibroblast_Repair_id_variable();



/** int count_Fibroblast_Repair_id_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_Fibroblast_Repair_id_variable(int count_value);

/** int min_Fibroblast_Repair_id_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_Fibroblast_Repair_id_variable();
/** int max_Fibroblast_Repair_id_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_Fibroblast_Repair_id_variable();

/** float reduce_Fibroblast_Repair_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_Fibroblast_Repair_x_variable();



/** float min_Fibroblast_Repair_x_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_Fibroblast_Repair_x_variable();
/** float max_Fibroblast_Repair_x_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_Fibroblast_Repair_x_variable();

/** float reduce_Fibroblast_Repair_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_Fibroblast_Repair_y_variable();



/** float min_Fibroblast_Repair_y_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_Fibroblast_Repair_y_variable();
/** float max_Fibroblast_Repair_y_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_Fibroblast_Repair_y_variable();

/** float reduce_Fibroblast_Repair_z_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_Fibroblast_Repair_z_variable();



/** float min_Fibroblast_Repair_z_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_Fibroblast_Repair_z_variable();
/** float max_Fibroblast_Repair_z_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_Fibroblast_Repair_z_variable();

/** int reduce_Fibroblast_Repair_damage_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_Fibroblast_Repair_damage_variable();



/** int count_Fibroblast_Repair_damage_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_Fibroblast_Repair_damage_variable(int count_value);

/** int min_Fibroblast_Repair_damage_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_Fibroblast_Repair_damage_variable();
/** int max_Fibroblast_Repair_damage_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_Fibroblast_Repair_damage_variable();

/** int reduce_Fibroblast_Repair_current_state_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_Fibroblast_Repair_current_state_variable();



/** int count_Fibroblast_Repair_current_state_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_Fibroblast_Repair_current_state_variable(int count_value);

/** int min_Fibroblast_Repair_current_state_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_Fibroblast_Repair_current_state_variable();
/** int max_Fibroblast_Repair_current_state_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_Fibroblast_Repair_current_state_variable();

/** int reduce_Fibroblast_Repair_go_to_state_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_Fibroblast_Repair_go_to_state_variable();



/** int count_Fibroblast_Repair_go_to_state_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_Fibroblast_Repair_go_to_state_variable(int count_value);

/** int min_Fibroblast_Repair_go_to_state_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_Fibroblast_Repair_go_to_state_variable();
/** int max_Fibroblast_Repair_go_to_state_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_Fibroblast_Repair_go_to_state_variable();


  
/* global constant variables */

__constant__ float TISSUE_DAMAGE_PROB;

__constant__ float QUIESCENT_MIGRATION_SCALE;

__constant__ float REPAIR_RANGE;

__constant__ float DAMAGE_DETECTION_RANGE;

/** set_TISSUE_DAMAGE_PROB
 * Sets the constant variable TISSUE_DAMAGE_PROB on the device which can then be used in the agent functions.
 * @param h_TISSUE_DAMAGE_PROB value to set the variable
 */
extern void set_TISSUE_DAMAGE_PROB(float* h_TISSUE_DAMAGE_PROB);

extern const float* get_TISSUE_DAMAGE_PROB();


extern float h_env_TISSUE_DAMAGE_PROB;

/** set_QUIESCENT_MIGRATION_SCALE
 * Sets the constant variable QUIESCENT_MIGRATION_SCALE on the device which can then be used in the agent functions.
 * @param h_QUIESCENT_MIGRATION_SCALE value to set the variable
 */
extern void set_QUIESCENT_MIGRATION_SCALE(float* h_QUIESCENT_MIGRATION_SCALE);

extern const float* get_QUIESCENT_MIGRATION_SCALE();


extern float h_env_QUIESCENT_MIGRATION_SCALE;

/** set_REPAIR_RANGE
 * Sets the constant variable REPAIR_RANGE on the device which can then be used in the agent functions.
 * @param h_REPAIR_RANGE value to set the variable
 */
extern void set_REPAIR_RANGE(float* h_REPAIR_RANGE);

extern const float* get_REPAIR_RANGE();


extern float h_env_REPAIR_RANGE;

/** set_DAMAGE_DETECTION_RANGE
 * Sets the constant variable DAMAGE_DETECTION_RANGE on the device which can then be used in the agent functions.
 * @param h_DAMAGE_DETECTION_RANGE value to set the variable
 */
extern void set_DAMAGE_DETECTION_RANGE(float* h_DAMAGE_DETECTION_RANGE);

extern const float* get_DAMAGE_DETECTION_RANGE();


extern float h_env_DAMAGE_DETECTION_RANGE;


/** getMaximumBound
 * Returns the maximum agent positions determined from the initial loading of agents
 * @return 	a three component float indicating the maximum x, y and z positions of all agents
 */
glm::vec3 getMaximumBounds();

/** getMinimumBounds
 * Returns the minimum agent positions determined from the initial loading of agents
 * @return 	a three component float indicating the minimum x, y and z positions of all agents
 */
glm::vec3 getMinimumBounds();
    
    
#ifdef VISUALISATION
/** initVisualisation
 * Prototype for method which initialises the visualisation. Must be implemented in separate file
 * @param argc	the argument count from the main function used with GLUT
 * @param argv	the argument values from the main function used with GLUT
 */
extern void initVisualisation();

extern void runVisualisation();


#endif

#if defined(PROFILE)
#include "nvToolsExt.h"

#define PROFILE_WHITE   0x00eeeeee
#define PROFILE_GREEN   0x0000ff00
#define PROFILE_BLUE    0x000000ff
#define PROFILE_YELLOW  0x00ffff00
#define PROFILE_MAGENTA 0x00ff00ff
#define PROFILE_CYAN    0x0000ffff
#define PROFILE_RED     0x00ff0000
#define PROFILE_GREY    0x00999999
#define PROFILE_LILAC   0xC8A2C8

const uint32_t profile_colors[] = {
  PROFILE_WHITE,
  PROFILE_GREEN,
  PROFILE_BLUE,
  PROFILE_YELLOW,
  PROFILE_MAGENTA,
  PROFILE_CYAN,
  PROFILE_RED,
  PROFILE_GREY,
  PROFILE_LILAC
};
const int num_profile_colors = sizeof(profile_colors) / sizeof(uint32_t);

// Externed value containing colour information.
extern unsigned int g_profile_colour_id;

#define PROFILE_PUSH_RANGE(name) { \
    unsigned int color_id = g_profile_colour_id % num_profile_colors;\
    nvtxEventAttributes_t eventAttrib = {0}; \
    eventAttrib.version = NVTX_VERSION; \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    eventAttrib.colorType = NVTX_COLOR_ARGB; \
    eventAttrib.color = profile_colors[color_id]; \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
    eventAttrib.message.ascii = name; \
    nvtxRangePushEx(&eventAttrib); \
    g_profile_colour_id++; \
}
#define PROFILE_POP_RANGE() nvtxRangePop();

// Class for simple fire-and-forget profile ranges (ie. functions with multiple return conditions.)
class ProfileScopedRange {
public:
    ProfileScopedRange(const char * name){
      PROFILE_PUSH_RANGE(name);
    }
    ~ProfileScopedRange(){
      PROFILE_POP_RANGE();
    }
};
#define PROFILE_SCOPED_RANGE(name) ProfileScopedRange uniq_name_using_macros(name);
#else
#define PROFILE_PUSH_RANGE(name)
#define PROFILE_POP_RANGE()
#define PROFILE_SCOPED_RANGE(name)
#endif


#endif //__HEADER

