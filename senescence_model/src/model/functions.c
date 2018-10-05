
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


#ifndef _FLAMEGPU_FUNCTIONS
#define _FLAMEGPU_FUNCTIONS

#include <header.h>

#define INSTRUMENT_ITERATIONS 1
#define INSTRUMENT_AGENT_FUNCTIONS 1
#define INSTRUMENT_INIT_FUNCTIONS 1
#define INSTRUMENT_STEP_FUNCTIONS 1
#define INSTRUMENT_EXIT_FUNCTIONS 1
#define OUTPUT_POPULATION_PER_ITERATION 1


// get reference to model constants
//extern "REPAIR_RANGE" float* get_REPAIR_RADIUS();
//extern "TISSUE_DAMAGE_PROB" float* get_TISSUE_DAMAGE_PROB();
//extern "QUIESCENT_MIGRATION_SCALE" float* get_QUIESCENT_MIGRATION_SCALE();
//extern "DAMAGE_DETECTION_RANGE" float* get_DAMAGE_DETECTION_RANGE();

//const float* REPAIR_RANGE = get_REPAIR_RANGE();


inline __device__ float dot(glm::vec3 a, glm::vec3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __device__ float length(glm::vec3 v)
{
    return sqrtf(dot(v, v));
}


inline __device__ glm::vec3 subtract_a_from_b(glm::vec3 a, glm::vec3 b)
{
return glm::vec3(b.x-a.x, b.y-a.y, b.z-a.z);
}

inline __device__ glm::vec3 add_a_to_b(glm::vec3 a, glm::vec3 b)
{
return glm::vec3(a.x+b.x, a.y+b.y, a.z+b.z);
}

inline __device__ glm::vec3 compute_next_fibroblast_location(
        glm::vec3 a, glm::vec3 b, float scale)
{
// Compute: a + scale * (b-a) where:
// a:       fibroblast location
// b:       damaged tissue location
// scale:   proportion of the distance between fibroblast and damaged tissue to traverse
// Note:    this means of fibroblast migration is not ideal and will probably change, once
//          have simulations running
return add_a_to_b(a, scale*(subtract_a_from_b(a, b)));
}

/**
 * declareGlobalConstants FLAMEGPU Init function
 * Automatically generated using functions.xslt
 */
__FLAME_GPU_INIT_FUNC__ void declareGlobalConstants(){
//    float REPAIR_RADIUS = get_REPAIR_RADIUS();
}


/**
 * Tissuelogs FLAMEGPU Step function
 * Automatically generated using functions.xslt
 */
__FLAME_GPU_STEP_FUNC__ void Tissuelogs(){

}

/**
 * FibroblastQuiescentlogs FLAMEGPU Step function
 * Automatically generated using functions.xslt
 */
__FLAME_GPU_STEP_FUNC__ void FibroblastQuiescentlogs(){

}

/**
 * FibroblastRepairlogs FLAMEGPU Step function
 * Automatically generated using functions.xslt
 */
__FLAME_GPU_STEP_FUNC__ void FibroblastRepairlogs(){

}

/**
 * TissueTakesDamage FLAMEGPU Agent Function
 * Automatically generated using functions.xsl
 * @param agent Pointer to an agent structure of type xmachine_memory_TissueBlock. This represents a single agent instance and can be modified directly.
 * @param tissue_damage_report_messages Pointer to output message list of type xmachine_message_tissue_damage_report_list. Must be passed as an argument to the add_tissue_damage_report_message function.* @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an argument to the rand48 function for generating random numbers on the GPU.
 */
__FLAME_GPU_FUNC__ int TissueTakesDamage(
    xmachine_memory_TissueBlock* agent,
    xmachine_message_tissue_damage_report_list* tissue_damage_report_messages,
    RNG_rand48* rand48){
    

    //Template for message output function
    int id = agent->id;
    float x = agent->x;
    float y = agent->y;
    float z = agent->z;
    int damage = agent->damage;

    float random_number = rnd<CONTINUOUS>(rand48);
    if (random_number < TISSUE_DAMAGE_PROB)
        agent->damage +=1;
        add_tissue_damage_report_message(
        tissue_damage_report_messages, id, x, y, z, damage);

    
    return 0;
}

/**
 * RepairDamage FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_TissueBlock. This represents a single agent instance and can be modified directly.
 * @param fibroblast_report_messages  fibroblast_report_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_fibroblast_report_message and get_next_fibroblast_report_message functions.* @param partition_matrix Pointer to the partition matrix of type xmachine_message_fibroblast_report_PBM. Used within the get_first__message and get_next__message functions for spatially partitioned message access.
 */
__FLAME_GPU_FUNC__ int RepairDamage(
    xmachine_memory_TissueBlock* agent,
    xmachine_message_fibroblast_report_list* fibroblast_report_messages,
    xmachine_message_fibroblast_report_PBM* partition_matrix){

    /// if agent has no damage, exit function
    if (agent->damage == 0){
        return 0;
    }

    ///get the location of the tissue block
    glm::vec3 tissue_location = glm::vec3(agent->x, agent->y, agent->z);
    
    //Template for input message iteration
    xmachine_message_fibroblast_report* current_message = get_first_fibroblast_report_message(
        fibroblast_report_messages,
        partition_matrix,
        agent->x,
        agent->y,
        agent->z);

    // Count number of messages that are from TissueBlocks
    // that get repaired (i.e. are within REPAIR_RADIUS of
    // fibroblast). If this equals the number of messages
    // that exist
    //
//    int not_repaired_count = 0;
//    int total_number_of_messages = xmachine_message_fibroblast_report_list.size();

    while (current_message)
    {
        //INSERT MESSAGE PROCESSING CODE HERE

        if (current_message->current_state == 2){
            glm::vec3 fibroblast_location = glm::vec3(
                current_message->x,
                current_message->y,
                current_message->z
            );
            float separation = length(tissue_location - fibroblast_location);
            /// if repairative fibroblast within REPAIR_RADIUS
            /// distance, subtract a point of damage. If not
//            float repair_radius = get_REPAIR_RADIUS();
            if (separation < REPAIR_RANGE){
                agent->damage = agent->damage - 1;
            }
        }
        
        current_message = get_next_fibroblast_report_message(current_message, fibroblast_report_messages, partition_matrix);
    }

//    if (not_repaired_count == total_number_of_messages){
//        agent->go_to_state = 1;  // go to quiescent state
//    };
    
    return 0;
}

/**
 * QuiescentMigration FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_Fibroblast. This represents a single agent instance and can be modified directly.
 * @param tissue_damage_report_messages  tissue_damage_report_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_tissue_damage_report_message and get_next_tissue_damage_report_message functions.* @param partition_matrix Pointer to the partition matrix of type xmachine_message_tissue_damage_report_PBM. Used within the get_first__message and get_next__message functions for spatially partitioned message access.* @param fibroblast_report_messages Pointer to output message list of type xmachine_message_fibroblast_report_list. Must be passed as an argument to the add_fibroblast_report_message function.
 */
__FLAME_GPU_FUNC__ int QuiescentMigration(
    xmachine_memory_Fibroblast* agent,
    xmachine_message_tissue_damage_report_list* tissue_damage_report_messages,
    xmachine_message_tissue_damage_report_PBM* partition_matrix,
    xmachine_message_fibroblast_report_list* fibroblast_report_messages){
    
    // Position within space
    float agent_x = agent->x;
    float agent_y = agent->y;
    float agent_z = agent->z;
    glm::vec3 current_fiboblast_location = glm::vec3(agent_x, agent_y, agent_z);

    
    //Template for input message iteration
    xmachine_message_tissue_damage_report* current_message = get_first_tissue_damage_report_message(tissue_damage_report_messages, partition_matrix, agent_x, agent_y, agent_z);

    /// create variable to store tissue block in radius with maximum damage
    xmachine_message_tissue_damage_report *message_with_max_damage = current_message;

    // if no messages exist then the while block is bypassed and the
    // fibroblast location gets updated to the current location
    // Find the message with maximum damage for coordinates.

    /// when do I transition to Repair?
    while (current_message)
    {
        // assign message with maximum damage to variable

        /// Get distance between fibroblast and damaged tissue
        // float separation = length(tissue_location - fibroblast_location);
        // Thnk about including if statement to only execute this
        // part if damaged tissue is below the range of detection for the
        // fibroblast. I think Spatial partitioning will take care of this
        // however so will not do this yet.
        if (current_message->damage > message_with_max_damage->damage) {
            message_with_max_damage = current_message;
        }

        current_message = get_next_tissue_damage_report_message(current_message, tissue_damage_report_messages, partition_matrix);
    }

    /// get position of
    glm::vec3 most_damaged_tissue_location = glm::vec3(
        message_with_max_damage->x,
        message_with_max_damage->y,
        message_with_max_damage->z
    );

    float separation = length(subtract_a_from_b(current_fiboblast_location,most_damaged_tissue_location));

    // if distance between fibroblast and tissue block smaller than REPAIR_RADIUS
    // setup the transition to repair mode.
    if (separation < REPAIR_RANGE){
        agent->go_to_state = 2;

    // otherwise migrate the fibroblast
    }else{
        // get coordinates for the next fibroblast location
        // a + scale*(b-a)
        glm::vec3 next_location = compute_next_fibroblast_location(
            current_fiboblast_location,
            most_damaged_tissue_location,
            QUIESCENT_MIGRATION_SCALE
        );
        // update location
        agent->x = next_location.x;
        agent->y = next_location.y;
        agent->z = next_location.z;
    }

    //Template for message output function
    int id = agent->id;
    float x = agent->x;
    float y = agent->y;
    float z = agent->z;
    int current_state = agent->current_state;
    int go_to_state = agent->go_to_state;

    add_fibroblast_report_message(
        fibroblast_report_messages,
        id, x, y, z, current_state, go_to_state);

    return 0;
}

/**
 * TransitionQuiescentToRepair FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_Fibroblast. This represents a single agent instance and can be modified directly.
 
 */
__FLAME_GPU_FUNC__ int TransitionToRepair(
        xmachine_memory_Fibroblast* agent
    ){

    /// set the currect state variable to 2
    agent->current_state = 2;

    /// set go_to_state variable back to 0
    agent->go_to_state = 0;
    return 0;
}

/**
 * TransitionRepairToQuiescent FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_Fibroblast. This represents a single agent instance and can be modified directly.
 
 */
__FLAME_GPU_FUNC__ int TransitionToQuiescent(xmachine_memory_Fibroblast* agent){
    
    /// set the currect state variable to 2
    agent->current_state = 1;

    /// set go_to_state variable back to 0
    agent->go_to_state = 0;
    return 0;
}

  


#endif //_FLAMEGPU_FUNCTIONS
