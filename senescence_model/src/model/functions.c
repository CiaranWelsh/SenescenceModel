
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

#define TRUE 1
#define FALSE 0

#define TIME_SCALE 0.00001f;
#define STEER_SCALE 0.65f

inline __device__ float dot(glm::vec3 a, glm::vec3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __device__ float magnitude(glm::vec3 v)
{
    return sqrtf(dot(v, v));
}


/**
 * setConstants FLAMEGPU Init function
 * Automatically generated using functions.xslt
 */
__FLAME_GPU_INIT_FUNC__ void setConstants(){
    float TISSUE_DAMAGE_PROB = 0.01f;

//    float MAXIMAL_MIGRATION_RATE =
    float QUIESCENT_MIGRATION_RATE = 1.0f; // 1 = full migration rate. Less than 1 is fractional migrating rate


    set_TISSUE_DAMAGE_PROB(&TISSUE_DAMAGE_PROB);
}

/**
 * TissueTakesDamage FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_TissueBlock. This represents a single agent instance and can be modified directly.
 * @param tissue_damage_report_messages  tissue_damage_report_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_tissue_damage_report_message and get_next_tissue_damage_report_message functions.* @param partition_matrix Pointer to the partition matrix of type xmachine_message_tissue_damage_report_PBM. Used within the get_first__message and get_next__message functions for spatially partitioned message access.* @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an argument to the rand48 function for generating random numbers on the GPU.
 */
__FLAME_GPU_FUNC__ int TissueTakesDamage(
    xmachine_memory_TissueBlock* agent,
    RNG_rand48* rand48){

    float random_number = rnd<CONTINUOUS>(rand48);
    if (random_number >= TISSUE_DAMAGE_PROB)
        if (agent->damage != 0)
            agent->damage = agent->damage - 1;

    return 0;
}

/**
 * TissueSendDamageReport FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_TissueBlock. This represents a single agent instance and can be modified directly.
 * @param tissue_damage_report_messages Pointer to output message list of type xmachine_message_tissue_damage_report_list. Must be passed as an argument to the add_tissue_damage_report_message function.
 */
__FLAME_GPU_FUNC__ int TissueSendDamageReport(
    xmachine_memory_TissueBlock* agent,
    xmachine_message_tissue_damage_report_list* tissue_damage_report_messages){
    

    //Template for message output function
    int id = agent->id;
    float x = agent->x;
    float y = agent->y;
    float z = agent->z;
    int damage = agent->damage;
    
    add_tissue_damage_report_message(tissue_damage_report_messages, id, x, y, z, damage);

    return 0;
}

/**
 * QuiescentMigration FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_Fibroblast. This represents a single agent instance and can be modified directly.
 * @param location_report_messages  location_report_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_location_report_message and get_next_location_report_message functions.* @param partition_matrix Pointer to the partition matrix of type xmachine_message_location_report_PBM. Used within the get_first__message and get_next__message functions for spatially partitioned message access.
 */

__FLAME_GPU_FUNC__ int QuiescentMigration(
    xmachine_memory_Fibroblast* agent,
    xmachine_message_tissue_damage_report_list* tissue_damage_report_messages,
    xmachine_message_tissue_damage_report_PBM* partition_matrix){
    
    /*
    // Position within space
    float agent_x = 0.0;
    float agent_y = 0.0;
    float agent_z = 0.0;
    
    //Template for input message iteration
    xmachine_message_location_report* current_message = get_first_location_report_message(location_report_messages, partition_matrix, agent_x, agent_y, agent_z);
    while (current_message)
    {
        //INSERT MESSAGE PROCESSING CODE HERE
        
        current_message = get_next_location_report_message(current_message, location_report_messages, partition_matrix);
    }
    */
    
    return 0;
}

/**
 * SenescentMigration FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_Fibroblast. This represents a single agent instance and can be modified directly.
 * @param location_report_messages  location_report_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_location_report_message and get_next_location_report_message functions.* @param partition_matrix Pointer to the partition matrix of type xmachine_message_location_report_PBM. Used within the get_first__message and get_next__message functions for spatially partitioned message access.
 */
__FLAME_GPU_FUNC__ int SenescentMigration(
    xmachine_memory_Fibroblast* agent,
    xmachine_message_tissue_damage_report_list* tissue_damage_report_messages,
    xmachine_message_tissue_damage_report_PBM* partition_matrix){
    

    // Position within space
    float agent_x = agent->x;
    float agent_y = agent->y;
    float agent_z = agent->z;

    //Template for input message iteration
    xmachine_message_tissue_damage_report* current_message = get_first_tissue_damage_report_message(
        tissue_damage_report_messages,
        partition_matrix,
        agent_x,
        agent_y,
        agent_z
    );
    xmachine_message_tissue_damage_report* message_with_max_damage = current_message;
    while (current_message)
    {
        //INSERT MESSAGE PROCESSING CODE HERE
        int current_message_damage = current_message->damage;
        // assign message with maximum damage to variable
        if (current_message_damage > message_with_max_damage->damage){
            message_with_max_damage* = current_message;
        }

        current_message = get_next_tissue_damage_report_message(current_message, tissue_damage_report_messages, partition_matrix);
    }

    float most_damaged_fibroblast_x = message_with_max_damage->x;
    float most_damaged_fibroblast_y = message_with_max_damage->y;
    float most_damaged_fibroblast_z = message_with_max_damage->z;

    glm::vec3 most_damaged_fibroblast_location = glm::vec3( most_damaged_fibroblast_x,
                                                            most_damaged_fibroblast_y,
                                                            most_damaged_fibroblast_z);
    return 0;
}


//__FLAME_GPU_FUNC__ int Migration(
//    xmachine_memory_Fibroblast* agent,
//    xmachine_message_tissue_damage_report_list* tissue_damage_report_messages,
//    xmachine_message_tissue_damage_report_PBM* partition_matrix,
//    float migration_rate){
//
//    }

/**
 * EarlySenescentMigration FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_Fibroblast. This represents a single agent instance and can be modified directly.
 * @param location_report_messages  location_report_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_location_report_message and get_next_location_report_message functions.* @param partition_matrix Pointer to the partition matrix of type xmachine_message_location_report_PBM. Used within the get_first__message and get_next__message functions for spatially partitioned message access.
 */
__FLAME_GPU_FUNC__ int EarlySenescentMigration(
    xmachine_memory_Fibroblast* agent,
    xmachine_message_tissue_damage_report_list* tissue_damage_report_messages,
    xmachine_message_tissue_damage_report_PBM* partition_matrix){

    // Position of damaged tissue within space
//    float agent_x = agent.x;
//    float agent_y = agent.y;
//    float agent_z = agent.z;
    
    //Template for input message iteration
//    xmachine_message_tissue_damage_report* current_message = get_first_tissue_damage_report_message(
//        tissue_damage_report_messages,
//        partition_matrix,
//        agent->x,
//        agent->y,
//        agent->z
//     );
//    while (current_message)
//    {
//        //INSERT MESSAGE PROCESSING CODE HERE
//
//        current_message = get_next_tissue_damage_report_message(current_message, tissue_damage_report_messages, partition_matrix);
//    }
//    */
    
    return 0;
}

/**
 * QuiescentTakesDamage FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_Fibroblast. This represents a single agent instance and can be modified directly.
 * @param fibroblast_damage_report_messages  fibroblast_damage_report_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_fibroblast_damage_report_message and get_next_fibroblast_damage_report_message functions.* @param partition_matrix Pointer to the partition matrix of type xmachine_message_fibroblast_damage_report_PBM. Used within the get_first__message and get_next__message functions for spatially partitioned message access.* @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an argument to the rand48 function for generating random numbers on the GPU.
 */
__FLAME_GPU_FUNC__ int QuiescentTakesDamage(xmachine_memory_Fibroblast* agent, xmachine_message_fibroblast_damage_report_list* fibroblast_damage_report_messages, xmachine_message_fibroblast_damage_report_PBM* partition_matrix, RNG_rand48* rand48){
    
    /*
    // Position within space
    float agent_x = 0.0;
    float agent_y = 0.0;
    float agent_z = 0.0;
    
    //Template for input message iteration
    xmachine_message_tissue_damage_report* current_message = get_first_tissue_damage_report_message(tissue_damage_report_messages, partition_matrix, agent_x, agent_y, agent_z);
    while (current_message)
    {
        //INSERT MESSAGE PROCESSING CODE HERE

        current_message = get_next_tissue_damage_report_message(current_message, tissue_damage_report_messages, partition_matrix);
    }
    */
//    */
    
    return 0;
}

/**
 * QuiescentSendDamageReport FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_Fibroblast. This represents a single agent instance and can be modified directly.
 * @param fibroblast_damage_report_messages Pointer to output message list of type xmachine_message_fibroblast_damage_report_list. Must be passed as an argument to the add_fibroblast_damage_report_message function.* @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an argument to the rand48 function for generating random numbers on the GPU.
 */
__FLAME_GPU_FUNC__ int QuiescentSendDamageReport(xmachine_memory_Fibroblast* agent, xmachine_message_fibroblast_damage_report_list* fibroblast_damage_report_messages, RNG_rand48* rand48){
    
    /* 
    //Template for message output function
    int id = 0;
    float x = 0;
    float y = 0;
    float z = 0;
    int damage = 0;
    
    add_fibroblast_damage_report_message(fibroblast_damage_report_messages, id, x, y, z, damage);
    */     
    
    return 0;
}

/**
 * Proliferation FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_Fibroblast. This represents a single agent instance and can be modified directly.
 * @param Fibroblast_agents Pointer to agent list of type xmachine_memory_Fibroblast_list. This must be passed as an argument to the add_Fibroblast_agent function to add a new agent.
 */
__FLAME_GPU_FUNC__ int Proliferation(xmachine_memory_Fibroblast* agent, xmachine_memory_Fibroblast_list* Fibroblast_agents){
    
    /* 
    //Template for agent output functions int id = 0;
    float position_x = 0;
    float position_y = 0;
    float position_z = 0;
    float direction_x = 0;
    float direction_y = 0;
    float direction_z = 0;
    float doublings = 0;
    float damage = 0;
    
    add_Fibroblast_agent(Fibroblast_agents, int id, float position_x, float position_y, float position_z, float direction_x, float direction_y, float direction_z, float doublings, float damage);
    */
    
    return 0;
}

/**
 * ProliferationCompletion FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_Fibroblast. This represents a single agent instance and can be modified directly.
 
 */
__FLAME_GPU_FUNC__ int ProliferationCompletion(xmachine_memory_Fibroblast* agent){
    
    return 0;
}

/**
 * BystanderEffect FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_Fibroblast. This represents a single agent instance and can be modified directly.
 * @param location_report_messages  location_report_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_location_report_message and get_next_location_report_message functions.* @param partition_matrix Pointer to the partition matrix of type xmachine_message_location_report_PBM. Used within the get_first__message and get_next__message functions for spatially partitioned message access.
 */
__FLAME_GPU_FUNC__ int BystanderEffect(xmachine_memory_Fibroblast* agent, xmachine_message_location_report_list* location_report_messages, xmachine_message_location_report_PBM* partition_matrix){
    
    /*
    // Position within space
    float agent_x = 0.0;
    float agent_y = 0.0;
    float agent_z = 0.0;
    
    //Template for input message iteration
    xmachine_message_location_report* current_message = get_first_location_report_message(location_report_messages, partition_matrix, agent_x, agent_y, agent_z);
    while (current_message)
    {
        //INSERT MESSAGE PROCESSING CODE HERE
        
        current_message = get_next_location_report_message(current_message, location_report_messages, partition_matrix);
    }
    */
    
    return 0;
}

/**
 * ExcessiveDamage FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_Fibroblast. This represents a single agent instance and can be modified directly.
 * @param fibroblast_damage_report_messages  fibroblast_damage_report_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_fibroblast_damage_report_message and get_next_fibroblast_damage_report_message functions.* @param partition_matrix Pointer to the partition matrix of type xmachine_message_fibroblast_damage_report_PBM. Used within the get_first__message and get_next__message functions for spatially partitioned message access.
 */
__FLAME_GPU_FUNC__ int ExcessiveDamage(xmachine_memory_Fibroblast* agent, xmachine_message_fibroblast_damage_report_list* fibroblast_damage_report_messages, xmachine_message_fibroblast_damage_report_PBM* partition_matrix){
    
    /*
    // Position within space
    float agent_x = 0.0;
    float agent_y = 0.0;
    float agent_z = 0.0;
    
    //Template for input message iteration
    xmachine_message_fibroblast_damage_report* current_message = get_first_fibroblast_damage_report_message(fibroblast_damage_report_messages, partition_matrix, agent_x, agent_y, agent_z);
    while (current_message)
    {
        //INSERT MESSAGE PROCESSING CODE HERE
        
        current_message = get_next_fibroblast_damage_report_message(current_message, fibroblast_damage_report_messages, partition_matrix);
    }
    */
    
    return 0;
}

/**
 * ReplicativeSenescence FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_Fibroblast. This represents a single agent instance and can be modified directly.
 * @param doublings_messages  doublings_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_doublings_message and get_next_doublings_message functions.
 */
__FLAME_GPU_FUNC__ int ReplicativeSenescence(xmachine_memory_Fibroblast* agent, xmachine_message_doublings_list* doublings_messages){
    
    /*
    //Template for input message iteration
    xmachine_message_doublings* current_message = get_first_doublings_message(doublings_messages);
    while (current_message)
    {
        //INSERT MESSAGE PROCESSING CODE HERE
        
        current_message = get_next_doublings_message(current_message, doublings_messages);
    }
    */
    
    return 0;
}

/**
 * FullSenescence FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_Fibroblast. This represents a single agent instance and can be modified directly.
 * @param count_messages  count_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_count_message and get_next_count_message functions.
 */
__FLAME_GPU_FUNC__ int FullSenescence(xmachine_memory_Fibroblast* agent, xmachine_message_count_list* count_messages){
    
    /*
    //Template for input message iteration
    xmachine_message_count* current_message = get_first_count_message(count_messages);
    while (current_message)
    {
        //INSERT MESSAGE PROCESSING CODE HERE
        
        current_message = get_next_count_message(current_message, count_messages);
    }
    */
    
    return 0;
}

/**
 * ClearanceOfEarlySenescent FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_Fibroblast. This represents a single agent instance and can be modified directly.
 
 */
__FLAME_GPU_FUNC__ int ClearanceOfEarlySenescent(xmachine_memory_Fibroblast* agent){
    
    return 0;
}

/**
 * ClearanceOfSenescent FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_Fibroblast. This represents a single agent instance and can be modified directly.
 
 */
__FLAME_GPU_FUNC__ int ClearanceOfSenescent(xmachine_memory_Fibroblast* agent){
    
    return 0;
}

/**
 * DetectDamage FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_Fibroblast. This represents a single agent instance and can be modified directly.
 * @param tissue_damage_report_messages  tissue_damage_report_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_tissue_damage_report_message and get_next_tissue_damage_report_message functions.* @param partition_matrix Pointer to the partition matrix of type xmachine_message_tissue_damage_report_PBM. Used within the get_first__message and get_next__message functions for spatially partitioned message access.
 */
__FLAME_GPU_FUNC__ int DetectDamage(xmachine_memory_Fibroblast* agent, xmachine_message_tissue_damage_report_list* tissue_damage_report_messages, xmachine_message_tissue_damage_report_PBM* partition_matrix){
    
    /*
    // Position within space
    float agent_x = 0.0;
    float agent_y = 0.0;
    float agent_z = 0.0;
    
    //Template for input message iteration
    xmachine_message_tissue_damage_report* current_message = get_first_tissue_damage_report_message(tissue_damage_report_messages, partition_matrix, agent_x, agent_y, agent_z);
    while (current_message)
    {
        //INSERT MESSAGE PROCESSING CODE HERE
        
        current_message = get_next_tissue_damage_report_message(current_message, tissue_damage_report_messages, partition_matrix);
    }
    */
    
    return 0;
}

/**
 * RepairDamage FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_Fibroblast. This represents a single agent instance and can be modified directly.
 
 */
__FLAME_GPU_FUNC__ int RepairDamage(xmachine_memory_Fibroblast* agent){
    
    return 0;
}

/**
 * DamageRepaired FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_Fibroblast. This represents a single agent instance and can be modified directly.
 
 */
__FLAME_GPU_FUNC__ int DamageRepaired(xmachine_memory_Fibroblast* agent){
    
    return 0;
}

  


#endif //_FLAMEGPU_FUNCTIONS
