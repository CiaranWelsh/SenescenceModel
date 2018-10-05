
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


/**
 * declareGlobalConstants FLAMEGPU Init function
 * Automatically generated using functions.xslt
 */
__FLAME_GPU_INIT_FUNC__ void declareGlobalConstants(){

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
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_TissueBlock. This represents a single agent instance and can be modified directly.
 * @param tissue_damage_report_messages Pointer to output message list of type xmachine_message_tissue_damage_report_list. Must be passed as an argument to the add_tissue_damage_report_message function.* @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an argument to the rand48 function for generating random numbers on the GPU.
 */
__FLAME_GPU_FUNC__ int TissueTakesDamage(xmachine_memory_TissueBlock* agent, xmachine_message_tissue_damage_report_list* tissue_damage_report_messages, RNG_rand48* rand48){
    
    /* 
    //Template for message output function
    int id = 0;
    float x = 0;
    float y = 0;
    float z = 0;
    int damage = 0;
    
    add_tissue_damage_report_message(tissue_damage_report_messages, id, x, y, z, damage);
    */     
    
    return 0;
}

/**
 * RepairDamage FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_TissueBlock. This represents a single agent instance and can be modified directly.
 * @param fibroblast_report_messages  fibroblast_report_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_fibroblast_report_message and get_next_fibroblast_report_message functions.* @param partition_matrix Pointer to the partition matrix of type xmachine_message_fibroblast_report_PBM. Used within the get_first__message and get_next__message functions for spatially partitioned message access.
 */
__FLAME_GPU_FUNC__ int RepairDamage(xmachine_memory_TissueBlock* agent, xmachine_message_fibroblast_report_list* fibroblast_report_messages, xmachine_message_fibroblast_report_PBM* partition_matrix){
    
    /*
    // Position within space
    float agent_x = 0.0;
    float agent_y = 0.0;
    float agent_z = 0.0;
    
    //Template for input message iteration
    xmachine_message_fibroblast_report* current_message = get_first_fibroblast_report_message(fibroblast_report_messages, partition_matrix, agent_x, agent_y, agent_z);
    while (current_message)
    {
        //INSERT MESSAGE PROCESSING CODE HERE
        
        current_message = get_next_fibroblast_report_message(current_message, fibroblast_report_messages, partition_matrix);
    }
    */
    
    return 0;
}

/**
 * QuiescentMigration FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_Fibroblast. This represents a single agent instance and can be modified directly.
 * @param tissue_damage_report_messages  tissue_damage_report_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_tissue_damage_report_message and get_next_tissue_damage_report_message functions.* @param partition_matrix Pointer to the partition matrix of type xmachine_message_tissue_damage_report_PBM. Used within the get_first__message and get_next__message functions for spatially partitioned message access.* @param fibroblast_report_messages Pointer to output message list of type xmachine_message_fibroblast_report_list. Must be passed as an argument to the add_fibroblast_report_message function.
 */
__FLAME_GPU_FUNC__ int QuiescentMigration(xmachine_memory_Fibroblast* agent, xmachine_message_tissue_damage_report_list* tissue_damage_report_messages, xmachine_message_tissue_damage_report_PBM* partition_matrix, xmachine_message_fibroblast_report_list* fibroblast_report_messages){
    
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
    
    /* 
    //Template for message output function
    int id = 0;
    float x = 0;
    float y = 0;
    float z = 0;
    int current_state = 0;
    int go_to_state = 0;
    
    add_fibroblast_report_message(fibroblast_report_messages, id, x, y, z, current_state, go_to_state);
    */     
    
    return 0;
}

/**
 * TransitionToRepair FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_Fibroblast. This represents a single agent instance and can be modified directly.
 
 */
__FLAME_GPU_FUNC__ int TransitionToRepair(xmachine_memory_Fibroblast* agent){
    
    return 0;
}

/**
 * TransitionToQuiescent FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_Fibroblast. This represents a single agent instance and can be modified directly.
 
 */
__FLAME_GPU_FUNC__ int TransitionToQuiescent(xmachine_memory_Fibroblast* agent){
    
    return 0;
}

  


#endif //_FLAMEGPU_FUNCTIONS
