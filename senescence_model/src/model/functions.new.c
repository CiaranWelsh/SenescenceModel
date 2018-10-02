
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
 * setConstants FLAMEGPU Init function
 * Automatically generated using functions.xslt
 */
__FLAME_GPU_INIT_FUNC__ void setConstants(){

}

/**
 * TissueTakesDamage FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_TissueBlock. This represents a single agent instance and can be modified directly.
 * @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an argument to the rand48 function for generating random numbers on the GPU.
 */
__FLAME_GPU_FUNC__ int TissueTakesDamage(xmachine_memory_TissueBlock* agent, RNG_rand48* rand48){
    
    return 0;
}

/**
 * TissueSendDamageReport FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_TissueBlock. This represents a single agent instance and can be modified directly.
 * @param tissue_damage_report_messages Pointer to output message list of type xmachine_message_tissue_damage_report_list. Must be passed as an argument to the add_tissue_damage_report_message function.
 */
__FLAME_GPU_FUNC__ int TissueSendDamageReport(xmachine_memory_TissueBlock* agent, xmachine_message_tissue_damage_report_list* tissue_damage_report_messages){
    
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
 * QuiescentMigration FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_Fibroblast. This represents a single agent instance and can be modified directly.
 * @param tissue_damage_report_messages  tissue_damage_report_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_tissue_damage_report_message and get_next_tissue_damage_report_message functions.* @param partition_matrix Pointer to the partition matrix of type xmachine_message_tissue_damage_report_PBM. Used within the get_first__message and get_next__message functions for spatially partitioned message access.
 */
__FLAME_GPU_FUNC__ int QuiescentMigration(xmachine_memory_Fibroblast* agent, xmachine_message_tissue_damage_report_list* tissue_damage_report_messages, xmachine_message_tissue_damage_report_PBM* partition_matrix){
    
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
 * SenescentMigration FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_Fibroblast. This represents a single agent instance and can be modified directly.
 * @param tissue_damage_report_messages  tissue_damage_report_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_tissue_damage_report_message and get_next_tissue_damage_report_message functions.* @param partition_matrix Pointer to the partition matrix of type xmachine_message_tissue_damage_report_PBM. Used within the get_first__message and get_next__message functions for spatially partitioned message access.
 */
__FLAME_GPU_FUNC__ int SenescentMigration(xmachine_memory_Fibroblast* agent, xmachine_message_tissue_damage_report_list* tissue_damage_report_messages, xmachine_message_tissue_damage_report_PBM* partition_matrix){
    
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
 * EarlySenescentMigration FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_Fibroblast. This represents a single agent instance and can be modified directly.
 * @param tissue_damage_report_messages  tissue_damage_report_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_tissue_damage_report_message and get_next_tissue_damage_report_message functions.* @param partition_matrix Pointer to the partition matrix of type xmachine_message_tissue_damage_report_PBM. Used within the get_first__message and get_next__message functions for spatially partitioned message access.
 */
__FLAME_GPU_FUNC__ int EarlySenescentMigration(xmachine_memory_Fibroblast* agent, xmachine_message_tissue_damage_report_list* tissue_damage_report_messages, xmachine_message_tissue_damage_report_PBM* partition_matrix){
    
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
 * Quiescent2Proliferating FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_Fibroblast. This represents a single agent instance and can be modified directly.
 * @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an argument to the rand48 function for generating random numbers on the GPU.
 */
__FLAME_GPU_FUNC__ int Quiescent2Proliferating(xmachine_memory_Fibroblast* agent, RNG_rand48* rand48){
    
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
    float x = 0;
    float y = 0;
    float z = 0;
    float doublings = 0;
    int damage = 0;
    int proliferate_bool = 0;
    int transition_to_early_sen = 0;
    
    add_Fibroblast_agent(Fibroblast_agents, int id, float x, float y, float z, float doublings, int damage, int proliferate_bool, int transition_to_early_sen);
    */
    
    return 0;
}

/**
 * BystanderEffect FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_Fibroblast. This represents a single agent instance and can be modified directly.
 * @param senescent_location_report_messages  senescent_location_report_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_senescent_location_report_message and get_next_senescent_location_report_message functions.* @param partition_matrix Pointer to the partition matrix of type xmachine_message_senescent_location_report_PBM. Used within the get_first__message and get_next__message functions for spatially partitioned message access.* @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an argument to the rand48 function for generating random numbers on the GPU.
 */
__FLAME_GPU_FUNC__ int BystanderEffect(xmachine_memory_Fibroblast* agent, xmachine_message_senescent_location_report_list* senescent_location_report_messages, xmachine_message_senescent_location_report_PBM* partition_matrix, RNG_rand48* rand48){
    
    /*
    // Position within space
    float agent_x = 0.0;
    float agent_y = 0.0;
    float agent_z = 0.0;
    
    //Template for input message iteration
    xmachine_message_senescent_location_report* current_message = get_first_senescent_location_report_message(senescent_location_report_messages, partition_matrix, agent_x, agent_y, agent_z);
    while (current_message)
    {
        //INSERT MESSAGE PROCESSING CODE HERE
        
        current_message = get_next_senescent_location_report_message(current_message, senescent_location_report_messages, partition_matrix);
    }
    */
    
    return 0;
}

/**
 * ExcessiveDamage FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_Fibroblast. This represents a single agent instance and can be modified directly.
 * @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an argument to the rand48 function for generating random numbers on the GPU.
 */
__FLAME_GPU_FUNC__ int ExcessiveDamage(
        xmachine_memory_Fibroblast* agent, RNG_rand48* rand48){
    
    return 0;
}

/**
 * ReplicativeSenescence FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_Fibroblast. This represents a single agent instance and can be modified directly.
 * @param doublings_messages  doublings_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_doublings_message and get_next_doublings_message functions.* @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an argument to the rand48 function for generating random numbers on the GPU.
 */
__FLAME_GPU_FUNC__ int ReplicativeSenescence(xmachine_memory_Fibroblast* agent, xmachine_message_doublings_list* doublings_messages, RNG_rand48* rand48){
    
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
 * TransitionToEarlySen FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_Fibroblast. This represents a single agent instance and can be modified directly.
 
 */
__FLAME_GPU_FUNC__ int TransitionToEarlySen(xmachine_memory_Fibroblast* agent){
    
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
