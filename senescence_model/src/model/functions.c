
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

inline __device__ float dot_prod(glm::vec3 a, glm::vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __device__ float magnitude_of_vec(glm::vec3 v) {
    return sqrtf(dot(v, v));
}

inline __device__ glm::vec3 subtract_b_from_a(glm::vec3 a, glm::vec3 b)
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
return add_a_to_b(a, (b-a)* scale);
}


/**
 * setConstants FLAMEGPU Init function
 * Automatically generated using functions.xslt
 */
__FLAME_GPU_INIT_FUNC__ void setConstants() {
    float TISSUE_DAMAGE_PROB = 0.1f;
    float EARLY_SENESCENT_MIGRATION_SCALE = 0.1f;
    float SENESCENT_MIGRATION_SCALE = 0.001f;
    float QUIESCENT_MIGRATION_SCALE = 0.0001f;
    float PROLIFERATION_PROB = 0.0001f;

    float BYSTANDER_DISTANCE = 0.1f;
    float BYSTANDER_PROB = 0.1f;

    int EXCESSIVE_DAMAGE_AMOUNT = 100;
    float EXCESSIVE_DAMAGE_PROB = 0.1f;

    int REPLICATIVE_SEN_AGE = 2500;
    float REPLICATIVE_SEN_PROB = 0.1f;

    int EARLY_SENESCENT_MATURATION_TIME = 10;

    float TRANSITION_TO_FULL_SENESCENCE_PROB = 0.1f;

    float CLEARANCE_EARLY_SEN_PROB = 0.1f;
    float CLEARANCE_SEN_PROB = 0.1f;

    float REPAIR_RADIUS = 0.01f;



//    float MAXIMAL_MIGRATION_RATE =
//    float QUIESCENT_MIGRATION_RATE = 1.0f; // 1 = full migration rate. Less than 1 is fractional migrating rate

    set_TISSUE_DAMAGE_PROB(&TISSUE_DAMAGE_PROB);
    set_SENESCENT_MIGRATION_SCALE(&SENESCENT_MIGRATION_SCALE);
    set_EARLY_SENESCENT_MIGRATION_SCALE(&EARLY_SENESCENT_MIGRATION_SCALE);
    set_QUIESCENT_MIGRATION_SCALE(&QUIESCENT_MIGRATION_SCALE);
    set_PROLIFERATION_PROB(&PROLIFERATION_PROB);

    set_BYSTANDER_DISTANCE(&BYSTANDER_DISTANCE);
    set_BYSTANDER_PROB(&BYSTANDER_PROB);

    set_EXCESSIVE_DAMAGE_AMOUNT(&EXCESSIVE_DAMAGE_AMOUNT);
    set_EXCESSIVE_DAMAGE_PROB(&EXCESSIVE_DAMAGE_PROB);

    set_REPLICATIVE_SEN_AGE(&REPLICATIVE_SEN_AGE);
    set_REPLICATIVE_SEN_PROB(&REPLICATIVE_SEN_PROB);

    set_EARLY_SENESCENT_MATURATION_TIME (&EARLY_SENESCENT_MATURATION_TIME);
    set_TRANSITION_TO_FULL_SENESCENCE_PROB(&TRANSITION_TO_FULL_SENESCENCE_PROB);

    set_CLEARANCE_EARLY_SEN_PROB(&CLEARANCE_EARLY_SEN_PROB);
    set_CLEARANCE_SEN_PROB(&CLEARANCE_SEN_PROB);
    set_REPAIR_RADIUS(&REPAIR_RADIUS);
}

/**
 * logs FLAMEGPU Step function
 * Automatically generated using functions.xslt
 */
__FLAME_GPU_STEP_FUNC__ void Tissuelogs(){

    // Get some values and construct an output path.
    const char * directory = getOutputDir();

//    fprintf(stdout, todirectory);

    unsigned int iteration = getIterationNumber();

    std::string outputFilename = std::string( std::string(directory) + "results/tissue/" + "tissue-logs-" + std::to_string(iteration) +".csv");

    // Get a file handle for output.
    FILE * fp = fopen(outputFilename.c_str(), "w");
    // If the file has been opened successfully
    if(fp != nullptr){
        fprintf(stdout, "Outputting some Tissue agent data to %s\n", outputFilename.c_str());

        // Output a header row for the CSV
        fprintf(fp, "ID, x, y, z, damage\n");

        // For each agent of a target type in a target state
        for(int index = 0; index < get_agent_TissueBlock_default_count(); index++){
            // Append a row to the CSV file.
            fprintf(
                    fp,
                    "%u, %f, %f, %f, %u\n",
                    get_TissueBlock_default_variable_id(index),
                    get_TissueBlock_default_variable_x(index),
                    get_TissueBlock_default_variable_y(index),
                    get_TissueBlock_default_variable_z(index),
                    get_TissueBlock_default_variable_damage(index)
            );
        }
        // Flush the file handle
        fflush(fp);
//        fprintf(stderr, "debug: file %s was created for customOutputStepFunction\n", outputFilename.c_str());

    } else {
        fprintf(stderr, "Error: file %s could not be created for customOutputStepFunction\n", outputFilename.c_str());
    }
    // Close the file handle if necessary.
    if (fp != nullptr && fp != stdout && fp != stderr){
        fclose(fp);
        fp = nullptr;
    }
}

__FLAME_GPU_STEP_FUNC__ void FibroblastQuiescentlogs(){

    // Get some values and construct an output path.
    const char * directory = getOutputDir();

//    fprintf(stdout, todirectory);

    unsigned int iteration = getIterationNumber();

    std::string outputFilename = std::string( std::string(directory) + "results/quiescent/"+"fibroblast-quiescent-logs-" + std::to_string(iteration) +".csv");

    // Get a file handle for output.
    FILE * fp = fopen(outputFilename.c_str(), "w");
    // If the file has been opened successfully
    if(fp != nullptr){
        fprintf(stdout, "Outputting some agent data to %s\n", outputFilename.c_str());

        // Output a header row for the CSV
        fprintf(fp, "ID, x, y, z, doublings, damage, early_sen_time_counter\n");

        // For each agent of a target type in a target state
        for(int index = 0; index < get_agent_Fibroblast_Quiescent_count(); index++){
            // Append a row to the CSV file.
            fprintf(
                    fp,
                    "%u, %f, %f, %f, %f, %u, %u \n",
                    get_Fibroblast_Quiescent_variable_id(index),
                    get_Fibroblast_Quiescent_variable_x(index),
                    get_Fibroblast_Quiescent_variable_y(index),
                    get_Fibroblast_Quiescent_variable_z(index),
                    get_Fibroblast_Quiescent_variable_doublings(index),
                    get_Fibroblast_Quiescent_variable_damage(index),
                    get_Fibroblast_Quiescent_variable_early_sen_time_counter(index)
            );
        }
        // Flush the file handle
        fflush(fp);
//        fprintf(stderr, "debug: file %s was created for customOutputStepFunction\n", outputFilename.c_str());

    } else {
        fprintf(stderr, "Error: file %s could not be created for customOutputStepFunction\n", outputFilename.c_str());
    }
    // Close the file handle if necessary.
    if (fp != nullptr && fp != stdout && fp != stderr){
        fclose(fp);
        fp = nullptr;
    }
}


/**
 * FibroblastEarlySenescentlogs FLAMEGPU Step function
 * Automatically generated using functions.xslt
 */
__FLAME_GPU_STEP_FUNC__ void FibroblastEarlySenescentlogs(){

    // Get some values and construct an output path.
    const char * directory = getOutputDir();

//    fprintf(stdout, todirectory);

    unsigned int iteration = getIterationNumber();

    std::string outputFilename = std::string( std::string(directory) + "results/early-sen/"+"fibroblast-early-senescent-logs-" + std::to_string(iteration) +".csv");

    // Get a file handle for output.
    FILE * fp = fopen(outputFilename.c_str(), "w");
    // If the file has been opened successfully
    if(fp != nullptr){
        fprintf(stdout, "Outputting some agent data to %s\n", outputFilename.c_str());

        // Output a header row for the CSV
        fprintf(fp, "ID, x, y, z, doublings, damage, early_sen_time_counter\n");

        // For each agent of a target type in a target state
        for(int index = 0; index < get_agent_Fibroblast_EarlySenescent_count(); index++){
            // Append a row to the CSV file.
            fprintf(
                    fp,
                    "%u, %f, %f, %f, %f, %u, %u \n",
                    get_Fibroblast_EarlySenescent_variable_id(index),
                    get_Fibroblast_EarlySenescent_variable_x(index),
                    get_Fibroblast_EarlySenescent_variable_y(index),
                    get_Fibroblast_EarlySenescent_variable_z(index),
                    get_Fibroblast_EarlySenescent_variable_doublings(index),
                    get_Fibroblast_EarlySenescent_variable_damage(index),
                    get_Fibroblast_EarlySenescent_variable_early_sen_time_counter(index)
            );
        }
        // Flush the file handle
        fflush(fp);
//        fprintf(stderr, "debug: file %s was created for customOutputStepFunction\n", outputFilename.c_str());

    } else {
        fprintf(stderr, "Error: file %s could not be created for customOutputStepFunction\n", outputFilename.c_str());
    }
    // Close the file handle if necessary.
    if (fp != nullptr && fp != stdout && fp != stderr){
        fclose(fp);
        fp = nullptr;
    }
}


/**
 * FibroblastSenescentlogs FLAMEGPU Step function
 * Automatically generated using functions.xslt
 */
__FLAME_GPU_STEP_FUNC__ void FibroblastSenescentlogs(){

    // Get some values and construct an output path.
    const char * directory = getOutputDir();

//    fprintf(stdout, todirectory);

    unsigned int iteration = getIterationNumber();

    std::string outputFilename = std::string( std::string(directory) + "results/sen/"+"fibroblast-senescent-logs-" + std::to_string(iteration) +".csv");

    // Get a file handle for output.
    FILE * fp = fopen(outputFilename.c_str(), "w");
    // If the file has been opened successfully
    if(fp != nullptr){
        fprintf(stdout, "Outputting some agent data to %s\n", outputFilename.c_str());

        // Output a header row for the CSV
        fprintf(fp, "ID, x, y, z, doublings, damage, early_sen_time_counter\n");

        // For each agent of a target type in a target state
        for(int index = 0; index < get_agent_Fibroblast_Senescent_count(); index++){
            // Append a row to the CSV file.
            fprintf(
                    fp,
                    "%u, %f, %f, %f, %f, %u, %u \n",
                    get_Fibroblast_Senescent_variable_id(index),
                    get_Fibroblast_Senescent_variable_x(index),
                    get_Fibroblast_Senescent_variable_y(index),
                    get_Fibroblast_Senescent_variable_z(index),
                    get_Fibroblast_Senescent_variable_doublings(index),
                    get_Fibroblast_Senescent_variable_damage(index),
                    get_Fibroblast_Senescent_variable_early_sen_time_counter(index)
            );
        }
        // Flush the file handle
        fflush(fp);
//        fprintf(stderr, "debug: file %s was created for customOutputStepFunction\n", outputFilename.c_str());

    } else {
        fprintf(stderr, "Error: file %s could not be created for customOutputStepFunction\n", outputFilename.c_str());
    }
    // Close the file handle if necessary.
    if (fp != nullptr && fp != stdout && fp != stderr){
        fclose(fp);
        fp = nullptr;
    }
}


/**
 * FibroblastProliferatinglogs FLAMEGPU Step function
 * Automatically generated using functions.xslt
 */
__FLAME_GPU_STEP_FUNC__ void FibroblastProliferatinglogs(){

    // Get some values and construct an output path.
    const char * directory = getOutputDir();

//    fprintf(stdout, todirectory);

    unsigned int iteration = getIterationNumber();

    std::string outputFilename = std::string( std::string(directory) + "results/prolif/"+"fibroblast-proliferating-logs-" + std::to_string(iteration) +".csv");

    // Get a file handle for output.
    FILE * fp = fopen(outputFilename.c_str(), "w");
    // If the file has been opened successfully
    if(fp != nullptr){
        fprintf(stdout, "Outputting some agent data to %s\n", outputFilename.c_str());

        // Output a header row for the CSV
        fprintf(fp, "ID, x, y, z, doublings, damage, early_sen_time_counter\n");

        // For each agent of a target type in a target state
        for(int index = 0; index < get_agent_Fibroblast_Proliferating_count(); index++){
            // Append a row to the CSV file.
            fprintf(
                    fp,
                    "%u, %f, %f, %f, %f, %u, %u \n",
                    get_Fibroblast_Proliferating_variable_id(index),
                    get_Fibroblast_Proliferating_variable_x(index),
                    get_Fibroblast_Proliferating_variable_y(index),
                    get_Fibroblast_Proliferating_variable_z(index),
                    get_Fibroblast_Proliferating_variable_doublings(index),
            get_Fibroblast_Proliferating_variable_damage(index),
            get_Fibroblast_Proliferating_variable_early_sen_time_counter(index)
            );
        }
        // Flush the file handle
        fflush(fp);
//        fprintf(stderr, "debug: file %s was created for customOutputStepFunction\n", outputFilename.c_str());

    } else {
        fprintf(stderr, "Error: file %s could not be created for customOutputStepFunction\n", outputFilename.c_str());
    }
    // Close the file handle if necessary.
    if (fp != nullptr && fp != stdout && fp != stderr){
        fclose(fp);
        fp = nullptr;
    }
}


/**
 * FibroblastRepairlogs FLAMEGPU Step function
 * Automatically generated using functions.xslt
 */
__FLAME_GPU_STEP_FUNC__ void FibroblastRepairlogs(){

    // Get some values and construct an output path.
    const char * directory = getOutputDir();

//    fprintf(stdout, todirectory);

    unsigned int iteration = getIterationNumber();

    std::string outputFilename = std::string( std::string(directory) + "results/repair/"+"fibroblast-repair-logs-" + std::to_string(iteration) +".csv");

    // Get a file handle for output.
    FILE * fp = fopen(outputFilename.c_str(), "w");
    // If the file has been opened successfully
    if(fp != nullptr){
        fprintf(stdout, "Outputting some agent data to %s\n", outputFilename.c_str());

        // Output a header row for the CSV
        fprintf(fp, "ID, x, y, z, doublings, damage, early_sen_time_counter\n");

        // For each agent of a target type in a target state
        for(int index = 0; index < get_agent_Fibroblast_Repair_count(); index++){
            // Append a row to the CSV file.
            fprintf(
                    fp,
                    "%u, %f, %f, %f, %f, %u, %u \n",
                    get_Fibroblast_Repair_variable_id(index),
                    get_Fibroblast_Repair_variable_x(index),
                    get_Fibroblast_Repair_variable_y(index),
                    get_Fibroblast_Repair_variable_z(index),
                    get_Fibroblast_Repair_variable_doublings(index),
            get_Fibroblast_Repair_variable_damage(index),
            get_Fibroblast_Repair_variable_early_sen_time_counter(index)
            );
        }
        // Flush the file handle
        fflush(fp);
//        fprintf(stderr, "debug: file %s was created for customOutputStepFunction\n", outputFilename.c_str());

    } else {
        fprintf(stderr, "Error: file %s could not be created for customOutputStepFunction\n", outputFilename.c_str());
    }
    // Close the file handle if necessary.
    if (fp != nullptr && fp != stdout && fp != stderr){
        fclose(fp);
        fp = nullptr;
    }
}


/**
 * TissueTakesDamage FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_TissueBlock. This represents a single agent instance and can be modified directly.
 * @param tissue_damage_report_messages  tissue_damage_report_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_tissue_damage_report_message and get_next_tissue_damage_report_message functions.* @param partition_matrix Pointer to the partition matrix of type xmachine_message_tissue_damage_report_PBM. Used within the get_first__message and get_next__message functions for spatially partitioned message access.* @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an argument to the rand48 function for generating random numbers on the GPU.
 */
__FLAME_GPU_FUNC__ int TissueTakesDamage(
        xmachine_memory_TissueBlock *agent,
        RNG_rand48 *rand48) {

    float random_number = rnd<CONTINUOUS>(rand48);
    if (random_number < TISSUE_DAMAGE_PROB)
        agent->damage = agent->damage + 1;

    return 0;
}

/**
 * TissueSendDamageReport FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_TissueBlock. This represents a single agent instance and can be modified directly.
 * @param tissue_damage_report_messages Pointer to output message list of type xmachine_message_tissue_damage_report_list. Must be passed as an argument to the add_tissue_damage_report_message function.
 */
__FLAME_GPU_FUNC__ int TissueSendDamageReport(
        xmachine_memory_TissueBlock *agent,
        xmachine_message_tissue_damage_report_list *tissue_damage_report_messages) {


    //Template for message output function
    int id = agent->id+1000;
    float x = agent->x;
    float y = agent->y;
    float z = agent->z;
    int damage = agent->damage;

    add_tissue_damage_report_message(
            tissue_damage_report_messages, id, x, y, z, damage);

    return 0;
}


/**
 * ReapirDamage FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_TissueBlock. This represents a single agent instance and can be modified directly.
 * @param fibroblast_location_report_messages  fibroblast_location_report_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_fibroblast_location_report_message and get_next_fibroblast_location_report_message functions.* @param partition_matrix Pointer to the partition matrix of type xmachine_message_fibroblast_location_report_PBM. Used within the get_first__message and get_next__message functions for spatially partitioned message access.
 */
__FLAME_GPU_FUNC__ int ReapirDamage(xmachine_memory_TissueBlock* agent, xmachine_message_fibroblast_location_report_list* fibroblast_location_report_messages, xmachine_message_fibroblast_location_report_PBM* partition_matrix){

    /// if agent has no damage, exit function
    if (agent->damage == 0){
        return 0;
    }

    glm::vec3 tissue_location = glm::vec3(agent->x,agent->y, agent->z);

    xmachine_message_fibroblast_location_report* current_message = get_first_fibroblast_location_report_message(
            fibroblast_location_report_messages, partition_matrix,
            agent->x, agent->y, agent->z);
    while (current_message)
    {
        /// if fibroblast in repairative state continue
        if (current_message->current_state == 5){
            glm::vec3 fibroblast_location = glm::vec3(
                    current_message->x,
                    current_message->y,
                    current_message->z);
            float separation = magnitude_of_vec(tissue_location - fibroblast_location);
            /// if repairative fibroblast within REPAIR_RADIUS distance, subtract a point of damage
            if (separation < REPAIR_RADIUS){
                agent->damage = agent->damage - 1;
            }


        }

        current_message = get_next_fibroblast_location_report_message(current_message, fibroblast_location_report_messages, partition_matrix);
    }


    return 0;
}


/**
 * QuiescentMigration FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_Fibroblast. This represents a single agent instance and can be modified directly.
 * @param location_report_messages  location_report_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_location_report_message and get_next_location_report_message functions.* @param partition_matrix Pointer to the partition matrix of type xmachine_message_location_report_PBM. Used within the get_first__message and get_next__message functions for spatially partitioned message access.
 */

__FLAME_GPU_FUNC__ int QuiescentMigration(
        xmachine_memory_Fibroblast *agent,
        xmachine_message_tissue_damage_report_list *tissue_damage_report_messages,
        xmachine_message_tissue_damage_report_PBM *partition_matrix) {


    // Position within space
    float agent_x = agent->x;
    float agent_y = agent->y;
    float agent_z = agent->z;
    glm::vec3 current_fiboblast_location = glm::vec3(agent_x, agent_y, agent_z);

    //Template for input message iteration
    xmachine_message_tissue_damage_report *current_message = get_first_tissue_damage_report_message(
            tissue_damage_report_messages,
            partition_matrix,
            agent_x,
            agent_y,
            agent_z
    );
    xmachine_message_tissue_damage_report *message_with_max_damage = current_message;
    while (current_message) {
        //INSERT MESSAGE PROCESSING CODE HERE
        int current_message_damage = current_message->damage;
        // assign message with maximum damage to variable
        if (current_message_damage > message_with_max_damage->damage) {
            message_with_max_damage = current_message;
        }
        current_message = get_next_tissue_damage_report_message(current_message, tissue_damage_report_messages,
                                                                partition_matrix);
    }

    float most_damaged_tissue_x = message_with_max_damage->x;
    float most_damaged_tissue_y = message_with_max_damage->y;
    float most_damaged_tissue_z = message_with_max_damage->z;

    glm::vec3 most_damaged_tissue_location = glm::vec3(most_damaged_tissue_x,
                                                       most_damaged_tissue_y,
                                                       most_damaged_tissue_z);

//    a + scale*(b-a)
    glm::vec3 next_location = compute_next_fibroblast_location(
            current_fiboblast_location,
            most_damaged_tissue_location,
            QUIESCENT_MIGRATION_SCALE
    );

    agent->x = next_location.x;
    agent->y = next_location.y;
    agent->z = next_location.z;

    return 0;
}

/**
 * SenescentMigration FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_Fibroblast. This represents a single agent instance and can be modified directly.
 * @param location_report_messages  location_report_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_location_report_message and get_next_location_report_message functions.* @param partition_matrix Pointer to the partition matrix of type xmachine_message_location_report_PBM. Used within the get_first__message and get_next__message functions for spatially partitioned message access.
 */
__FLAME_GPU_FUNC__ int SenescentMigration(
        xmachine_memory_Fibroblast *agent,
        xmachine_message_tissue_damage_report_list *tissue_damage_report_messages,
        xmachine_message_tissue_damage_report_PBM *partition_matrix) {


    // Position within space
    float agent_x = agent->x;
    float agent_y = agent->y;
    float agent_z = agent->z;
    glm::vec3 current_fiboblast_location = glm::vec3(agent_x, agent_y, agent_z);

    //Template for input message iteration
    xmachine_message_tissue_damage_report *current_message = get_first_tissue_damage_report_message(
            tissue_damage_report_messages,
            partition_matrix,
            agent_x,
            agent_y,
            agent_z
    );
    xmachine_message_tissue_damage_report *message_with_max_damage = current_message;
    while (current_message) {
        //INSERT MESSAGE PROCESSING CODE HERE
        int current_message_damage = current_message->damage;
        // assign message with maximum damage to variable
        if (current_message_damage > message_with_max_damage->damage) {
            message_with_max_damage = current_message;
        }
        current_message = get_next_tissue_damage_report_message(current_message, tissue_damage_report_messages,
                                                                partition_matrix);
    }

    float most_damaged_tissue_x = message_with_max_damage->x;
    float most_damaged_tissue_y = message_with_max_damage->y;
    float most_damaged_tissue_z = message_with_max_damage->z;

    glm::vec3 most_damaged_tissue_location = glm::vec3(most_damaged_tissue_x,
                                                       most_damaged_tissue_y,
                                                       most_damaged_tissue_z);

//    a + scale*(b-a)
    glm::vec3 next_location = compute_next_fibroblast_location(
            current_fiboblast_location,
            most_damaged_tissue_location,
            SENESCENT_MIGRATION_SCALE
    );

    agent->x = next_location.x;
    agent->y = next_location.y;
    agent->z = next_location.z;

    return 0;
}

/**
 * EarlySenescentMigration FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_Fibroblast. This represents a single agent instance and can be modified directly.
 * @param location_report_messages  location_report_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_location_report_message and get_next_location_report_message functions.* @param partition_matrix Pointer to the partition matrix of type xmachine_message_location_report_PBM. Used within the get_first__message and get_next__message functions for spatially partitioned message access.
 */
__FLAME_GPU_FUNC__ int EarlySenescentMigration(
        xmachine_memory_Fibroblast *agent,
        xmachine_message_tissue_damage_report_list *tissue_damage_report_messages,
        xmachine_message_tissue_damage_report_PBM *partition_matrix) {


    // Position within space
    float agent_x = agent->x;
    float agent_y = agent->y;
    float agent_z = agent->z;
    glm::vec3 current_fiboblast_location = glm::vec3(agent_x, agent_y, agent_z);

    //Template for input message iteration
    xmachine_message_tissue_damage_report *current_message = get_first_tissue_damage_report_message(
            tissue_damage_report_messages,
            partition_matrix,
            agent_x,
            agent_y,
            agent_z
    );
    xmachine_message_tissue_damage_report *message_with_max_damage = current_message;
    while (current_message) {
        //INSERT MESSAGE PROCESSING CODE HERE
        int current_message_damage = current_message->damage;
        // assign message with maximum damage to variable
        if (current_message_damage > message_with_max_damage->damage) {
            message_with_max_damage = current_message;
        }
        current_message = get_next_tissue_damage_report_message(current_message, tissue_damage_report_messages,
                                                                partition_matrix);
    }

    float most_damaged_tissue_x = message_with_max_damage->x;
    float most_damaged_tissue_y = message_with_max_damage->y;
    float most_damaged_tissue_z = message_with_max_damage->z;

    glm::vec3 most_damaged_tissue_location = glm::vec3(most_damaged_tissue_x,
                                                       most_damaged_tissue_y,
                                                       most_damaged_tissue_z);

//    a + scale*(b-a)
    glm::vec3 next_location = compute_next_fibroblast_location(
            current_fiboblast_location,
            most_damaged_tissue_location,
            EARLY_SENESCENT_MIGRATION_SCALE
    );

    agent->x = next_location.x;
    agent->y = next_location.y;
    agent->z = next_location.z;


    return 0;
}

/**
 * QuiescentTakesDamage FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_Fibroblast. This represents a single agent instance and can be modified directly.
 * @param fibroblast_damage_report_messages  fibroblast_damage_report_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_fibroblast_damage_report_message and get_next_fibroblast_damage_report_message functions.* @param partition_matrix Pointer to the partition matrix of type xmachine_message_fibroblast_damage_report_PBM. Used within the get_first__message and get_next__message functions for spatially partitioned message access.* @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an argument to the rand48 function for generating random numbers on the GPU.
 */
__FLAME_GPU_FUNC__ int QuiescentTakesDamage(
        xmachine_memory_Fibroblast *agent,
        RNG_rand48 *rand48) {


    float random_number = rnd<CONTINUOUS>(rand48);
    if (random_number < TISSUE_DAMAGE_PROB)
        agent->damage = agent->damage + 1;

    return 0;
}

/**
 * TransitionToProliferating FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_Fibroblast. This represents a single agent instance and can be modified directly.
 * @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an argument to the rand48 function for generating random numbers on the GPU.
    0: quiescent
    1: early senescent
    2: senescent
    4: proliferating
    5: repairing

 */
__FLAME_GPU_FUNC__ int TransitionToProliferating(
        xmachine_memory_Fibroblast *agent,
        RNG_rand48 *rand48) {

    float random_number = rnd<CONTINUOUS>(rand48);
    if (random_number < PROLIFERATION_PROB) {
        agent->current_state = 4;
        agent->colour = 4;
    }

    return 0;
}

/**
 * Proliferation FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_Fibroblast. This represents a single agent instance and can be modified directly.
 * @param Fibroblast_agents Pointer to agent list of type xmachine_memory_Fibroblast_list. This must be passed as an argument to the add_Fibroblast_agent function to add a new agent.
 */
__FLAME_GPU_FUNC__ int Proliferation(
        xmachine_memory_Fibroblast *agent,
        xmachine_memory_Fibroblast_list *Fibroblast_agents) {

    // When spawning new agent from existing, how do you ensure unique IDs?
    int id=agent->id;
    float x = agent->x;
    float y = agent->y;
    float z = agent->z;
    float doublings = 0;
    int damage = 0;
    int early_sen_time_counter = 0;
    int current_state = 0;
    int colour = 0;

    add_Fibroblast_agent(
            Fibroblast_agents,
            id, x, y, z,
            doublings,
            damage,
            early_sen_time_counter,
            current_state,
            colour);

    agent->current_state = 0;
    agent->colour = 0;
    agent->doublings += 1;
    return 0;
}


/**
 * BystanderEffect FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_Fibroblast. This represents a single agent instance and can be modified directly.
 * @param fibroblast_location_report_messages  fibroblast_location_report_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_fibroblast_location_report_message and get_next_fibroblast_location_report_message functions.* @param partition_matrix Pointer to the partition matrix of type xmachine_message_fibroblast_location_report_PBM. Used within the get_first__message and get_next__message functions for spatially partitioned message access.

     0: quiescent
    1: early senescent
    2: senescent
    4: proliferating
    5: repairing
 */
__FLAME_GPU_FUNC__ int BystanderEffect(
        xmachine_memory_Fibroblast *agent,
        xmachine_message_fibroblast_location_report_list *fibroblast_location_report_messages,
        xmachine_message_fibroblast_location_report_PBM *partition_matrix,
        RNG_rand48* rand48) {

    // Position within space
    float agent_x = agent->x;
    float agent_y = agent->y;
    float agent_z = agent->z;

    glm::vec3 fibroblast_loc = glm::vec3(agent_x, agent_y, agent_z);

    //Template for input message iteration
    xmachine_message_fibroblast_location_report *current_message = get_first_fibroblast_location_report_message(
            fibroblast_location_report_messages,
            partition_matrix,
            agent_x, agent_y, agent_z);

    while (current_message) {
        if (current_message->current_state == 2) {

            glm::vec3 senescent_fib_loc = glm::vec3(
                    current_message->x,
                    current_message->y,
                    current_message->z);

            //        glm::vec3 distance = subtract_b_from_a(fibroblast_loc, senescent_fib_loc);
            float separation = length(senescent_fib_loc - fibroblast_loc);
            if (separation > BYSTANDER_DISTANCE) {
                float random_number = rnd<CONTINUOUS>(rand48);
                if (random_number < BYSTANDER_PROB) {
                    agent->current_state = 1;
                }
            }
        }
        current_message = get_next_fibroblast_location_report_message(
                current_message,
                fibroblast_location_report_messages,
                partition_matrix);
    }
    return 0;
}


/**
 * ExcessiveDamage FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_Fibroblast. This represents a single agent instance and can be modified directly.
 * @param fibroblast_damage_report_messages  fibroblast_damage_report_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_fibroblast_damage_report_message and get_next_fibroblast_damage_report_message functions.* @param partition_matrix Pointer to the partition matrix of type xmachine_message_fibroblast_damage_report_PBM. Used within the get_first__message and get_next__message functions for spatially partitioned message access.
 */
__FLAME_GPU_FUNC__ int ExcessiveDamage(
        xmachine_memory_Fibroblast* agent,
        RNG_rand48* rand48) {

    if (agent->damage > EXCESSIVE_DAMAGE_AMOUNT) {
        float random_number = rnd<CONTINUOUS>(rand48);
        if (random_number < EXCESSIVE_DAMAGE_PROB){
            agent->current_state = 1;
        }
    };

    return 0;
}

/**
 * ReplicativeSenescence FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_Fibroblast. This represents a single agent instance and can be modified directly.
 * @param doublings_messages  doublings_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_doublings_message and get_next_doublings_message functions.
 */
__FLAME_GPU_FUNC__ int ReplicativeSenescence(
        xmachine_memory_Fibroblast* agent,
        RNG_rand48* rand48){

    if (agent->doublings > REPLICATIVE_SEN_AGE) {
        float random_number = rnd<CONTINUOUS>(rand48);
        if (random_number < REPLICATIVE_SEN_PROB){
            agent->current_state = 1;
        }
    };
    return 0;
}


/**
 * EarlySenCountTime FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_Fibroblast. This represents a single agent instance and can be modified directly.

 */
__FLAME_GPU_FUNC__ int EarlySenCountTime(xmachine_memory_Fibroblast* agent){
    if (agent->early_sen_time_counter < EARLY_SENESCENT_MATURATION_TIME ){
        agent->early_sen_time_counter += 1;
    }
    return 0;
}


/**
 * FullSenescence FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_Fibroblast. This represents a single agent instance and can be modified directly.

    0: quiescent
    1: early senescent
    2: senescent
    4: proliferating
    5: repairing
 */
__FLAME_GPU_FUNC__ int TransitionToFullSenescence(
        xmachine_memory_Fibroblast* agent,
        RNG_rand48 *rand48){

    float random_number = rnd<CONTINUOUS>(rand48);
    if (random_number < TRANSITION_TO_FULL_SENESCENCE_PROB) {
        agent->current_state = 2;
        agent->colour = 2;
    }

    return 0;
}


/**
 * ClearanceOfEarlySenescent FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_Fibroblast. This represents a single agent instance and can be modified directly.
 * @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an argument to the rand48 function for generating random numbers on the GPU.

     0: quiescent
    1: early senescent
    2: senescent
    4: proliferating
    5: repairing
 */
__FLAME_GPU_FUNC__ int ClearanceOfEarlySenescent(xmachine_memory_Fibroblast* agent, RNG_rand48* rand48){
    if (agent->current_state == 1){
        if (rnd<CONTINUOUS>(rand48) < CLEARANCE_EARLY_SEN_PROB){
            return 1; /// non 0 exit status marks agent for removal
        }
    }
    return 0;
}



/**
 * ClearanceOfSenescent FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_Fibroblast. This represents a single agent instance and can be modified directly.
 * @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an argument to the rand48 function for generating random numbers on the GPU.
 */
__FLAME_GPU_FUNC__ int ClearanceOfSenescent(xmachine_memory_Fibroblast* agent, RNG_rand48* rand48){

    if (agent->current_state == 1){
        if (rnd<CONTINUOUS>(rand48) < CLEARANCE_SEN_PROB){
            return 1; /// non 0 exit status marks agent for removal
        }
    }
    return 0;
}

/**
 * DetectDamage FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_Fibroblast. This represents a single agent instance and can be modified directly.
 * @param tissue_damage_report_messages  tissue_damage_report_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_tissue_damage_report_message and get_next_tissue_damage_report_message functions.* @param partition_matrix Pointer to the partition matrix of type xmachine_message_tissue_damage_report_PBM. Used within the get_first__message and get_next__message functions for spatially partitioned message access.
 */
__FLAME_GPU_FUNC__ int DetectDamage(
        xmachine_memory_Fibroblast* agent,
        xmachine_message_tissue_damage_report_list* tissue_damage_report_messages,
        xmachine_message_tissue_damage_report_PBM* partition_matrix){


    // Position within space
//    float agent_x = 0.0;
//    float agent_y = 0.0;
//    float agent_z = 0.0;
    glm::vec3 fibroblast_position = glm::vec3(
            agent->x, agent->y, agent->z
            );

    //Template for input message iteration
    xmachine_message_tissue_damage_report* current_message = get_first_tissue_damage_report_message(
            tissue_damage_report_messages, partition_matrix, agent->x, agent->y, agent->z);
    while (current_message)
    {
        //INSERT MESSAGE PROCESSING CODE HERE
        glm::vec3 damaged_tissue_position = glm::vec3(
                current_message->x,current_message->y, current_message->z
                );
        float separation = magnitude_of_vec(fibroblast_position - damaged_tissue_position);

        if (separation < REPAIR_RADIUS){
            agent->current_state = 5;
            agent->colour = 5;
        }

        current_message = get_next_tissue_damage_report_message(
                current_message,
                tissue_damage_report_messages,
                partition_matrix);
    }


    return 0;
}














#endif //_FLAMEGPU_FUNCTIONS
